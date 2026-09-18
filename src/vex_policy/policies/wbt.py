from __future__ import annotations

import json
import sys
from enum import StrEnum

import numpy as np
import onnxruntime
from loguru import logger
from termcolor import colored

from vex_policy.config.config_types.inference import InferenceConfig
from vex_policy.policies.base import BasePolicy, PolicyRuntimeFault
from vex_policy.policies.guard.wbt import WbtGuard
from vex_policy.policies.utils.inference import load_metadata, resolve_control_gains
from vex_policy.policies.utils.joint_command import PositionAction, position_command
from vex_policy.policies.observations import ObservationHistory
from vex_policy.policies.utils.wbt_utils import MotionClockUtil, NpzTargetSource, PinocchioRobot, TimestepUtil
from vex_policy.robots import G1_JOINT_LOWER, G1_JOINT_UPPER, G1_JOINT_VELOCITY
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.clock import ClockSub
from vex_policy.utils.joint_interpolation import JointPositionInterpolator
from vex_policy.utils.latency import LatencyStage
from vex_policy.utils.math.quat import (
    matrix_from_quat,
    quat_mul,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class WbtStage(StrEnum):
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    TRACKING = "tracking"


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.observations = ObservationHistory(config.observation)
        self.actions = PositionAction(
            self.num_dofs,
            self.action_mask,
            require_full_body=config.action_mask is not None,
            force_zero=config.task.debug.force_zero_action,
        )
        self._stage = WbtStage.INACTIVE
        self._target_source = None

        # initialize motion state
        self.motion_clip_progressing = False
        self.curr_motion_timestep = config.task.motion_start_timestep
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None
        self._activation_q: np.ndarray | None = None
        self._startup_interpolator = JointPositionInterpolator(
            joint_lower=G1_JOINT_LOWER,
            joint_upper=G1_JOINT_UPPER,
            joint_velocity=G1_JOINT_VELOCITY,
            rate_hz=config.task.rl_rate,
            duration_s=config.task.init_duration_s,
            slew_safety_factor=config.robot.joint_interpolation_slew_safety_factor,
        )

        # Initialize clock for sim-time synchronization
        self.clock_sub = ClockSub()
        clock_util = MotionClockUtil(self.clock_sub)
        self.timestep_util = TimestepUtil(
            clock=clock_util,
            interval_ms=1000.0 / config.task.rl_rate,
            start_timestep=config.task.motion_start_timestep,
        )

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self.per_joint_policy_action_scale: np.ndarray | None = None

        self.setup_policy(config.task.model_path)
        self.robot_config = resolve_control_gains(self.robot_config, self.onnx_kp, self.onnx_kd)
        self._configure_action_scales()
        if self.config.guard:
            self.guard = WbtGuard(self.config.guard, self)

        # Retain the configured stdin gate, but preload never sends a command.
        if not self.config.task.skip_stiff_prompt:
            if sys.stdin.isatty():
                logger.info("WBT preloaded. Press Enter to allow startup.")
                try:
                    input()
                except EOFError:
                    logger.warning("WBT startup confirmation skipped: stdin reached EOF")
            else:
                logger.warning("WBT startup confirmation skipped: non-interactive stdin")

        try:
            if self.use_sim_time:
                self.clock_sub.start()
        except BaseException:
            self.clock_sub.close()
            raise

    def _get_ref_body_orientation_in_world(self, robot_state_data: LowState):
        # Create configuration for pinocchio robot
        # Note:
        # 1. pinocchio quaternion is in xyzw format, robot_state_data is in wxyz format
        # 2. joint sequences in pinocchio robot and real robot are different

        # free base pos, does not matter
        root_pos = robot_state_data.base_pos[0]

        # free base ori, wxyz -> xyzw
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data.base_quat)[0]

        # dof pos in real robot -> pinocchio robot
        dof_pos_in_real = robot_state_data.joint_pos[0]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)

        ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_orientation_in_world(configuration)
        return xyzw_to_wxyz(ref_ori_xyzw)

    def setup_policy(self, model_path):
        if self.config.task.motion_data_path:
            self._target_source = NpzTargetSource(
                self.config.task.motion_data_path,
                dof_names=self.dof_names,
                start_frame=self.config.task.motion_start_timestep,
            )
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Load model-specific metadata and kinematics.
        metadata = load_metadata(model_path)

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            from pathlib import Path

            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        # get initial command and ref quat xyzw at the configured start timestep
        time_step = np.array([[self.config.task.motion_start_timestep]], dtype=np.float32)
        if self._target_source is not None:
            self.motion_command_t, self.ref_quat_xyzw_t = self._target_source.get_target()
        else:
            # Use configured observation dimensions (including history) instead of a hard-coded value.
            actor_obs_template = self.observations.obs_buf_dict.get("actor_obs")
            if actor_obs_template is None:
                raise ValueError("Observation group 'actor_obs' must be configured for WBT policy.")
            obs = actor_obs_template.copy()
            input_feed = {"obs": obs, "time_step": time_step}
            outputs = self.onnx_policy_session.run(["joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)

            # motion_command_t/ref_quat_xyzw_t will be used in get_current_obs_buffer_dict
            self.motion_command_t = np.concatenate(outputs[0:2], axis=1)  # (1, 58)
            self.ref_quat_xyzw_t = outputs[2]
        # Keep immutable startup targets for subsequent episodes.
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()

        def policy_act(input_feed):
            output = self.onnx_policy_session.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)
            action = output[0]
            motion_command = np.concatenate(output[1:3], axis=1)
            ref_quat_xyzw = output[3]
            return action, motion_command, ref_quat_xyzw

        self.policy = policy_act

    def _initialization_target(self, robot_state_data: LowState):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data.joint_pos
        if self._stage == WbtStage.INITIALIZING:
            if self._activation_q is None:
                raise PolicyRuntimeFault("wbt_startup_pose_unavailable")
            try:
                interpolation = self._startup_interpolator.next(dof_pos[0])
            except (RuntimeError, ValueError) as error:
                raise PolicyRuntimeFault(f"wbt_interpolation_failed: {error}") from error
            q_target = interpolation.q_target.reshape(1, -1)
            if interpolation.complete:
                self._activation_q = None
                self._start_tracking(robot_state_data)
                self.logger.info("WBT initialization complete; policy action enabled")
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data: LowState):
        current_obs_buffer_dict = {}

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        # robot_ref_ori
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)

        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        # base_ang_vel
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data.base_ang_vel

        # dof_pos
        current_obs_buffer_dict["dof_pos"] = robot_state_data.joint_pos - self.default_dof_angles

        # dof_vel
        current_obs_buffer_dict["dof_vel"] = robot_state_data.joint_vel

        # actions
        current_obs_buffer_dict["actions"] = self.actions.last

        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the configured start while waiting to trigger the clip.
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep

        obs = self.observations.prepare(self.get_current_obs_buffer_dict(robot_state_data))
        if self.config.task.print_observations:
            self.observations.print_observations(obs, self.dof_names, self.actions.scaled)

        input_feed = {"time_step": np.array([[self.curr_motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)

        # Override the ONNX's self-generated clip target with the injected source.
        if self._target_source is not None:
            self.motion_command_t, self.ref_quat_xyzw_t = self._target_source.get_target()

        scale = (
            self.config.task.policy_action_scale
            if self.per_joint_policy_action_scale is None
            else self.per_joint_policy_action_scale
        )
        self.actions.process(policy_action, scale)
        # update motion timestep
        self._set_motion_timestep()

        return self.actions.scaled

    def _configure_action_scales(self) -> None:
        """Configure action scales, prioritising ONNX metadata over config fallbacks.

        Resolution order:
        1. ONNX metadata ``action_scale`` (scalar or per-joint list)
        2. ``robot.default_per_joint_action_scale`` when
           ``task.action_scales_by_effort_limit_over_p_gain`` is True
        3. Fall back to the scalar ``task.policy_action_scale``
        """
        raw_metadata = dict(self.onnx_policy_session.get_modelmeta().custom_metadata_map)
        onnx_action_scale = self._parse_action_scale_metadata(raw_metadata.get("action_scale"))

        if onnx_action_scale is not None:
            scales = onnx_action_scale.astype(np.float32, copy=False).reshape(-1)
        elif self.config.task.action_scales_by_effort_limit_over_p_gain:
            fallback = self.config.robot.default_per_joint_action_scale
            if fallback is None:
                raise ValueError(
                    "task.action_scales_by_effort_limit_over_p_gain=True requires ONNX metadata key "
                    "'action_scale' (scalar or per-joint list) or "
                    "robot.default_per_joint_action_scale."
                )
            scales = np.asarray(fallback, dtype=np.float32).reshape(-1)
            logger.warning("ONNX metadata 'action_scale' missing; using robot.default_per_joint_action_scale.")
        else:
            self.per_joint_policy_action_scale = None
            return

        if scales.size == 1:
            scales = np.full(self.num_dofs, scales.item(), dtype=np.float32)
        elif scales.size != self.num_dofs:
            raise ValueError(f"Action scale must contain 1 or {self.num_dofs} values, got {scales.size}.")

        self.per_joint_policy_action_scale = scales.reshape(1, -1)

    @staticmethod
    def _parse_action_scale_metadata(raw_value: str | None) -> np.ndarray | None:
        """Parse action_scale metadata from JSON-serialized or CSV string formats."""
        if raw_value is None:
            return None

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = raw_value

        if isinstance(parsed, (int, float)):
            return np.array([float(parsed)], dtype=np.float32)
        if isinstance(parsed, str):
            values = [float(token.strip()) for token in parsed.split(",") if token.strip()]
            if not values:
                raise ValueError("ONNX metadata action_scale is an empty string.")
            return np.array(values, dtype=np.float32)

        values = np.asarray(parsed, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("ONNX metadata action_scale is empty.")
        return values

    def get_reference_state(self) -> np.ndarray | None:
        """Return the current WBT clip target in the normal state-message layout."""
        if self.motion_command_t is None or self.ref_quat_xyzw_t is None:
            return None
        joint_pos = np.asarray(self.motion_command_t[:, : self.num_dofs], dtype=np.float64).reshape(1, -1)
        quat_xyzw = np.asarray(self.ref_quat_xyzw_t, dtype=np.float64).reshape(1, 4)
        quat_wxyz = xyzw_to_wxyz(quat_xyzw)
        return np.concatenate((np.zeros((1, 3)), quat_wxyz, joint_pos), axis=1)

    def _start_tracking(self, robot_state_data: LowState):
        self._stage = WbtStage.TRACKING
        self._capture_robot_yaw_offset(robot_state_data)
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)
        self._handle_start_motion_clip()

    def _on_activate(self, robot_state_data: LowState) -> str | None:
        """Start immediately or interpolate to the motion's first pose."""
        self.observations.reset()
        self.actions.reset()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()

        if self.config.task.startup_mode == "immediate":
            self._activation_q = None
            self._startup_interpolator.clear()
            self._start_tracking(robot_state_data)
            self.logger.info("WBT immediate startup; policy action enabled")
        else:
            self._activation_q = np.asarray(robot_state_data.joint_pos[0], dtype=np.float64).copy()
            try:
                self._startup_interpolator.reset(
                    self._activation_q,
                    np.asarray(self.motion_command_0[0, : self.num_dofs], dtype=np.float64),
                )
            except ValueError as error:
                self._activation_q = None
                self._startup_interpolator.clear()
                return f"wbt_start_failed: {error}"
            self._stage = WbtStage.INITIALIZING
            self.motion_clip_progressing = False
            self.logger.info(f"WBT initialization started ({self.config.task.init_duration_s:.1f}s)")
        return None

    def _set_motion_timestep(self):
        if self.motion_clip_progressing:
            prev = self.curr_motion_timestep

            if self.use_sim_time:
                self.curr_motion_timestep = self.timestep_util.get_timestep(log=self.logger)
            else:
                self.curr_motion_timestep += 1

            if self.curr_motion_timestep != prev:
                self.logger.debug(f"Motion timestep: {prev} → {self.curr_motion_timestep}")

            # Stop motion clip at configured end timestep (keep policy running at final pose)
            if (end := self.config.task.motion_end_timestep) and self.curr_motion_timestep >= end:
                if self.config.task.motion_loop:
                    self.curr_motion_timestep = self.config.task.motion_start_timestep
                    self.logger.info(colored(f"Loop=True, set to {self.curr_motion_timestep}", "green"))
                else:
                    self.logger.info(colored(f"Reached end timestep {end}, keeping last clip", "yellow"))
                    # self.motion_clip_progressing = False
                    self.curr_motion_timestep = end

    def _on_deactivate(self):
        self.observations.reset()
        self.actions.reset()
        self._stage = WbtStage.INACTIVE
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self._activation_q = None
        self._startup_interpolator.clear()

    def _on_close(self) -> None:
        self.clock_sub.close()

    def _compute_command(self, robot_state_data):
        if self._stage == WbtStage.INITIALIZING:
            with self.latency_tracker.measure(LatencyStage.PREPROCESSING):
                q_target = self._initialization_target(robot_state_data)
        else:
            with self.latency_tracker.measure(LatencyStage.INFERENCE):
                self.rl_inference(robot_state_data)
            q_target = self.actions.target(self.default_dof_angles)
        with self.latency_tracker.measure(LatencyStage.POSTPROCESSING):
            return position_command(
                q_target, self.robot_config.motor_kp, self.robot_config.motor_kd, self.controlled_joint_mask
            )

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
        if self._target_source:
            self._target_source.reset(self.config.task.motion_start_timestep)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.motion_clip_progressing = True
        if self.config.task.motion_start_timestep > 0 or self.config.task.motion_end_timestep is not None:
            start_str = str(self.config.task.motion_start_timestep)
            end_str = str(self.config.task.motion_end_timestep) if self.config.task.motion_end_timestep else "end"
            self.logger.info(colored(f"Starting motion clip from timestep {start_str} to {end_str}", "blue"))
        else:
            self.logger.info(colored("Starting motion clip", "blue"))

    def _capture_robot_yaw_offset(self, robot_state_data: LowState):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _capture_motion_yaw_offset(self, ref_quat_xyzw_0: np.ndarray) -> float:
        """Capture motion yaw when policy starts to use as reference offset."""
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(ref_quat_xyzw_0))
        self.logger.info(colored(f"Motion yaw offset captured at {np.degrees(self.motion_yaw_offset):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        """Remove stored yaw offset from robot orientation quaternion."""
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        """Extract yaw angle from quaternion array of shape (1, 4)."""
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)
