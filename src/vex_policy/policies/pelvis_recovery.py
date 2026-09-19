"""Holosoma pelvis recovery with compensated right-ankle FK and position PD."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import ClassVar

import numpy as np

from vex_policy.config.config_types import (
    GuardConfig,
    InferenceConfig,
    PelvisRecoveryTaskConfig,
    SliderInput,
    input_parameters,
)
from vex_policy.policies.guard.initial_pose import InitialPoseGuard
from vex_policy.policies.utils.inference import OnnxActor, resolve_control_gains
from vex_policy.policies.utils.joint_command import PositionAction, position_command
from vex_policy.policies.utils.locomotion_utils import RightAnkleKinematics, load_motion_last_pose
from vex_policy.policies.utils.observations import ObservationHistory
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.latency import LatencyStage
from vex_policy.utils.math.quat import quat_inverse, quat_mul, quat_rotate_inverse

from .base import BasePolicy, PolicyRuntimeFault

BRIDGE_JOINTS = (
    "waist_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
)


def _align_reference_heading(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Apply only the best-fitting world-Z rotation, preserving reference tilt.

    Projecting the relative rotation onto its W/Z components avoids Euler yaw
    discontinuities when a bridge pose crosses a pitch of -pi/2.
    """
    delta = quat_mul(current, quat_inverse(reference))
    yaw = delta.copy()
    yaw[:, 1:3] = 0
    norm = float(np.linalg.norm(yaw))
    if not np.isfinite(norm) or norm < 1e-8:
        raise PolicyRuntimeFault("recovery_invalid_reference_heading")
    return quat_mul(yaw / norm, reference)


class PelvisRecoveryPolicy(BasePolicy):
    """Single-frame actor; total motor torque clipping is not implemented here."""

    _OBS_DIMS: ClassVar[dict[str, int]] = {
        "base_angular_velocity": 3,
        "base_orientation": 4,
        "command": 5,
        "joint_position": 29,
        "joint_velocity": 29,
        "previous_action": 29,
    }
    _OBS_SCALES: ClassVar[dict[str, float]] = {
        "base_angular_velocity": 0.25,
        "base_orientation": 1.0,
        "command": 1.0,
        "joint_position": 1.0,
        "joint_velocity": 0.05,
        "previous_action": 1.0,
    }
    _COMMAND_NAMES = ("peak_speed", "target_height")
    _SLOWDOWN_HEIGHT = 0.03
    _MAX_ACCELERATION = 0.5

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, PelvisRecoveryTaskConfig):
            raise TypeError("PelvisRecoveryPolicy requires PelvisRecoveryTaskConfig")
        if not isinstance(config.guard, GuardConfig):
            raise TypeError("PelvisRecoveryPolicy requires GuardConfig")
        if config.action_mask is not None or config.task.action_mask_path is not None:
            raise ValueError("Pelvis recovery does not support action masks")
        if config.robot.robot != "g1" or config.robot.num_joints != 29 or config.robot.num_motors != 29:
            raise ValueError("Pelvis recovery requires G1 with 29 joints and motors")
        super().__init__(config)
        self.parameters = {p.name: p for p in input_parameters(config.inputs)}
        if (
            len(config.inputs) != 2
            or not all(isinstance(component, SliderInput) for component in config.inputs)
            or set(self.parameters) != set(self._COMMAND_NAMES)
        ):
            raise ValueError(f"Pelvis recovery requires two sliders: {self._COMMAND_NAMES}")
        speed = self.parameters["peak_speed"]
        if speed.min < 0.05 or speed.max > 0.3:
            raise ValueError("Pelvis recovery peak_speed range must be within [0.05, 0.3] m/s")
        if self.parameters["target_height"].min <= 0:
            raise ValueError("Pelvis recovery target_height range must be positive")

        self.initial_pose = load_motion_last_pose(config.task.motion_data_path)
        names = self.initial_pose.dof_names
        if len(names) != 29 or set(names) != set(self.dof_names):
            raise ValueError("Pelvis recovery motion joint names do not match the robot")
        self.default_dof_angles = np.asarray([self.initial_pose.dof_pos[names.index(n)] for n in self.dof_names])
        # Hardware limits are defined in the canonical G1 order.
        if tuple(self.dof_names) != tuple(G1_29DOF.dof_names):
            raise ValueError("Pelvis recovery requires the canonical G1 joint order")
        if np.any(self.default_dof_angles < G1_JOINT_LOWER) or np.any(self.default_dof_angles > G1_JOINT_UPPER):
            raise ValueError("Pelvis recovery reference pose exceeds G1 joint limits")
        self.bridge_ids = np.asarray([self.dof_names.index(name) for name in BRIDGE_JOINTS])
        self._validate_observations()
        self.observations = ObservationHistory(config.observation)
        self.actor = OnnxActor(config.task.model_path)
        self._validate_model()
        self.robot_config = resolve_control_gains(config.robot, self.actor.metadata["kp"], self.actor.metadata["kd"])
        kp = np.asarray(self.robot_config.motor_kp, dtype=np.float64).copy()
        kd = np.asarray(self.robot_config.motor_kd, dtype=np.float64).copy()
        for name, values in (("motor_kp", kp), ("motor_kd", kd)):
            if values.shape != (29,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Pelvis recovery {name} must contain 29 finite nonnegative gains")
        # PPO exports robot-config gains, not the action term's weakened gains.
        kp[self.bridge_ids] *= 0.35
        kd[self.bridge_ids] *= 0.6
        self.robot_config = replace(self.robot_config, motor_kp=tuple(kp), motor_kd=tuple(kd))
        self.kinematics = RightAnkleKinematics(self.actor.metadata.get("robot_urdf"), self.dof_names)
        self.actions = PositionAction(
            29, self.action_mask, require_full_body=True, force_zero=config.task.debug.force_zero_action
        )
        self.recovery_command = np.zeros((1, 5), dtype=np.float32)
        self._reset_episode()
        self.guard = InitialPoseGuard(
            config.guard,
            self.initial_pose,
            self.dof_names,
            self.logger,
            reason_prefix="pelvis_recovery_start_check_failed",
        )

    def _validate_observations(self) -> None:
        obs = self.config.observation
        if set(obs.obs_dict) != {"actor_obs"} or sorted(obs.obs_dict["actor_obs"]) != sorted(self._OBS_DIMS):
            raise ValueError("Pelvis recovery actor_obs must contain exactly the six training terms")
        if obs.obs_dims != self._OBS_DIMS or obs.obs_scales != self._OBS_SCALES:
            raise ValueError("Pelvis recovery observation dimensions/scales must match training")
        if obs.history_length_dict != {"actor_obs": 1}:
            raise ValueError("Pelvis recovery requires one frame of actor_obs history")

    def _validate_model(self) -> None:
        for tensors, name, shape in (
            (self.actor.session.get_inputs(), "actor_obs", [1, 99]),
            (self.actor.session.get_outputs(), "action", [1, 29]),
        ):
            if (
                len(tensors) != 1
                or tensors[0].name != name
                or list(tensors[0].shape) != shape
                or tensors[0].type != "tensor(float)"
            ):
                raise ValueError(f"Pelvis recovery model must expose float32 {name}{shape}")
        if tuple(self.actor.metadata.get("dof_names", ())) != tuple(self.dof_names):
            raise ValueError("Pelvis recovery ONNX dof_names do not match the robot joint order")
        scale = np.asarray(self.actor.metadata.get("action_scale", ()), dtype=np.float64)
        if scale.shape not in ((), (29,)) or not np.isfinite(scale).all() or not np.allclose(scale, 0.25):
            raise ValueError("Pelvis recovery ONNX action_scale must equal 0.25 for all 29 joints")
        for name in ("kp", "kd"):
            values = np.asarray(self.actor.metadata.get(name, ()), dtype=np.float64)
            if values.shape != (29,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Pelvis recovery ONNX {name} must contain 29 finite nonnegative gains")

    def validated_state(self, state: LowState) -> LowState:
        for name, shape in (
            ("joint_pos", (1, 29)),
            ("joint_vel", (1, 29)),
            ("base_ang_vel", (1, 3)),
            ("base_quat", (1, 4)),
        ):
            value = np.asarray(getattr(state, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise PolicyRuntimeFault(f"recovery_invalid_state:{name}")
        quat = np.asarray(state.base_quat, dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if not np.isfinite(norm) or norm < 1e-8:
            raise PolicyRuntimeFault("recovery_invalid_state:base_quat")
        quat = quat / norm
        if self.config.task.debug.force_upright_imu:
            quat = np.array([[1.0, 0.0, 0.0, 0.0]])
        return replace(state, base_quat=quat)

    def _height(self, state: LowState) -> float:
        gravity = quat_rotate_inverse(state.base_quat, np.array([[0.0, 0.0, -1.0]]))
        height = np.asarray(self.kinematics.height_difference(state, gravity))
        if height.shape != (1, 1) or not np.isfinite(height).all():
            raise PolicyRuntimeFault("recovery_invalid_height")
        # Unitree does not provide root position. Anchor FK in world Z using
        # the configured ankle link origin height, in the target's frame.
        return float(height[0, 0]) + self.config.task.right_ankle_height_m

    def _reset_episode(self) -> None:
        self.observations.reset()
        self.actions.reset()
        self.peak_speed = self.parameters["peak_speed"].default
        self.target_height = self.parameters["target_height"].default
        self.reference_quat: np.ndarray | None = None
        self.current_height = 0.0
        self.start_height = 0.0
        self.target_velocity = 0.0
        self._first_step = True
        self.recovery_command.fill(0)
        self.recovery_command[0, 4] = self.target_height

    def _on_activate(self, robot_state_data: LowState) -> None:
        self._reset_episode()
        state = self.validated_state(robot_state_data)
        self.reference_quat = _align_reference_heading(np.asarray([self.initial_pose.root_quat_wxyz]), state.base_quat)
        self.current_height = self.start_height = self._height(state)

    def _on_deactivate(self) -> None:
        self._reset_episode()

    def _apply_control(self, control: Mapping[str, float]) -> None:
        try:
            speed, target = (float(control[name]) for name in self._COMMAND_NAMES)
            for name, value in zip(self._COMMAND_NAMES, (speed, target), strict=True):
                param = self.parameters[name]
                if not np.isfinite(value) or not param.min <= value <= param.max:
                    raise ValueError(f"{name} is outside its configured range")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise PolicyRuntimeFault(f"recovery_invalid_command:{error}") from error
        if target != self.target_height:
            self.start_height = self.current_height
            self.target_velocity = 0.0
            self._first_step = True
        self.peak_speed, self.target_height = speed, target

    def _update_command(self, height: float) -> None:
        self.current_height = height
        error = self.target_height - height
        desired = self.peak_speed * np.tanh(max(error, 0.0) / self._SLOWDOWN_HEIGHT)
        allowance = 0.0 if self._first_step else self._MAX_ACCELERATION / self.rl_rate
        self.target_velocity += float(np.clip(desired - self.target_velocity, -allowance, allowance))
        self._first_step = False
        drop = self.target_height - self.start_height
        scale = drop if drop > 1e-4 else self._SLOWDOWN_HEIGHT
        phase = np.pi * np.clip(1.0 - error / scale, 0.0, 1.0)
        self.recovery_command[0] = (error, self.target_velocity, np.sin(phase), np.cos(phase), self.target_height)

    def get_current_obs_buffer_dict(self, robot_state_data: LowState) -> dict[str, np.ndarray]:
        state = self.validated_state(robot_state_data)
        if self.reference_quat is None:
            raise PolicyRuntimeFault("recovery_missing_reference_orientation")
        relative = quat_mul(quat_inverse(self.reference_quat), state.base_quat)
        relative *= np.where(relative[:, :1] < 0, -1.0, 1.0)
        return {
            "base_angular_velocity": np.zeros((1, 3))
            if self.config.task.debug.force_zero_angular_velocity
            else state.base_ang_vel,
            "base_orientation": relative[:, [1, 2, 3, 0]],
            "command": self.recovery_command,
            "joint_position": state.joint_pos - self.default_dof_angles,
            "joint_velocity": state.joint_vel,
            "previous_action": self.actions.last,
        }

    def _compute_command(self, robot_state_data: LowState):
        try:
            with self.latency_tracker.measure(LatencyStage.PREPROCESSING):
                state = self.validated_state(robot_state_data)
                self._update_command(self._height(state))
                obs = self.observations.prepare(self.get_current_obs_buffer_dict(state))
                if obs["actor_obs"].shape != (1, 99) or not np.isfinite(obs["actor_obs"]).all():
                    raise PolicyRuntimeFault("recovery_invalid_observation")
                np.clip(obs["actor_obs"], -100, 100, out=obs["actor_obs"])
                self.observations.obs_buf_dict = {"actor_obs": obs["actor_obs"].copy()}
                if self.config.task.print_observations:
                    self.observations.print_observations(obs, self.dof_names, self.actions.scaled)
            with self.latency_tracker.measure(LatencyStage.INFERENCE):
                raw_action = np.asarray(self.actor(obs), dtype=np.float32)
            with self.latency_tracker.measure(LatencyStage.POSTPROCESSING):
                if raw_action.shape != (1, 29) or not np.isfinite(raw_action).all():
                    raise PolicyRuntimeFault("recovery_invalid_action")
                self.actions.process(raw_action, self.config.task.policy_action_scale)
                # Training observes ActionManager.action before joint-control clipping.
                self.actions.last = (
                    np.zeros_like(raw_action) if self.config.task.debug.force_zero_action else raw_action.copy()
                )
                # The current upstream action term only weakens gains; its
                # bridge-specific offset and joint-limit clamps are disabled.
                target = self.actions.target(self.default_dof_angles)
                if not np.isfinite(target).all():
                    raise PolicyRuntimeFault("recovery_invalid_target")
                self.actions.scaled = target - self.default_dof_angles
                return position_command(
                    target, self.robot_config.motor_kp, self.robot_config.motor_kd, self.controlled_joint_mask
                )
        except PolicyRuntimeFault:
            raise
        except Exception as error:
            raise PolicyRuntimeFault(f"recovery_inference_failed:{error}") from error
