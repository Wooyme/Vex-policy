"""Pure-Python GEAR-SONIC policy, encoder, and planner integration."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping

import numpy as np
import onnxruntime
from loguru import logger

from vex_policy.config.config_types import InferenceConfig, SonicTaskConfig
from vex_policy.policies.base import BasePolicy
from vex_policy.policies.sonic_motion import load_motion_directory
from vex_policy.policies.sonic_planner import (
    HW_TO_POLICY,
    MODE_NAMES,
    ONE_SHOT_MODES,
    POLICY_TO_HW,
    STATIC_MODES,
    MotionSequence,
    MovementCommand,
    SonicPlanner,
    heading_quaternion,
    ort_providers,
    quaternion_conjugate,
    quaternion_matrix,
    quaternion_multiply,
    quaternion_slerp,
)
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse

_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], onnxruntime.InferenceSession] = {}
_SESSION_CACHE_LOCK = threading.Lock()

_ACTOR_TERMS = (
    "token_state",
    "his_base_angular_velocity_10frame_step1",
    "his_body_joint_positions_10frame_step1",
    "his_body_joint_velocities_10frame_step1",
    "his_last_actions_10frame_step1",
    "his_gravity_dir_10frame_step1",
)
_ENCODER_TERMS = (
    "encoder_mode_4",
    "motion_joint_positions_10frame_step5",
    "motion_joint_velocities_10frame_step5",
    "motion_root_z_position_10frame_step5",
    "motion_root_z_position",
    "motion_anchor_orientation",
    "motion_anchor_orientation_10frame_step5",
    "motion_joint_positions_lowerbody_10frame_step5",
    "motion_joint_velocities_lowerbody_10frame_step5",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
    "smpl_joints_10frame_step1",
    "smpl_anchor_orientation_10frame_step1",
    "motion_joint_positions_wrists_10frame_step1",
)


def _shared_session(path: str, provider: str) -> onnxruntime.InferenceSession:
    providers = tuple(ort_providers(provider))
    key = (path, providers)
    with _SESSION_CACHE_LOCK:
        session = _SESSION_CACHE.get(key)
        if session is None:
            session = onnxruntime.InferenceSession(path, providers=list(providers))
            _SESSION_CACHE[key] = session
        return session


def _source_control_gains() -> tuple[np.ndarray, np.ndarray]:
    natural_frequency = 10.0 * 2.0 * np.pi
    damping_ratio = 2.0
    armature = {
        "5020": 0.003609725,
        "7520_14": 0.010177520,
        "7520_22": 0.025101925,
        "4010": 0.00425,
    }
    stiffness = {name: value * natural_frequency ** 2 for name, value in armature.items()}
    damping = {name: 2.0 * damping_ratio * value * natural_frequency for name, value in armature.items()}
    types = (
        "7520_22",
        "7520_22",
        "7520_14",
        "7520_22",
        "5020",
        "5020",
        "7520_22",
        "7520_22",
        "7520_14",
        "7520_22",
        "5020",
        "5020",
        "7520_14",
        "5020",
        "5020",
        "5020",
        "5020",
        "5020",
        "5020",
        "5020",
        "4010",
        "4010",
        "5020",
        "5020",
        "5020",
        "5020",
        "5020",
        "4010",
        "4010",
    )
    doubled = frozenset({4, 5, 10, 11, 13, 14})
    kp = np.asarray([stiffness[kind] * (2.0 if index in doubled else 1.0) for index, kind in enumerate(types)])
    kd = np.asarray([damping[kind] * (2.0 if index in doubled else 1.0) for index, kind in enumerate(types)])
    return kp.astype(np.float32), kd.astype(np.float32)


def _same_command(left: MovementCommand | None, right: MovementCommand) -> bool:
    return (
            left is not None
            and left.mode == right.mode
            and left.speed == right.speed
            and left.height == right.height
            and np.array_equal(left.movement_direction, right.movement_direction)
            and np.array_equal(left.facing_direction, right.facing_direction)
    )


class SonicPolicy(BasePolicy):
    """GEAR-SONIC locomotion policy driven through the existing MQTT inputs."""

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, SonicTaskConfig):
            raise TypeError("SonicPolicy requires SonicTaskConfig")
        if tuple(config.observation.obs_dict.get("actor_obs", ())) != _ACTOR_TERMS:
            raise ValueError("SONIC actor_obs terms or order do not match the exported decoder")
        if tuple(config.observation.obs_dict.get("encoder_obs", ())) != _ENCODER_TERMS:
            raise ValueError("SONIC encoder_obs terms or order do not match the exported encoder")
        self.sonic_task = config.task
        self._state_history: deque[tuple[np.ndarray, ...]] = deque(maxlen=10)
        self._motion_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._planner_wakeup = threading.Event()
        self._planner_stop: threading.Event | None = None
        self._planner_thread: threading.Thread | None = None
        self._planner_robot_state: tuple[np.ndarray, np.ndarray] | None = None
        self._reference_motion: MotionSequence | None = None
        self._reference_motion_name: str | None = None
        self._motion: MotionSequence | None = None
        self._pending_motion: MotionSequence | None = None
        self._motion_frame = 0
        self._heading_robot_initial: np.ndarray | None = None
        self._heading_reference_initial: np.ndarray | None = None
        self._desired_heading = 0.0
        self._one_shot_complete = False
        self._movement_command = MovementCommand(
            mode=self.sonic_task.planner_mode,
            speed=0.0,
            height=-1.0,
            movement_direction=np.zeros(3, dtype=np.float32),
            facing_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        )
        super().__init__(config)
        self._action_scale_hw = np.asarray(self.robot_config.default_per_joint_action_scale, dtype=np.float32)

    @staticmethod
    def _validate_session(session, input_size: int, output_size: int, label: str) -> None:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or int(np.prod(inputs[0].shape)) != input_size:
            raise ValueError(f"{label} must have one {input_size}-element input")
        if len(outputs) != 1 or int(np.prod(outputs[0].shape)) != output_size:
            raise ValueError(f"{label} must have one {output_size}-element output")

    def _mask_policy_order_action(self, action_policy: np.ndarray) -> np.ndarray:
        """Apply the hardware-order mask to a SONIC/IsaacLab-order output."""
        masked = np.asarray(action_policy, dtype=np.float32) * self.action_mask[:, POLICY_TO_HW]
        if self.config.task.debug.force_zero_action:
            masked.fill(0.0)
        return masked

    def setup_policy(self, model_path):
        provider = self.sonic_task.inference_provider
        self.onnx_policy_session = _shared_session(model_path, provider)
        self.encoder_session = _shared_session(self.sonic_task.encoder_model_path, provider)
        self._validate_session(self.onnx_policy_session, 994, 29, "SONIC decoder")
        self._validate_session(self.encoder_session, 1762, 64, "SONIC encoder")
        self.onnx_input_names = [item.name for item in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [item.name for item in self.onnx_policy_session.get_outputs()]
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_name = self.encoder_session.get_outputs()[0].name
        if self.sonic_task.motion_source == "planner":
            planner_session = _shared_session(self.sonic_task.planner_model_path, provider)
            self.planner: SonicPlanner | None = SonicPlanner(
                self.sonic_task.planner_model_path,
                provider=provider,
                version=self.sonic_task.planner_version,
                look_ahead_steps=self.sonic_task.motion_look_ahead_steps,
                default_height=self.sonic_task.planner_default_height,
                seed=self.sonic_task.planner_seed,
                session=planner_session,
            )
            self._reference_motion = None
            self._reference_motion_name = None
        else:
            self.planner = None
            motion_data_path = self.sonic_task.motion_data_path
            if not motion_data_path:
                raise ValueError("SONIC directory motion source requires motion_data_path")
            self._reference_motion_name, self._reference_motion = load_motion_directory(
                motion_data_path,
                motion_name=self.sonic_task.motion_name,
                start_frame=self.sonic_task.motion_start_timestep,
                end_frame=self.sonic_task.motion_end_timestep,
            )
        self.onnx_kp, self.onnx_kd = _source_control_gains()

        def policy_act(observation):
            return self.onnx_policy_session.run(
                self.onnx_output_names, {self.onnx_input_names[0]: np.asarray(observation, dtype=np.float32)}
            )[0]

        self.policy = policy_act

    def _reset_sonic_state(self) -> None:
        self._state_history.clear()
        with self._motion_lock:
            self._motion = None
            self._pending_motion = None
            self._motion_frame = 0
        self._planner_robot_state = None
        self._heading_robot_initial = None
        self._heading_reference_initial = None
        self._desired_heading = 0.0
        self._one_shot_complete = False
        self.last_policy_action = np.zeros((1, self.num_dofs), dtype=np.float32)
        self.scaled_policy_action = np.zeros((1, self.num_dofs), dtype=np.float32)

    def activate(self, robot_state_data: LowState) -> str | None:
        self._stop_planner()
        self._reset_sonic_state()
        reason = super().activate(robot_state_data)
        if reason:
            return reason
        if self.sonic_task.motion_source == "directory":
            with self._motion_lock:
                self._motion = self._reference_motion
            logger.info(f"SONIC playing reference motion: {self._reference_motion_name}")
        else:
            self._planner_stop = threading.Event()
            self._planner_thread = threading.Thread(target=self._planner_loop, name="sonic-planner-10hz", daemon=True)
            self._planner_thread.start()

    def deactivate(self) -> None:
        self._stop_planner()
        super().deactivate()

    def close(self) -> None:
        self._stop_planner()

    def _stop_planner(self) -> None:
        stop = self._planner_stop
        thread = self._planner_thread
        if stop is not None:
            stop.set()
            self._planner_wakeup.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._planner_stop = None
        self._planner_thread = None
        self._planner_wakeup.clear()

    def apply_control(self, control: Mapping[str, float]) -> None:
        if self.sonic_task.motion_source == "directory":
            return
        # Preserve the repository's panel-to-robot convention: (vy, -vx, -yaw).
        local = np.asarray([control["vy"], -control["vx"]], dtype=np.float32)
        speed = float(np.linalg.norm(local))
        self._desired_heading += float(-control["yaw"]) / self.rl_rate
        cosine = float(np.cos(self._desired_heading))
        sine = float(np.sin(self._desired_heading))
        facing = np.asarray([cosine, sine, 0.0], dtype=np.float32)
        if speed > 1e-6 and self.sonic_task.planner_mode not in STATIC_MODES:
            local /= speed
            movement = np.asarray(
                [cosine * local[0] - sine * local[1], sine * local[0] + cosine * local[1], 0.0],
                dtype=np.float32,
            )
        else:
            movement = np.zeros(3, dtype=np.float32)
            speed = 0.0
        command = MovementCommand(
            mode=self.sonic_task.planner_mode,
            speed=speed,
            height=float(control["height"]) if control["height"] > 0 else -1.0,
            movement_direction=movement,
            facing_direction=facing,
        )
        with self._command_lock:
            changed = not _same_command(self._movement_command, command)
            self._movement_command = command
        if changed:
            self._planner_wakeup.set()

    def _command_snapshot(self) -> MovementCommand:
        with self._command_lock:
            command = self._movement_command
            return MovementCommand(
                command.mode,
                command.speed,
                command.height,
                command.movement_direction.copy(),
                command.facing_direction.copy(),
            )

    @staticmethod
    def _replan_interval(mode: int) -> float:
        if mode == 3:
            return 0.1
        if mode == 8:
            return 0.2
        if mode in {11, 12, 13, 15, 16}:
            return 1.0
        return 1.0

    def _planner_loop(self) -> None:
        stop = self._planner_stop
        if stop is None:
            return
        period = 1.0 / self.sonic_task.planner_rate
        last_command: MovementCommand | None = None
        last_plan_at = 0.0
        while not stop.is_set():
            started = time.monotonic()
            robot = self._planner_robot_state
            command = self._command_snapshot()
            with self._motion_lock:
                motion = self._motion
                frame = self._motion_frame
            due = (
                    robot is not None
                    and not self._one_shot_complete
                    and (
                            motion is None
                            or not _same_command(last_command, command)
                            or (
                                    command.mode not in STATIC_MODES
                                    and command.speed != 0.0
                                    and started - last_plan_at >= self._replan_interval(command.mode)
                            )
                    )
            )
            if due:
                try:
                    if self.planner is None:
                        raise RuntimeError("SONIC planner loop started without a planner")
                    context = (
                        self.planner.initial_context(robot[1])
                        if motion is None
                        else self.planner.motion_context(motion, frame)
                    )
                    generated = self.planner.infer(context, command)
                    with self._motion_lock:
                        self._pending_motion = generated
                    last_command = command
                    last_plan_at = time.monotonic()
                except Exception:
                    logger.exception("SONIC planner inference failed; keeping the last safe trajectory")
            delay = max(0.0, period - (time.monotonic() - started))
            self._planner_wakeup.wait(delay)
            self._planner_wakeup.clear()

    @staticmethod
    def _blend_motion(old: MotionSequence, old_frame: int, new: MotionSequence) -> MotionSequence:
        remaining = max(1, old.frames - old_frame)
        count = max(remaining, new.frames)
        old_index = np.minimum(old_frame + np.arange(count), old.frames - 1)
        new_index = np.minimum(np.arange(count), new.frames - 1)
        weight = np.minimum((np.arange(count, dtype=np.float32) + 1.0) / 8.0, 1.0)
        return MotionSequence(
            old.root_positions[old_index] * (1 - weight[:, None]) + new.root_positions[new_index] * weight[:, None],
            quaternion_slerp(old.root_quaternions[old_index], new.root_quaternions[new_index], weight),
            old.joint_positions[old_index] * (1 - weight[:, None]) + new.joint_positions[new_index] * weight[:, None],
            old.joint_velocities[old_index] * (1 - weight[:, None]) + new.joint_velocities[new_index] * weight[:, None],
        )

    def _consume_pending_motion(self, robot_quaternion: np.ndarray) -> MotionSequence | None:
        with self._motion_lock:
            if self._pending_motion is not None:
                if self._motion is None:
                    self._motion = self._pending_motion
                else:
                    self._motion = self._blend_motion(self._motion, self._motion_frame, self._pending_motion)
                self._pending_motion = None
                self._motion_frame = 0
            motion = self._motion
        if motion is not None and self._heading_robot_initial is None:
            self._heading_robot_initial = robot_quaternion.copy()
            self._heading_reference_initial = motion.root_quaternions[0].copy()
        return motion

    def _append_history(self, robot_state_data: LowState) -> None:
        q_hw = robot_state_data.joint_pos[0]
        dq_hw = robot_state_data.joint_vel[0]
        q_policy = q_hw[POLICY_TO_HW] - self.default_dof_angles[POLICY_TO_HW]
        dq_policy = dq_hw[POLICY_TO_HW]
        angular_velocity = robot_state_data.base_ang_vel[0]
        quaternion = robot_state_data.base_quat
        gravity = quat_rotate_inverse(quaternion, np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32))[0]
        self._state_history.append(
            (
                angular_velocity.astype(np.float32),
                q_policy.astype(np.float32),
                dq_policy.astype(np.float32),
                self.last_policy_action[0].astype(np.float32).copy(),
                gravity.astype(np.float32),
            )
        )

    def _decoder_observation(self, token: np.ndarray) -> np.ndarray:
        missing = 10 - len(self._state_history)
        zeros = (
            np.zeros(3, dtype=np.float32),
            np.zeros(29, dtype=np.float32),
            np.zeros(29, dtype=np.float32),
            np.zeros(29, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        history = [zeros] * missing + list(self._state_history)
        terms = [np.concatenate([frame[index] for frame in history]) for index in range(5)]
        observation = np.concatenate((np.asarray(token, dtype=np.float32).reshape(-1), *terms)).reshape(1, -1)
        if observation.shape != (1, 994):
            raise RuntimeError(f"Unexpected SONIC decoder observation shape: {observation.shape}")
        return observation

    def _encoder_observation(self, motion: MotionSequence, frame: int, robot_quaternion: np.ndarray) -> np.ndarray:
        observation = np.zeros(1762, dtype=np.float32)
        observation[0] = float(self.sonic_task.planner_encoder_mode)
        indices = np.minimum(frame + np.arange(10) * 5, motion.frames - 1)
        observation[4:294] = motion.joint_positions[indices].reshape(-1)
        observation[294:584] = motion.joint_velocities[indices].reshape(-1)
        initial_robot = self._heading_robot_initial
        initial_reference = self._heading_reference_initial
        if initial_robot is None or initial_reference is None:
            return observation.reshape(1, -1)
        apply_heading = quaternion_multiply(
            heading_quaternion(initial_robot), quaternion_conjugate(heading_quaternion(initial_reference))
        )
        corrected_reference = quaternion_multiply(apply_heading, motion.root_quaternions[indices])
        current = np.broadcast_to(robot_quaternion, corrected_reference.shape)
        relative = quaternion_multiply(quaternion_conjugate(current), corrected_reference)
        observation[601:661] = quaternion_matrix(relative)[..., :2].reshape(-1)
        return observation.reshape(1, -1)

    def rl_inference(self, robot_state_data: LowState):
        self._append_history(robot_state_data)
        quaternion = np.asarray(robot_state_data.base_quat[0], dtype=np.float32)
        joint_positions_hw = np.asarray(robot_state_data.joint_pos[0], dtype=np.float32)
        if self.sonic_task.motion_source == "planner":
            self._planner_robot_state = (quaternion.copy(), joint_positions_hw.copy())
            self._planner_wakeup.set()
        motion = self._consume_pending_motion(quaternion)
        if motion is None:
            return np.zeros((1, self.num_dofs), dtype=np.float32)

        with self._motion_lock:
            frame = min(self._motion_frame, motion.frames - 1)
        encoder_observation = self._encoder_observation(motion, frame, quaternion)
        token = self.encoder_session.run([self.encoder_output_name], {self.encoder_input_name: encoder_observation})[0]
        decoder_observation = self._decoder_observation(token)
        action_policy = np.asarray(self.policy(decoder_observation), dtype=np.float32)
        action_policy = np.clip(action_policy, -100.0, 100.0)
        action_policy = self._mask_policy_order_action(action_policy)
        self.last_policy_action = action_policy.copy()
        action_hw = action_policy[:, HW_TO_POLICY] * self._action_scale_hw
        self.scaled_policy_action = action_hw

        max_frame = motion.frames - 1
        with self._motion_lock:
            if self._motion_frame < max_frame:
                self._motion_frame += 1
            elif self.sonic_task.motion_source == "directory" and self.sonic_task.motion_loop:
                self._motion_frame = 0
                self._heading_robot_initial = None
                self._heading_reference_initial = None
            elif self.sonic_task.planner_mode in ONE_SHOT_MODES:
                self._one_shot_complete = True
        return action_hw

    def get_reference_state(self) -> np.ndarray | None:
        with self._motion_lock:
            motion = self._motion
            frame = self._motion_frame
        if motion is None:
            return None
        frame = min(frame, motion.frames - 1)
        state = np.zeros((1, 7 + self.num_dofs), dtype=np.float32)
        state[0, :3] = motion.root_positions[frame]
        state[0, 3:7] = motion.root_quaternions[frame]
        state[0, 7:] = motion.joint_positions[frame, HW_TO_POLICY]
        return state


__all__ = ["MODE_NAMES", "SonicPolicy"]
