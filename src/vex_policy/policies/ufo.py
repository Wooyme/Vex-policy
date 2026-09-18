"""Vex runtime adapter for the released UFO-Deploy G1 policy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import yaml

from vex_policy.config.config_types import (
    InferenceConfig,
    UfoGoalContextConfig,
    UfoRewardContextConfig,
    UfoTaskConfig,
    UfoTrackingContextConfig,
)
from vex_policy.policies.base import BasePolicy, PolicyJointCommand, PolicyRuntimeFault
from vex_policy.policies.guard.initial_pose import InitialPoseGuard
from vex_policy.policies.sonic_planner import ort_providers
from vex_policy.policies.utils.inference import shared_session
from vex_policy.policies.utils.initial_pose import InitialPose
from vex_policy.policies.utils.joint_command import position_command
from vex_policy.robots import G1_JOINT_LOWER, G1_JOINT_UPPER, G1_JOINT_VELOCITY
from vex_policy.robots.g1 import DOF_NAMES
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.joint_interpolation import JointPositionInterpolator, limit_joint_position_target
from vex_policy.utils.latency import LatencyStage
from vex_policy.utils.math.quat import quat_rotate_inverse

_ACTOR_TERMS = (
    "dof_pos_minus_default",
    "dof_vel",
    "projected_gravity",
    "base_ang_vel",
    "prev_actions",
    "prev_actions_history",
    "base_ang_vel_history",
    "dof_pos_minus_default_history",
    "dof_vel_history",
    "projected_gravity_history",
)
_OBS_DIMS = {
    "dof_pos_minus_default": 29,
    "dof_vel": 29,
    "projected_gravity": 3,
    "base_ang_vel": 3,
    "prev_actions": 29,
    "prev_actions_history": 116,
    "base_ang_vel_history": 12,
    "dof_pos_minus_default_history": 116,
    "dof_vel_history": 116,
    "projected_gravity_history": 12,
}


@dataclass(frozen=True)
class _UfoModelConfig:
    action_rescale: float
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    default_dof_angles: np.ndarray


def _finite_float(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"UFO model config {label} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"UFO model config {label} must be finite")
    return result


def _joint_values(
    value,
    label: str,
    dof_names: tuple[str, ...],
    *,
    default: float | None = None,
    minimum: float | None = None,
) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ValueError(f"UFO model config {label} must be a mapping")
    result = np.full(len(dof_names), np.nan if default is None else default, dtype=np.float64)
    matched_by: list[str | None] = [None] * len(dof_names)
    for pattern, raw_value in value.items():
        if not isinstance(pattern, str):
            raise ValueError(f"UFO model config {label} patterns must be strings")
        try:
            matcher = re.compile(pattern)
        except re.error as error:
            raise ValueError(f"UFO model config {label} has invalid pattern {pattern!r}: {error}") from error
        indexes = [index for index, name in enumerate(dof_names) if matcher.fullmatch(name)]
        if not indexes:
            raise ValueError(f"UFO model config {label} pattern {pattern!r} matches no joints")
        joint_value = _finite_float(raw_value, f"{label}[{pattern!r}]")
        if minimum is not None and joint_value < minimum:
            raise ValueError(f"UFO model config {label}[{pattern!r}] must be >= {minimum}")
        for index in indexes:
            if matched_by[index] is not None:
                raise ValueError(
                    f"UFO model config {label} patterns {matched_by[index]!r} and {pattern!r} "
                    f"both match {dof_names[index]!r}"
                )
            result[index] = joint_value
            matched_by[index] = pattern
    missing = [name for index, name in enumerate(dof_names) if not np.isfinite(result[index])]
    if missing:
        raise ValueError(f"UFO model config {label} has no value for joints: {missing}")
    return result


def _load_model_config(path: str, dof_names: tuple[str, ...]) -> _UfoModelConfig:
    config_path = Path(path).expanduser()
    try:
        with config_path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Failed to load UFO model config {config_path}: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("UFO model config root must be a mapping")

    required = {"action_rescale", "action_scale", "joint_kp", "joint_kd", "default_joint_pos"}
    missing = sorted(required - loaded.keys())
    if missing:
        raise ValueError(f"UFO model config is missing fields: {missing}")
    action_rescale = _finite_float(loaded["action_rescale"], "action_rescale")
    if action_rescale <= 0.0:
        raise ValueError("UFO model config action_rescale must be > 0")
    return _UfoModelConfig(
        action_rescale=action_rescale,
        action_scale=_joint_values(loaded["action_scale"], "action_scale", dof_names, minimum=0.0),
        kp=_joint_values(loaded["joint_kp"], "joint_kp", dof_names, minimum=0.0),
        kd=_joint_values(loaded["joint_kd"], "joint_kd", dof_names, minimum=0.0),
        default_dof_angles=_joint_values(loaded["default_joint_pos"], "default_joint_pos", dof_names, default=0.0),
    )


def _latent_vector(value, label: str) -> np.ndarray:
    latent = np.asarray(value, dtype=np.float32)
    if latent.shape == (1, 256):
        latent = latent[0]
    if latent.shape != (256,):
        raise ValueError(f"{label} must have shape (256,) or (1, 256), got {latent.shape}")
    if not np.isfinite(latent).all():
        raise ValueError(f"{label} contains non-finite values")
    return latent.copy()


class UfoPolicy(BasePolicy):
    """Offline tracking/reward/goal adapter for UFO-Deploy's G1 policy."""

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, UfoTaskConfig):
            raise TypeError("UfoPolicy requires UfoTaskConfig")
        if tuple(config.robot.dof_names) != DOF_NAMES:
            raise ValueError("UFO requires the released G1 29-DoF joint order")
        actor_terms = tuple(config.observation.obs_dict.get("actor_obs", ()))
        if actor_terms != _ACTOR_TERMS or set(config.observation.obs_dict) != {"actor_obs"}:
            raise ValueError("UFO actor_obs terms or order do not match the released model")
        for term, expected in _OBS_DIMS.items():
            if config.observation.obs_dims.get(term) != expected:
                raise ValueError(f"UFO observation {term!r} must have dimension {expected}")
            if term not in config.observation.obs_scales:
                raise ValueError(f"UFO observation {term!r} must define a scale")
        if config.observation.history_length_dict != {"actor_obs": 1}:
            raise ValueError("UFO observation history_length_dict must be {'actor_obs': 1}")

        super().__init__(config)
        self.ufo_task = config.task
        model_config = _load_model_config(config.task.model_config, DOF_NAMES)
        self.action_rescale = model_config.action_rescale
        self.action_scale = model_config.action_scale
        self.kp = model_config.kp
        self.kd = model_config.kd
        self.default_dof_angles = model_config.default_dof_angles
        self.initial_pose = InitialPose(
            dof_names=self.dof_names,
            dof_pos=tuple(self.default_dof_angles),
            root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        self.rl_rate = config.task.rl_rate
        self.rl_dt = 1.0 / self.rl_rate
        if config.guard:
            self.guard = InitialPoseGuard(
                config.guard,
                self.initial_pose,
                self.dof_names,
                self.logger,
                reason_prefix="ufo_start_check_failed",
            )
        else:
            self.guard = None

        self.onnx_policy_session = shared_session(config.task.model_path, ort_providers(config.task.inference_provider))
        self._validate_session()
        self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name

        self._tracking_context: np.ndarray | None = None
        self._fixed_context: np.ndarray | None = None
        self._tracking_end = 0
        self._load_context()

        self._history = {
            "prev_actions": np.zeros((4, 29), dtype=np.float32),
            "base_ang_vel": np.zeros((4, 3), dtype=np.float32),
            "dof_pos_minus_default": np.zeros((4, 29), dtype=np.float32),
            "dof_vel": np.zeros((4, 29), dtype=np.float32),
            "projected_gravity": np.zeros((4, 3), dtype=np.float32),
        }
        self.last_action = np.zeros(29, dtype=np.float32)
        self._last_cmd_q: np.ndarray | None = None
        self._activation_q: np.ndarray | None = None
        self._initializing = False
        self._startup_interpolator = JointPositionInterpolator(
            joint_lower=G1_JOINT_LOWER,
            joint_upper=G1_JOINT_UPPER,
            joint_velocity=G1_JOINT_VELOCITY,
            rate_hz=self.rl_rate,
            duration_s=config.task.init_duration_s,
            slew_safety_factor=config.robot.joint_interpolation_slew_safety_factor,
        )
        self._tracking_frame = 0
        self._tracking_playing = False

    def _validate_session(self) -> None:
        inputs = self.onnx_policy_session.get_inputs()
        outputs = self.onnx_policy_session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 721] or inputs[0].type != "tensor(float)":
            raise ValueError("UFO policy must have one float input with shape [1, 721]")
        if len(outputs) != 1 or outputs[0].shape != [1, 29] or outputs[0].type != "tensor(float)":
            raise ValueError("UFO policy must have one float output with shape [1, 29]")

    def _read_context(self):
        try:
            return joblib.load(self.ufo_task.context.path)
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise ValueError(
                    "UFO reward context requires Torch; use the released reward_locomotion_numpy.pkl file"
                ) from error
            raise

    def _load_context(self) -> None:
        context_config = self.ufo_task.context
        loaded = self._read_context()
        if isinstance(context_config, UfoTrackingContextConfig):
            context = np.asarray(loaded, dtype=np.float32)
            if context.ndim != 2 or context.shape[1] != 256 or context.shape[0] == 0:
                raise ValueError(f"UFO tracking context must have shape (frames, 256), got {context.shape}")
            if not np.isfinite(context).all():
                raise ValueError("UFO tracking context contains non-finite values")
            end = context.shape[0] if context_config.end_frame is None else context_config.end_frame
            if context_config.start_frame >= end or end > context.shape[0]:
                raise ValueError("UFO tracking start/end frames are outside the context")
            if context_config.stop_frame >= context.shape[0]:
                raise ValueError("UFO tracking stop_frame is outside the context")
            self._tracking_context = context
            self._tracking_end = end
            self._tracking_frame = context_config.start_frame
            return
        if not isinstance(loaded, dict):
            raise ValueError("UFO reward/goal context must contain a dictionary")
        if context_config.name not in loaded:
            raise ValueError(f"UFO context does not contain {context_config.name!r}")
        selected = loaded[context_config.name]
        if isinstance(context_config, UfoRewardContextConfig):
            if not isinstance(selected, (list, tuple)) or context_config.z_id >= len(selected):
                raise ValueError(f"UFO reward {context_config.name!r} does not contain z_id={context_config.z_id}")
            selected = selected[context_config.z_id]
        elif not isinstance(context_config, UfoGoalContextConfig):
            raise TypeError(f"Unsupported UFO context config: {type(context_config).__name__}")
        self._fixed_context = _latent_vector(selected, f"UFO context {context_config.name!r}")

    def _reset_history(self) -> None:
        for history in self._history.values():
            history.fill(0.0)
        self.last_action.fill(0.0)

    @staticmethod
    def _projected_gravity(robot_state: LowState) -> np.ndarray:
        quaternion = np.asarray(robot_state.base_quat, dtype=np.float64)
        quaternion_norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if not np.isfinite(quaternion_norm).all() or np.any(quaternion_norm < 1e-8):
            raise PolicyRuntimeFault("ufo_invalid_base_quaternion")
        gravity = quat_rotate_inverse(
            quaternion / quaternion_norm,
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64),
        )[0]
        if not np.isfinite(gravity).all():
            raise PolicyRuntimeFault("ufo_invalid_projected_gravity")
        return gravity.astype(np.float32)

    def _prefill_startup_history(self, robot_state: LowState, joint_pos: np.ndarray) -> None:
        startup_action = np.clip(
            (joint_pos - self.default_dof_angles) / self.action_scale,
            -self.action_rescale,
            self.action_rescale,
        ).astype(np.float32)
        self.last_action = startup_action
        terms = self._observation_terms(robot_state)
        for name, history in self._history.items():
            history[:] = np.asarray(terms[name], dtype=np.float32)

    def _on_activate(self, robot_state: LowState) -> str | None:
        joint_pos = np.asarray(robot_state.joint_pos[0], dtype=np.float64)
        self._reset_history()
        self._activation_q = joint_pos.copy()
        self._last_cmd_q = None
        if isinstance(self.ufo_task.context, UfoTrackingContextConfig):
            self._tracking_frame = self.ufo_task.context.start_frame
        if self.ufo_task.startup_mode == "prefill":
            self._startup_interpolator.clear()
            try:
                self._prefill_startup_history(robot_state, joint_pos)
            except PolicyRuntimeFault as error:
                self._activation_q = None
                self._reset_history()
                return f"ufo_start_failed: {error}"
            self._initializing = False
            self._tracking_playing = isinstance(self.ufo_task.context, UfoTrackingContextConfig)
            self.logger.info("UFO observation history prefilled; policy action enabled")
        else:
            try:
                self._startup_interpolator.reset(joint_pos, self.default_dof_angles)
            except ValueError as error:
                self._activation_q = None
                self._startup_interpolator.clear()
                self._reset_history()
                return f"ufo_start_failed: {error}"
            self._initializing = True
            self._tracking_playing = False
            self.logger.info(f"UFO initialization started ({self.ufo_task.init_duration_s:.1f}s)")
        return None

    def _on_deactivate(self) -> None:
        self._initializing = False
        self._tracking_playing = False
        self._activation_q = None
        self._last_cmd_q = None
        self._startup_interpolator.clear()
        self._reset_history()

    def _apply_control(self, control: Mapping[str, float]) -> None:
        if control:
            raise PolicyRuntimeFault("ufo_unexpected_control_input")

    def _tracking_latent(self) -> np.ndarray:
        context_config = self.ufo_task.context
        context = self._tracking_context
        if not isinstance(context_config, UfoTrackingContextConfig) or context is None:
            raise PolicyRuntimeFault("ufo_tracking_context_unavailable")
        if self._initializing or not self._tracking_playing:
            return context[context_config.stop_frame].copy()

        start = self._tracking_frame
        stop = min(start + context_config.window_size, self._tracking_end)
        window = context[start:stop]
        discounts = context_config.gamma ** np.arange(window.shape[0], dtype=np.float32)
        discounts /= discounts.sum()
        latent = np.sum(window * discounts[:, None], axis=0)
        norm = float(np.linalg.norm(latent))
        reference_norm = float(np.linalg.norm(context[0]))
        if not np.isfinite(norm) or norm < 1e-8 or not np.isfinite(reference_norm):
            raise PolicyRuntimeFault("ufo_tracking_context_normalization_failed")
        latent = latent / norm * reference_norm

        self._tracking_frame += 1
        if self._tracking_frame >= self._tracking_end:
            self._tracking_playing = False
            self._tracking_frame = context_config.stop_frame
        return latent.astype(np.float32, copy=False)

    def _latent(self) -> np.ndarray:
        if isinstance(self.ufo_task.context, UfoTrackingContextConfig):
            return self._tracking_latent()
        if self._fixed_context is None:
            raise PolicyRuntimeFault("ufo_fixed_context_unavailable")
        return self._fixed_context

    def _observation_terms(self, robot_state: LowState) -> dict[str, np.ndarray]:
        joint_pos = np.asarray(robot_state.joint_pos[0], dtype=np.float32)
        joint_vel = np.asarray(robot_state.joint_vel[0], dtype=np.float32)
        if self.ufo_task.debug.force_zero_angular_velocity:
            base_ang_vel = np.zeros(3, dtype=np.float32)
        else:
            base_ang_vel = np.asarray(robot_state.base_ang_vel[0], dtype=np.float32)
        if self.ufo_task.debug.force_upright_imu:
            projected_gravity = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        else:
            projected_gravity = self._projected_gravity(robot_state)
        dof_pos_minus_default = joint_pos - self.default_dof_angles.astype(np.float32)
        current = {
            "prev_actions": self.last_action.copy(),
            "base_ang_vel": base_ang_vel,
            "dof_pos_minus_default": dof_pos_minus_default,
            "dof_vel": joint_vel,
            "projected_gravity": projected_gravity,
        }
        if not all(np.isfinite(value).all() for value in current.values()):
            raise PolicyRuntimeFault("ufo_non_finite_robot_state")
        for name, value in current.items():
            history = self._history[name]
            history[1:] = history[:-1]
            history[0] = value
        return {
            "dof_pos_minus_default": dof_pos_minus_default,
            "dof_vel": joint_vel,
            "projected_gravity": projected_gravity,
            "base_ang_vel": base_ang_vel,
            "prev_actions": self.last_action,
            "prev_actions_history": self._history["prev_actions"].reshape(-1),
            "base_ang_vel_history": self._history["base_ang_vel"].reshape(-1),
            "dof_pos_minus_default_history": self._history["dof_pos_minus_default"].reshape(-1),
            "dof_vel_history": self._history["dof_vel"].reshape(-1),
            "projected_gravity_history": self._history["projected_gravity"].reshape(-1),
        }

    def prepare_obs_for_rl(self, robot_state: LowState) -> np.ndarray:
        terms = self._observation_terms(robot_state)
        scales = self.config.observation.obs_scales
        proprioception = np.concatenate(
            [np.asarray(terms[name], dtype=np.float32) * scales[name] for name in _ACTOR_TERMS]
        ).reshape(1, -1)
        inputs = np.concatenate((proprioception, self._latent().reshape(1, 256)), axis=1).astype(np.float32, copy=False)
        if inputs.shape != (1, 721) or not np.isfinite(inputs).all():
            raise PolicyRuntimeFault(f"ufo_invalid_observation:{inputs.shape}")
        if self.ufo_task.print_observations:
            self.logger.info(f"UFO actor_obs={inputs}")
        return inputs

    def _infer(self, inputs: np.ndarray) -> np.ndarray:
        try:
            output = self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: inputs})[0]
        except Exception as error:
            raise PolicyRuntimeFault(f"ufo_inference_failed:{type(error).__name__}") from error
        action = np.asarray(output, dtype=np.float32)
        if action.shape != (1, 29):
            raise PolicyRuntimeFault(f"ufo_invalid_action_shape:{action.shape}")
        if not np.isfinite(action).all():
            raise PolicyRuntimeFault("ufo_non_finite_action")
        action = np.clip(action, -1.0, 1.0)
        if self.ufo_task.debug.force_zero_action:
            action.fill(0.0)
        self.last_action = (self.action_rescale * action[0]).astype(np.float32, copy=False)
        return self.last_action.astype(np.float64) * self.action_scale + self.default_dof_angles

    def _slew_limit(self, q_target: np.ndarray, robot_state: LowState) -> np.ndarray:
        try:
            return limit_joint_position_target(
                q_target,
                current_q=robot_state.joint_pos[0],
                previous_q=self._last_cmd_q,
                joint_lower=G1_JOINT_LOWER,
                joint_upper=G1_JOINT_UPPER,
                joint_velocity=G1_JOINT_VELOCITY,
                dt=self.rl_dt,
                slew_safety_factor=self.ufo_task.q_target_slew_safety_factor,
            )
        except ValueError as error:
            raise PolicyRuntimeFault(f"ufo_invalid_q_target: {error}") from error

    def _compute_command(self, robot_state: LowState) -> PolicyJointCommand:
        if self._activation_q is None:
            raise PolicyRuntimeFault("ufo_policy_not_active")
        with self.latency_tracker.measure(LatencyStage.PREPROCESSING):
            inputs = self.prepare_obs_for_rl(robot_state)
        with self.latency_tracker.measure(LatencyStage.INFERENCE):
            policy_target = self._infer(inputs)
        with self.latency_tracker.measure(LatencyStage.POSTPROCESSING):
            if self._initializing:
                try:
                    interpolation = self._startup_interpolator.next(robot_state.joint_pos[0])
                except (RuntimeError, ValueError) as error:
                    raise PolicyRuntimeFault(f"ufo_interpolation_failed: {error}") from error
                q_target = interpolation.q_target
                if interpolation.complete:
                    self._initializing = False
                    if isinstance(self.ufo_task.context, UfoTrackingContextConfig):
                        self._tracking_frame = self.ufo_task.context.start_frame
                        self._tracking_playing = True
                    self.logger.info("UFO initialization complete; policy action enabled")
            else:
                q_target = self._slew_limit(policy_target, robot_state)
            if not np.isfinite(q_target).all():
                raise PolicyRuntimeFault("ufo_non_finite_q_target")
            self._last_cmd_q = q_target.copy()
        return position_command(q_target, self.kp, self.kd, self.controlled_joint_mask)


__all__ = ["UfoPolicy"]
