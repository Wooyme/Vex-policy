"""Vex runtime adapter for the released UFO-Deploy G1 policy."""

from __future__ import annotations

import threading
from collections.abc import Mapping

import joblib
import numpy as np
import onnxruntime
from loguru import logger

from vex_policy.config.config_types import (
    InferenceConfig,
    UfoGoalContextConfig,
    UfoGuardConfig,
    UfoRewardContextConfig,
    UfoTaskConfig,
    UfoTrackingContextConfig,
)
from vex_policy.policies.base import BasePolicy, PolicyJointCommand, PolicyRuntimeFault
from vex_policy.policies.guard.ufo import UfoGuard
from vex_policy.policies.sonic_planner import ort_providers
from vex_policy.robots.g1 import DOF_NAMES
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse

_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], onnxruntime.InferenceSession] = {}
_SESSION_CACHE_LOCK = threading.Lock()

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
_ACTION_RESCALE = 5.0

# These values are the policy contract from UFO-Deploy's G1 release, not Vex's
# generic G1 standing configuration.
_DEFAULT_DOF_ANGLES = np.asarray(
    (
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    dtype=np.float64,
)
_ACTION_SCALE = np.asarray(
    (
        0.35066146,
        0.35066146,
        0.54754644,
        0.35066146,
        0.43857726,
        0.43857726,
        0.35066146,
        0.35066146,
        0.54754644,
        0.35066146,
        0.43857726,
        0.43857726,
        0.07333333,
        0.04166667,
        0.04166667,
        0.43857741,
        0.43857741,
        0.43857741,
        0.43857741,
        0.43857741,
        0.38903592,
        0.38903592,
        0.43857741,
        0.43857741,
        0.43857741,
        0.43857741,
        0.43857741,
        0.38903592,
        0.38903592,
    ),
    dtype=np.float64,
)
_KP = np.asarray(
    (
        99.09843,
        99.0984,
        40.1792,
        99.0984,
        28.5012,
        28.5012,
        99.09843,
        99.0984,
        40.1792,
        99.0984,
        28.5012,
        28.5012,
        300.0,
        300.0,
        300.0,
        14.2506,
        14.2506,
        14.2506,
        14.2506,
        14.25062,
        8.61103,
        8.61103,
        14.2506,
        14.2506,
        14.2506,
        14.2506,
        14.25062,
        8.61103,
        8.61103,
    ),
    dtype=np.float64,
)
_KD = np.asarray(
    (
        6.3088,
        6.3088,
        2.5579,
        6.3088,
        1.8145,
        1.8145,
        6.3088,
        6.3088,
        2.5579,
        6.3088,
        1.8145,
        1.8145,
        5.0,
        5.0,
        5.0,
        0.9072,
        0.9072,
        0.9072,
        0.9072,
        0.9072,
        0.5482,
        0.5482,
        0.9072,
        0.9072,
        0.9072,
        0.9072,
        0.9072,
        0.5482,
        0.5482,
    ),
    dtype=np.float64,
)
_JOINT_LOWER = np.asarray(
    (
        -2.5307,
        -0.5236,
        -2.7576,
        -0.0873,
        -0.8727,
        -0.2618,
        -2.5307,
        -2.9671,
        -2.7576,
        -0.0873,
        -0.8727,
        -0.2618,
        -2.618,
        -0.52,
        -0.52,
        -3.0892,
        -1.5882,
        -2.618,
        -1.0472,
        -1.9722,
        -1.6144,
        -1.6144,
        -3.0892,
        -2.2515,
        -2.618,
        -1.0472,
        -1.9722,
        -1.6144,
        -1.6144,
    ),
    dtype=np.float64,
)
_JOINT_UPPER = np.asarray(
    (
        2.8798,
        2.9671,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.8798,
        0.5236,
        2.7576,
        2.8798,
        0.5236,
        0.2618,
        2.618,
        0.52,
        0.52,
        2.6704,
        2.2515,
        2.618,
        2.0944,
        1.9722,
        1.6144,
        1.6144,
        2.6704,
        1.5882,
        2.618,
        2.0944,
        1.9722,
        1.6144,
        1.6144,
    ),
    dtype=np.float64,
)
_JOINT_VELOCITY = np.asarray(
    (
        32.0,
        32.0,
        32.0,
        20.0,
        37.0,
        37.0,
        32.0,
        32.0,
        32.0,
        20.0,
        37.0,
        37.0,
        32.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        22.0,
        22.0,
        37.0,
        37.0,
        37.0,
        37.0,
        37.0,
        22.0,
        22.0,
    ),
    dtype=np.float64,
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

        self.config = config
        self.ufo_task = config.task
        self.logger = logger
        self._init_robot_config(config.robot)
        self.default_dof_angles = _DEFAULT_DOF_ANGLES.copy()
        self.rl_rate = config.task.rl_rate
        self.rl_dt = 1.0 / self.rl_rate
        self.use_phase = False
        self._init_latency_tracking()
        if config.guard:
            self.guard = UfoGuard(config.guard, self)
        else:
            self.guard = None

        self.onnx_policy_session = _shared_session(config.task.model_path, config.task.inference_provider)
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
        self._init_step = 0
        self._init_steps = max(1, round(config.task.init_duration_s * self.rl_rate))
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
            (joint_pos - _DEFAULT_DOF_ANGLES) / _ACTION_SCALE,
            -_ACTION_RESCALE,
            _ACTION_RESCALE,
        ).astype(np.float32)
        self.last_action = startup_action
        terms = self._observation_terms(robot_state)
        for name, history in self._history.items():
            history[:] = np.asarray(terms[name], dtype=np.float32)

    def activate(self, robot_state: LowState) -> str | None:
        if self.guard:
            result, reason = self.guard.start_check(robot_state)
            if not result:
                return reason
        joint_pos = np.asarray(robot_state.joint_pos[0], dtype=np.float64)
        self._reset_history()
        self._activation_q = joint_pos.copy()
        self._last_cmd_q = None
        self._init_step = 0
        if isinstance(self.ufo_task.context, UfoTrackingContextConfig):
            self._tracking_frame = self.ufo_task.context.start_frame
        if self.ufo_task.startup_mode == "prefill":
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
            self._initializing = True
            self._tracking_playing = False
            self.logger.info(f"UFO initialization started ({self.ufo_task.init_duration_s:.1f}s)")
        return None

    def deactivate(self) -> None:
        self._initializing = False
        self._tracking_playing = False
        self._activation_q = None
        self._last_cmd_q = None
        self._reset_history()

    def apply_control(self, control: Mapping[str, float]) -> None:
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
        dof_pos_minus_default = joint_pos - _DEFAULT_DOF_ANGLES.astype(np.float32)
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
        self.last_action = (_ACTION_RESCALE * action[0]).astype(np.float32, copy=False)
        return self.last_action.astype(np.float64) * _ACTION_SCALE + _DEFAULT_DOF_ANGLES

    def _slew_limit(self, q_target: np.ndarray, robot_state: LowState) -> np.ndarray:
        baseline = self._last_cmd_q
        if baseline is None:
            baseline = np.asarray(robot_state.joint_pos[0], dtype=np.float64)
        max_delta = _JOINT_VELOCITY * self.rl_dt * self.ufo_task.q_target_slew_safety_factor
        return baseline + np.clip(q_target - baseline, -max_delta, max_delta)

    def step(self, robot_state: LowState) -> PolicyJointCommand:
        if self._activation_q is None:
            raise PolicyRuntimeFault("ufo_policy_not_active")
        self.latency_tracker.start_cycle()
        try:
            with self.latency_tracker.measure("preprocessing"):
                inputs = self.prepare_obs_for_rl(robot_state)
            with self.latency_tracker.measure("inference"):
                policy_target = self._infer(inputs)
            with self.latency_tracker.measure("postprocessing"):
                if self._initializing:
                    alpha = min((self._init_step + 1) / self._init_steps, 1.0)
                    q_target = self._activation_q + (_DEFAULT_DOF_ANGLES - self._activation_q) * alpha
                    self._init_step += 1
                    if self._init_step >= self._init_steps:
                        self._initializing = False
                        if isinstance(self.ufo_task.context, UfoTrackingContextConfig):
                            self._tracking_frame = self.ufo_task.context.start_frame
                            self._tracking_playing = True
                        self.logger.info("UFO initialization complete; policy action enabled")
                else:
                    q_target = policy_target
                if not np.isfinite(q_target).all():
                    raise PolicyRuntimeFault("ufo_non_finite_q_target")
                q_target = np.clip(q_target, _JOINT_LOWER, _JOINT_UPPER)
                q_target = self._slew_limit(q_target, robot_state)
                q_target = np.clip(q_target, _JOINT_LOWER, _JOINT_UPPER)
                if not np.isfinite(q_target).all():
                    raise PolicyRuntimeFault("ufo_non_finite_q_target_after_slew")
                self._last_cmd_q = q_target.copy()
            return PolicyJointCommand(
                q=q_target,
                dq=np.zeros(29, dtype=np.float64),
                tau=np.zeros(29, dtype=np.float64),
                kp=_KP.copy(),
                kd=_KD.copy(),
                controlled_joints=self.controlled_joint_mask.copy(),
            )
        finally:
            self.latency_tracker.end_cycle()

    def get_reference_state(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        """UFO offline contexts do not own background resources."""


__all__ = ["UfoPolicy"]
