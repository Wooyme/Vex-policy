"""Inference adapter for Holosoma's pelvis-sine waist locomotion task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from vex_policy.config.config_types import (
    InferenceConfig,
    SliderInput,
    WaistLocomotionGuardConfig,
    WaistLocomotionTaskConfig,
    input_parameters,
)
from vex_policy.policies.guard.waist_locomotion import WaistLocomotionGuard
from vex_policy.robots import G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.latency import LatencyStage

from .base import BasePolicy, PolicyRuntimeFault
from vex_policy.policies.utils.inference import OnnxActor, resolve_control_gains
from vex_policy.policies.utils.joint_command import PositionAction, position_command
from vex_policy.policies.utils.locomotion_utils import MotionInitialPose as WaistInitialPose  # noqa: F401
from vex_policy.policies.utils.locomotion_utils import RightAnkleKinematics
from vex_policy.policies.utils.locomotion_utils import load_motion_last_pose as load_waist_motion_last_pose
from .observations import ObservationHistory, robot_observation_terms


def _quat_to_rotation_vector(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert normalized WXYZ quaternions to shortest-path rotation vectors."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion *= np.where(quaternion[:, :1] < 0.0, -1.0, 1.0)
    vector = quaternion[:, 1:]
    magnitude = np.linalg.norm(vector, axis=1)
    half_angle = np.arctan2(magnitude, quaternion[:, 0])
    angle = 2.0 * half_angle
    scale = np.empty_like(angle)
    regular = np.abs(angle) > 1e-8
    scale[regular] = angle[regular] / np.sin(half_angle[regular])
    scale[~regular] = 1.0 / (0.5 - angle[~regular] ** 2 / 48.0)
    return vector * scale[:, None]


def _relative_rotation_vector(initial_wxyz: np.ndarray, current_wxyz: np.ndarray) -> np.ndarray:
    """Return ``initial^-1 * current`` as a shortest-path rotation vector."""
    initial_scalar, initial_vector = initial_wxyz[:, :1], initial_wxyz[:, 1:]
    current_scalar, current_vector = current_wxyz[:, :1], current_wxyz[:, 1:]
    relative_quat = np.concatenate(
        (
            initial_scalar * current_scalar + np.sum(initial_vector * current_vector, axis=1, keepdims=True),
            initial_scalar * current_vector
            - current_scalar * initial_vector
            - np.cross(initial_vector, current_vector),
        ),
        axis=1,
    )
    return _quat_to_rotation_vector(relative_quat)


class WaistLocomotionPolicy(BasePolicy):
    """Full-body inference for a commanded sinusoidal pelvis trajectory."""

    _OBS_DIMS: ClassVar[dict[str, int]] = {
        "actions": 29,
        "base_ang_vel": 3,
        "base_right_foot_height_difference": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "pelvis_orientation_error": 3,
        "pelvis_sine_command": 8,
        "projected_gravity": 3,
    }
    _OBS_SCALES: ClassVar[dict[str, float]] = {
        "actions": 1.0,
        "base_ang_vel": 0.25,
        "base_right_foot_height_difference": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "pelvis_orientation_error": 1.0,
        "pelvis_sine_command": 1.0,
        "projected_gravity": 1.0,
    }

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, WaistLocomotionTaskConfig):
            raise TypeError("WaistLocomotionPolicy requires WaistLocomotionTaskConfig")
        if not isinstance(config.guard, WaistLocomotionGuardConfig):
            raise TypeError("WaistLocomotionPolicy requires WaistLocomotionGuardConfig")
        if not all(isinstance(component, SliderInput) for component in config.inputs):
            raise ValueError("Waist locomotion inputs must contain only sliders")
        parameters = {parameter.name: parameter for parameter in input_parameters(config.inputs)}
        expected_parameters = {
            "amplitude",
            "frequency",
            "height_delta",
            "x",
            "y",
            "z",
        }
        if set(parameters) != expected_parameters:
            raise ValueError(f"Waist locomotion inputs must be {sorted(expected_parameters)}")
        if parameters["amplitude"].min <= 0.0 or parameters["frequency"].min <= 0.0:
            raise ValueError("Waist locomotion amplitude and frequency ranges must be positive")
        default_direction = np.asarray([parameters[name].default for name in ("x", "y", "z")], dtype=np.float32)
        if np.linalg.norm(default_direction) < 1e-8:
            raise ValueError("Waist locomotion default direction must be non-zero")
        self.waist_input_parameters = parameters
        self.waist_task = config.task
        self.initial_pose = load_waist_motion_last_pose(self.waist_task.motion_data_path)
        super().__init__(config)
        self._load_initial_joint_pose()
        self.observations = ObservationHistory(config.observation)
        self._validate_observations()
        if not config.task.use_phase:
            raise ValueError("Waist locomotion requires task.use_phase=true")
        self.actor = OnnxActor(config.task.model_path)
        self._validate_model()
        self.robot_config = resolve_control_gains(
            config.robot, self.actor.metadata.get("kp"), self.actor.metadata.get("kd")
        )
        self.actions = PositionAction(
            self.num_dofs,
            self.action_mask,
            require_full_body=config.action_mask is not None,
            force_zero=config.task.debug.force_zero_action,
        )
        self._init_commands()
        self.guard = WaistLocomotionGuard(config.guard, self)

    def _load_initial_joint_pose(self) -> None:
        source_names = self.initial_pose.dof_names
        expected_names = tuple(self.dof_names)
        missing = sorted(set(expected_names) - set(source_names))
        extra = sorted(set(source_names) - set(expected_names))
        if missing or extra or len(source_names) != self.num_dofs:
            raise ValueError(f"Waist locomotion motion joint names mismatch: missing={missing}, extra={extra}")
        source_indices = {name: index for index, name in enumerate(source_names)}
        hardware_order = [source_indices[name] for name in expected_names]
        self.default_dof_angles = np.asarray(self.initial_pose.dof_pos, dtype=np.float64)[hardware_order]

    def _validate_observations(self) -> None:
        actor_terms = self.observations.obs_terms_sorted.get("actor_obs")
        expected_terms = sorted(self._OBS_DIMS)
        if actor_terms != expected_terms:
            raise ValueError(f"Waist locomotion actor_obs terms must be {expected_terms}, got {actor_terms}")
        if self.observations.history_length_dict.get("actor_obs", 1) != 1:
            raise ValueError("Waist locomotion actor_obs history length must be 1")
        for term, expected_dim in self._OBS_DIMS.items():
            actual_dim = self.observations.obs_dims.get(term)
            if actual_dim != expected_dim:
                raise ValueError(f"Waist locomotion observation {term!r} must have dimension {expected_dim}")
            actual_scale = self.observations.obs_scales.get(term)
            if actual_scale is None or not np.isclose(actual_scale, self._OBS_SCALES[term]):
                raise ValueError(f"Waist locomotion observation {term!r} must use scale {self._OBS_SCALES[term]}")
        if self.observations.obs_dim_dict["actor_obs"] != 105:
            raise ValueError("Waist locomotion actor_obs must have dimension 105")

    def _init_commands(self) -> None:
        self.pelvis_sine_phase = 0.0
        self.pelvis_sine_command = np.zeros((1, 8), dtype=np.float32)
        self.pelvis_orientation_reference_quat: np.ndarray | None = None
        self.initial_base_right_foot_height_difference: float | None = None
        self._reset_pelvis_sine_command()

    def _validate_model(self) -> None:
        inputs = self.actor.session.get_inputs()
        outputs = self.actor.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "actor_obs" or list(inputs[0].shape) != [1, 105]:
            exposed = [(item.name, item.shape) for item in inputs]
            raise ValueError(f"Waist locomotion model must expose actor_obs[1, 105], got {exposed}")
        if len(outputs) != 1 or outputs[0].name != "action" or list(outputs[0].shape) != [1, 29]:
            exposed = [(item.name, item.shape) for item in outputs]
            raise ValueError(f"Waist locomotion model must expose action[1, 29], got {exposed}")

        metadata = self.actor.metadata
        model_dof_names = tuple(metadata.get("dof_names", ()))
        if model_dof_names != tuple(self.dof_names):
            raise ValueError("Waist locomotion ONNX dof_names do not match the robot joint order")
        action_scale = np.asarray(metadata.get("action_scale", ()), dtype=np.float64)
        if action_scale.shape != (self.num_dofs,):
            raise ValueError(f"Waist locomotion ONNX action_scale must have {self.num_dofs} values")
        if not np.allclose(action_scale, self.config.task.policy_action_scale):
            raise ValueError(
                "Waist locomotion ONNX action_scale does not match task.policy_action_scale "
                f"({self.config.task.policy_action_scale})"
            )

        self._ankle_kinematics = RightAnkleKinematics(metadata.get("robot_urdf"), self.dof_names)

    def _reset_pelvis_sine_command(self) -> None:
        self.pelvis_sine_phase = 0.0
        direction = np.asarray(
            [self.waist_input_parameters[name].default for name in ("x", "y", "z")],
            dtype=np.float32,
        )
        direction /= np.linalg.norm(direction)
        initial_height = self.initial_base_right_foot_height_difference or 0.0
        self.pelvis_sine_command[0] = (
            0.0,
            1.0,
            self.waist_input_parameters["amplitude"].default,
            self.waist_input_parameters["frequency"].default,
            *direction,
            initial_height + self.waist_input_parameters["height_delta"].default,
        )

    def _on_activate(self, robot_state_data: LowState) -> None:
        reference_quat = np.asarray(robot_state_data.base_quat, dtype=np.float64).copy()
        reference_norm = np.linalg.norm(reference_quat, axis=1, keepdims=True)
        if not np.isfinite(reference_quat).all() or np.any(reference_norm < 1e-8):
            raise PolicyRuntimeFault("Cannot capture pelvis orientation from an invalid quaternion")
        self.pelvis_orientation_reference_quat = reference_quat / reference_norm
        base_observations = robot_observation_terms(robot_state_data, self.default_dof_angles, self.config.task.debug)
        initial_height = self._base_right_foot_height_difference(
            robot_state_data, base_observations["projected_gravity"]
        )
        self.initial_base_right_foot_height_difference = float(initial_height[0, 0])
        self.actions.reset()
        self.observations.reset()
        self._reset_pelvis_sine_command()

    def update_phase_time(self) -> None:
        frequency_hz = float(self.pelvis_sine_command[0, 3])
        self.pelvis_sine_phase += 2.0 * np.pi * frequency_hz / self.rl_rate
        self.pelvis_sine_phase = float(np.fmod(self.pelvis_sine_phase + np.pi, 2.0 * np.pi) - np.pi)
        self.pelvis_sine_command[0, 0] = np.sin(self.pelvis_sine_phase)
        self.pelvis_sine_command[0, 1] = np.cos(self.pelvis_sine_phase)

    def _apply_control(self, control: Mapping[str, float]) -> None:
        amplitude = float(control["amplitude"])
        frequency = float(control["frequency"])
        direction = np.asarray([control["x"], control["y"], control["z"]], dtype=np.float32)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            direction = np.asarray(
                [self.waist_input_parameters[name].default for name in ("x", "y", "z")],
                dtype=np.float32,
            )
            direction_norm = float(np.linalg.norm(direction))
        direction /= direction_norm
        initial_height = self.initial_base_right_foot_height_difference
        if initial_height is None:
            raise RuntimeError("Initial base-right-foot height is unavailable; activate the policy first")
        target_height = initial_height + float(control["height_delta"])
        self.pelvis_sine_command[0, 2:] = (amplitude, frequency, *direction, target_height)

    def _base_right_foot_height_difference(
        self, robot_state_data: LowState, projected_gravity: np.ndarray
    ) -> np.ndarray:
        return self._ankle_kinematics.height_difference(robot_state_data, projected_gravity)

    def get_current_obs_buffer_dict(self, robot_state_data: LowState):
        observations = robot_observation_terms(robot_state_data, self.default_dof_angles, self.config.task.debug)
        observations["actions"] = self.actions.last
        observations["base_right_foot_height_difference"] = self._base_right_foot_height_difference(
            robot_state_data, observations["projected_gravity"]
        )
        current_quat = np.asarray(robot_state_data.base_quat, dtype=np.float64).copy()
        current_quat /= np.linalg.norm(current_quat, axis=1, keepdims=True).clip(min=1e-8)
        reference_quat = self.pelvis_orientation_reference_quat
        if reference_quat is None:
            raise RuntimeError("Pelvis orientation reference is unavailable; activate the policy first")
        observations["pelvis_orientation_error"] = _relative_rotation_vector(reference_quat, current_quat)
        observations["pelvis_sine_command"] = self.pelvis_sine_command
        return observations

    def _on_deactivate(self):
        self.observations.reset()
        self.actions.reset()
        self.pelvis_orientation_reference_quat = None
        self.initial_base_right_foot_height_difference = None
        self._reset_pelvis_sine_command()

    def _compute_command(self, robot_state_data):
        with self.latency_tracker.measure(LatencyStage.PREPROCESSING):
            self.update_phase_time()
            obs = self.observations.prepare(self.get_current_obs_buffer_dict(robot_state_data))
            if self.config.task.print_observations:
                self.observations.print_observations(obs, self.dof_names, self.actions.scaled)
        with self.latency_tracker.measure(LatencyStage.INFERENCE):
            action = self.actor({"actor_obs": obs["actor_obs"]})
        with self.latency_tracker.measure(LatencyStage.POSTPROCESSING):
            self.actions.process(action, self.config.task.policy_action_scale)
            q_target = np.clip(self.actions.target(self.default_dof_angles), G1_JOINT_LOWER, G1_JOINT_UPPER)
            self.actions.scaled = q_target - self.default_dof_angles
            return position_command(
                q_target, self.robot_config.motor_kp, self.robot_config.motor_kd, self.controlled_joint_mask
            )
