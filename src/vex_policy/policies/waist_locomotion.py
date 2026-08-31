"""Inference adapter for Holosoma's pelvis-sine waist locomotion task."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import numpy as np
import onnx
import pinocchio as pin
from pydantic import BaseModel, ConfigDict, model_validator

from vex_policy.config.config_types import (
    InferenceConfig,
    SliderInput,
    WaistLocomotionGuardConfig,
    WaistLocomotionTaskConfig,
    input_parameters,
)
from vex_policy.policies.guard.waist_locomotion import WaistLocomotionGuard
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse

from .base import BasePolicy


class WaistInitialPose(BaseModel):
    """Minimal training-pose data needed by inference and its startup guard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dof_names: tuple[str, ...]
    dof_pos: tuple[float, ...]
    root_quat_wxyz: tuple[float, float, float, float]
    projected_gravity: tuple[float, float, float]

    @model_validator(mode="after")
    def validate_values(self) -> WaistInitialPose:
        if not self.dof_names or len(self.dof_names) != len(self.dof_pos):
            raise ValueError("dof_names and dof_pos must have the same non-zero length")
        if len(set(self.dof_names)) != len(self.dof_names):
            raise ValueError("dof_names must not contain duplicates")
        values = np.asarray((*self.dof_pos, *self.root_quat_wxyz, *self.projected_gravity), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("initial pose values must be finite")
        quaternion_norm = float(np.linalg.norm(self.root_quat_wxyz))
        if not np.isclose(quaternion_norm, 1.0, atol=1e-3):
            raise ValueError("root_quat_wxyz must be a unit quaternion")
        gravity_norm = float(np.linalg.norm(self.projected_gravity))
        if not np.isclose(gravity_norm, 1.0, atol=1e-3):
            raise ValueError("projected_gravity must be a unit vector")
        return self


def load_waist_motion_last_pose(path: str | Path) -> WaistInitialPose:
    """Load and validate the final root/joint pose from a Holosoma motion NPZ."""
    motion_path = Path(path)
    if not motion_path.is_file():
        raise ValueError(f"Waist locomotion motion file does not exist: {motion_path}")
    try:
        with np.load(motion_path, allow_pickle=False) as motion:
            missing = {"joint_names", "joint_pos"} - set(motion.files)
            if missing:
                raise ValueError(f"Waist locomotion motion is missing arrays: {sorted(missing)}")
            joint_names = tuple(str(name) for name in motion["joint_names"].tolist())
            joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Waist locomotion motion"):
            raise
        raise ValueError(f"Failed to load waist locomotion motion {motion_path}: {error}") from error

    expected_width = len(joint_names) + 7
    if joint_pos.ndim != 2 or joint_pos.shape[0] == 0 or joint_pos.shape[1] != expected_width:
        raise ValueError(
            f"Waist locomotion joint_pos must have shape (frames, 7 + {len(joint_names)}), got {joint_pos.shape}"
        )
    final_pose = joint_pos[-1]
    root_quat_wxyz = final_pose[3:7]
    quaternion_norm = float(np.linalg.norm(root_quat_wxyz))
    if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-8:
        raise ValueError("Waist locomotion final-frame root quaternion is invalid")
    root_quat_wxyz = (root_quat_wxyz / quaternion_norm).reshape(1, 4)
    projected_gravity = quat_rotate_inverse(root_quat_wxyz, np.asarray([[0.0, 0.0, -1.0]]))[0]
    return WaistInitialPose(
        dof_names=joint_names,
        dof_pos=tuple(final_pose[7:]),
        root_quat_wxyz=tuple(root_quat_wxyz[0]),
        projected_gravity=tuple(projected_gravity),
    )


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
        self.guard = WaistLocomotionGuard(config.guard, self)

    def _init_robot_config(self, robot_config) -> None:
        super()._init_robot_config(robot_config)
        source_names = self.initial_pose.dof_names
        expected_names = tuple(self.dof_names)
        missing = sorted(set(expected_names) - set(source_names))
        extra = sorted(set(source_names) - set(expected_names))
        if missing or extra or len(source_names) != self.num_dofs:
            raise ValueError(f"Waist locomotion motion joint names mismatch: missing={missing}, extra={extra}")
        source_indices = {name: index for index, name in enumerate(source_names)}
        hardware_order = [source_indices[name] for name in expected_names]
        self.default_dof_angles = np.asarray(self.initial_pose.dof_pos, dtype=np.float64)[hardware_order]

    def _init_obs_config(self) -> None:
        super()._init_obs_config()
        actor_terms = self.obs_terms_sorted.get("actor_obs")
        expected_terms = sorted(self._OBS_DIMS)
        if actor_terms != expected_terms:
            raise ValueError(f"Waist locomotion actor_obs terms must be {expected_terms}, got {actor_terms}")
        if self.history_length_dict.get("actor_obs", 1) != 1:
            raise ValueError("Waist locomotion actor_obs history length must be 1")
        for term, expected_dim in self._OBS_DIMS.items():
            actual_dim = self.obs_dims.get(term)
            if actual_dim != expected_dim:
                raise ValueError(f"Waist locomotion observation {term!r} must have dimension {expected_dim}")
            actual_scale = self.obs_scales.get(term)
            if actual_scale is None or not np.isclose(actual_scale, self._OBS_SCALES[term]):
                raise ValueError(f"Waist locomotion observation {term!r} must use scale {self._OBS_SCALES[term]}")
        if self.obs_dim_dict["actor_obs"] != 105:
            raise ValueError("Waist locomotion actor_obs must have dimension 105")

    def _init_command_components(self) -> None:
        super()._init_command_components()
        self.pelvis_sine_phase = 0.0
        self.pelvis_sine_command = np.zeros((1, 8), dtype=np.float32)
        self.pelvis_orientation_reference_quat: np.ndarray | None = None
        self.initial_base_right_foot_height_difference: float | None = None
        self._reset_pelvis_sine_command()

    def _init_phase_components(self) -> None:
        self.use_phase = self.config.task.use_phase
        if not self.use_phase:
            raise ValueError("Waist locomotion requires task.use_phase=true")

    def _reset_inference_episode_state(self) -> None:
        """Remove action and observation state carried over from a prior activation."""

        self.last_policy_action.fill(0.0)
        self.scaled_policy_action.fill(0.0)
        for group_buffers in self.obs_history_buffers.values():
            for buffer in group_buffers.values():
                buffer.clear()
        self.obs_buf_dict = {group: np.zeros_like(buffer) for group, buffer in self.obs_buf_dict.items()}

    def setup_policy(self, model_path) -> None:
        super().setup_policy(model_path)
        inputs = self.onnx_policy_session.get_inputs()
        outputs = self.onnx_policy_session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "actor_obs" or list(inputs[0].shape) != [1, 105]:
            exposed = [(item.name, item.shape) for item in inputs]
            raise ValueError(f"Waist locomotion model must expose actor_obs[1, 105], got {exposed}")
        if len(outputs) != 1 or outputs[0].name != "action" or list(outputs[0].shape) != [1, 29]:
            exposed = [(item.name, item.shape) for item in outputs]
            raise ValueError(f"Waist locomotion model must expose action[1, 29], got {exposed}")

        model = onnx.load(model_path, load_external_data=False)
        metadata = {prop.key: json.loads(prop.value) for prop in model.metadata_props}
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

        robot_urdf = metadata.get("robot_urdf")
        if not isinstance(robot_urdf, str) or not robot_urdf.strip():
            raise ValueError("Waist locomotion ONNX must contain non-empty robot_urdf metadata")
        try:
            kinematics_model = pin.buildModelFromXML(robot_urdf)
        except Exception as error:
            raise ValueError(f"Failed to build waist locomotion kinematics from ONNX robot_urdf: {error}") from error

        q_indices: list[int] = []
        for name in self.dof_names:
            joint_id = kinematics_model.getJointId(name)
            if joint_id == 0 or kinematics_model.names[joint_id] != name:
                raise ValueError(f"Waist locomotion robot_urdf is missing joint {name!r}")
            joint = kinematics_model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"Waist locomotion robot_urdf joint {name!r} must have one configuration value")
            q_indices.append(joint.idx_q)

        ankle_frame_id = kinematics_model.getFrameId("right_ankle_roll_link", pin.FrameType.BODY)
        if ankle_frame_id >= kinematics_model.nframes:
            raise ValueError("Waist locomotion robot_urdf is missing body frame 'right_ankle_roll_link'")
        self._kinematics_model = kinematics_model
        self._kinematics_data = kinematics_model.createData()
        self._kinematics_q_indices = np.asarray(q_indices, dtype=np.int64)
        self._right_ankle_frame_id = ankle_frame_id

    def _capture_policy_state(self) -> dict:
        state = super()._capture_policy_state()
        state.update(
            {
                "kinematics_model": self._kinematics_model,
                "kinematics_data": self._kinematics_data,
                "kinematics_q_indices": self._kinematics_q_indices,
                "right_ankle_frame_id": self._right_ankle_frame_id,
            }
        )
        return state

    def _restore_policy_state(self, state: dict) -> None:
        super()._restore_policy_state(state)
        self._kinematics_model = state["kinematics_model"]
        self._kinematics_data = state["kinematics_data"]
        self._kinematics_q_indices = state["kinematics_q_indices"]
        self._right_ankle_frame_id = state["right_ankle_frame_id"]

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

    def _handle_start_policy(self, robot_state_data: LowState) -> None:
        reference_quat = np.asarray(robot_state_data.base_quat, dtype=np.float64).copy()
        reference_norm = np.linalg.norm(reference_quat, axis=1, keepdims=True)
        if not np.isfinite(reference_quat).all() or np.any(reference_norm < 1e-8):
            raise RuntimeError("Cannot capture pelvis orientation from an invalid quaternion")
        self.pelvis_orientation_reference_quat = reference_quat / reference_norm
        base_observations = super().get_current_obs_buffer_dict(robot_state_data)
        initial_height = self._base_right_foot_height_difference(
            robot_state_data, base_observations["projected_gravity"]
        )
        self.initial_base_right_foot_height_difference = float(initial_height[0, 0])
        self._reset_inference_episode_state()
        self._reset_pelvis_sine_command()
        super()._handle_start_policy(robot_state_data)

    def update_phase_time(self) -> None:
        frequency_hz = float(self.pelvis_sine_command[0, 3])
        self.pelvis_sine_phase += 2.0 * np.pi * frequency_hz / self.rl_rate
        self.pelvis_sine_phase = float(np.fmod(self.pelvis_sine_phase + np.pi, 2.0 * np.pi) - np.pi)
        self.pelvis_sine_command[0, 0] = np.sin(self.pelvis_sine_phase)
        self.pelvis_sine_command[0, 1] = np.cos(self.pelvis_sine_phase)

    def apply_control(self, control: Mapping[str, float]) -> None:
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
        """Solve base-minus-right-ankle world height from joints and IMU gravity."""

        gravity_b = np.asarray(projected_gravity, dtype=np.float64)
        if gravity_b.shape != (1, 3) or not np.isfinite(gravity_b).all():
            raise ValueError("Cannot solve right-ankle height with invalid projected gravity")

        joint_pos = np.asarray(robot_state_data.joint_pos[0], dtype=np.float64)
        if not np.isfinite(joint_pos).all():
            raise ValueError("Cannot solve right-ankle height with invalid joint positions")
        q = pin.neutral(self._kinematics_model)
        q[self._kinematics_q_indices] = joint_pos
        pin.forwardKinematics(self._kinematics_model, self._kinematics_data, q)
        ankle_placement = pin.updateFramePlacement(
            self._kinematics_model, self._kinematics_data, self._right_ankle_frame_id
        )
        ankle_position_b = np.asarray(ankle_placement.translation, dtype=np.float64)

        # projected_gravity is world-down expressed in the base frame. Its dot
        # product with (ankle - base) is therefore base_z - ankle_z in world.
        return np.asarray([[np.dot(gravity_b[0], ankle_position_b)]], dtype=np.float64)

    def get_current_obs_buffer_dict(self, robot_state_data: LowState):
        observations = super().get_current_obs_buffer_dict(robot_state_data)
        observations["actions"] = self.last_policy_action
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
