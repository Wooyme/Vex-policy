"""Inference adapter for Holosoma's pelvis-sine waist locomotion task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import onnx
from pydantic import BaseModel, ConfigDict, model_validator

from vex_policy.config.config_types import (
    InferenceConfig,
    WaistLocomotionGuardConfig,
    WaistLocomotionTaskConfig,
)
from vex_policy.inputs.api.commands import ControlValues
from vex_policy.policies.guard.waist_locomotion import WaistLocomotionGuard
from vex_policy.utils.math.quat import quat_rotate_inverse

from .base import BasePolicy


class WaistInitialPose(BaseModel):
    """Minimal training-pose data needed by inference and its startup guard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dof_names: tuple[str, ...]
    dof_pos: tuple[float, ...]
    projected_gravity: tuple[float, float, float]

    @model_validator(mode="after")
    def validate_values(self) -> WaistInitialPose:
        if not self.dof_names or len(self.dof_names) != len(self.dof_pos):
            raise ValueError("dof_names and dof_pos must have the same non-zero length")
        if len(set(self.dof_names)) != len(self.dof_names):
            raise ValueError("dof_names must not contain duplicates")
        values = np.asarray((*self.dof_pos, *self.projected_gravity), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("initial pose values must be finite")
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
            "Waist locomotion joint_pos must have shape "
            f"(frames, 7 + {len(joint_names)}), got {joint_pos.shape}"
        )
    final_pose = joint_pos[-1]
    root_quat_xyzw = final_pose[3:7]
    quaternion_norm = float(np.linalg.norm(root_quat_xyzw))
    if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-8:
        raise ValueError("Waist locomotion final-frame root quaternion is invalid")
    root_quat_wxyz = (root_quat_xyzw / quaternion_norm)[[3, 0, 1, 2]].reshape(1, 4)
    projected_gravity = quat_rotate_inverse(root_quat_wxyz, np.asarray([[0.0, 0.0, -1.0]]))[0]
    return WaistInitialPose(
        dof_names=joint_names,
        dof_pos=tuple(final_pose[7:]),
        projected_gravity=tuple(projected_gravity),
    )


class WaistLocomotionPolicy(BasePolicy):
    """Full-body inference for a commanded sinusoidal pelvis trajectory."""

    _OBS_DIMS: ClassVar[dict[str, int]] = {
        "actions": 29,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "pelvis_sine_command": 7,
        "projected_gravity": 3,
    }
    _OBS_SCALES: ClassVar[dict[str, float]] = {
        "actions": 1.0,
        "base_ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "pelvis_sine_command": 1.0,
        "projected_gravity": 1.0,
    }

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, WaistLocomotionTaskConfig):
            raise TypeError("WaistLocomotionPolicy requires WaistLocomotionTaskConfig")
        if not isinstance(config.guard, WaistLocomotionGuardConfig):
            raise TypeError("WaistLocomotionPolicy requires WaistLocomotionGuardConfig")
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
        if self.obs_dim_dict["actor_obs"] != 100:
            raise ValueError("Waist locomotion actor_obs must have dimension 100")

    def _init_command_components(self) -> None:
        super()._init_command_components()
        self.pelvis_sine_phase = 0.0
        self.pelvis_sine_command = np.zeros((1, 7), dtype=np.float32)
        self._reset_pelvis_sine_command()

    def _init_phase_components(self) -> None:
        self.use_phase = self.config.task.use_phase
        if not self.use_phase:
            raise ValueError("Waist locomotion requires task.use_phase=true")

    def setup_policy(self, model_path) -> None:
        super().setup_policy(model_path)
        inputs = self.onnx_policy_session.get_inputs()
        outputs = self.onnx_policy_session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "actor_obs" or list(inputs[0].shape) != [1, 100]:
            exposed = [(item.name, item.shape) for item in inputs]
            raise ValueError(f"Waist locomotion model must expose actor_obs[1, 100], got {exposed}")
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

    def _reset_pelvis_sine_command(self) -> None:
        self.pelvis_sine_phase = 0.0
        direction = np.asarray(self.waist_task.default_direction, dtype=np.float32)
        direction /= np.linalg.norm(direction)
        frequency_min, frequency_max = self.waist_task.frequency_range_hz
        self.pelvis_sine_command[0] = (
            0.0,
            1.0,
            self.waist_task.default_amplitude_m,
            0.5 * (frequency_min + frequency_max),
            *direction,
        )

    def _handle_start_policy(self) -> None:
        self._reset_pelvis_sine_command()
        super()._handle_start_policy()

    def update_phase_time(self) -> None:
        frequency_hz = float(self.pelvis_sine_command[0, 3])
        self.pelvis_sine_phase += 2.0 * np.pi * frequency_hz / self.rl_rate
        self.pelvis_sine_phase = float(np.fmod(self.pelvis_sine_phase + np.pi, 2.0 * np.pi) - np.pi)
        self.pelvis_sine_command[0, 0] = np.sin(self.pelvis_sine_phase)
        self.pelvis_sine_command[0, 1] = np.cos(self.pelvis_sine_phase)

    def apply_control(self, control: ControlValues) -> None:
        amplitude_min, amplitude_max = self.waist_task.amplitude_range_m
        amplitude = (
            self.waist_task.default_amplitude_m
            if control.height == 0.0
            else float(np.clip(control.height, amplitude_min, amplitude_max))
        )

        frequency_min, frequency_max = self.waist_task.frequency_range_hz
        normalized_pitch = float(np.clip(control.pitch, -1.0, 1.0))
        frequency = frequency_min + 0.5 * (normalized_pitch + 1.0) * (frequency_max - frequency_min)

        direction = np.asarray([control.vy, -control.vx, -control.yaw], dtype=np.float32)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            direction = np.asarray(self.waist_task.default_direction, dtype=np.float32)
            direction_norm = float(np.linalg.norm(direction))
        direction /= direction_norm
        self.pelvis_sine_command[0, 2:] = (amplitude, frequency, *direction)

    def get_current_obs_buffer_dict(self, robot_state_data):
        observations = super().get_current_obs_buffer_dict(robot_state_data)
        observations["actions"] = self.last_policy_action
        observations["pelvis_sine_command"] = self.pelvis_sine_command
        return observations
