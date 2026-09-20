"""Deployment of Holosoma's G1 pelvis height recovery actor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import ClassVar

import numpy as np

from vex_policy.config.config_types import InferenceConfig, PelvisRecoveryTaskConfig, SliderInput, input_parameters
from vex_policy.policies.guard.initial_pose import InitialPoseGuard
from vex_policy.policies.utils.inference import OnnxActor, resolve_control_gains
from vex_policy.policies.utils.initial_pose import normalize_quaternion_wxyz
from vex_policy.policies.utils.joint_command import PositionAction, position_command
from vex_policy.policies.utils.locomotion_utils import (
    RightAnkleKinematics,
    load_motion_last_pose,
    relative_rotation_vector,
)
from vex_policy.policies.utils.observations import ObservationHistory, robot_observation_terms
from vex_policy.robots import G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.latency import LatencyStage
from vex_policy.utils.math.quat import quat_mul, quat_to_rpy, rpy_to_quat

from .base import BasePolicy, PolicyRuntimeFault


class PelvisRecoveryPolicy(BasePolicy):
    """Recover toward a commanded height while the right foot supports the robot."""

    _OBS_DIMS: ClassVar[dict[str, int]] = {
        "actions": 29,
        "base_ang_vel": 3,
        "base_right_foot_height_difference": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "pelvis_orientation_error": 3,
        "pelvis_recovery_command": 2,
        "projected_gravity": 3,
    }
    _OBS_SCALES: ClassVar[dict[str, float]] = {
        "actions": 1.0,
        "base_ang_vel": 0.25,
        "base_right_foot_height_difference": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "pelvis_orientation_error": 1.0,
        "pelvis_recovery_command": 1.0,
        "projected_gravity": 1.0,
    }
    BRIDGE_JOINTS: ClassVar[tuple[str, ...]] = (
        "waist_pitch_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
    )

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, PelvisRecoveryTaskConfig):
            raise TypeError("PelvisRecoveryPolicy requires PelvisRecoveryTaskConfig")
        super().__init__(config)
        if self.num_dofs != 29:
            raise ValueError("Pelvis recovery requires a G1 29-DOF robot")
        self.task = config.task
        self.input_parameters = {parameter.name: parameter for parameter in input_parameters(config.inputs)}
        if (
            not all(isinstance(component, SliderInput) for component in config.inputs)
            or len(config.inputs) != 2
            or set(self.input_parameters) != {"target_height", "peak_speed"}
        ):
            raise ValueError("Pelvis recovery requires target_height and peak_speed sliders")
        if any(parameter.min <= 0.0 for parameter in self.input_parameters.values()):
            raise ValueError("Pelvis recovery input ranges must be positive")

        self.initial_pose = load_motion_last_pose(self.task.motion_data_path)
        if set(self.initial_pose.dof_names) != set(self.dof_names):
            raise ValueError("Pelvis recovery reference joint names do not match the robot")
        source_indices = {name: index for index, name in enumerate(self.initial_pose.dof_names)}
        self.default_dof_angles = np.clip(
            [self.initial_pose.dof_pos[source_indices[name]] for name in self.dof_names],
            G1_JOINT_LOWER,
            G1_JOINT_UPPER,
        )
        self.initial_pose = self.initial_pose.model_copy(
            update={"dof_names": tuple(self.dof_names), "dof_pos": tuple(self.default_dof_angles)}
        )
        self.observations = ObservationHistory(config.observation)
        self._validate_observations()
        self.actor = OnnxActor(self.task.model_path)
        self._validate_model()
        self._ankle_kinematics = RightAnkleKinematics(self.actor.metadata.get("robot_urdf"), self.dof_names)

        gains = resolve_control_gains(config.robot, self.actor.metadata["kp"], self.actor.metadata["kd"])
        kp, kd = np.asarray(gains.motor_kp).copy(), np.asarray(gains.motor_kd).copy()
        bridge_ids = [self.dof_names.index(name) for name in self.BRIDGE_JOINTS]
        kp[bridge_ids] *= self.task.bridge_kp_scale
        kd[bridge_ids] *= self.task.bridge_kd_scale
        self._validate_gains(kp, kd)
        self.robot_config = replace(gains, motor_kp=tuple(kp), motor_kd=tuple(kd))
        self.actions = PositionAction(
            self.num_dofs, self.action_mask, require_full_body=True, force_zero=self.task.debug.force_zero_action
        )
        if config.guard is not None:
            self.guard = InitialPoseGuard(
                config.guard,
                self.initial_pose,
                self.dof_names,
                self.logger,
                reason_prefix="pelvis_recovery_start_check_failed",
            )
        self._reset_episode()

    def _validate_observations(self) -> None:
        observations = self.observations
        if observations.obs_terms_sorted != {"actor_obs": sorted(self._OBS_DIMS)}:
            raise ValueError("Pelvis recovery requires exactly the eight actor_obs terms")
        if observations.history_length_dict.get("actor_obs", 1) != 1:
            raise ValueError("Pelvis recovery actor_obs history length must be 1")
        for term, dimension in self._OBS_DIMS.items():
            if observations.obs_dims.get(term) != dimension:
                raise ValueError(f"Pelvis recovery observation {term!r} must have dimension {dimension}")
            scale = observations.obs_scales.get(term)
            if scale is None or not np.isclose(scale, self._OBS_SCALES[term]):
                raise ValueError(f"Pelvis recovery observation {term!r} must use scale {self._OBS_SCALES[term]}")

    @staticmethod
    def _validate_gains(kp, kd) -> None:
        for name, values in (("kp", kp), ("kd", kd)):
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (29,) or not np.isfinite(values).all() or np.any(values < 0.0):
                raise ValueError(f"Pelvis recovery {name} must contain 29 finite nonnegative gains")

    def _validate_model(self) -> None:
        inputs, outputs = self.actor.session.get_inputs(), self.actor.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "actor_obs" or list(inputs[0].shape) != [1, 99]:
            raise ValueError("Pelvis recovery ONNX must expose actor_obs[1, 99]")
        if len(outputs) != 1 or outputs[0].name != "action" or list(outputs[0].shape) != [1, 29]:
            raise ValueError("Pelvis recovery ONNX must expose action[1, 29]")
        metadata = self.actor.metadata
        if tuple(metadata.get("dof_names", ())) != tuple(self.dof_names):
            raise ValueError("Pelvis recovery ONNX dof_names do not match the robot joint order")
        scale = np.asarray(metadata.get("action_scale", ()), dtype=np.float64)
        if (
            scale.shape != (29,)
            or not np.isfinite(scale).all()
            or not np.allclose(scale, self.task.policy_action_scale)
        ):
            raise ValueError("Pelvis recovery ONNX action_scale must match task.policy_action_scale for all 29 joints")
        self._validate_gains(metadata.get("kp", ()), metadata.get("kd", ()))

    def _reset_episode(self) -> None:
        self.actions.reset()
        self.observations.reset()
        # Training feeds back raw actor outputs, before clipping/scaling targets.
        self.last_action = np.zeros((1, 29), dtype=np.float32)
        self.pelvis_orientation_reference_quat = None
        self.recovery_command = np.array([[0.0, self.input_parameters["target_height"].default]])
        self.peak_speed = self.input_parameters["peak_speed"].default
        self._elapsed_steps = 0

    def _validated_state(self, state: LowState) -> LowState:
        for name, shape in (
            ("joint_pos", (1, 29)),
            ("joint_vel", (1, 29)),
            ("base_ang_vel", (1, 3)),
            ("base_quat", (1, 4)),
        ):
            values = np.asarray(getattr(state, name))
            if values.shape != shape or not np.isfinite(values).all():
                raise PolicyRuntimeFault(f"Invalid pelvis recovery state: {name}")
        try:
            quaternion = normalize_quaternion_wxyz(state.base_quat[0])
        except ValueError as error:
            raise PolicyRuntimeFault(f"Invalid pelvis recovery quaternion: {error}") from error
        return replace(state, base_quat=quaternion[None, :])

    def _on_activate(self, robot_state_data: LowState) -> None:
        state = self._validated_state(robot_state_data)
        self._reset_episode()
        reference = np.asarray(self.initial_pose.root_quat_wxyz)
        yaw_delta = quat_to_rpy(state.base_quat[0])[2] - quat_to_rpy(reference)[2]
        self.pelvis_orientation_reference_quat = quat_mul(rpy_to_quat((0.0, 0.0, yaw_delta))[None], reference[None])

    def _on_deactivate(self) -> None:
        self._reset_episode()

    def _apply_control(self, control: Mapping[str, float]) -> None:
        values = {}
        for name, parameter in self.input_parameters.items():
            value = float(control[name])
            if not np.isfinite(value) or not parameter.min <= value <= parameter.max:
                raise PolicyRuntimeFault(f"Pelvis recovery input {name!r} is outside its configured range")
            values[name] = value
        self.recovery_command[0, 1] = values["target_height"]
        self.peak_speed = values["peak_speed"]

    def _prepare_observations(self, state: LowState) -> dict[str, np.ndarray]:
        terms = robot_observation_terms(state, self.default_dof_angles, self.task.debug)
        height_difference = self._ankle_kinematics.height_difference(state, terms["projected_gravity"])
        if not np.isfinite(height_difference).all():
            raise PolicyRuntimeFault("Invalid pelvis recovery height estimate")
        height = float(height_difference[0, 0]) + self.task.right_ankle_height_m
        # The reset observation has zero velocity. Each subsequent policy step
        # uses the new state to compute the command for the next action.
        if self._elapsed_steps:
            error = max(float(self.recovery_command[0, 1]) - height, 0.0)
            desired = self.peak_speed * np.tanh(error / self.task.slowdown_height_m)
            allowance = self.task.max_acceleration_m_s2 / self.rl_rate
            self.recovery_command[0, 0] += np.clip(desired - self.recovery_command[0, 0], -allowance, allowance)
        terms["actions"] = self.last_action
        terms["base_right_foot_height_difference"] = height_difference
        terms["pelvis_orientation_error"] = relative_rotation_vector(
            self.pelvis_orientation_reference_quat, state.base_quat
        )
        terms["pelvis_recovery_command"] = self.recovery_command
        observations = self.observations.prepare(terms)
        if not np.isfinite(observations["actor_obs"]).all():
            raise PolicyRuntimeFault("Nonfinite pelvis recovery observations")
        return observations

    def _compute_command(self, robot_state_data: LowState):
        with self.latency_tracker.measure(LatencyStage.PREPROCESSING):
            state = self._validated_state(robot_state_data)
            observations = self._prepare_observations(state)
            if self.task.print_observations:
                self.observations.print_observations(observations, self.dof_names, self.actions.scaled)
        with self.latency_tracker.measure(LatencyStage.INFERENCE):
            action = np.asarray(self.actor(observations), dtype=np.float32)
            if action.shape != (1, 29) or not np.isfinite(action).all():
                raise PolicyRuntimeFault("Pelvis recovery actor must return 29 finite actions")
        with self.latency_tracker.measure(LatencyStage.POSTPROCESSING):
            self.actions.process(action, self.task.policy_action_scale)
            self.last_action = np.zeros_like(action) if self.task.debug.force_zero_action else action.copy()
            q_target = np.clip(self.actions.target(self.default_dof_angles), G1_JOINT_LOWER, G1_JOINT_UPPER)
            self.actions.scaled = q_target - self.default_dof_angles
            self._elapsed_steps += 1
            return position_command(
                q_target, self.robot_config.motor_kp, self.robot_config.motor_kd, self.controlled_joint_mask
            )
