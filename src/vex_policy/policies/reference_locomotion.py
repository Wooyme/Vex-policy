"""Deploy Holosoma kneeling and crawling actors around a training reference pose."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np

from vex_policy.config.config_types import InferenceConfig, ReferenceLocomotionTaskConfig, input_parameters
from vex_policy.policies.guard.initial_pose import InitialPoseGuard
from vex_policy.policies.utils.inference import OnnxActor, resolve_control_gains
from vex_policy.policies.utils.initial_pose import normalize_quaternion_wxyz
from vex_policy.policies.utils.joint_command import PositionAction, position_command
from vex_policy.policies.utils.locomotion_utils import load_motion_pose
from vex_policy.policies.utils.observations import ObservationHistory, robot_observation_terms
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.latency import LatencyStage
from .base import BasePolicy, PolicyRuntimeFault


class ReferenceLocomotionPolicy(BasePolicy):
    """Velocity-controlled 100→29 actor using body-frame IMU observations."""

    _OBS_DIMS: ClassVar[dict[str, int]] = {
        "actions": 29,
        "base_ang_vel": 3,
        "command_ang_vel": 1,
        "command_lin_vel": 2,
        "cos_phase": 2,
        "dof_pos": 29,
        "dof_vel": 29,
        "projected_gravity": 3,
        "sin_phase": 2,
    }
    _OBS_SCALES: ClassVar[dict[str, float]] = {
        "actions": 1.0,
        "base_ang_vel": 0.25,
        "command_ang_vel": 1.0,
        "command_lin_vel": 1.0,
        "cos_phase": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "projected_gravity": 1.0,
        "sin_phase": 1.0,
    }

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, ReferenceLocomotionTaskConfig):
            raise TypeError("ReferenceLocomotionPolicy requires ReferenceLocomotionTaskConfig")
        super().__init__(config)
        if tuple(self.dof_names) != tuple(G1_29DOF.dof_names):
            raise ValueError("Reference locomotion requires the G1 29-DOF hardware joint order")
        parameters = input_parameters(config.inputs)
        self._input_parameters = {parameter.name: parameter for parameter in parameters}
        if len(parameters) != 3 or self._input_parameters.keys() != {"vx", "vy", "yaw"}:
            raise ValueError("Reference locomotion requires exactly vx, vy and yaw inputs")
        self.task = config.task
        self.initial_pose = load_motion_pose(self.task.motion_data_path, self.task.reference_pose_frame)
        if set(self.initial_pose.dof_names) != set(self.dof_names):
            raise ValueError("Reference locomotion reference joint names do not match the robot")
        positions = dict(zip(self.initial_pose.dof_names, self.initial_pose.dof_pos, strict=True))
        self.default_dof_angles = np.asarray([positions[name] for name in self.dof_names])
        if np.any(self.default_dof_angles < G1_JOINT_LOWER) or np.any(self.default_dof_angles > G1_JOINT_UPPER):
            raise ValueError("Reference locomotion reference contains out-of-limit joints")

        self._validate_observations()
        self.observations = ObservationHistory(config.observation)
        self.actor = OnnxActor(self.task.model_path)
        self._validate_model()
        self.robot_config = resolve_control_gains(config.robot, self.actor.metadata["kp"], self.actor.metadata["kd"])
        self._validate_gains(self.robot_config.motor_kp, self.robot_config.motor_kd)
        self.actions = PositionAction(
            self.num_dofs, self.action_mask, require_full_body=True, force_zero=self.task.debug.force_zero_action
        )
        if config.guard is not None:
            self.guard = InitialPoseGuard(
                config.guard,
                self.initial_pose,
                self.dof_names,
                self.logger,
                reason_prefix="reference_locomotion_start_check_failed",
            )
        self._reset_episode()

    def _validate_observations(self) -> None:
        config = self.config.observation
        if {group: sorted(terms) for group, terms in config.obs_dict.items()} != {"actor_obs": sorted(self._OBS_DIMS)}:
            raise ValueError("Reference locomotion requires exactly the nine actor_obs terms")
        if config.history_length_dict.get("actor_obs", 1) != 1:
            raise ValueError("Reference locomotion actor_obs history length must be 1")
        for term, dimension in self._OBS_DIMS.items():
            if config.obs_dims.get(term) != dimension:
                raise ValueError(f"Reference locomotion observation {term!r} must have dimension {dimension}")
            scale = config.obs_scales.get(term)
            if scale is None or not np.isclose(scale, self._OBS_SCALES[term]):
                raise ValueError(f"Reference locomotion observation {term!r} must use scale {self._OBS_SCALES[term]}")

    @staticmethod
    def _validate_gains(kp, kd) -> None:
        for name, values in (("kp", kp), ("kd", kd)):
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (29,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Reference locomotion {name} must contain 29 finite nonnegative gains")

    def _validate_model(self) -> None:
        inputs, outputs = self.actor.session.get_inputs(), self.actor.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "actor_obs" or list(inputs[0].shape) != [1, 100]:
            raise ValueError("Reference locomotion ONNX must expose actor_obs[1, 100]")
        if len(outputs) != 1 or outputs[0].name != "action" or list(outputs[0].shape) != [1, 29]:
            raise ValueError("Reference locomotion ONNX must expose action[1, 29]")
        metadata = self.actor.metadata
        if tuple(metadata.get("dof_names", ())) != tuple(self.dof_names):
            raise ValueError("Reference locomotion ONNX dof_names do not match the robot joint order")
        scale = np.asarray(metadata.get("action_scale", ()), dtype=np.float64)
        if (
                scale.shape != (29,)
                or not np.isfinite(scale).all()
                or not np.allclose(scale, self.task.policy_action_scale)
        ):
            raise ValueError(
                "Reference locomotion ONNX action_scale must match task.policy_action_scale for all 29 joints"
            )
        self._validate_gains(metadata.get("kp", ()), metadata.get("kd", ()))

    def _reset_episode(self) -> None:
        self.actions.reset()
        self.observations.reset()
        self.last_action = np.zeros((1, 29), dtype=np.float32)
        self.lin_vel_command = np.zeros((1, 2))
        self.ang_vel_command = np.zeros((1, 1))
        self._episode_step = 0

    def _validated_state(self, state: LowState) -> LowState:
        for name, shape in (
                ("joint_pos", (1, 29)),
                ("joint_vel", (1, 29)),
                ("base_ang_vel", (1, 3)),
                ("base_quat", (1, 4)),
        ):
            values = np.asarray(getattr(state, name))
            if values.shape != shape or not np.isfinite(values).all():
                raise PolicyRuntimeFault(f"Invalid reference locomotion state: {name}")
        try:
            quaternion = normalize_quaternion_wxyz(state.base_quat[0])
        except ValueError as error:
            raise PolicyRuntimeFault(f"Invalid reference locomotion quaternion: {error}") from error
        return replace(state, base_quat=quaternion[None, :])

    def _on_activate(self, robot_state_data: LowState) -> None:
        self._validated_state(robot_state_data)
        self._reset_episode()
        self._apply_control({name: parameter.default for name, parameter in self._input_parameters.items()})

    def _on_deactivate(self) -> None:
        self._reset_episode()

    def _apply_control(self, control) -> None:
        if control.keys() != self._input_parameters.keys():
            raise PolicyRuntimeFault("Reference locomotion control requires exactly vx, vy and yaw")
        for name, parameter in self._input_parameters.items():
            value = control[name]
            if not np.isfinite(value) or not parameter.min <= value <= parameter.max:
                raise PolicyRuntimeFault(f"Invalid reference locomotion input: {name}")
        self.lin_vel_command[0] = (control["vy"], -control["vx"])
        self.ang_vel_command[0, 0] = -control["yaw"]

    def _observation_phase(self) -> np.ndarray:
        """Standing overrides observations, not the continuously advancing episode clock."""
        if np.linalg.norm(self.lin_vel_command) < 0.01 and abs(self.ang_vel_command[0, 0]) < 0.01:
            return np.full((1, 2), np.pi)
        phase = self._episode_step * (2 * np.pi / (self.rl_rate * self.task.gait_period)) + np.array([[0.0, -np.pi]])
        return np.fmod(phase + np.pi, 2 * np.pi) - np.pi

    def _compute_command(self, robot_state_data: LowState):
        with self.latency_tracker.measure(LatencyStage.PREPROCESSING):
            state = self._validated_state(robot_state_data)
            terms = robot_observation_terms(state, self.default_dof_angles, self.task.debug)
            phase = self._observation_phase()
            terms.update(
                actions=self.last_action,
                command_lin_vel=self.lin_vel_command,
                command_ang_vel=self.ang_vel_command,
                sin_phase=np.sin(phase),
                cos_phase=np.cos(phase),
            )
            observations = self.observations.prepare(terms)
            if not np.isfinite(observations["actor_obs"]).all():
                raise PolicyRuntimeFault("Nonfinite reference locomotion observations")
            if self.task.print_observations:
                self.observations.print_observations(observations, self.dof_names, self.actions.scaled)
        with self.latency_tracker.measure(LatencyStage.INFERENCE):
            action = np.asarray(self.actor(observations), dtype=np.float32)
            if action.shape != (1, 29) or not np.isfinite(action).all():
                raise PolicyRuntimeFault("Reference locomotion actor must return 29 finite actions")
        with self.latency_tracker.measure(LatencyStage.POSTPROCESSING):
            self.actions.process(action, self.task.policy_action_scale)
            # Training feeds back the actor output before clipping, masking or scaling.
            self.last_action = np.zeros_like(action) if self.task.debug.force_zero_action else action.copy()
            target = np.clip(self.actions.target(self.default_dof_angles), G1_JOINT_LOWER, G1_JOINT_UPPER)
            self.actions.scaled = target - self.default_dof_angles
            self._episode_step += 1
            return position_command(
                target, self.robot_config.motor_kp, self.robot_config.motor_kd, self.controlled_joint_mask
            )
