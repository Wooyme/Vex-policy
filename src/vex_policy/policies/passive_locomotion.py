"""Holosoma passive_waist_loco G1 actor with term-major 20-frame history."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import ClassVar

import numpy as np

from vex_policy.config.config_types import (
    InferenceConfig,
    PassiveLocomotionGuardConfig,
    PassiveLocomotionTaskConfig,
    SliderInput,
    input_parameters,
)
from vex_policy.policies.guard.waist_locomotion import WaistLocomotionGuard
from vex_policy.robots import G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.latency import LatencyStage

from .base import BasePolicy, PolicyRuntimeFault
from vex_policy.policies.utils.inference import OnnxActor, resolve_control_gains
from vex_policy.policies.utils.joint_command import PositionAction, position_command
from vex_policy.policies.utils.locomotion_utils import RightAnkleKinematics, load_motion_last_pose
from .observations import ObservationHistory, robot_observation_terms


class PassiveLocomotionGuard(WaistLocomotionGuard):
    reason_prefix = "passive_locomotion_start_check_failed"

    def start_check(self, robot_state_data: LowState) -> tuple[bool, str | None]:
        try:
            state = self.policy.validated_state(robot_state_data)
        except PolicyRuntimeFault as error:
            return self._fail(f"{self.reason_prefix}: {error}")
        return super().start_check(state)


class PassiveLocomotionPolicy(BasePolicy):
    """Full-body residual position controller; force inference stays inside ONNX."""

    _OBS_DIMS: ClassVar[dict[str, int]] = {
        "actions": 29,
        "base_ang_vel": 3,
        "base_right_foot_height_difference": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "passive_command": 4,
        "projected_gravity": 3,
    }
    _OBS_SCALES: ClassVar[dict[str, float]] = {
        "actions": 1.0,
        "base_ang_vel": 0.25,
        "base_right_foot_height_difference": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "passive_command": 1.0,
        "projected_gravity": 1.0,
    }
    _COMMAND_NAMES = ("down_vel", "up_vel", "target_height", "min_height")

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, PassiveLocomotionTaskConfig):
            raise TypeError("PassiveLocomotionPolicy requires PassiveLocomotionTaskConfig")
        if not isinstance(config.guard, PassiveLocomotionGuardConfig):
            raise TypeError("PassiveLocomotionPolicy requires PassiveLocomotionGuardConfig")
        if config.action_mask is not None or config.task.action_mask_path is not None:
            raise ValueError("Passive locomotion does not support action masks")
        if config.robot.num_joints != 29 or config.robot.num_motors != 29:
            raise ValueError("Passive locomotion requires 29 joints and motors")
        super().__init__(config)
        self.parameters = {p.name: p for p in input_parameters(config.inputs)}
        if (
            len(config.inputs) != 4
            or not all(isinstance(component, SliderInput) for component in config.inputs)
            or set(self.parameters) != set(self._COMMAND_NAMES)
        ):
            raise ValueError(f"Passive locomotion requires four sliders: {self._COMMAND_NAMES}")
        if any(p.min < 0 for p in self.parameters.values()):
            raise ValueError("Passive command ranges must be nonnegative")


        self.initial_pose = load_motion_last_pose(config.task.motion_data_path)
        names = self.initial_pose.dof_names
        if len(names) != 29 or set(names) != set(self.dof_names):
            raise ValueError("Passive locomotion motion joint names do not match the robot")
        self.default_dof_angles = np.asarray(
            [self.initial_pose.dof_pos[names.index(name)] for name in self.dof_names], dtype=np.float64
        )
        if np.any(self.default_dof_angles < G1_JOINT_LOWER) or np.any(self.default_dof_angles > G1_JOINT_UPPER):
            raise ValueError("Passive locomotion reference pose exceeds G1 joint limits")
        self._validate_observations()
        self.observations = ObservationHistory(config.observation)
        self.actor = OnnxActor(config.task.model_path)
        self._validate_model()
        self.robot_config = resolve_control_gains(
            config.robot, self.actor.metadata.get("kp"), self.actor.metadata.get("kd")
        )
        for name in ("motor_kp", "motor_kd"):
            gains = np.asarray(getattr(self.robot_config, name))
            if gains.shape != (29,) or not np.isfinite(gains).all() or np.any(gains < 0):
                raise ValueError(f"Passive locomotion {name} must contain 29 finite nonnegative gains")
        self.kinematics = RightAnkleKinematics(self.actor.metadata.get("robot_urdf"), self.dof_names)
        self.actions = PositionAction(
            29, self.action_mask, require_full_body=True, force_zero=config.task.debug.force_zero_action
        )
        self.passive_command = np.zeros((1, 4), dtype=np.float32)
        self._reset_episode()
        self.guard = PassiveLocomotionGuard(config.guard, self)

    def _validate_observations(self) -> None:
        obs = self.config.observation
        if set(obs.obs_dict) != {"actor_obs"} or sorted(obs.obs_dict["actor_obs"]) != sorted(self._OBS_DIMS):
            raise ValueError("Passive locomotion actor_obs must contain exactly the seven training terms")
        if obs.obs_dims != self._OBS_DIMS or obs.obs_scales != self._OBS_SCALES:
            raise ValueError("Passive locomotion observation dimensions/scales must match training")
        if obs.history_length_dict != {"actor_obs": 20}:
            raise ValueError("Passive locomotion requires 20 frames of actor_obs history")

    def _validate_model(self) -> None:
        for tensors, name, shape in (
            (self.actor.session.get_inputs(), "actor_obs", [1, 1960]),
            (self.actor.session.get_outputs(), "action", [1, 29]),
        ):
            if (
                len(tensors) != 1
                or tensors[0].name != name
                or list(tensors[0].shape) != shape
                or tensors[0].type != "tensor(float)"
            ):
                raise ValueError(f"Passive locomotion model must expose float32 {name}{shape}")
        if tuple(self.actor.metadata.get("dof_names", ())) != tuple(self.dof_names):
            raise ValueError("Passive locomotion ONNX dof_names do not match the robot joint order")
        scale = np.asarray(self.actor.metadata.get("action_scale", ()), dtype=np.float64)
        if scale.shape not in ((), (29,)) or not np.isfinite(scale).all() or not np.allclose(scale, 0.25):
            raise ValueError("Passive locomotion ONNX action_scale must equal 0.25 for all 29 joints")

    def validated_state(self, state: LowState) -> LowState:
        for name, shape in (
            ("joint_pos", (1, 29)),
            ("joint_vel", (1, 29)),
            ("base_ang_vel", (1, 3)),
            ("base_quat", (1, 4)),
        ):
            value = np.asarray(getattr(state, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise PolicyRuntimeFault(f"passive_invalid_state:{name}")
        quat = np.asarray(state.base_quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or norm < 1e-8:
            raise PolicyRuntimeFault("passive_invalid_state:base_quat")
        return replace(state, base_quat=quat / norm)

    def _reset_episode(self) -> None:
        self.observations.reset()
        self.actions.reset()
        self.passive_command[0] = [self.parameters[name].default for name in self._COMMAND_NAMES]

    def _on_activate(self, robot_state_data: LowState) -> None:
        self._reset_episode()

    def _on_deactivate(self) -> None:
        self._reset_episode()

    def _apply_control(self, control: Mapping[str, float]) -> None:
        try:
            values = [float(control[name]) for name in self._COMMAND_NAMES]
            for name, value in zip(self._COMMAND_NAMES, values, strict=True):
                param = self.parameters[name]
                if not np.isfinite(value) or not param.min <= value <= param.max:
                    raise ValueError(f"{name} is outside its configured range")
            if values[3] > values[2]:
                raise ValueError("min_height exceeds target_height")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise PolicyRuntimeFault(f"passive_invalid_command:{error}") from error
        self.passive_command[0] = values

    def get_current_obs_buffer_dict(self, robot_state_data: LowState) -> dict[str, np.ndarray]:
        state = self.validated_state(robot_state_data)
        terms = robot_observation_terms(state, self.default_dof_angles, self.config.task.debug)
        terms["actions"] = self.actions.last
        terms["passive_command"] = self.passive_command
        terms["base_right_foot_height_difference"] = self.kinematics.height_difference(
            state, terms["projected_gravity"]
        )
        return terms

    def _compute_command(self, robot_state_data: LowState):
        try:
            with self.latency_tracker.measure(LatencyStage.PREPROCESSING):
                obs = self.observations.prepare(self.get_current_obs_buffer_dict(robot_state_data))
                if obs["actor_obs"].shape != (1, 1960) or not np.isfinite(obs["actor_obs"]).all():
                    raise PolicyRuntimeFault("passive_invalid_observation")
                if self.config.task.print_observations:
                    self.observations.print_observations(obs, self.dof_names, self.actions.scaled)
            with self.latency_tracker.measure(LatencyStage.INFERENCE):
                raw_action = np.asarray(self.actor(obs), dtype=np.float32)
            with self.latency_tracker.measure(LatencyStage.POSTPROCESSING):
                if raw_action.shape != (1, 29) or not np.isfinite(raw_action).all():
                    raise PolicyRuntimeFault("passive_invalid_action")
                self.actions.process(raw_action, self.config.task.policy_action_scale)
                # Training observes ActionManager.action, before joint-control clipping.
                self.actions.last = (
                    np.zeros_like(raw_action) if self.config.task.debug.force_zero_action else raw_action.copy()
                )
                target = np.clip(self.actions.target(self.default_dof_angles), G1_JOINT_LOWER, G1_JOINT_UPPER)
                if not np.isfinite(target).all():
                    raise PolicyRuntimeFault("passive_invalid_target")
                self.actions.scaled = target - self.default_dof_angles
                return position_command(
                    target, self.robot_config.motor_kp, self.robot_config.motor_kd, self.controlled_joint_mask
                )
        except PolicyRuntimeFault:
            raise
        except Exception as error:
            raise PolicyRuntimeFault(f"passive_inference_failed:{error}") from error
