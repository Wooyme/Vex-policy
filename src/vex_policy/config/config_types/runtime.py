"""MQTT service configuration loaded from YAML."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .control import PolicyInput, input_parameters
from .GuardConfig import GuardConfig
from .observation import ObservationConfig
from .robot import RobotConfig
from .safety import EmergencyStopConfig, LimiterConfig
from .task import (
    HoldPositionTaskConfig,
    InterpolationTaskConfig,
    PassiveLocomotionTaskConfig,
    SonicTaskConfig,
    TaskConfig,
    UfoTaskConfig,
    WaistLocomotionTaskConfig,
    WbtTaskConfig,
)

PolicyType = Literal["full_body", "lower_body", "upper_body"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RobotRuntimeConfig(StrictModel):
    interface: str = "auto"
    domain_id: int = Field(default=0, ge=0)
    config: RobotConfig


class MqttConfig(StrictModel):
    broker: str = "mqtt://localhost:1883"
    command_topic: str = "robot/commands"
    policies_topic: str = "robot/policies"
    status_topic: str = "robot/status"
    state_topic: str = "robot/g1/real/state"
    reference_state_topic: str = "robot/g1/reference/state"
    command_timeout_s: float = Field(default=1.0, gt=0)
    state_frequency_hz: float = Field(default=50.0, gt=0)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    client_id: str | None = None

    @field_validator(
        "broker", "command_topic", "policies_topic", "status_topic", "state_topic", "reference_state_topic"
    )
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("command_topic", "policies_topic", "status_topic", "state_topic", "reference_state_topic")
    @classmethod
    def no_wildcard(cls, value: str) -> str:
        if "+" in value or "#" in value:
            raise ValueError("publish/command topics must not contain MQTT wildcards")
        return value


class PolicySpec(StrictModel):
    name: str
    implementation: str
    type: PolicyType = "full_body"
    inputs: tuple[PolicyInput, ...] = ()
    observation: ObservationConfig
    task: (
        TaskConfig
        | WbtTaskConfig
        | SonicTaskConfig
        | WaistLocomotionTaskConfig
        | HoldPositionTaskConfig
        | InterpolationTaskConfig
        | UfoTaskConfig
        | PassiveLocomotionTaskConfig
    )
    guard: GuardConfig | None = None
    limiter: LimiterConfig | None = None
    estop: EmergencyStopConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def select_config_types(cls, value):
        if not isinstance(value, dict) or not isinstance(value.get("task"), dict):
            return value
        task_types = {
            "hold_position": HoldPositionTaskConfig,
            "interpolation": InterpolationTaskConfig,
            "sonic": SonicTaskConfig,
            "ufo": UfoTaskConfig,
            "waist_locomotion": WaistLocomotionTaskConfig,
            "passive_locomotion": PassiveLocomotionTaskConfig,
            "wbt": WbtTaskConfig,
        }
        implementation = value.get("implementation")
        task_type = task_types.get(implementation, TaskConfig)
        selected = {**value, "task": task_type(**value["task"])}
        return selected

    @field_validator("name", "implementation")
    @classmethod
    def clean_name(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("must be non-empty and must not have surrounding whitespace")
        return value

    @field_validator("inputs")
    @classmethod
    def unique_inputs(cls, value: tuple[PolicyInput, ...]) -> tuple[PolicyInput, ...]:
        names = [parameter.name for parameter in input_parameters(value)]
        if len(set(names)) != len(names):
            raise ValueError("input parameter names must not contain duplicates")
        return value

    @property
    def input_parameters(self):
        return input_parameters(self.inputs)


def _validate_policy_set(policies: tuple[PolicySpec, ...]) -> None:
    if not policies:
        raise ValueError("at least one policy must be configured")
    names = [policy.name for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError("policy names must be unique")


class RuntimeConfig(StrictModel):
    robot: RobotRuntimeConfig
    mqtt: MqttConfig = MqttConfig()
    policies: tuple[PolicySpec, ...]

    @model_validator(mode="after")
    def validate_policies(self) -> RuntimeConfig:
        _validate_policy_set(self.policies)
        specs = {spec.name: spec for spec in self.policies}
        for spec in self.policies:
            for limits in (spec.limiter, spec.estop):
                if limits is not None:
                    limits.validate_joints(self.robot.config.dof_names)
            fallback = spec.estop.fallback if spec.estop is not None else None
            if fallback is None:
                continue
            target = specs.get(fallback.policy)
            if target is None or target.name == spec.name or target.type != "full_body":
                raise ValueError(f"Policy {spec.name!r} fallback must name another loaded full_body policy")
            parameters = {p.name: p for p in target.input_parameters}
            unknown = fallback.inputs.keys() - parameters.keys()
            if unknown:
                raise ValueError(f"Unknown fallback inputs for {target.name!r}: {sorted(unknown)}")
            for name, value in fallback.inputs.items():
                if not parameters[name].min <= value <= parameters[name].max:
                    raise ValueError(f"Fallback input {target.name}.{name} is outside its configured range")
        return self
