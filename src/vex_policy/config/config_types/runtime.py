"""MQTT service configuration loaded from YAML."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .GuardConfig import GuardConfig, WaistLocomotionGuardConfig
from .observation import ObservationConfig
from .robot import RobotConfig
from .task import SonicTaskConfig, TaskConfig, WaistLocomotionTaskConfig

PolicyType = Literal["full_body", "lower_body", "upper_body"]
PolicyInput = Literal["vx", "vy", "yaw", "pitch", "height"]


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
    task: TaskConfig | SonicTaskConfig | WaistLocomotionTaskConfig
    guard: GuardConfig | WaistLocomotionGuardConfig | None = None

    @field_validator("name", "implementation")
    @classmethod
    def clean_name(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("must be non-empty and must not have surrounding whitespace")
        return value

    @field_validator("inputs")
    @classmethod
    def unique_inputs(cls, value: tuple[PolicyInput, ...]) -> tuple[PolicyInput, ...]:
        if len(set(value)) != len(value):
            raise ValueError("inputs must not contain duplicates")
        return value


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
        return self
