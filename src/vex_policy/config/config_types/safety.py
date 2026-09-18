"""Optional per-policy output limits and measured-state emergency checks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Speed = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class SafetyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Bounds(SafetyModel):
    min: Finite
    max: Finite

    @model_validator(mode="after")
    def ordered(self) -> Bounds:
        if self.min > self.max:
            raise ValueError("min must not exceed max")
        return self


class JointLimit(SafetyModel):
    pos: Bounds | None = None
    vel: Speed | None = None


class LimiterConfig(SafetyModel):
    joints: dict[str, JointLimit] = Field(default_factory=dict)

    def validate_joints(self, names: Sequence[str]) -> None:
        unknown = self.joints.keys() - set(names)
        if unknown:
            raise ValueError(f"Unknown joint limits: {sorted(unknown)}")


class RpyLimits(SafetyModel):
    roll: Bounds | None = None
    pitch: Bounds | None = None
    yaw: Bounds | None = None

    @model_validator(mode="after")
    def principal_ranges(self) -> RpyLimits:
        for axis, extent in (("roll", math.pi), ("pitch", math.pi / 2), ("yaw", math.pi)):
            bounds = getattr(self, axis)
            if bounds is not None and (bounds.min < -extent or bounds.max > extent):
                raise ValueError(f"{axis} bounds must be within [{-extent}, {extent}]")
        return self


class FallbackConfig(SafetyModel):
    policy: str = Field(min_length=1)
    inputs: dict[str, Finite] = Field(default_factory=dict)


class EmergencyStopConfig(LimiterConfig):
    rpy: RpyLimits = Field(default_factory=RpyLimits)
    fallback: FallbackConfig | None = None
