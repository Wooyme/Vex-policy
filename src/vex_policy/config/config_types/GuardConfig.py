import math

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GuardConfig:
    bad_lower_joint_pos_threshold: float = 0.8
    bad_ref_ori_threshold: float = 0.0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class WaistLocomotionGuardConfig:
    """Startup pose thresholds for waist locomotion."""

    startup_joint_tolerance_rad: float = 0.2
    startup_gravity_tolerance: float = 0.2

    def __post_init__(self) -> None:
        if not math.isfinite(self.startup_joint_tolerance_rad) or self.startup_joint_tolerance_rad <= 0.0:
            raise ValueError("startup_joint_tolerance_rad must be finite and positive")
        if not math.isfinite(self.startup_gravity_tolerance) or self.startup_gravity_tolerance <= 0.0:
            raise ValueError("startup_gravity_tolerance must be finite and positive")


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class UfoGuardConfig:
    """Startup pose thresholds for UFO policies."""

    startup_joint_tolerance_rad: float = 0.2
    startup_gravity_tolerance: float = 0.2

    def __post_init__(self) -> None:
        if not math.isfinite(self.startup_joint_tolerance_rad) or self.startup_joint_tolerance_rad <= 0.0:
            raise ValueError("startup_joint_tolerance_rad must be finite and positive")
        if not math.isfinite(self.startup_gravity_tolerance) or self.startup_gravity_tolerance <= 0.0:
            raise ValueError("startup_gravity_tolerance must be finite and positive")
