from typing import Annotated

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

PositiveTolerance = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GuardConfig:
    """Startup pose tolerances shared by all guarded policies."""

    startup_joint_tolerance_rad: PositiveTolerance = 0.2
    startup_gravity_tolerance: PositiveTolerance = 0.2
    startup_joint_tolerances_rad: dict[str, PositiveTolerance] = Field(default_factory=dict)
    """Joint-name overrides; omitted joints use startup_joint_tolerance_rad."""
