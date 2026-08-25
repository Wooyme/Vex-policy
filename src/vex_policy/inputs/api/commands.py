"""Transport-independent command helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VelCmd:
    lin_vel: tuple[float, float]
    ang_vel: float
