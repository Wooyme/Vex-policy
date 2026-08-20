"""Transport-independent values delivered by the MQTT runtime."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VelCmd:
    lin_vel: tuple[float, float]
    ang_vel: float


@dataclass(frozen=True)
class ControlValues:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    height: float = 0.0
