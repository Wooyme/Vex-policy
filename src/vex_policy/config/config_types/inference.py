"""Configuration passed to one policy implementation."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from .observation import ObservationConfig
from .robot import RobotConfig
from .task import SonicTaskConfig, TaskConfig


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class InferenceConfig:
    robot: RobotConfig
    observation: ObservationConfig
    task: TaskConfig | SonicTaskConfig
