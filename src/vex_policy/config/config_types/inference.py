"""Configuration passed to one policy implementation."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from .action_mask import ActionMaskConfig
from .control import PolicyInput
from .GuardConfig import GuardConfig, WaistLocomotionGuardConfig
from .observation import ObservationConfig
from .robot import RobotConfig
from .task import SonicTaskConfig, TaskConfig, WaistLocomotionTaskConfig


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class InferenceConfig:
    robot: RobotConfig
    inputs: tuple[PolicyInput, ...]
    observation: ObservationConfig
    task: TaskConfig | SonicTaskConfig | WaistLocomotionTaskConfig
    guard: GuardConfig | WaistLocomotionGuardConfig | None = None
    action_mask: ActionMaskConfig | None = None
