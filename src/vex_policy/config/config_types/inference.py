"""Configuration passed to one policy implementation."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from .action_mask import ActionMaskConfig
from .control import PolicyInput
from .GuardConfig import GuardConfig, UfoGuardConfig, WaistLocomotionGuardConfig
from .observation import ObservationConfig
from .robot import RobotConfig
from .task import HoldPositionTaskConfig, SonicTaskConfig, TaskConfig, UfoTaskConfig, WaistLocomotionTaskConfig


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class InferenceConfig:
    robot: RobotConfig
    inputs: tuple[PolicyInput, ...]
    observation: ObservationConfig
    task: TaskConfig | SonicTaskConfig | WaistLocomotionTaskConfig | HoldPositionTaskConfig | UfoTaskConfig
    guard: GuardConfig | WaistLocomotionGuardConfig | UfoGuardConfig | None = None
    action_mask: ActionMaskConfig | None = None
