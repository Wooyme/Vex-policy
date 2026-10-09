"""Configuration passed to one policy implementation."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from .action_mask import ActionMaskConfig
from .control import PolicyInput
from .GuardConfig import GuardConfig
from .observation import ObservationConfig
from .robot import RobotConfig
from .safety import EmergencyStopConfig, LimiterConfig
from .task import (
    BfmPoseTransitionTaskConfig,
    BfmTaskConfig,
    HoldPositionTaskConfig,
    InterpolationTaskConfig,
    PelvisRecoveryTaskConfig,
    PoseHoldTaskConfig,
    ReferenceLocomotionTaskConfig,
    SonicTaskConfig,
    TaskConfig,
    UfoTaskConfig,
    WaistLocomotionTaskConfig,
    WbtTaskConfig,
)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class InferenceConfig:
    robot: RobotConfig
    inputs: tuple[PolicyInput, ...]
    observation: ObservationConfig
    task: (
        TaskConfig
        | BfmTaskConfig
        | BfmPoseTransitionTaskConfig
        | WbtTaskConfig
        | SonicTaskConfig
        | WaistLocomotionTaskConfig
        | PelvisRecoveryTaskConfig
        | PoseHoldTaskConfig
        | ReferenceLocomotionTaskConfig
        | HoldPositionTaskConfig
        | InterpolationTaskConfig
        | UfoTaskConfig
    )
    guard: GuardConfig | None = None
    action_mask: ActionMaskConfig | None = None
    limiter: LimiterConfig | None = None
    estop: EmergencyStopConfig | None = None
