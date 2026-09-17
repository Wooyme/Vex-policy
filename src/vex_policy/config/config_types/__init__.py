"""Type definitions for vex_policy configuration system."""

from .action_mask import ActionMaskConfig
from .control import InputParameter, JoystickInput, PolicyInput, SliderInput, input_parameters
from .GuardConfig import GuardConfig, PassiveLocomotionGuardConfig, UfoGuardConfig, WaistLocomotionGuardConfig
from .inference import InferenceConfig
from .observation import ObservationConfig
from .robot import RobotConfig
from .runtime import (
    MqttConfig,
    PolicySpec,
    PolicyType,
    RobotRuntimeConfig,
    RuntimeConfig,
)
from .task import (
    DebugConfig,
    HoldPositionTaskConfig,
    InterpolationTaskConfig,
    PassiveLocomotionTaskConfig,
    SonicTaskConfig,
    TaskConfig,
    UfoContextConfig,
    UfoGoalContextConfig,
    UfoRewardContextConfig,
    UfoTaskConfig,
    UfoTrackingContextConfig,
    WaistLocomotionTaskConfig,
    WbtTaskConfig,
)

__all__ = [
    "ActionMaskConfig",
    "DebugConfig",
    "GuardConfig",
    "HoldPositionTaskConfig",
    "InferenceConfig",
    "InputParameter",
    "InterpolationTaskConfig",
    "JoystickInput",
    "MqttConfig",
    "ObservationConfig",
    "PassiveLocomotionGuardConfig",
    "PassiveLocomotionTaskConfig",
    "PolicyInput",
    "PolicySpec",
    "PolicyType",
    "RobotConfig",
    "RobotRuntimeConfig",
    "RuntimeConfig",
    "SliderInput",
    "SonicTaskConfig",
    "TaskConfig",
    "UfoContextConfig",
    "UfoGoalContextConfig",
    "UfoGuardConfig",
    "UfoRewardContextConfig",
    "UfoTaskConfig",
    "UfoTrackingContextConfig",
    "WaistLocomotionGuardConfig",
    "WaistLocomotionTaskConfig",
    "WbtTaskConfig",
    "input_parameters",
]
