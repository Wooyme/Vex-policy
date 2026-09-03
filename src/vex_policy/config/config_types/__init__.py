"""Type definitions for vex_policy configuration system."""

from .action_mask import ActionMaskConfig
from .control import InputParameter, JoystickInput, PolicyInput, SliderInput, input_parameters
from .GuardConfig import GuardConfig, UfoGuardConfig, WaistLocomotionGuardConfig
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
    SonicTaskConfig,
    TaskConfig,
    UfoContextConfig,
    UfoGoalContextConfig,
    UfoRewardContextConfig,
    UfoTaskConfig,
    UfoTrackingContextConfig,
    WaistLocomotionTaskConfig,
)

__all__ = [
    "ActionMaskConfig",
    "DebugConfig",
    "GuardConfig",
    "HoldPositionTaskConfig",
    "InferenceConfig",
    "InputParameter",
    "JoystickInput",
    "MqttConfig",
    "ObservationConfig",
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
    "input_parameters",
]
