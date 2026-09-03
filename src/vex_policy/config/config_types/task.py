"""Per-policy inference task configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class DebugConfig:
    force_upright_imu: bool = False
    force_zero_angular_velocity: bool = False
    force_zero_action: bool = False


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class HoldPositionTaskConfig:
    """Configuration for the model-free hold policy."""

    action_mask_path: str | None = None
    rl_rate: float = Field(default=50.0, gt=0)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class TaskConfig:
    """Parameters that affect one policy instance.

    Control input transport is deliberately absent. MQTT is owned once by the
    runtime and shared by every policy.
    """

    model_path: str
    action_mask_path: str | None = None
    rl_rate: float = 50.0
    policy_action_scale: float = 0.25
    action_scales_by_effort_limit_over_p_gain: bool = False
    use_phase: bool = True
    gait_period: float = 1.0
    skip_stiff_prompt: bool = True
    auto_walk_on_vel_cmd: bool = True
    use_sim_time: bool = False
    desired_base_height: float = 0.75
    residual_upper_body_action: bool = False
    print_observations: bool = False
    motion_data_path: str | None = None
    motion_start_timestep: int = 0
    motion_end_timestep: int | None = None
    motion_loop: bool = False
    debug: DebugConfig = DebugConfig()


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class WbtTaskConfig(TaskConfig):
    """Whole-body tracking startup and motion configuration."""

    startup_mode: Literal["interpolate", "immediate"] = "interpolate"
    init_duration_s: float = Field(default=10.0, gt=0.0, allow_inf_nan=False)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SonicTaskConfig(TaskConfig):
    """GEAR-SONIC decoder, encoder, planner, and timing configuration."""

    encoder_model_path: str = "models/sonic/model_encoder.onnx"
    planner_model_path: str = "models/sonic/planner_sonic.onnx"
    inference_provider: Literal["auto", "cpu", "cuda"] = "auto"
    motion_source: Literal["planner", "directory"] = "planner"
    motion_data_path: str | None = None
    motion_name: str | None = None
    motion_loop: bool = False
    planner_mode: int = Field(default=0, ge=0, le=26)
    planner_version: Literal[0, 1, 2] = 2
    planner_rate: float = Field(default=10.0, gt=0)
    motion_look_ahead_steps: int = Field(default=2, ge=0)
    planner_seed: int = 1234
    planner_default_height: float = 0.78874
    planner_encoder_mode: Literal[0] = 0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class WaistLocomotionTaskConfig(TaskConfig):
    """Pelvis-sine waist locomotion command and startup configuration."""

    def __post_init__(self) -> None:
        if not self.motion_data_path or not self.motion_data_path.strip():
            raise ValueError("motion_data_path must not be empty")


class UfoContextBase(BaseModel):
    """Strict local latent-context configuration for a UFO policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str

    @field_validator("path")
    @classmethod
    def nonempty_path(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("path must be non-empty and have no surrounding whitespace")
        return value


class UfoTrackingContextConfig(UfoContextBase):
    type: Literal["tracking"]
    start_frame: int = Field(default=0, ge=0)
    end_frame: int | None = Field(default=None, gt=0)
    stop_frame: int = Field(default=0, ge=0)
    gamma: float = Field(default=0.8, gt=0.0, le=1.0, allow_inf_nan=False)
    window_size: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def valid_frame_range(self) -> UfoTrackingContextConfig:
        if self.end_frame is not None and self.end_frame <= self.start_frame:
            raise ValueError("end_frame must be greater than start_frame")
        return self


class UfoRewardContextConfig(UfoContextBase):
    type: Literal["reward"]
    name: str
    z_id: int = Field(default=0, ge=0)

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("name must be non-empty and have no surrounding whitespace")
        return value


class UfoGoalContextConfig(UfoContextBase):
    type: Literal["goal"]
    name: str

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("name must be non-empty and have no surrounding whitespace")
        return value


UfoContextConfig = Annotated[
    UfoTrackingContextConfig | UfoRewardContextConfig | UfoGoalContextConfig,
    Field(discriminator="type"),
]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class UfoTaskConfig:
    """UFO-Deploy G1 policy and latent-context configuration."""

    model_path: str
    context: UfoContextConfig
    action_mask_path: str | None = None
    rl_rate: float = Field(default=50.0, gt=0.0, allow_inf_nan=False)
    inference_provider: Literal["auto", "cpu", "cuda"] = "cpu"
    startup_mode: Literal["prefill", "interpolate"] = "interpolate"
    init_duration_s: float = Field(default=10.0, gt=0.0, allow_inf_nan=False)
    q_target_slew_safety_factor: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    print_observations: bool = False
    debug: DebugConfig = DebugConfig()
