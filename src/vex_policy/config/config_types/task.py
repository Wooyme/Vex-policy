"""Per-policy inference task configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class DebugConfig:
    force_upright_imu: bool = False
    force_zero_angular_velocity: bool = False
    force_zero_action: bool = False


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class HoldPositionTaskConfig:
    """Timing and writer configuration for the model-free hold policy."""

    action_mask_path: str | None = None
    rl_rate: float = Field(default=50.0, gt=0)
    lowcmd_publish_rate: float = Field(default=500.0, gt=0)
    low_state_timeout_s: float = Field(default=0.1, gt=0)


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
    lowcmd_publish_rate: float = Field(default=500.0, gt=0)
    low_state_timeout_s: float = Field(default=0.1, gt=0)
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
