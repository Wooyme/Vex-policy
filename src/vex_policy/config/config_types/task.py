"""Per-policy inference task configuration."""

from __future__ import annotations

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class DebugConfig:
    force_upright_imu: bool = False
    force_zero_angular_velocity: bool = False
    force_zero_action: bool = False


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class TaskConfig:
    """Parameters that affect one policy instance.

    Control input transport is deliberately absent. MQTT is owned once by the
    runtime and shared by every policy.
    """

    model_path: str
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
    motion_start_timestep: int = 0
    motion_end_timestep: int | None = None
    debug: DebugConfig = DebugConfig()
