"""Model-free policy that holds the joint positions captured at activation."""

from __future__ import annotations

import numpy as np

from vex_policy.config.config_types import HoldPositionTaskConfig, InferenceConfig
from vex_policy.sdk.base.base_interface import LowState

from .base import BasePolicy, PolicyJointCommand
from vex_policy.policies.utils.joint_command import position_command


class HoldPositionPolicy(BasePolicy):
    """Hold the activation-time DOF positions without loading a network."""

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, HoldPositionTaskConfig):
            raise TypeError("HoldPositionPolicy requires HoldPositionTaskConfig")
        super().__init__(config)
        self.held_dof_pos: np.ndarray | None = None
        kp = self.robot_config.motor_kp or self.robot_config.stiff_startup_kp
        kd = self.robot_config.motor_kd or self.robot_config.stiff_startup_kd
        if kp is None or kd is None:
            raise ValueError("Hold position requires motor or stiff-startup KP/KD in the robot config")
        if len(kp) != self.num_dofs or len(kd) != self.num_dofs:
            raise ValueError("Hold position KP/KD must match the robot DOF count")
        self.hold_kp = np.asarray(kp, dtype=np.float64)
        self.hold_kd = np.asarray(kd, dtype=np.float64)

    def _on_activate(self, robot_state: LowState) -> str | None:
        """Capture the current DOF positions and begin holding them."""
        self.held_dof_pos = None
        dof_pos = np.asarray(robot_state.joint_pos[0], dtype=np.float64)
        if not np.isfinite(dof_pos).all():
            return "hold_position_start_failed: invalid_dof_pos"
        self.held_dof_pos = dof_pos.copy()
        self.logger.info("Holding activation-time DOF positions")
        return None

    def _on_deactivate(self) -> None:
        """Discard the snapshot so the next activation captures a fresh pose."""
        self.held_dof_pos = None

    def _compute_command(self, robot_state_data: LowState) -> PolicyJointCommand:
        """Return the captured pose in the shared pre-offset command convention."""
        del robot_state_data
        if self.held_dof_pos is None:
            raise RuntimeError("Hold position policy is not active")
        return position_command(
            self.held_dof_pos - self.joint_offsets, self.hold_kp, self.hold_kd, self.controlled_joint_mask
        )
