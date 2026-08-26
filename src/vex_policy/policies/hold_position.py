"""Model-free policy that holds the joint positions captured at activation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from loguru import logger

from vex_policy.config.config_types import HoldPositionTaskConfig, InferenceConfig

from .base import BasePolicy, PolicyJointCommand


class HoldPositionPolicy(BasePolicy):
    """Hold the activation-time DOF positions without loading a network."""

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, HoldPositionTaskConfig):
            raise TypeError("HoldPositionPolicy requires HoldPositionTaskConfig")
        self.config = config
        self.logger = logger
        self._init_robot_config(config.robot)
        self._init_sdk_components()
        self._init_communication_components()
        self.rl_rate = config.task.rl_rate
        self.use_phase = False
        self._init_latency_tracking()
        self.guard = None
        self.held_dof_pos: np.ndarray | None = None
        self.cmd_q = np.zeros(self.num_dofs, dtype=np.float64)
        self.cmd_dq = np.zeros(self.num_dofs, dtype=np.float64)
        self.cmd_tau = np.zeros(self.num_dofs, dtype=np.float64)
        kp = self.robot_config.motor_kp or self.robot_config.stiff_startup_kp
        kd = self.robot_config.motor_kd or self.robot_config.stiff_startup_kd
        if kp is None or kd is None:
            raise ValueError("Hold position requires motor or stiff-startup KP/KD in the robot config")
        if len(kp) != self.num_dofs or len(kd) != self.num_dofs:
            raise ValueError("Hold position KP/KD must match the robot DOF count")
        self.hold_kp = np.asarray(kp, dtype=np.float64)
        self.hold_kd = np.asarray(kd, dtype=np.float64)

    def activate(self) -> str | None:
        """Capture the current DOF positions and begin holding them."""
        self.held_dof_pos = None
        robot_state = self.interface.get_low_state()
        if robot_state is None:
            return "hold_position_start_failed: low_state_unavailable"
        state = np.asarray(robot_state)
        required = 7 + self.num_dofs
        if state.ndim != 2 or state.shape[0] != 1 or state.shape[1] < required:
            return f"hold_position_start_failed: invalid_low_state_shape={state.shape}"
        dof_pos = np.asarray(state[0, 7:required], dtype=np.float64)
        if not np.isfinite(dof_pos).all():
            return "hold_position_start_failed: invalid_dof_pos"
        self.held_dof_pos = dof_pos.copy()
        self.logger.info("Holding activation-time DOF positions")
        if not getattr(self, "_manager_owns_interface_lifecycle", False):
            self._configure_interface_writer()
            start_writer = getattr(self.interface, "start_command_writer", None)
            if start_writer is not None:
                start_writer()
            if hasattr(self.interface, "no_action"):
                self.interface.no_action = 0
        return None

    def deactivate(self) -> None:
        """Discard the snapshot so the next activation captures a fresh pose."""
        self.held_dof_pos = None
        if not getattr(self, "_manager_owns_interface_lifecycle", False):
            stop_writer = getattr(self.interface, "stop_command_writer", None)
            if stop_writer is not None:
                stop_writer()
            if hasattr(self.interface, "no_action"):
                self.interface.no_action = 1

    def apply_control(self, control: Mapping[str, float]) -> None:
        """The hold policy has no runtime control parameters."""
        del control

    def compute_joint_command(self, robot_state_data: np.ndarray) -> PolicyJointCommand:
        """Return the captured pose in the shared pre-offset command convention."""
        del robot_state_data
        if self.held_dof_pos is None:
            raise RuntimeError("Hold position policy is not active")
        return PolicyJointCommand(
            q=self.held_dof_pos - self.joint_offsets,
            dq=np.zeros(self.num_dofs, dtype=np.float64),
            tau=np.zeros(self.num_dofs, dtype=np.float64),
            kp=self.hold_kp.copy(),
            kd=self.hold_kd.copy(),
            controlled_joints=self.controlled_joint_mask.copy(),
        )
