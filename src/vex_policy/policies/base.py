"""Policy lifecycle and the hardware-order command contract.

PolicyStateMachine owns scheduling and hardware IO. A policy owns one active
episode at a time; optional inference components belong to its implementation.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, final

import numpy as np
from loguru import logger

from vex_policy.config.config_types.inference import InferenceConfig
from vex_policy.policies.emergency_stop import EmergencyStop
from vex_policy.policies.limiter import JointCommandLimiter
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.latency import LatencyTracker

if TYPE_CHECKING:
    from vex_policy.policies.guard.base import BaseGuard


@dataclass(frozen=True)
class PolicyJointCommand:
    """One policy's full hardware-order joint command before calibration offsets."""

    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    controlled_joints: np.ndarray


class PolicyRuntimeFault(RuntimeError):
    """Expected policy data/inference fault that must latch the runtime safely."""


class PolicyLifecycleState(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    CLOSED = "closed"


class BasePolicy(ABC):
    """Common lifecycle; constructors never dispatch into subclass hooks.

    Lifecycle calls are serialized with step/control on each instance. Hooks
    must not call lifecycle methods or acquire this lock from a worker thread.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = logger
        self.robot_config = config.robot
        self.num_dofs = config.robot.num_joints
        self.dof_names = config.robot.dof_names
        self.default_dof_angles = np.array(config.robot.default_dof_angles)
        offsets = config.robot.joint_offsets_deg
        self.joint_offsets = np.deg2rad(offsets) if offsets is not None else np.zeros(self.num_dofs)
        self.action_mask = np.ones((1, self.num_dofs), dtype=np.float32)
        if config.action_mask is not None:
            for name in config.action_mask.masked_joints:
                self.action_mask[0, self.dof_names.index(name)] = 0.0
        self.controlled_joint_mask = self.action_mask[0].astype(bool)
        self.rl_rate = config.task.rl_rate
        self.latency_tracker = LatencyTracker(window_size=max(1, int(self.rl_rate)))
        self.guard: BaseGuard | None = None
        self.limiter = JointCommandLimiter(config.limiter, config.robot) if config.limiter is not None else None
        self.estop = EmergencyStop(config.estop, self.dof_names) if config.estop is not None else None
        self._lifecycle_state = PolicyLifecycleState.INACTIVE
        self._lifecycle_lock = threading.RLock()

    @property
    def is_active(self) -> bool:
        return self._lifecycle_state == PolicyLifecycleState.ACTIVE

    @final
    def activate(self, robot_state_data: LowState) -> str | None:
        """Start a fresh episode, or leave an already active episode untouched."""
        with self._lifecycle_lock:
            if self._lifecycle_state == PolicyLifecycleState.CLOSED:
                raise RuntimeError("Cannot activate a closed policy")
            if self.is_active:
                return None
            if self.guard is not None:
                accepted, reason = self.guard.start_check(robot_state_data)
                if not accepted:
                    return reason or "policy_start_rejected"
            try:
                reason = self._on_activate(robot_state_data)
            except BaseException:
                try:
                    self._on_deactivate()
                except Exception:
                    self.logger.exception("Policy activation rollback failed")
                raise
            if reason is not None:
                self._on_deactivate()
                return reason
            self.latency_tracker.reset()
            self._lifecycle_state = PolicyLifecycleState.ACTIVE
            return None

    @final
    def apply_control(self, control: Mapping[str, float]) -> None:
        with self._lifecycle_lock:
            if not self.is_active:
                raise RuntimeError("Policy is not active")
            self._apply_control(control)

    @final
    def step(self, robot_state_data: LowState) -> PolicyJointCommand:
        with self._lifecycle_lock:
            if not self.is_active:
                raise RuntimeError("Policy is not active")
            self.latency_tracker.start_cycle()
            try:
                command = self._compute_command(robot_state_data)
                if self.limiter is not None:
                    try:
                        command = self.limiter.postprocess(command)
                    except (TypeError, ValueError) as error:
                        raise PolicyRuntimeFault(str(error)) from error
                return command
            finally:
                self.latency_tracker.end_cycle()

    @final
    def deactivate(self) -> None:
        """Stop this episode without emitting a command; safe to call repeatedly."""
        with self._lifecycle_lock:
            if not self.is_active:
                return
            try:
                self._on_deactivate()
            finally:
                self._lifecycle_state = PolicyLifecycleState.INACTIVE

    @final
    def close(self) -> None:
        """Stop the episode and release instance resources exactly once."""
        with self._lifecycle_lock:
            if self._lifecycle_state == PolicyLifecycleState.CLOSED:
                return
            try:
                self.deactivate()
            finally:
                try:
                    self._on_close()
                finally:
                    self._lifecycle_state = PolicyLifecycleState.CLOSED

    def _on_activate(self, robot_state_data: LowState) -> str | None:
        """Reset all episode state, capture the current pose, then start workers."""
        return None

    def _apply_control(self, control: Mapping[str, float]) -> None:
        """Implementations without runtime inputs need no control handler."""
        return None

    @abstractmethod
    def _compute_command(self, robot_state_data: LowState) -> PolicyJointCommand:
        """Compute one full hardware-order command without publishing it."""
        raise NotImplementedError

    def _on_deactivate(self) -> None:
        """Stop/join episode workers and discard episode state, including partial startup."""
        return None

    def _on_close(self) -> None:
        """Release instance resources; must also work on an inactive instance."""
        return None

    def get_reference_state(self) -> np.ndarray | None:
        return None
