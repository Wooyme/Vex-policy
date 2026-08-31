"""Base interface for robot control."""

import time
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from vex_policy.config.config_types import RobotConfig
from vex_policy.sdk.high_frequency_logger import HighFrequencyLogger


@dataclass(frozen=True, slots=True)
class LowState:
    """One normalized low-level robot state in configured joint order."""

    base_pos: np.ndarray
    base_quat: np.ndarray
    joint_pos: np.ndarray
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    joint_vel: np.ndarray
    q: np.ndarray | None = None
    dq: np.ndarray | None = None
    ddq: np.ndarray | None = None
    tau_est: np.ndarray | None = None

    def __post_init__(self) -> None:
        expected_shapes = {
            "base_pos": (1, 3),
            "base_quat": (1, 4),
            "base_lin_vel": (1, 3),
            "base_ang_vel": (1, 3),
        }
        for name, expected_shape in expected_shapes.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != expected_shape:
                shape = getattr(value, "shape", None)
                raise ValueError(f"LowState.{name} must have shape {expected_shape}, got {shape}")

        for name in ("joint_pos", "joint_vel"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[0] != 1:
                shape = getattr(value, "shape", None)
                raise ValueError(f"LowState.{name} must have shape (1, N), got {shape}")

        if self.joint_pos.shape != self.joint_vel.shape:
            raise ValueError(
                "LowState.joint_pos and LowState.joint_vel must have matching shapes, "
                f"got {self.joint_pos.shape} and {self.joint_vel.shape}"
            )

        for name in ("q", "dq", "ddq", "tau_est"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, np.ndarray) or value.shape != self.joint_pos.shape):
                shape = getattr(value, "shape", None)
                raise ValueError(
                    f"LowState.{name} must be None or have shape {self.joint_pos.shape}, got {shape}"
                )


@dataclass(frozen=True, slots=True)
class LowCommand:
    """A backend command plus the final motor-order values sent to its SDK."""

    payload: Any
    q_target: np.ndarray
    dq_target: np.ndarray
    tau_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    def __post_init__(self) -> None:
        expected_shape = self.q_target.shape if isinstance(self.q_target, np.ndarray) else None
        for name in ("q_target", "dq_target", "tau_ff", "kp", "kd"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.ndim != 1 or value.shape != expected_shape:
                shape = getattr(value, "shape", None)
                raise ValueError(f"LowCommand.{name} must have shape {expected_shape}, got {shape}")


class BaseInterface(ABC):
    """
    Abstract base class for robot control interfaces.
    """

    def __init__(self, robot_config: RobotConfig, domain_id=0, interface_str=None, use_joystick=True):
        self.robot_config = robot_config
        self.domain_id = domain_id
        self.interface_str = interface_str
        self.use_joystick = use_joystick
        # Initialize key state tracking for joystick
        self._key_states: dict[str, bool] = {}
        self._last_key_states: dict[str, bool] = {}
        self._high_frequency_logger: HighFrequencyLogger | None = None
        log_config = self.robot_config.high_frequency_log
        if log_config is not None:
            try:
                self._high_frequency_logger = HighFrequencyLogger(
                    log_config,
                    num_joints=self.robot_config.num_joints,
                    num_motors=self.robot_config.num_motors,
                )
            except Exception as exc:
                logger.error(f"SDK high-frequency logger could not start: {type(exc).__name__}: {exc}")

    def get_low_state(self) -> LowState | None:
        """
        Get the raw low-level robot state.

        Returns:
            A structured state with a WXYZ base quaternion and configured-order
            joints, or ``None`` when the backend cannot return a state. Backends
            may return the same cached state in consecutive calls.
        """
        state = self._get_low_state()
        high_frequency_logger = self._high_frequency_logger
        if high_frequency_logger is not None:
            with suppress(Exception):
                high_frequency_logger.log_low_state(state)
        return state

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray = None,
        kp_override: np.ndarray = None,
        kd_override: np.ndarray = None,
    ) -> None:
        """
        Send low-level command to robot.

        Args:
            cmd_q: target joint positions (N,)
            cmd_dq: target joint velocities (N,)
            cmd_tau: feedforward torques (N,)
            dof_pos_latest: latest joint positions (N,)
            kp_override: optional KP gains override (N,)
            kd_override: optional KD gains override (N,)
        """
        command = self._prepare_low_command(
            cmd_q,
            cmd_dq,
            cmd_tau,
            dof_pos_latest,
            kp_override,
            kd_override,
        )
        high_frequency_logger = self._high_frequency_logger
        if high_frequency_logger is None:
            self._write_low_command(command)
            return

        wall_time_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        try:
            self._write_low_command(command)
        except Exception as exc:
            self._log_low_command(
                high_frequency_logger,
                command,
                success=False,
                duration_ns=time.monotonic_ns() - monotonic_ns,
                error=exc,
                wall_time_ns=wall_time_ns,
                monotonic_ns=monotonic_ns,
            )
            raise
        self._log_low_command(
            high_frequency_logger,
            command,
            success=True,
            duration_ns=time.monotonic_ns() - monotonic_ns,
            wall_time_ns=wall_time_ns,
            monotonic_ns=monotonic_ns,
        )

    def _get_low_state(self) -> LowState | None:
        """Read and normalize one backend-specific state sample."""
        raise NotImplementedError

    def _prepare_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray | None,
        kp_override: np.ndarray | None,
        kd_override: np.ndarray | None,
    ) -> LowCommand:
        """Build a backend payload and expose its final motor-order values."""
        raise NotImplementedError

    def _write_low_command(self, command: LowCommand) -> None:
        """Write one prepared command to the backend SDK."""
        raise NotImplementedError

    @staticmethod
    def _log_low_command(
        high_frequency_logger: HighFrequencyLogger,
        command: LowCommand,
        **result: Any,
    ) -> None:
        with suppress(Exception):
            high_frequency_logger.log_low_command(
                q_target=command.q_target,
                dq_target=command.dq_target,
                tau_ff=command.tau_ff,
                kp=command.kp,
                kd=command.kd,
                **result,
            )

    def close(self) -> None:
        """Flush and stop the optional SDK logger."""
        high_frequency_logger = self._high_frequency_logger
        self._high_frequency_logger = None
        if high_frequency_logger is not None:
            with suppress(Exception):
                high_frequency_logger.close()

    @property
    @abstractmethod
    def kp_level(self):
        raise NotImplementedError

    @kp_level.setter
    @abstractmethod
    def kp_level(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def kd_level(self):
        raise NotImplementedError

    @kd_level.setter
    @abstractmethod
    def kd_level(self, value):
        raise NotImplementedError
