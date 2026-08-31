"""Base interface for robot control."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from vex_policy.config.config_types import RobotConfig


@dataclass(frozen=True, slots=True)
class LowState:
    """One batch of raw low-level robot state in hardware joint order."""

    base_pos: np.ndarray
    base_quat: np.ndarray
    joint_pos: np.ndarray
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    joint_vel: np.ndarray

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

    @abstractmethod
    def get_low_state(self) -> LowState | None:
        """
        Get the raw low-level robot state.

        Returns:
            A structured state with a WXYZ base quaternion and hardware-order
            joints, or ``None`` when no fresh state is available.
        """
        raise NotImplementedError

    @abstractmethod
    def send_low_command(
            self,
            cmd_q: np.ndarray,
            cmd_dq: np.ndarray,
            cmd_tau: np.ndarray,
            dof_pos_latest: np.ndarray = None,
            kp_override: np.ndarray = None,
            kd_override: np.ndarray = None,
    ):
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
        raise NotImplementedError

    def update_config(self, robot_config: RobotConfig):
        """
        Update the robot configuration and propagate to internal components.

        Override in subclasses that need to update internal SDK components
        when the config changes (e.g., after loading KP/KD from ONNX metadata).

        Args:
            robot_config: The new robot configuration.
        """
        self.robot_config = robot_config

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
