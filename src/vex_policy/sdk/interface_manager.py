"""Process-wide ownership of the robot interface."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from vex_policy.config.config_types import RobotConfig
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.sdk.factory import _create_interface


class InterfaceManager:
    """Explicit singleton factory and the only application-level SDK I/O owner."""

    _instance: ClassVar[InterfaceManager | None] = None

    def __init__(self, interface: Any):
        self._interface = interface

    @classmethod
    def initialize(
        cls,
        robot_config: RobotConfig,
        domain_id: int = 0,
        interface_str: str | None = None,
        use_joystick: bool = False,
        *,
        interface: Any | None = None,
    ) -> InterfaceManager:
        """Create the process singleton, returning the existing one if initialized."""
        if cls._instance is None:
            backend = (
                interface
                if interface is not None
                else _create_interface(
                    robot_config,
                    domain_id,
                    interface_str,
                    use_joystick,
                )
            )
            cls._instance = cls(backend)
        return cls._instance

    @classmethod
    def get(cls) -> InterfaceManager:
        """Return the initialized singleton."""
        if cls._instance is None:
            raise RuntimeError("InterfaceManager has not been initialized")
        return cls._instance

    @classmethod
    def close(cls) -> None:
        """Close and forget the singleton backend."""
        instance = cls._instance
        if instance is None:
            return
        cls._instance = None
        close = getattr(instance._interface, "close", None)
        if close is not None:
            close()

    def get_low_state(self) -> LowState | None:
        return self._interface.get_low_state()

    def send_low_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        dof_pos_latest: np.ndarray | None = None,
        kp_override: np.ndarray | None = None,
        kd_override: np.ndarray | None = None,
    ) -> None:
        self._interface.send_low_command(
            cmd_q,
            cmd_dq,
            cmd_tau,
            dof_pos_latest=dof_pos_latest,
            kp_override=kp_override,
            kd_override=kd_override,
        )


__all__ = ["InterfaceManager"]
