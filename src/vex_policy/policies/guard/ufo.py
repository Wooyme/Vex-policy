"""Startup guard for UFO policies."""

from __future__ import annotations

import numpy as np

from vex_policy.config.config_types import UfoGuardConfig
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse

from .base import BaseGuard


class UfoGuard(BaseGuard):
    """Require a valid upright pose close to UFO's default pose."""

    def __init__(self, config: UfoGuardConfig, policy):
        super().__init__(config, policy)
        self.config = config

    def _fail(self, reason: str) -> tuple[bool, str]:
        self.policy.logger.warning(reason)
        return False, reason

    def _current_projected_gravity(self, robot_state_data: LowState) -> np.ndarray:
        quaternion = np.asarray(robot_state_data.base_quat, dtype=np.float64).copy()
        quaternion_norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if not np.isfinite(quaternion_norm).all() or np.any(quaternion_norm < 1e-8):
            raise ValueError("robot base quaternion is invalid")
        quaternion /= quaternion_norm
        gravity = quat_rotate_inverse(quaternion, np.asarray([[0.0, 0.0, -1.0]]))[0]
        if not np.isfinite(gravity).all():
            raise ValueError("robot projected gravity is invalid")
        return gravity

    def start_check(self, robot_state_data: LowState) -> tuple[bool, str | None]:
        joint_pos = np.asarray(robot_state_data.joint_pos[0], dtype=np.float64)
        if joint_pos.shape != (self.policy.num_dofs,) or not np.isfinite(joint_pos).all():
            return self._fail("ufo_start_check_failed: invalid_joint_position")
        joint_errors = np.abs(joint_pos - self.policy.default_dof_angles)
        worst_index = int(np.argmax(joint_errors))
        worst_error = float(joint_errors[worst_index])
        if worst_error > self.config.startup_joint_tolerance_rad:
            return self._fail(
                "ufo_start_check_failed: "
                f"{self.policy.dof_names[worst_index]} error={worst_error:.3f}rad "
                f"> {self.config.startup_joint_tolerance_rad:.3f}rad"
            )

        try:
            current_gravity = self._current_projected_gravity(robot_state_data)
        except ValueError as error:
            return self._fail(f"ufo_start_check_failed: {error}")
        expected_gravity = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
        gravity_error = float(np.linalg.norm(current_gravity - expected_gravity))
        if gravity_error > self.config.startup_gravity_tolerance:
            return self._fail(
                "ufo_start_check_failed: "
                f"projected_gravity error={gravity_error:.3f} "
                f"> {self.config.startup_gravity_tolerance:.3f}"
            )
        return True, None
