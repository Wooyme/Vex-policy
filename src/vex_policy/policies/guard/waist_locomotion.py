"""Startup guard for pelvis-sine waist locomotion."""

from __future__ import annotations

import numpy as np

from vex_policy.config.config_types import WaistLocomotionGuardConfig
from vex_policy.utils.math.quat import quat_rotate_inverse

from .base import BaseGuard


class WaistLocomotionGuard(BaseGuard):
    """Require the robot to match the motion's final pose before inference."""

    def __init__(self, config: WaistLocomotionGuardConfig, policy):
        super().__init__(config, policy)
        self.config = config

    def _fail(self, reason: str) -> tuple[bool, str]:
        self.policy.logger.warning(reason)
        return False, reason

    def _current_projected_gravity(self, robot_state_data: np.ndarray) -> np.ndarray:
        base_state_size = 7 + self.policy.num_dofs + 6 + self.policy.num_dofs
        if robot_state_data.shape[1] == base_state_size + 3:
            gravity = np.asarray(robot_state_data[0, base_state_size : base_state_size + 3], dtype=np.float64)
        else:
            quaternion = np.asarray(robot_state_data[:, 3:7], dtype=np.float64)
            quaternion_norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
            if not np.isfinite(quaternion_norm).all() or np.any(quaternion_norm < 1e-8):
                raise ValueError("robot base quaternion is invalid")
            quaternion /= quaternion_norm
            gravity = quat_rotate_inverse(quaternion, np.asarray([[0.0, 0.0, -1.0]]))[0]
        if not np.isfinite(gravity).all():
            raise ValueError("robot projected gravity is invalid")
        return gravity

    def start_check(self) -> tuple[bool, str | None]:
        robot_state_data = self.policy.interface.get_low_state()
        if robot_state_data is None:
            return self._fail("waist_locomotion_start_check_failed: low_state_unavailable")
        robot_state_data = np.asarray(robot_state_data)
        minimum_state_size = 7 + self.policy.num_dofs + 6 + self.policy.num_dofs
        if (
            robot_state_data.ndim != 2
            or robot_state_data.shape[0] != 1
            or robot_state_data.shape[1] < minimum_state_size
        ):
            return self._fail(
                f"waist_locomotion_start_check_failed: invalid_low_state_shape={robot_state_data.shape}"
            )

        joint_pos = robot_state_data[0, 7 : 7 + self.policy.num_dofs]
        if not np.isfinite(joint_pos).all():
            return self._fail("waist_locomotion_start_check_failed: invalid_joint_position")
        joint_errors = np.abs(joint_pos - self.policy.default_dof_angles)
        worst_index = int(np.argmax(joint_errors))
        worst_error = float(joint_errors[worst_index])
        if worst_error > self.config.startup_joint_tolerance_rad:
            return self._fail(
                "waist_locomotion_start_check_failed: "
                f"{self.policy.dof_names[worst_index]} error={worst_error:.3f}rad "
                f"> {self.config.startup_joint_tolerance_rad:.3f}rad"
            )

        try:
            current_gravity = self._current_projected_gravity(robot_state_data)
        except ValueError as error:
            return self._fail(f"waist_locomotion_start_check_failed: {error}")
        expected_gravity = np.asarray(self.policy.initial_pose.projected_gravity, dtype=np.float64)
        gravity_error = float(np.linalg.norm(current_gravity - expected_gravity))
        if gravity_error > self.config.startup_gravity_tolerance:
            return self._fail(
                "waist_locomotion_start_check_failed: "
                f"projected_gravity error={gravity_error:.3f} > {self.config.startup_gravity_tolerance:.3f}"
            )
        return True, None
