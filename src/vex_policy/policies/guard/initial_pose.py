"""Common full-body startup check against a named reference pose."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from vex_policy.config.config_types.GuardConfig import GuardConfig
from vex_policy.policies.utils.initial_pose import InitialPose, normalize_quaternion_wxyz
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse

from .base import BaseGuard

if TYPE_CHECKING:
    from loguru import Logger


class InitialPoseGuard(BaseGuard):
    def __init__(
        self,
        config: GuardConfig,
        initial_pose: InitialPose,
        dof_names: Sequence[str],
        logger: Logger,
        *,
        reason_prefix: str = "initial_pose_start_check_failed",
    ):
        self.dof_names = tuple(dof_names)
        if not self.dof_names or len(set(self.dof_names)) != len(self.dof_names):
            raise ValueError("Hardware joint names must be nonempty and unique")
        missing = set(self.dof_names) - set(initial_pose.dof_names)
        extra = set(initial_pose.dof_names) - set(self.dof_names)
        if missing or extra:
            raise ValueError(f"Initial pose joint names mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        unknown = config.startup_joint_tolerances_rad.keys() - set(self.dof_names)
        if unknown:
            raise ValueError(f"Unknown startup_joint_tolerances_rad joints: {sorted(unknown)}")
        positions = dict(zip(initial_pose.dof_names, initial_pose.dof_pos, strict=True))
        self._joint_pos = np.asarray([positions[name] for name in self.dof_names], dtype=np.float64)
        self._joint_tolerances = np.asarray(
            [
                config.startup_joint_tolerances_rad.get(name, config.startup_joint_tolerance_rad)
                for name in self.dof_names
            ]
        )
        self._gravity = np.asarray(initial_pose.projected_gravity)
        self._gravity_tolerance = config.startup_gravity_tolerance
        self.logger = logger
        self.reason_prefix = reason_prefix

    def _fail(self, detail: str) -> tuple[bool, str]:
        reason = f"{self.reason_prefix}: {detail}"
        self.logger.warning(reason)
        return False, reason

    def start_check(self, robot_state_data: LowState) -> tuple[bool, str | None]:
        num_dofs = len(self.dof_names)
        for name, shape in (
            ("joint_pos", (1, num_dofs)),
            ("joint_vel", (1, num_dofs)),
            ("base_ang_vel", (1, 3)),
            ("base_quat", (1, 4)),
        ):
            values = np.asarray(getattr(robot_state_data, name), dtype=np.float64)
            if values.shape != shape or not np.isfinite(values).all():
                return self._fail(f"invalid_state:{name}")
        try:
            quaternion = normalize_quaternion_wxyz(robot_state_data.base_quat[0])
        except ValueError as error:
            return self._fail(f"invalid_state:base_quat ({error})")

        errors = np.abs(robot_state_data.joint_pos[0] - self._joint_pos)
        excess = errors - self._joint_tolerances
        worst_index = int(np.argmax(excess))
        if excess[worst_index] > 0:
            return self._fail(
                f"{self.dof_names[worst_index]} error={errors[worst_index]:.3f}rad "
                f"> {self._joint_tolerances[worst_index]:.3f}rad"
            )

        gravity = quat_rotate_inverse(quaternion[None, :], np.asarray([[0.0, 0.0, -1.0]]))[0]
        error = float(np.linalg.norm(gravity - self._gravity))
        if error > self._gravity_tolerance:
            return self._fail(f"projected_gravity error={error:.3f} > {self._gravity_tolerance:.3f}")
        return True, None
