"""Clip physical position targets and dq, without position slew limiting."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from vex_policy.config.config_types.robot import RobotConfig
from vex_policy.config.config_types.safety import LimiterConfig

if TYPE_CHECKING:
    from vex_policy.policies.base import PolicyJointCommand


class JointCommandLimiter:
    def __init__(self, config: LimiterConfig, robot: RobotConfig):
        config.validate_joints(robot.dof_names)
        self._size = robot.num_joints
        self._offsets = np.zeros(self._size) if robot.joint_offsets_deg is None else np.deg2rad(robot.joint_offsets_deg)
        if self._offsets.shape != (self._size,) or not np.isfinite(self._offsets).all():
            raise ValueError("Invalid joint offsets for limiter")
        self._lower = np.full(self._size, -np.inf)
        self._upper = np.full(self._size, np.inf)
        self._velocity = np.full(self._size, np.inf)
        hardware = {}
        if robot.robot == "g1":
            from vex_policy.robots._g1_config import DOF_NAMES, JOINT_PARAMETERS

            hardware = {
                name: (JOINT_PARAMETERS["lower"][i], JOINT_PARAMETERS["upper"][i], JOINT_PARAMETERS["velocity"][i])
                for i, name in enumerate(DOF_NAMES)
            }
        for name, limits in config.joints.items():
            index = robot.dof_names.index(name)
            lower, upper, velocity = hardware.get(name, (-np.inf, np.inf, np.inf))
            if limits.pos is not None:
                self._lower[index] = max(limits.pos.min, lower)
                self._upper[index] = min(limits.pos.max, upper)
                if self._lower[index] > self._upper[index]:
                    raise ValueError(f"{name} limiter.pos does not intersect hardware limits")
            if limits.vel is not None:
                self._velocity[index] = min(limits.vel, velocity)

    def postprocess(self, command: PolicyJointCommand) -> PolicyJointCommand:
        q = np.asarray(command.q, dtype=np.float64)
        dq = np.asarray(command.dq, dtype=np.float64)
        selected = np.asarray(command.controlled_joints)
        for name, value in (("q", q), ("dq", dq)):
            if value.shape != (self._size,) or not np.isfinite(value).all():
                raise ValueError(f"limiter: invalid {name}")
        if selected.shape != (self._size,) or selected.dtype != np.bool_:
            raise ValueError("limiter: invalid controlled_joints")
        q, dq = q.copy(), dq.copy()
        position_mask = selected & (np.isfinite(self._lower) | np.isfinite(self._upper))
        velocity_mask = selected & np.isfinite(self._velocity)
        physical = q[position_mask] + self._offsets[position_mask]
        q[position_mask] = (
            np.clip(physical, self._lower[position_mask], self._upper[position_mask]) - self._offsets[position_mask]
        )
        dq[velocity_mask] = np.clip(dq[velocity_mask], -self._velocity[velocity_mask], self._velocity[velocity_mask])
        if not np.isfinite(q).all():
            raise ValueError("limiter: invalid physical position")
        return replace(command, q=q, dq=dq)
