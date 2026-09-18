"""Joint and world-frame orientation checks, independent of policy lifecycle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vex_policy.config.config_types.safety import EmergencyStopConfig
from vex_policy.policies.utils.initial_pose import normalize_quaternion_wxyz
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_to_rpy


@dataclass(frozen=True)
class Violation:
    field: str
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    invalid: bool = False

    def __str__(self) -> str:
        if self.invalid:
            return f"invalid_state:{self.field}"
        return f"{self.field}={self.value:.6g} outside [{self.lower:.6g}, {self.upper:.6g}]"


class EmergencyStop:
    def __init__(self, config: EmergencyStopConfig, dof_names: Sequence[str]):
        config.validate_joints(dof_names)
        self._size = len(dof_names)
        self._joints = [(name, dof_names.index(name), limits) for name, limits in config.joints.items()]
        self._rpy = [
            (axis, i, getattr(config.rpy, axis))
            for i, axis in enumerate(("roll", "pitch", "yaw"))
            if getattr(config.rpy, axis) is not None
        ]

    def check(self, state: LowState) -> tuple[Violation, ...]:
        required = {}
        if any(limits.pos is not None for _, _, limits in self._joints):
            required["joint_pos"] = (1, self._size)
        if any(limits.vel is not None for _, _, limits in self._joints):
            required["joint_vel"] = (1, self._size)
        if self._rpy:
            required["base_quat"] = (1, 4)
        values = {}
        for field, shape in required.items():
            try:
                value = np.asarray(getattr(state, field), dtype=np.float64)
            except (TypeError, ValueError, AttributeError):
                return (Violation(field, invalid=True),)
            if value.shape != shape or not np.isfinite(value).all():
                return (Violation(field, invalid=True),)
            values[field] = value[0]
        rpy = None
        if self._rpy:
            try:
                quaternion = normalize_quaternion_wxyz(values["base_quat"])
            except ValueError:
                return (Violation("base_quat", invalid=True),)
            rpy = np.asarray(quat_to_rpy(quaternion))
            rpy[[0, 2]] = (rpy[[0, 2]] + np.pi) % (2 * np.pi) - np.pi

        violations = []
        for name, i, limits in self._joints:
            if limits.pos is not None:
                q = float(values["joint_pos"][i])
                if q < limits.pos.min or q > limits.pos.max:
                    violations.append(Violation(f"{name}.pos", q, limits.pos.min, limits.pos.max))
            if limits.vel is not None:
                dq = float(values["joint_vel"][i])
                if abs(dq) > limits.vel:
                    violations.append(Violation(f"{name}.vel", dq, -limits.vel, limits.vel))
        for axis, i, bounds in self._rpy:
            value = float(rpy[i])
            if value < bounds.min or value > bounds.max:
                violations.append(Violation(f"rpy.{axis}", value, bounds.min, bounds.max))
        return tuple(violations)
