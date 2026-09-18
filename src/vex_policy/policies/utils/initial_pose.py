"""Named startup reference pose, independent of motion loading and kinematics."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from vex_policy.utils.math.quat import quat_rotate_inverse


def normalize_quaternion_wxyz(value) -> np.ndarray:
    """Return a normalized (4,) quaternion without modifying the input."""
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("quaternion norm must be finite and nonzero")
    return quaternion / norm


class InitialPose(BaseModel):
    """Joint positions and world-frame base orientation for startup checks."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    dof_names: tuple[str, ...]
    dof_pos: tuple[float, ...]
    root_quat_wxyz: tuple[float, float, float, float]

    @field_validator("root_quat_wxyz")
    @classmethod
    def normalize_root_quaternion(cls, value):
        return tuple(normalize_quaternion_wxyz(value))

    @model_validator(mode="after")
    def validate_joints(self) -> InitialPose:
        if not self.dof_names or len(self.dof_names) != len(self.dof_pos):
            raise ValueError("dof_names and dof_pos must have the same non-zero length")
        if any(not name.strip() for name in self.dof_names):
            raise ValueError("dof_names must not contain empty names")
        if len(set(self.dof_names)) != len(self.dof_names):
            raise ValueError("dof_names must not contain duplicates")
        return self

    @property
    def projected_gravity(self) -> tuple[float, float, float]:
        gravity = quat_rotate_inverse(np.asarray([self.root_quat_wxyz]), np.asarray([[0.0, 0.0, -1.0]]))[0]
        return tuple(float(value) for value in gravity)
