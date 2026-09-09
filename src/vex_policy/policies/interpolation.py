"""Model-free joint interpolation to the first or last pose of a motion NPZ."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from vex_policy.config.config_types import InferenceConfig, InterpolationTaskConfig
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER, G1_JOINT_VELOCITY
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.joint_interpolation import JointPositionInterpolator

from .base import BasePolicy, PolicyJointCommand, PolicyRuntimeFault


def load_motion_pose(
    path: str | Path, dof_names: tuple[str, ...], target_frame: Literal["first", "last"]
) -> np.ndarray:
    """Read endpoint joint angles in hardware order, ignoring optional root data."""
    if target_frame not in {"first", "last"}:
        raise ValueError("target_frame must be first or last")
    motion_path = Path(path)
    if motion_path.suffix.lower() != ".npz":
        raise ValueError("Interpolation motion must be an NPZ file")
    try:
        with np.load(motion_path, allow_pickle=False) as motion:
            missing = {"joint_names", "joint_pos"} - set(motion.files)
            if missing:
                raise ValueError(f"missing arrays: {sorted(missing)}")
            names = motion["joint_names"]
            if names.ndim != 1 or names.size == 0 or names.dtype.kind not in {"U", "S"}:
                raise ValueError("joint_names must be a non-empty one-dimensional string array")
            joint_names = names.astype(str).tolist()
            if len(set(joint_names)) != len(joint_names):
                raise ValueError("joint_names contains duplicates")
            missing_joints = set(dof_names) - set(joint_names)
            if missing_joints:
                raise ValueError(f"missing robot joints: {sorted(missing_joints)}")
            positions = motion["joint_pos"]
            width = len(joint_names)
            if (
                positions.ndim != 2
                or positions.shape[0] == 0
                or positions.shape[1] not in {width, width + 7}
                or positions.dtype.kind not in {"f", "i", "u"}
            ):
                raise ValueError(f"joint_pos must be a numeric array of shape (T, {width}) or (T, {width + 7}), T > 0")
            frame = positions[0 if target_frame == "first" else -1, -width:]
            order = [joint_names.index(name) for name in dof_names]
            target = np.asarray(frame[order], dtype=np.float64)
            if not np.isfinite(target).all():
                raise ValueError("selected joint positions contain non-finite values")
            return target.copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"Failed to load interpolation motion {motion_path}: {error}") from error


class InterpolationPolicy(BasePolicy):
    """Interpolate from an activation snapshot, then hold the bounded endpoint."""

    def __init__(self, config: InferenceConfig):
        if not isinstance(config.task, InterpolationTaskConfig):
            raise TypeError("InterpolationPolicy requires InterpolationTaskConfig")
        self.config = config
        self.logger = logger
        self._init_robot_config(config.robot)
        self.rl_rate = config.task.rl_rate
        self.use_phase = False
        self._init_latency_tracking()
        self.guard = None
        self._active = False

        kp = self.robot_config.motor_kp
        kd = self.robot_config.motor_kd
        kp = self.robot_config.stiff_startup_kp if kp is None else kp
        kd = self.robot_config.stiff_startup_kd if kd is None else kd
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        for name, gains in (("KP", self._kp), ("KD", self._kd)):
            if gains.shape != (self.num_dofs,) or not np.isfinite(gains).all() or np.any(gains < 0):
                raise ValueError(f"Interpolation {name} must contain one finite non-negative gain per robot joint")

        # Constraint constants are in G1 hardware order; match by name explicitly.
        try:
            order = [G1_29DOF.dof_names.index(name) for name in self.dof_names]
        except ValueError as error:
            raise ValueError("Interpolation requires joints with known G1 limits") from error
        lower = np.asarray(G1_JOINT_LOWER)[order]
        upper = np.asarray(G1_JOINT_UPPER)[order]
        target = load_motion_pose(config.task.motion_data_path, self.dof_names, config.task.target_frame)
        self._target_q = np.clip(target + self.joint_offsets, lower, upper)
        self._interpolator = JointPositionInterpolator(
            joint_lower=lower,
            joint_upper=upper,
            joint_velocity=np.asarray(G1_JOINT_VELOCITY)[order],
            rate_hz=self.rl_rate,
            duration_s=config.task.duration_s,
            slew_safety_factor=config.robot.joint_interpolation_slew_safety_factor,
        )

    def _current_q(self, state: LowState) -> np.ndarray:
        positions = np.asarray(state.joint_pos, dtype=np.float64)
        if positions.shape != (1, self.num_dofs) or not np.isfinite(positions).all():
            raise ValueError("invalid joint positions in LowState")
        return positions[0]

    def activate(self, robot_state: LowState) -> str | None:
        self.deactivate()
        try:
            self._interpolator.reset(self._current_q(robot_state), self._target_q)
        except ValueError as error:
            return f"interpolation_start_failed: {error}"
        self._active = True
        self.logger.info(
            f"Interpolating to motion {self.config.task.target_frame} frame ({self.config.task.duration_s:.1f}s)"
        )
        return None

    def deactivate(self) -> None:
        self._active = False
        self._interpolator.clear()

    def apply_control(self, control: Mapping[str, float]) -> None:
        """This policy has no runtime control parameters."""
        del control

    def compute_joint_command(self, robot_state_data: LowState) -> PolicyJointCommand:
        if not self._active:
            raise RuntimeError("Interpolation policy is not active")
        try:
            # `complete` only signals elapsed nominal duration. Keep advancing
            # the bounded target afterward until it arrives, then hold it.
            step = self._interpolator.next(self._current_q(robot_state_data))
        except ValueError as error:
            raise PolicyRuntimeFault(f"interpolation_failed: {error}") from error
        return PolicyJointCommand(
            q=step.q_target - self.joint_offsets,
            dq=np.zeros(self.num_dofs, dtype=np.float64),
            tau=np.zeros(self.num_dofs, dtype=np.float64),
            kp=self._kp.copy(),
            kd=self._kd.copy(),
            controlled_joints=self.controlled_joint_mask.copy(),
        )
