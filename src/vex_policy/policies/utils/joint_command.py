"""Opt-in action transforms and position command construction."""

from __future__ import annotations

import numpy as np

from vex_policy.policies.base import PolicyJointCommand


def position_command(q, kp, kd, controlled_joints) -> PolicyJointCommand:
    """Copy a full hardware-order target; calibration is applied by the runtime."""
    q = np.asarray(q, dtype=np.float64).reshape(-1).copy()
    return PolicyJointCommand(
        q=q,
        dq=np.zeros_like(q),
        tau=np.zeros_like(q),
        kp=np.asarray(kp, dtype=np.float64).copy(),
        kd=np.asarray(kd, dtype=np.float64).copy(),
        controlled_joints=np.asarray(controlled_joints, dtype=bool).copy(),
    )


class PositionAction:
    """PPO action history, hardware-order masking and residual position scaling."""

    def __init__(self, num_dofs, mask, *, require_full_body=False, force_zero=False):
        self.num_dofs = num_dofs
        self.mask = mask
        self.require_full_body = require_full_body
        self.force_zero = force_zero
        self.reset()

    def reset(self):
        self.last = np.zeros((1, self.num_dofs))
        self.scaled = np.zeros((1, self.num_dofs))

    def process(self, action, scale):
        action = np.clip(np.asarray(action, dtype=np.float32), -100, 100)
        if action.ndim != 2 or action.shape[0] != 1:
            raise ValueError(f"Policy action must have shape (1, N), got {action.shape}")
        if action.shape[1] == self.num_dofs:
            action = action * self.mask
        elif self.require_full_body:
            raise ValueError(
                f"Action masks require a {self.num_dofs}-element full-body model output, got {action.shape[1]}"
            )
        if self.force_zero:
            action.fill(0.0)
        self.last = action.copy()
        self.scaled = action * scale
        return self.scaled

    def target(self, default_angles):
        # Preserve the legacy leading-zero padding for partial PPO outputs.
        action = self.scaled
        if action.shape[1] != self.num_dofs:
            action = np.concatenate([np.zeros((1, self.num_dofs - action.shape[1])), action], axis=1)
        return action + default_angles
