"""Bounded, velocity-limited joint-position interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector(value, name: str, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector, got shape {vector.shape}")
    if shape is not None and vector.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


def limit_joint_position_target(
    q_target,
    *,
    current_q,
    previous_q,
    joint_lower,
    joint_upper,
    joint_velocity,
    dt: float,
    slew_safety_factor: float,
) -> np.ndarray:
    """Clip a joint target to hardware position and per-cycle velocity limits."""
    target = _vector(q_target, "q_target")
    shape = target.shape
    current = _vector(current_q, "current_q", shape=shape)
    previous = current if previous_q is None else _vector(previous_q, "previous_q", shape=shape)
    lower = _vector(joint_lower, "joint_lower", shape=shape)
    upper = _vector(joint_upper, "joint_upper", shape=shape)
    velocity = _vector(joint_velocity, "joint_velocity", shape=shape)

    if np.any(lower > upper):
        raise ValueError("joint_lower must not exceed joint_upper")
    if np.any(velocity < 0.0):
        raise ValueError("joint_velocity must be non-negative")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(slew_safety_factor) or slew_safety_factor < 0.0:
        raise ValueError("slew_safety_factor must be finite and non-negative")

    bounded_target = np.clip(target, lower, upper)
    max_delta = velocity * dt * slew_safety_factor
    limited_target = previous + np.clip(bounded_target - previous, -max_delta, max_delta)
    result = np.clip(limited_target, lower, upper)
    if not np.isfinite(result).all():
        raise ValueError("limited joint target contains non-finite values")
    return result


@dataclass(frozen=True)
class JointInterpolationStep:
    q_target: np.ndarray
    complete: bool


class JointPositionInterpolator:
    """Fixed-duration interpolation with joint position and velocity constraints."""

    def __init__(
        self,
        *,
        joint_lower,
        joint_upper,
        joint_velocity,
        rate_hz: float,
        duration_s: float,
        slew_safety_factor: float,
    ) -> None:
        self._joint_lower = _vector(joint_lower, "joint_lower").copy()
        shape = self._joint_lower.shape
        self._joint_upper = _vector(joint_upper, "joint_upper", shape=shape).copy()
        self._joint_velocity = _vector(joint_velocity, "joint_velocity", shape=shape).copy()
        if np.any(self._joint_lower > self._joint_upper):
            raise ValueError("joint_lower must not exceed joint_upper")
        if np.any(self._joint_velocity < 0.0):
            raise ValueError("joint_velocity must be non-negative")
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and positive")
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if not np.isfinite(slew_safety_factor) or not 0.0 < slew_safety_factor <= 1.0:
            raise ValueError("slew_safety_factor must be in (0, 1]")

        self._dt = 1.0 / rate_hz
        self._total_steps = max(1, round(duration_s * rate_hz))
        self._slew_safety_factor = slew_safety_factor
        self._start_q: np.ndarray | None = None
        self._target_q: np.ndarray | None = None
        self._previous_q: np.ndarray | None = None
        self._step = 0

    def reset(self, start_q, target_q) -> None:
        shape = self._joint_lower.shape
        validated_start = _vector(start_q, "start_q", shape=shape).copy()
        validated_target = _vector(target_q, "target_q", shape=shape).copy()
        self._start_q = validated_start
        self._target_q = validated_target
        self._previous_q = None
        self._step = 0

    def clear(self) -> None:
        self._start_q = None
        self._target_q = None
        self._previous_q = None
        self._step = 0

    def next(self, current_q) -> JointInterpolationStep:
        if self._start_q is None or self._target_q is None:
            raise RuntimeError("joint interpolator has not been reset")

        alpha = min((self._step + 1) / self._total_steps, 1.0)
        raw_target = self._start_q + (self._target_q - self._start_q) * alpha
        q_target = limit_joint_position_target(
            raw_target,
            current_q=current_q,
            previous_q=self._previous_q,
            joint_lower=self._joint_lower,
            joint_upper=self._joint_upper,
            joint_velocity=self._joint_velocity,
            dt=self._dt,
            slew_safety_factor=self._slew_safety_factor,
        )
        self._previous_q = q_target.copy()
        self._step += 1
        return JointInterpolationStep(q_target=q_target, complete=self._step >= self._total_steps)


__all__ = ["JointInterpolationStep", "JointPositionInterpolator", "limit_joint_position_target"]
