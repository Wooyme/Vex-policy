"""Pure analysis helpers shared by the viewer and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from log_viewer.data import SessionData


@dataclass(frozen=True, slots=True)
class SessionMetrics:
    duration_s: float
    state_count: int
    command_count: int
    state_frequency_hz: float | None
    command_frequency_hz: float | None
    state_interval_p95_ms: float | None
    command_interval_p95_ms: float | None
    invalid_state_count: int
    failed_command_count: int
    command_duration_p50_us: float | None
    command_duration_p95_us: float | None
    command_duration_p99_us: float | None
    command_duration_max_us: float | None


def compute_metrics(data: SessionData) -> SessionMetrics:
    arrays = data.arrays
    state_intervals_ms = positive_intervals(arrays["state_monotonic_ns"]) / 1e6
    command_intervals_ms = positive_intervals(arrays["command_monotonic_ns"]) / 1e6
    command_duration_us = np.asarray(arrays["command_duration_ns"], dtype=np.float64) / 1e3
    return SessionMetrics(
        duration_s=max(0.0, (data.info.ended_wall_time_ns - data.info.started_wall_time_ns) / 1e9),
        state_count=len(arrays["state_monotonic_ns"]),
        command_count=len(arrays["command_monotonic_ns"]),
        state_frequency_hz=_frequency_from_intervals(state_intervals_ms),
        command_frequency_hz=_frequency_from_intervals(command_intervals_ms),
        state_interval_p95_ms=_percentile(state_intervals_ms, 95),
        command_interval_p95_ms=_percentile(command_intervals_ms, 95),
        invalid_state_count=int(np.count_nonzero(~arrays["state_valid"])),
        failed_command_count=int(np.count_nonzero(~arrays["command_success"])),
        command_duration_p50_us=_percentile(command_duration_us, 50),
        command_duration_p95_us=_percentile(command_duration_us, 95),
        command_duration_p99_us=_percentile(command_duration_us, 99),
        command_duration_max_us=_percentile(command_duration_us, 100),
    )


def positive_intervals(timestamps_ns: np.ndarray) -> np.ndarray:
    values = np.asarray(timestamps_ns, dtype=np.int64)
    if values.size < 2:
        return np.empty(0, dtype=np.float64)
    intervals = np.diff(values)
    return intervals[intervals > 0].astype(np.float64)


def relative_seconds(timestamps_ns: np.ndarray, origin_ns: int) -> np.ndarray:
    values = np.asarray(timestamps_ns, dtype=np.int64)
    return (values - int(origin_ns)).astype(np.float64) / 1e9


def timeline_origin_ns(data: SessionData) -> int:
    candidates = []
    for name in ("state_monotonic_ns", "command_monotonic_ns"):
        values = data.arrays[name]
        if values.size:
            candidates.append(int(values[0]))
    return min(candidates, default=0)


def quaternion_wxyz_to_euler_degrees(quaternions: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternions to XYZ intrinsic roll/pitch/yaw degrees."""
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (N, 4), got {values.shape}")
    if not len(values):
        return np.empty((0, 3), dtype=np.float64)

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.full_like(values, np.nan), where=norms > 0)
    w, x, y, z = normalized.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.rad2deg(np.column_stack((roll, pitch, yaw)))


def decimate_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int = 4_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max-bin a line while retaining its endpoints and local extrema."""
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if x_values.ndim != 1 or y_values.ndim != 1 or len(x_values) != len(y_values):
        raise ValueError("x and y must be equally sized one-dimensional arrays")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    if len(x_values) <= max_points:
        return x_values, y_values
    if max_points == 2:
        return x_values[[0, -1]], y_values[[0, -1]]

    interior = np.arange(1, len(x_values) - 1)
    bin_count = max(1, (max_points - 2) // 2)
    selected = [0]
    for indices in np.array_split(interior, bin_count):
        if not len(indices):
            continue
        finite = indices[np.isfinite(y_values[indices])]
        if not len(finite):
            selected.append(int(indices[0]))
            continue
        minimum = int(finite[np.argmin(y_values[finite])])
        maximum = int(finite[np.argmax(y_values[finite])])
        selected.extend(sorted({minimum, maximum}))
    selected.append(len(x_values) - 1)
    unique = np.asarray(sorted(set(selected)), dtype=np.int64)
    if len(unique) > max_points:
        unique = unique[np.linspace(0, len(unique) - 1, max_points, dtype=np.int64)]
    return x_values[unique], y_values[unique]


def _frequency_from_intervals(intervals_ms: np.ndarray) -> float | None:
    median = _percentile(intervals_ms, 50)
    return None if median is None or median <= 0 else 1_000.0 / median


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    return float(np.percentile(finite, percentile))


__all__ = [
    "SessionMetrics",
    "compute_metrics",
    "decimate_series",
    "positive_intervals",
    "quaternion_wxyz_to_euler_degrees",
    "relative_seconds",
    "timeline_origin_ns",
]
