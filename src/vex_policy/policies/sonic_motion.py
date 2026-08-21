"""Load GEAR-SONIC reference motions from the deploy CSV directory format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vex_policy.policies.sonic_planner import MotionSequence

_REQUIRED_FILES = ("joint_pos.csv", "joint_vel.csv", "body_pos.csv", "body_quat.csv")


def _is_motion_directory(path: Path) -> bool:
    return all((path / filename).is_file() for filename in _REQUIRED_FILES)


def _select_motion_directory(base_directory: Path, motion_name: str | None) -> Path:
    if motion_name:
        motion_directory = base_directory / motion_name
        if not motion_directory.is_dir():
            raise ValueError(f"SONIC motion {motion_name!r} does not exist under {base_directory}")
        return motion_directory

    if _is_motion_directory(base_directory):
        return base_directory

    candidates = sorted(
        (path for path in base_directory.iterdir() if path.is_dir() and _is_motion_directory(path)),
        key=lambda path: path.name,
    )
    if not candidates:
        raise ValueError(f"No valid SONIC motion directories found under {base_directory}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Multiple SONIC motions found under {base_directory}; configure motion_name from: {names}")
    return candidates[0]


def _read_csv(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read SONIC motion CSV {path}: {exc}") from exc
    if values.size == 0 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"SONIC motion CSV is empty: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"SONIC motion CSV contains non-finite values: {path}")
    return values


def load_motion_directory(
    base_directory: str | Path,
    *,
    motion_name: str | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[str, MotionSequence]:
    """Load one 50 Hz reference motion in the gear_sonic_deploy CSV format.

    ``base_directory`` may point either to one clip containing the four CSV
    files or to a collection of clip subdirectories. Joint CSV columns must be
    in the 29-DoF IsaacLab/policy order; body CSVs may contain multiple bodies,
    with the root stored first. ``end_frame`` is exclusive.
    """

    base_path = Path(base_directory).expanduser().resolve()
    if not base_path.is_dir():
        raise ValueError(f"SONIC motion directory does not exist: {base_path}")
    motion_directory = _select_motion_directory(base_path, motion_name)

    joint_positions = _read_csv(motion_directory / "joint_pos.csv")
    joint_velocities = _read_csv(motion_directory / "joint_vel.csv")
    body_positions = _read_csv(motion_directory / "body_pos.csv")
    body_quaternions = _read_csv(motion_directory / "body_quat.csv")

    if joint_positions.shape[1] != 29:
        raise ValueError(f"SONIC joint_pos.csv must have 29 columns in policy order, got {joint_positions.shape[1]}")
    if joint_velocities.shape[1] != 29:
        raise ValueError(f"SONIC joint_vel.csv must have 29 columns in policy order, got {joint_velocities.shape[1]}")
    if body_positions.shape[1] % 3:
        raise ValueError(f"SONIC body_pos.csv column count must be divisible by 3, got {body_positions.shape[1]}")
    if body_quaternions.shape[1] % 4:
        raise ValueError(f"SONIC body_quat.csv column count must be divisible by 4, got {body_quaternions.shape[1]}")

    frame_counts = {
        "joint_pos.csv": joint_positions.shape[0],
        "joint_vel.csv": joint_velocities.shape[0],
        "body_pos.csv": body_positions.shape[0],
        "body_quat.csv": body_quaternions.shape[0],
    }
    if len(set(frame_counts.values())) != 1:
        details = ", ".join(f"{name}={count}" for name, count in frame_counts.items())
        raise ValueError(f"SONIC motion CSV frame counts do not match: {details}")

    frames = joint_positions.shape[0]
    stop = frames if end_frame is None else end_frame
    if start_frame < 0 or stop <= start_frame or stop > frames:
        raise ValueError(f"Invalid SONIC motion frame range [{start_frame}, {stop}) for {frames} frames")
    frame_slice = slice(start_frame, stop)
    root_positions = body_positions.reshape(frames, -1, 3)[frame_slice, 0]
    root_quaternions = body_quaternions.reshape(frames, -1, 4)[frame_slice, 0]
    quaternion_norms = np.linalg.norm(root_quaternions, axis=1, keepdims=True)
    if np.any(quaternion_norms < 1e-8):
        raise ValueError(f"SONIC body_quat.csv contains a zero-length root quaternion: {motion_directory}")
    root_quaternions = root_quaternions / quaternion_norms

    motion = MotionSequence(
        root_positions.astype(np.float32, copy=False),
        root_quaternions.astype(np.float32, copy=False),
        joint_positions[frame_slice].astype(np.float32, copy=False),
        joint_velocities[frame_slice].astype(np.float32, copy=False),
    )
    return motion_directory.name, motion


__all__ = ["load_motion_directory"]
