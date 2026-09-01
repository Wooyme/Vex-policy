"""Discovery, validation, and loading for versioned SDK log sessions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

SUPPORTED_SCHEMA_VERSION = 1

_STATE_VECTORS = {
    "state_base_pos": 3,
    "state_base_quat": 4,
    "state_joint_pos": "joints",
    "state_base_lin_vel": 3,
    "state_base_ang_vel": 3,
    "state_joint_vel": "joints",
    "state_q": "joints",
    "state_dq": "joints",
    "state_ddq": "joints",
    "state_tau_est": "joints",
}
_COMMAND_VECTORS = {
    "command_q_target": "motors",
    "command_dq_target": "motors",
    "command_tau_ff": "motors",
    "command_kp": "motors",
    "command_kd": "motors",
}
_STATE_ROWS = (
    "state_wall_time_ns",
    "state_monotonic_ns",
    "state_valid",
    "state_q_present",
    "state_dq_present",
    "state_ddq_present",
    "state_tau_est_present",
)
_COMMAND_ROWS = (
    "command_wall_time_ns",
    "command_monotonic_ns",
    "command_duration_ns",
    "command_success",
    "command_error_type",
    "command_error_message",
)
_ALL_ARRAYS = (*_STATE_ROWS, *_STATE_VECTORS, *_COMMAND_ROWS, *_COMMAND_VECTORS)


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Validated, immutable metadata for one discovered log session."""

    key: str
    path: Path
    session_id: str
    chunk_paths: tuple[Path, ...]
    chunk_indices: tuple[int, ...]
    started_wall_time_ns: int
    ended_wall_time_ns: int
    num_joints: int
    num_motors: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Catalog:
    """Snapshot of sessions found below a configured log directory."""

    root: Path
    sessions: tuple[SessionInfo, ...]
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class SessionData:
    """Concatenated arrays and diagnostics for a selected session."""

    info: SessionInfo
    arrays: dict[str, np.ndarray]
    dropped_state_count: int
    dropped_command_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ChunkMetadata:
    path: Path
    index: int
    session_id: str
    started_wall_time_ns: int
    ended_wall_time_ns: int
    num_joints: int
    num_motors: int


def discover_sessions(log_dir: Path | str) -> Catalog:
    """Take a fixed snapshot of complete chunk files below ``log_dir``."""
    root = Path(log_dir).expanduser().resolve()
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in root.rglob("chunk_*.npz"):
        if path.is_file():
            grouped[path.parent].append(path)

    sessions: list[SessionInfo] = []
    catalog_warnings: list[str] = []
    for session_path, paths in sorted(grouped.items(), key=lambda item: str(item[0])):
        metadata: list[_ChunkMetadata] = []
        warnings: list[str] = []
        for path in sorted(paths):
            try:
                metadata.append(_read_chunk_metadata(path))
            except (OSError, ValueError, KeyError) as exc:
                warnings.append(f"{path.name}: 无法读取元数据 ({exc})")

        if not metadata:
            catalog_warnings.append(f"{session_path}: 没有可读取的 schema v1 chunk")
            continue

        metadata.sort(key=lambda item: item.index)
        reference = metadata[0]
        accepted: list[_ChunkMetadata] = []
        seen_indices: set[int] = set()
        for chunk in metadata:
            if chunk.session_id != reference.session_id:
                warnings.append(f"{chunk.path.name}: session_id 与本会话不一致, 已跳过")
                continue
            if (chunk.num_joints, chunk.num_motors) != (
                reference.num_joints,
                reference.num_motors,
            ):
                warnings.append(f"{chunk.path.name}: 关节或电机维度不一致, 已跳过")
                continue
            if chunk.index in seen_indices:
                warnings.append(f"{chunk.path.name}: chunk_index={chunk.index} 重复, 已跳过")
                continue
            seen_indices.add(chunk.index)
            accepted.append(chunk)

        if not accepted:
            catalog_warnings.append(f"{session_path}: 没有一致且可用的 chunk")
            continue

        indices = tuple(chunk.index for chunk in accepted)
        for previous, current in pairwise(indices):
            if current != previous + 1:
                warnings.append(f"chunk_index {previous} 与 {current} 之间存在缺口")

        resolved_path = session_path.resolve()
        sessions.append(
            SessionInfo(
                key=str(resolved_path),
                path=resolved_path,
                session_id=reference.session_id,
                chunk_paths=tuple(chunk.path.resolve() for chunk in accepted),
                chunk_indices=indices,
                started_wall_time_ns=min(chunk.started_wall_time_ns for chunk in accepted),
                ended_wall_time_ns=max(chunk.ended_wall_time_ns for chunk in accepted),
                num_joints=reference.num_joints,
                num_motors=reference.num_motors,
                warnings=tuple(warnings),
            )
        )

    sessions.sort(key=lambda session: session.started_wall_time_ns, reverse=True)
    if not sessions and not catalog_warnings:
        catalog_warnings.append(f"{root}: 未发现 chunk_*.npz")
    return Catalog(root=root, sessions=tuple(sessions), warnings=tuple(catalog_warnings))


def load_session(info: SessionInfo) -> SessionData:
    """Load and concatenate all valid chunks from one discovered session."""
    columns: dict[str, list[np.ndarray]] = {name: [] for name in _ALL_ARRAYS}
    warnings = list(info.warnings)
    dropped_state_count = 0
    dropped_command_count = 0

    for path, expected_index in zip(info.chunk_paths, info.chunk_indices, strict=True):
        try:
            with np.load(path, allow_pickle=False) as chunk:
                _validate_chunk_arrays(chunk, info, expected_index)
                for name in _ALL_ARRAYS:
                    columns[name].append(np.asarray(chunk[name]))
                dropped_state_count = max(dropped_state_count, int(chunk["dropped_state_count"]))
                dropped_command_count = max(dropped_command_count, int(chunk["dropped_command_count"]))
        except (OSError, ValueError, KeyError) as exc:
            warnings.append(f"{path.name}: 数据校验失败, 已跳过 ({exc})")

    arrays: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if values:
            arrays[name] = np.concatenate(values, axis=0)
        else:
            arrays[name] = _empty_array(name, info.num_joints, info.num_motors)

    return SessionData(
        info=info,
        arrays=arrays,
        dropped_state_count=dropped_state_count,
        dropped_command_count=dropped_command_count,
        warnings=tuple(warnings),
    )


def _read_chunk_metadata(path: Path) -> _ChunkMetadata:
    with np.load(path, allow_pickle=False) as chunk:
        schema_version = int(chunk["schema_version"])
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"不支持 schema_version={schema_version}, 当前仅支持 {SUPPORTED_SCHEMA_VERSION}"
            )
        return _ChunkMetadata(
            path=path,
            index=int(chunk["chunk_index"]),
            session_id=str(chunk["session_id"]),
            started_wall_time_ns=int(chunk["chunk_started_wall_time_ns"]),
            ended_wall_time_ns=int(chunk["chunk_ended_wall_time_ns"]),
            num_joints=int(chunk["num_joints"]),
            num_motors=int(chunk["num_motors"]),
        )


def _validate_chunk_arrays(chunk: np.lib.npyio.NpzFile, info: SessionInfo, expected_index: int) -> None:
    if int(chunk["schema_version"]) != SUPPORTED_SCHEMA_VERSION:
        raise ValueError("schema_version 已变化")
    if str(chunk["session_id"]) != info.session_id:
        raise ValueError("session_id 已变化")
    if int(chunk["chunk_index"]) != expected_index:
        raise ValueError("chunk_index 已变化")
    if int(chunk["num_joints"]) != info.num_joints or int(chunk["num_motors"]) != info.num_motors:
        raise ValueError("关节或电机维度已变化")

    state_count = _row_count(chunk, "state_wall_time_ns")
    command_count = _row_count(chunk, "command_wall_time_ns")
    for name in _STATE_ROWS:
        _expect_shape(chunk, name, (state_count,))
    for name in _COMMAND_ROWS:
        _expect_shape(chunk, name, (command_count,))
    for name, width in _STATE_VECTORS.items():
        expected_width = info.num_joints if width == "joints" else width
        _expect_shape(chunk, name, (state_count, expected_width))
    for name, width in _COMMAND_VECTORS.items():
        expected_width = info.num_motors if width == "motors" else width
        _expect_shape(chunk, name, (command_count, expected_width))


def _row_count(chunk: np.lib.npyio.NpzFile, name: str) -> int:
    value = np.asarray(chunk[name])
    if value.ndim != 1:
        raise ValueError(f"{name} 应为一维数组, 实际 shape={value.shape}")
    return len(value)


def _expect_shape(chunk: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...]) -> None:
    actual = np.asarray(chunk[name]).shape
    if actual != shape:
        raise ValueError(f"{name} shape 应为 {shape}, 实际为 {actual}")


def _empty_array(name: str, num_joints: int, num_motors: int) -> np.ndarray:
    if name in _STATE_VECTORS:
        width = _STATE_VECTORS[name]
        return np.empty((0, num_joints if width == "joints" else width), dtype=np.float64)
    if name in _COMMAND_VECTORS:
        width = _COMMAND_VECTORS[name]
        return np.empty((0, num_motors if width == "motors" else width), dtype=np.float64)
    if name in {"command_error_type", "command_error_message"}:
        return np.empty(0, dtype=np.str_)
    if name in {
        "state_valid",
        "state_q_present",
        "state_dq_present",
        "state_ddq_present",
        "state_tau_est_present",
        "command_success",
    }:
        return np.empty(0, dtype=np.bool_)
    return np.empty(0, dtype=np.uint64)


__all__ = [
    "SUPPORTED_SCHEMA_VERSION",
    "Catalog",
    "SessionData",
    "SessionInfo",
    "discover_sessions",
    "load_session",
]
