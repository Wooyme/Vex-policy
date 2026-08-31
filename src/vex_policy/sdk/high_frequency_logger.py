"""Buffered high-frequency logging for SDK state and command traffic."""

from __future__ import annotations

import io
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@dataclass(frozen=True, slots=True)
class HighFrequencyLogConfig:
    """Configuration for asynchronous SDK logging."""

    directory: Path
    chunk_interval_s: float = 5.0
    queue_capacity: int = 10_000
    compressed: bool = True

    def __post_init__(self) -> None:
        directory = Path(self.directory).expanduser().resolve()
        object.__setattr__(self, "directory", directory)
        if self.chunk_interval_s <= 0:
            raise ValueError("HighFrequencyLogConfig.chunk_interval_s must be positive")
        if self.queue_capacity <= 0:
            raise ValueError("HighFrequencyLogConfig.queue_capacity must be positive")


@dataclass(frozen=True, slots=True)
class _StateRecord:
    wall_time_ns: int
    monotonic_ns: int
    valid: bool
    values: dict[str, np.ndarray]
    present: dict[str, bool]


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    wall_time_ns: int
    monotonic_ns: int
    duration_ns: int
    success: bool
    error_type: str
    error_message: str
    q_target: np.ndarray
    dq_target: np.ndarray
    tau_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


class HighFrequencyLogger:
    """Move SDK log serialization and disk I/O off the control thread."""

    SCHEMA_VERSION = 1
    _OPTIONAL_STATE_FIELDS = ("q", "dq", "ddq", "tau_est")

    def __init__(self, config: HighFrequencyLogConfig, *, num_joints: int, num_motors: int):
        if num_joints <= 0 or num_motors <= 0:
            raise ValueError("HighFrequencyLogger joint and motor counts must be positive")

        self.config = config
        self.num_joints = int(num_joints)
        self.num_motors = int(num_motors)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.session_id = f"{timestamp}_{os.getpid()}_{secrets.token_hex(4)}"
        self.session_directory = config.directory / self.session_id

        self._condition = threading.Condition()
        self._queue: deque[_StateRecord | _CommandRecord] = deque()
        self._closed = False
        self._last_error: str | None = None
        self._dropped_state_count = 0
        self._dropped_command_count = 0
        self._chunk_index = 0
        self._thread = threading.Thread(
            target=self._run,
            name="vex-sdk-high-frequency-logger",
            daemon=True,
        )
        self._thread.start()

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    @property
    def dropped_counts(self) -> tuple[int, int]:
        """Return cumulative ``(state, command)`` queue-drop counts."""
        with self._condition:
            return self._dropped_state_count, self._dropped_command_count

    def log_low_state(
        self,
        state: Any | None,
        *,
        wall_time_ns: int | None = None,
        monotonic_ns: int | None = None,
    ) -> None:
        """Queue one returned low state, including an explicit ``None`` result."""
        if self._closed:
            return
        try:
            record = self._copy_state_record(
                state,
                wall_time_ns=time.time_ns() if wall_time_ns is None else wall_time_ns,
                monotonic_ns=time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
            )
            self._enqueue(record)
        except Exception as exc:
            self._record_producer_error("state", exc)

    def log_low_command(
        self,
        *,
        q_target: Any,
        dq_target: Any,
        tau_ff: Any,
        kp: Any,
        kd: Any,
        success: bool,
        duration_ns: int,
        error: BaseException | None = None,
        wall_time_ns: int | None = None,
        monotonic_ns: int | None = None,
    ) -> None:
        """Queue one final motor-order SDK command and its write result."""
        if self._closed:
            return
        try:
            record = _CommandRecord(
                wall_time_ns=time.time_ns() if wall_time_ns is None else int(wall_time_ns),
                monotonic_ns=time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns),
                duration_ns=max(0, int(duration_ns)),
                success=bool(success),
                error_type="" if error is None else type(error).__name__,
                error_message="" if error is None else str(error)[:1024],
                q_target=self._copy_vector(q_target, self.num_motors, "q_target"),
                dq_target=self._copy_vector(dq_target, self.num_motors, "dq_target"),
                tau_ff=self._copy_vector(tau_ff, self.num_motors, "tau_ff"),
                kp=self._copy_vector(kp, self.num_motors, "kp"),
                kd=self._copy_vector(kd, self.num_motors, "kd"),
            )
            self._enqueue(record)
        except Exception as exc:
            self._record_producer_error("command", exc)

    def close(self, timeout: float = 10.0) -> None:
        """Stop accepting records and flush the remaining queue."""
        with self._condition:
            if not self._closed:
                self._closed = True
                self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            logger.warning("SDK high-frequency logger did not stop within the close timeout")

    def _copy_state_record(self, state: Any | None, *, wall_time_ns: int, monotonic_ns: int) -> _StateRecord:
        field_widths = self._state_field_widths()
        if state is None:
            values = {name: np.zeros(width, dtype=np.float64) for name, width in field_widths.items()}
            present = {name: False for name in self._OPTIONAL_STATE_FIELDS}
            return _StateRecord(int(wall_time_ns), int(monotonic_ns), False, values, present)

        values: dict[str, np.ndarray] = {}
        for name, width in field_widths.items():
            value = getattr(state, name, None)
            if value is None:
                values[name] = np.zeros(width, dtype=np.float64)
            else:
                values[name] = self._copy_vector(value, width, name)
        present = {name: getattr(state, name, None) is not None for name in self._OPTIONAL_STATE_FIELDS}
        return _StateRecord(int(wall_time_ns), int(monotonic_ns), True, values, present)

    def _copy_vector(self, value: Any, width: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != width:
            raise ValueError(f"{name} must contain {width} values, got {array.size}")
        return array.copy()

    def _state_field_widths(self) -> dict[str, int]:
        return {
            "base_pos": 3,
            "base_quat": 4,
            "joint_pos": self.num_joints,
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "joint_vel": self.num_joints,
            "q": self.num_joints,
            "dq": self.num_joints,
            "ddq": self.num_joints,
            "tau_est": self.num_joints,
        }

    def _enqueue(self, record: _StateRecord | _CommandRecord) -> None:
        with self._condition:
            if self._closed:
                return
            if len(self._queue) >= self.config.queue_capacity:
                dropped = self._queue.popleft()
                self._increment_dropped(dropped)
            self._queue.append(record)

    def _record_producer_error(self, kind: str, exc: Exception) -> None:
        with self._condition:
            if kind == "state":
                self._dropped_state_count += 1
            else:
                self._dropped_command_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"

    def _increment_dropped(self, record: _StateRecord | _CommandRecord) -> None:
        if isinstance(record, _StateRecord):
            self._dropped_state_count += 1
        else:
            self._dropped_command_count += 1

    def _run(self) -> None:
        records: list[_StateRecord | _CommandRecord] = []
        try:
            self.session_directory.mkdir(parents=True, exist_ok=False)
            deadline = time.monotonic() + self.config.chunk_interval_s
            while True:
                with self._condition:
                    while not self._closed:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)

                    records = list(self._queue)
                    self._queue.clear()
                    closed = self._closed
                    dropped_counts = (self._dropped_state_count, self._dropped_command_count)

                if records:
                    self._write_chunk(records, dropped_counts)
                    records = []
                if closed:
                    return
                deadline = time.monotonic() + self.config.chunk_interval_s
        except Exception as exc:
            self._fail(exc, records)

    def _fail(self, exc: Exception, pending_records: list[_StateRecord | _CommandRecord]) -> None:
        message = f"{type(exc).__name__}: {exc}"
        with self._condition:
            self._last_error = message
            self._closed = True
            for record in pending_records:
                self._increment_dropped(record)
            while self._queue:
                self._increment_dropped(self._queue.popleft())
        logger.error(f"SDK high-frequency logger disabled after an internal error: {message}")

    def _write_chunk(
        self,
        records: list[_StateRecord | _CommandRecord],
        dropped_counts: tuple[int, int],
    ) -> None:
        payload = self._build_payload(records, dropped_counts)
        buffer = io.BytesIO()
        save = np.savez_compressed if self.config.compressed else np.savez
        save(buffer, **payload)

        final_path = self.session_directory / f"chunk_{self._chunk_index:06d}.npz"
        partial_path = final_path.with_suffix(".npz.partial")
        with partial_path.open("wb") as stream:
            stream.write(buffer.getbuffer())
        partial_path.replace(final_path)
        self._chunk_index += 1

    def _build_payload(
        self,
        records: list[_StateRecord | _CommandRecord],
        dropped_counts: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        states = [record for record in records if isinstance(record, _StateRecord)]
        commands = [record for record in records if isinstance(record, _CommandRecord)]
        wall_times = [record.wall_time_ns for record in records]
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(self.SCHEMA_VERSION, dtype=np.uint16),
            "session_id": np.asarray(self.session_id),
            "chunk_index": np.asarray(self._chunk_index, dtype=np.uint64),
            "chunk_started_wall_time_ns": np.asarray(min(wall_times), dtype=np.uint64),
            "chunk_ended_wall_time_ns": np.asarray(max(wall_times), dtype=np.uint64),
            "num_joints": np.asarray(self.num_joints, dtype=np.uint32),
            "num_motors": np.asarray(self.num_motors, dtype=np.uint32),
            "dropped_state_count": np.asarray(dropped_counts[0], dtype=np.uint64),
            "dropped_command_count": np.asarray(dropped_counts[1], dtype=np.uint64),
            "state_wall_time_ns": np.asarray([record.wall_time_ns for record in states], dtype=np.uint64),
            "state_monotonic_ns": np.asarray([record.monotonic_ns for record in states], dtype=np.uint64),
            "state_valid": np.asarray([record.valid for record in states], dtype=np.bool_),
        }

        for name, width in self._state_field_widths().items():
            payload[f"state_{name}"] = self._stack_vectors(
                [record.values[name] for record in states], width
            )
        for name in self._OPTIONAL_STATE_FIELDS:
            payload[f"state_{name}_present"] = np.asarray(
                [record.present[name] for record in states], dtype=np.bool_
            )

        payload.update(
            {
                "command_wall_time_ns": np.asarray(
                    [record.wall_time_ns for record in commands], dtype=np.uint64
                ),
                "command_monotonic_ns": np.asarray(
                    [record.monotonic_ns for record in commands], dtype=np.uint64
                ),
                "command_duration_ns": np.asarray(
                    [record.duration_ns for record in commands], dtype=np.uint64
                ),
                "command_success": np.asarray([record.success for record in commands], dtype=np.bool_),
                "command_error_type": np.asarray([record.error_type for record in commands], dtype=np.str_),
                "command_error_message": np.asarray(
                    [record.error_message for record in commands], dtype=np.str_
                ),
            }
        )
        for name in ("q_target", "dq_target", "tau_ff", "kp", "kd"):
            payload[f"command_{name}"] = self._stack_vectors(
                [getattr(record, name) for record in commands], self.num_motors
            )
        return payload

    @staticmethod
    def _stack_vectors(values: list[np.ndarray], width: int) -> np.ndarray:
        if not values:
            return np.empty((0, width), dtype=np.float64)
        return np.stack(values, axis=0)


__all__ = ["HighFrequencyLogConfig", "HighFrequencyLogger"]
