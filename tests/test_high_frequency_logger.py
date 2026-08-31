from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vex_policy.cli import build_parser
from vex_policy.sdk.base.base_interface import BaseInterface, LowState
from vex_policy.sdk.high_frequency_logger import HighFrequencyLogConfig, HighFrequencyLogger
from vex_policy.sdk.unitree.unitree_interface import UnitreeInterface


def _low_state(offset: float = 0.0) -> LowState:
    joint_pos = np.arange(3, dtype=np.float64).reshape(1, 3) + offset
    joint_vel = joint_pos + 10
    return LowState(
        base_pos=np.asarray([[1.0, 2.0, 3.0]]) + offset,
        base_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=joint_pos,
        base_lin_vel=np.asarray([[4.0, 5.0, 6.0]]) + offset,
        base_ang_vel=np.asarray([[7.0, 8.0, 9.0]]) + offset,
        joint_vel=joint_vel,
        q=joint_pos.copy(),
        dq=joint_vel.copy(),
        ddq=joint_vel + 10,
        tau_est=joint_vel + 20,
    )


def _load_only_chunk(logger: HighFrequencyLogger):
    chunks = sorted(logger.session_directory.glob("chunk_*.npz"))
    assert len(chunks) == 1
    return np.load(chunks[0], allow_pickle=False)


def test_high_frequency_logger_flushes_versioned_state_and_command_chunk(tmp_path):
    high_frequency_logger = HighFrequencyLogger(
        HighFrequencyLogConfig(directory=tmp_path, chunk_interval_s=60.0),
        num_joints=3,
        num_motors=3,
    )
    high_frequency_logger.log_low_state(_low_state(), wall_time_ns=100, monotonic_ns=10)
    high_frequency_logger.log_low_state(None, wall_time_ns=200, monotonic_ns=20)
    write_error = RuntimeError("write failed")
    high_frequency_logger.log_low_command(
        q_target=[1, 2, 3],
        dq_target=[4, 5, 6],
        tau_ff=[7, 8, 9],
        kp=[10, 11, 12],
        kd=[13, 14, 15],
        success=False,
        duration_ns=42,
        error=write_error,
        wall_time_ns=300,
        monotonic_ns=30,
    )
    high_frequency_logger.close()

    with _load_only_chunk(high_frequency_logger) as chunk:
        assert int(chunk["schema_version"]) == 1
        assert int(chunk["num_joints"]) == 3
        assert int(chunk["num_motors"]) == 3
        np.testing.assert_array_equal(chunk["state_wall_time_ns"], [100, 200])
        np.testing.assert_array_equal(chunk["state_valid"], [True, False])
        np.testing.assert_array_equal(chunk["state_joint_pos"][0], [0, 1, 2])
        np.testing.assert_array_equal(chunk["state_joint_pos"][1], [0, 0, 0])
        np.testing.assert_array_equal(chunk["state_q_present"], [True, False])
        np.testing.assert_array_equal(chunk["command_q_target"], [[1, 2, 3]])
        np.testing.assert_array_equal(chunk["command_success"], [False])
        assert chunk["command_error_type"].tolist() == ["RuntimeError"]
        assert chunk["command_error_message"].tolist() == ["write failed"]
        np.testing.assert_array_equal(chunk["command_duration_ns"], [42])

    assert not list(high_frequency_logger.session_directory.glob("*.partial"))


def test_high_frequency_logger_drops_oldest_record_without_blocking(tmp_path):
    high_frequency_logger = HighFrequencyLogger(
        HighFrequencyLogConfig(directory=tmp_path, chunk_interval_s=60.0, queue_capacity=2),
        num_joints=3,
        num_motors=3,
    )
    for timestamp in (1, 2, 3):
        high_frequency_logger.log_low_state(
            _low_state(float(timestamp)),
            wall_time_ns=timestamp,
            monotonic_ns=timestamp,
        )
    high_frequency_logger.close()

    with _load_only_chunk(high_frequency_logger) as chunk:
        np.testing.assert_array_equal(chunk["state_wall_time_ns"], [2, 3])
        assert int(chunk["dropped_state_count"]) == 1
        assert int(chunk["dropped_command_count"]) == 0


def test_high_frequency_logger_disk_failure_is_contained(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        del args, kwargs
        raise OSError("disk unavailable")

    monkeypatch.setattr(np, "savez_compressed", fail_save)
    high_frequency_logger = HighFrequencyLogger(
        HighFrequencyLogConfig(directory=tmp_path, chunk_interval_s=60.0),
        num_joints=3,
        num_motors=3,
    )
    high_frequency_logger.log_low_state(_low_state())
    high_frequency_logger.close()
    high_frequency_logger.close()

    assert high_frequency_logger.last_error == "OSError: disk unavailable"
    assert not list(high_frequency_logger.session_directory.glob("chunk_*.npz"))


class _RecordingLogger:
    def __init__(self):
        self.states = []
        self.commands = []
        self.closed = 0

    def log_low_state(self, state):
        self.states.append(state)

    def log_low_command(self, **values):
        self.commands.append(values)

    def close(self):
        self.closed += 1


class _FakeUnitreeSdk:
    def __init__(self):
        self.state = SimpleNamespace(
            imu=SimpleNamespace(quat=[1, 0, 0, 0], omega=[1, 2, 3]),
            motor=SimpleNamespace(
                q=[10, 20, 30],
                dq=[40, 50, 60],
                ddq=[70, 80, 90],
                tau_est=[100, 110, 120],
            ),
        )
        self.commands = []
        self.write_error = None

    def read_low_state(self):
        return self.state

    def create_zero_command(self):
        return SimpleNamespace()

    def write_low_command(self, command):
        self.commands.append(command)
        if self.write_error is not None:
            raise self.write_error


def _unitree_interface():
    interface = object.__new__(UnitreeInterface)
    interface.robot_config = SimpleNamespace(
        num_joints=3,
        num_motors=3,
        joint2motor=(2, 0, 1),
        motor_kp=(1.0, 2.0, 3.0),
        motor_kd=(4.0, 5.0, 6.0),
    )
    interface._unitree_motor_order = None
    interface._kp_level = 2.0
    interface._kd_level = 0.5
    interface.unitree_interface = _FakeUnitreeSdk()
    interface._high_frequency_logger = _RecordingLogger()
    return interface


def test_unitree_interface_logs_returned_state_and_final_motor_command():
    interface = _unitree_interface()
    low_state = interface.get_low_state()
    assert interface._high_frequency_logger.states == [low_state]
    np.testing.assert_array_equal(low_state.q, [[30, 10, 20]])

    interface.send_low_command(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([4.0, 5.0, 6.0]),
        np.asarray([7.0, 8.0, 9.0]),
    )
    logged = interface._high_frequency_logger.commands[-1]
    np.testing.assert_array_equal(logged["q_target"], [2, 3, 1])
    np.testing.assert_array_equal(logged["dq_target"], [5, 6, 4])
    np.testing.assert_array_equal(logged["tau_ff"], [8, 9, 7])
    np.testing.assert_array_equal(logged["kp"], [2, 4, 6])
    np.testing.assert_array_equal(logged["kd"], [2, 2.5, 3])
    assert logged["success"] is True
    assert logged.get("error") is None

    interface.unitree_interface.state = None
    assert interface.get_low_state() is None
    assert interface._high_frequency_logger.states[-1] is None


def test_unitree_interface_logs_failed_write_and_preserves_sdk_exception():
    interface = _unitree_interface()
    write_error = RuntimeError("SDK write failed")
    interface.unitree_interface.write_error = write_error

    with pytest.raises(RuntimeError, match="SDK write failed"):
        interface.send_low_command(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([4.0, 5.0, 6.0]),
            np.asarray([7.0, 8.0, 9.0]),
        )

    logged = interface._high_frequency_logger.commands[-1]
    assert logged["success"] is False
    assert logged["error"] is write_error


def test_unitree_interface_isolated_from_logger_errors():
    interface = _unitree_interface()

    class BrokenLogger:
        def log_low_state(self, state):
            del state
            raise RuntimeError("logger failed")

        def log_low_command(self, **values):
            del values
            raise RuntimeError("logger failed")

        def close(self):
            raise RuntimeError("logger failed")

    interface._high_frequency_logger = BrokenLogger()
    assert interface.get_low_state() is not None
    interface.send_low_command(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([4.0, 5.0, 6.0]),
        np.asarray([7.0, 8.0, 9.0]),
    )
    interface.close()
    assert len(interface.unitree_interface.commands) == 1


def test_cli_accepts_sdk_log_directory(tmp_path):
    args = build_parser().parse_args(["--sdk-log-dir", str(tmp_path)])
    assert args.sdk_log_dir == tmp_path


def test_unitree_uses_base_interface_logging_wrappers():
    assert UnitreeInterface.get_low_state is BaseInterface.get_low_state
    assert UnitreeInterface.send_low_command is BaseInterface.send_low_command
