from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from log_viewer.cli import build_parser
from log_viewer.data import discover_sessions, load_session
from log_viewer.joints import G1_29_DOF_NAMES
from log_viewer.metrics import (
    compute_metrics,
    decimate_series,
    quaternion_wxyz_to_euler_degrees,
)


def _write_chunk(
    directory: Path,
    index: int,
    *,
    state_times: list[int],
    command_times: list[int],
    width: int = 3,
    session_id: str = "test-session",
    dropped_state_count: int = 0,
    dropped_command_count: int = 0,
    malformed: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    state_count = len(state_times)
    command_count = len(command_times)
    state_vectors = np.arange(state_count * width, dtype=np.float64).reshape(state_count, width)
    command_vectors = np.arange(command_count * width, dtype=np.float64).reshape(command_count, width)
    wall_times = [*state_times, *command_times] or [index]
    payload = {
        "schema_version": np.asarray(1, dtype=np.uint16),
        "session_id": np.asarray(session_id),
        "chunk_index": np.asarray(index, dtype=np.uint64),
        "chunk_started_wall_time_ns": np.asarray(min(wall_times), dtype=np.uint64),
        "chunk_ended_wall_time_ns": np.asarray(max(wall_times), dtype=np.uint64),
        "num_joints": np.asarray(width, dtype=np.uint32),
        "num_motors": np.asarray(width, dtype=np.uint32),
        "dropped_state_count": np.asarray(dropped_state_count, dtype=np.uint64),
        "dropped_command_count": np.asarray(dropped_command_count, dtype=np.uint64),
        "state_wall_time_ns": np.asarray(state_times, dtype=np.uint64),
        "state_monotonic_ns": np.asarray(state_times, dtype=np.uint64),
        "state_valid": np.ones(state_count, dtype=np.bool_),
        "state_base_pos": np.zeros((state_count, 3)),
        "state_base_quat": np.tile([1.0, 0.0, 0.0, 0.0], (state_count, 1)),
        "state_joint_pos": state_vectors,
        "state_base_lin_vel": np.zeros((state_count, 3)),
        "state_base_ang_vel": np.zeros((state_count, 3)),
        "state_joint_vel": state_vectors + 10,
        "state_q": state_vectors,
        "state_dq": state_vectors + 10,
        "state_ddq": state_vectors + 20,
        "state_tau_est": state_vectors + 30,
        "state_q_present": np.ones(state_count, dtype=np.bool_),
        "state_dq_present": np.ones(state_count, dtype=np.bool_),
        "state_ddq_present": np.ones(state_count, dtype=np.bool_),
        "state_tau_est_present": np.ones(state_count, dtype=np.bool_),
        "command_wall_time_ns": np.asarray(command_times, dtype=np.uint64),
        "command_monotonic_ns": np.asarray(command_times, dtype=np.uint64),
        "command_duration_ns": np.arange(command_count, dtype=np.uint64) + 1_000,
        "command_success": np.ones(command_count, dtype=np.bool_),
        "command_error_type": np.full(command_count, "", dtype=np.str_),
        "command_error_message": np.full(command_count, "", dtype=np.str_),
        "command_q_target": command_vectors,
        "command_dq_target": command_vectors + 10,
        "command_tau_ff": command_vectors + 20,
        "command_kp": command_vectors + 30,
        "command_kd": command_vectors + 40,
    }
    if malformed:
        payload["state_joint_pos"] = np.zeros((state_count, width + 1))
    path = directory / f"chunk_{index:06d}.npz"
    np.savez(path, **payload)
    return path


def test_discovers_and_loads_sorted_complete_chunks(tmp_path):
    session = tmp_path / "session-a"
    _write_chunk(
        session,
        1,
        state_times=[30, 40],
        command_times=[35],
        dropped_state_count=3,
        dropped_command_count=2,
    )
    _write_chunk(session, 0, state_times=[10, 20], command_times=[])
    (session / "chunk_000002.npz.partial").write_bytes(b"incomplete")

    catalog = discover_sessions(tmp_path)
    assert len(catalog.sessions) == 1
    assert catalog.sessions[0].chunk_indices == (0, 1)

    data = load_session(catalog.sessions[0])
    np.testing.assert_array_equal(data.arrays["state_monotonic_ns"], [10, 20, 30, 40])
    np.testing.assert_array_equal(data.arrays["command_monotonic_ns"], [35])
    assert data.arrays["command_q_target"].shape == (1, 3)
    assert data.dropped_state_count == 3
    assert data.dropped_command_count == 2
    assert data.warnings == ()


def test_discovery_reports_corrupt_schema_and_chunk_gaps(tmp_path):
    session = tmp_path / "session-a"
    _write_chunk(session, 0, state_times=[10], command_times=[])
    _write_chunk(session, 2, state_times=[20], command_times=[])
    (session / "chunk_000001.npz").write_bytes(b"not an npz")
    unsupported = tmp_path / "unsupported"
    path = _write_chunk(unsupported, 0, state_times=[10], command_times=[])
    with np.load(path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["schema_version"] = np.asarray(99, dtype=np.uint16)
    np.savez(path, **payload)

    catalog = discover_sessions(tmp_path)
    assert len(catalog.sessions) == 1
    assert any("无法读取元数据" in warning for warning in catalog.sessions[0].warnings)
    assert any("存在缺口" in warning for warning in catalog.sessions[0].warnings)
    assert any("没有可读取" in warning for warning in catalog.warnings)


def test_load_skips_malformed_chunk_and_keeps_valid_data(tmp_path):
    session = tmp_path / "session-a"
    _write_chunk(session, 0, state_times=[10, 20], command_times=[])
    _write_chunk(session, 1, state_times=[30], command_times=[], malformed=True)

    data = load_session(discover_sessions(tmp_path).sessions[0])
    np.testing.assert_array_equal(data.arrays["state_monotonic_ns"], [10, 20])
    assert any("数据校验失败" in warning for warning in data.warnings)


def test_metrics_use_median_period_and_cumulative_drop_counts(tmp_path):
    session = tmp_path / "session-a"
    _write_chunk(
        session,
        0,
        state_times=[1_000_000_000, 1_020_000_000, 1_040_000_000],
        command_times=[1_010_000_000, 1_030_000_000],
    )
    data = load_session(discover_sessions(tmp_path).sessions[0])
    data.arrays["state_valid"][1] = False
    data.arrays["command_success"][0] = False

    metrics = compute_metrics(data)
    assert metrics.state_frequency_hz == pytest.approx(50.0)
    assert metrics.command_frequency_hz == pytest.approx(50.0)
    assert metrics.state_interval_p95_ms == pytest.approx(20.0)
    assert metrics.invalid_state_count == 1
    assert metrics.failed_command_count == 1
    assert metrics.command_duration_p50_us == pytest.approx(1.0005)


def test_quaternion_conversion_and_zero_quaternion():
    angle = np.deg2rad(90) / 2
    quaternions = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.cos(angle), 0.0, 0.0, np.sin(angle)],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    euler = quaternion_wxyz_to_euler_degrees(quaternions)
    np.testing.assert_allclose(euler[0], [0, 0, 0], atol=1e-10)
    np.testing.assert_allclose(euler[1], [0, 0, 90], atol=1e-10)
    assert np.isnan(euler[2]).all()


def test_decimation_retains_endpoints_and_global_extrema():
    x = np.arange(10_000)
    y = np.sin(x / 20)
    y[4_321] = 100
    y[7_654] = -100

    reduced_x, reduced_y = decimate_series(x, y, max_points=200)
    assert len(reduced_x) <= 200
    assert reduced_x[0] == x[0]
    assert reduced_x[-1] == x[-1]
    assert 100 in reduced_y
    assert -100 in reduced_y


def test_cli_defaults_are_independent_from_policy_cli():
    args = build_parser().parse_args([])
    assert args.log_dir == Path("logs")
    assert args.host == "127.0.0.1"
    assert args.port == 8050


def test_g1_display_metadata_is_kept_inside_viewer_package():
    assert len(G1_29_DOF_NAMES) == 29
    assert G1_29_DOF_NAMES[0] == "left_hip_pitch_joint"
    assert G1_29_DOF_NAMES[-1] == "right_wrist_yaw_joint"


def test_dash_app_serves_catalog_snapshot(tmp_path):
    pytest.importorskip("dash")
    from log_viewer.app import create_app, make_joint_figure

    _write_chunk(tmp_path / "session-a", 0, state_times=[10, 20], command_times=[15])
    catalog = discover_sessions(tmp_path)
    data = load_session(catalog.sessions[0])
    assert len(make_joint_figure(data, "position", [0]).data) == 1
    assert len(make_joint_figure(data, "gains", [0]).data) == 2
    app = create_app(catalog)
    response = app.server.test_client().get("/")
    assert response.status_code == 200
    assert b"dash-renderer" in response.data
    assert len(app.callback_map) == 3


def test_g1_joint_plot_overlays_actual_and_motor_target(tmp_path):
    pytest.importorskip("plotly")
    from log_viewer.app import make_joint_figure

    _write_chunk(
        tmp_path / "session-a",
        0,
        state_times=[10, 20],
        command_times=[15],
        width=29,
    )
    data = load_session(discover_sessions(tmp_path).sessions[0])
    figure = make_joint_figure(data, "position", [0])
    assert [trace.name for trace in figure.data] == [
        "left_hip_pitch_joint · 实际位置",
        "left_hip_pitch_joint · q_target",
    ]
    assert figure.data[0].line.color == figure.data[1].line.color
    assert figure.data[1].line.dash == "dash"

    torque_figure = make_joint_figure(data, "torque", [0])
    assert torque_figure.layout.title.text == "估算力矩 tau_est / 前馈力矩 tau_ff"
    assert [trace.name for trace in torque_figure.data] == [
        "left_hip_pitch_joint · tau_est",
        "left_hip_pitch_joint · tau_ff",
    ]
