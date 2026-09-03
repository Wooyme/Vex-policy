"""Dash application for offline SDK log inspection."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from log_viewer.data import Catalog, SessionData, SessionInfo, load_session
from log_viewer.joints import G1_29_DOF_NAMES
from log_viewer.metrics import (
    compute_metrics,
    decimate_series,
    quaternion_wxyz_to_euler_degrees,
    relative_seconds,
    timeline_origin_ns,
)

MAX_PLOT_POINTS = 4_000
TRACE_COLORS = tuple(qualitative.Plotly)


def create_app(catalog: Catalog) -> Dash:
    """Create a viewer over the immutable catalog snapshot."""
    app = Dash(
        __name__,
        title="Vex 日志分析",
        assets_folder=str(Path(__file__).with_name("assets")),
    )
    sessions_by_key = {session.key: session for session in catalog.sessions}

    @lru_cache(maxsize=4)
    def get_data(key: str) -> SessionData:
        return load_session(sessions_by_key[key])

    options = [
        {
            "label": f"{_format_wall_time(session.started_wall_time_ns)} · {session.session_id}",
            "value": session.key,
        }
        for session in catalog.sessions
    ]
    initial_key = catalog.sessions[0].key if catalog.sessions else None
    app.layout = html.Main(
        className="viewer-shell",
        children=[
            html.Header(
                className="viewer-header",
                children=[
                    html.Div(
                        [
                            html.H1("Vex SDK 日志分析"),
                            html.P("离线快照 · 仅加载启动时已完成的 NPZ 分块"),
                        ]
                    ),
                    html.Div(
                        className="session-picker",
                        children=[
                            html.Label("会话", htmlFor="session-selector"),
                            dcc.Dropdown(
                                id="session-selector",
                                options=options,
                                value=initial_key,
                                clearable=False,
                                placeholder="没有可用会话",
                            ),
                        ],
                    ),
                ],
            ),
            _warning_list(catalog.warnings, "catalog-warnings"),
            html.Div(id="session-meta", className="session-meta"),
            html.Div(id="summary-cards", className="metric-grid"),
            dcc.Tabs(
                className="viewer-tabs",
                children=[
                    dcc.Tab(
                        label="概览与时序",
                        children=[
                            html.Section(
                                className="panel",
                                children=[
                                    html.H2("控制周期与写入耗时"),
                                    dcc.Graph(id="timing-figure", config=_graph_config()),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="关节曲线",
                        children=[
                            html.Section(
                                className="panel",
                                children=[
                                    html.Div(
                                        className="chart-controls",
                                        children=[
                                            html.Div(
                                                [
                                                    html.Label("信号", htmlFor="joint-signal"),
                                                    dcc.Dropdown(
                                                        id="joint-signal",
                                                        clearable=False,
                                                        value="position",
                                                        options=[
                                                            {"label": "位置", "value": "position"},
                                                            {"label": "速度", "value": "velocity"},
                                                            {
                                                                "label": "估算力矩 tau_est / 前馈力矩 tau_ff",
                                                                "value": "torque",
                                                            },
                                                            {
                                                                "label": "电机状态码 motorstate",
                                                                "value": "motorstate",
                                                            },
                                                            {"label": "KP / KD", "value": "gains"},
                                                        ],
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("关节", htmlFor="joint-selector"),
                                                    dcc.Dropdown(id="joint-selector", multi=True),
                                                ]
                                            ),
                                        ],
                                    ),
                                    html.Div(id="joint-order-note", className="inline-note"),
                                    dcc.Graph(id="joint-figure", config=_graph_config()),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="基座状态",
                        children=[
                            html.Section(
                                className="panel",
                                children=[
                                    html.H2("基座运动与姿态"),
                                    dcc.Graph(id="base-figure", config=_graph_config()),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="错误与质量",
                        children=[
                            html.Section(
                                className="panel",
                                children=[
                                    html.H2("数据质量"),
                                    html.Div(id="quality-details"),
                                    html.H2("异常事件"),
                                    html.Div(id="event-table"),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("summary-cards", "children"),
        Output("timing-figure", "figure"),
        Output("joint-order-note", "children"),
        Output("base-figure", "figure"),
        Output("quality-details", "children"),
        Output("event-table", "children"),
        Output("session-meta", "children"),
        Input("session-selector", "value"),
    )
    def update_session(key: str | None):
        if key is None or key not in sessions_by_key:
            empty = _empty_figure("没有可用日志会话")
            return [], empty, "", empty, _empty_message("没有数据"), "", ""

        data = get_data(key)
        note = (
            "当前 G1 的 joint/motor 映射为恒等映射, 可直接对照实际值与目标值。"
            if _can_overlay_state_and_command(data.info)
            else "日志未携带 joint/motor 映射; 为避免误导, 仅显示状态实际值, KP/KD 仍按电机索引显示。"
        )
        meta = (
            f"{data.info.path} · schema v{data.info.schema_version} · {len(data.info.chunk_paths)} 个分块 · "
            f"{data.info.num_joints} joints / {data.info.num_motors} motors"
        )
        return (
            _summary_cards(data),
            make_timing_figure(data),
            note,
            make_base_figure(data),
            _quality_details(data),
            _event_table(data),
            meta,
        )

    @app.callback(
        Output("joint-selector", "options"),
        Output("joint-selector", "value"),
        Input("session-selector", "value"),
        Input("joint-signal", "value"),
    )
    def update_joint_options(key: str | None, signal: str):
        if key is None or key not in sessions_by_key:
            return [], []
        info = sessions_by_key[key]
        names = _axis_names(info, motors=signal == "gains")
        options = [{"label": name, "value": index} for index, name in enumerate(names)]
        return options, [0] if names else []

    @app.callback(
        Output("joint-figure", "figure"),
        Input("session-selector", "value"),
        Input("joint-signal", "value"),
        Input("joint-selector", "value"),
    )
    def update_joint_figure(key: str | None, signal: str, joints: list[int] | int | None):
        if key is None or key not in sessions_by_key:
            return _empty_figure("没有可用日志会话")
        if joints is None:
            selected: list[int] = []
        elif isinstance(joints, int):
            selected = [joints]
        else:
            selected = [int(index) for index in joints]
        return make_joint_figure(get_data(key), signal, selected)

    return app


def make_timing_figure(data: SessionData) -> go.Figure:
    arrays = data.arrays
    origin = timeline_origin_ns(data)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("调用间隔", "命令写入耗时"),
    )
    for name, label in (
        ("state_monotonic_ns", "State 间隔"),
        ("command_monotonic_ns", "Command 间隔"),
    ):
        timestamps = arrays[name]
        if len(timestamps) < 2:
            continue
        raw_intervals = np.diff(timestamps.astype(np.int64))
        valid = raw_intervals > 0
        x = relative_seconds(timestamps[1:][valid], origin)
        y = raw_intervals[valid].astype(np.float64) / 1e6
        x, y = decimate_series(x, y, max_points=MAX_PLOT_POINTS)
        figure.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=label), row=1, col=1)

    command_time = relative_seconds(arrays["command_monotonic_ns"], origin)
    duration = arrays["command_duration_ns"].astype(np.float64) / 1e3
    command_time, duration = decimate_series(command_time, duration, max_points=MAX_PLOT_POINTS)
    figure.add_trace(
        go.Scattergl(x=command_time, y=duration, mode="lines", name="write_low_command"),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="间隔 (ms)", row=1, col=1)
    figure.update_yaxes(title_text="耗时 (µs)", row=2, col=1)
    figure.update_xaxes(title_text="相对时间 (s)", row=2, col=1)
    return _style_figure(figure, height=650)


def make_joint_figure(data: SessionData, signal: str, joints: list[int]) -> go.Figure:
    names = _axis_names(data.info, motors=signal == "gains")
    selected = [index for index in joints if 0 <= index < len(names)]
    if not selected:
        return _empty_figure("请选择至少一个关节")

    arrays = data.arrays
    origin = timeline_origin_ns(data)
    state_time = relative_seconds(arrays["state_monotonic_ns"], origin)
    command_time = relative_seconds(arrays["command_monotonic_ns"], origin)
    figure = go.Figure()
    if signal == "gains":
        for series_index, index in enumerate(selected):
            color = TRACE_COLORS[series_index % len(TRACE_COLORS)]
            for field, suffix, dash in (
                ("command_kp", "KP", "solid"),
                ("command_kd", "KD", "dash"),
            ):
                x, y = decimate_series(
                    command_time,
                    arrays[field][:, index],
                    max_points=MAX_PLOT_POINTS,
                )
                figure.add_trace(
                    go.Scattergl(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{names[index]} · {suffix}",
                        line={"color": color, "dash": dash},
                    )
                )
        unit = "gain"
        title = "命令增益"
    else:
        definitions = {
            "position": (
                "state_joint_pos",
                "command_q_target",
                "位置",
                "rad",
                None,
                "实际位置",
                "q_target",
            ),
            "velocity": (
                "state_joint_vel",
                "command_dq_target",
                "速度",
                "rad/s",
                None,
                "实际速度",
                "dq_target",
            ),
            "torque": (
                "state_tau_est",
                "command_tau_ff",
                "估算力矩 tau_est / 前馈力矩 tau_ff",
                "N·m",
                "state_tau_est_present",
                "tau_est",
                "tau_ff",
            ),
            "motorstate": (
                "state_motorstate",
                None,
                "电机状态码 motorstate",
                "状态码",
                "state_motorstate_present",
                "motorstate",
                None,
            ),
        }
        state_field, command_field, title, unit, present_field, actual_label, target_label = definitions.get(
            signal, definitions["position"]
        )
        state_valid = arrays["state_valid"].copy()
        if present_field is not None:
            state_valid &= arrays[present_field]
        if signal == "motorstate" and not np.any(state_valid):
            return _empty_figure("当前日志未记录 motorstate")
        for series_index, index in enumerate(selected):
            color = TRACE_COLORS[series_index % len(TRACE_COLORS)]
            actual = arrays[state_field][:, index].astype(np.float64).copy()
            actual[~state_valid] = np.nan
            x, y = decimate_series(state_time, actual, max_points=MAX_PLOT_POINTS)
            figure.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"{names[index]} · {actual_label}",
                    line={"color": color, "shape": "hv" if signal == "motorstate" else "linear"},
                )
            )
            if command_field is not None and target_label is not None and _can_overlay_state_and_command(data.info):
                x, y = decimate_series(
                    command_time,
                    arrays[command_field][:, index],
                    max_points=MAX_PLOT_POINTS,
                )
                figure.add_trace(
                    go.Scattergl(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{names[index]} · {target_label}",
                        line={"color": color, "dash": "dash"},
                    )
                )

    figure.update_layout(title=title)
    figure.update_xaxes(title_text="相对时间 (s)")
    figure.update_yaxes(title_text=unit)
    if signal == "motorstate":
        figure.update_yaxes(tickformat="d")
    return _style_figure(figure, height=620)


def make_base_figure(data: SessionData) -> go.Figure:
    arrays = data.arrays
    origin = timeline_origin_ns(data)
    x = relative_seconds(arrays["state_monotonic_ns"], origin)
    valid = arrays["state_valid"]
    euler = quaternion_wxyz_to_euler_degrees(arrays["state_base_quat"])
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("位置", "线速度", "角速度", "姿态角"),
    )
    groups = (
        ("state_base_pos", ("x", "y", "z"), "m", 1),
        ("state_base_lin_vel", ("vx", "vy", "vz"), "m/s", 2),
        ("state_base_ang_vel", ("wx", "wy", "wz"), "rad/s", 3),
    )
    for field, labels, unit, row in groups:
        for index, label in enumerate(labels):
            values = arrays[field][:, index].astype(np.float64).copy()
            values[~valid] = np.nan
            x_plot, y_plot = decimate_series(x, values, max_points=MAX_PLOT_POINTS)
            figure.add_trace(
                go.Scattergl(x=x_plot, y=y_plot, mode="lines", name=label), row=row, col=1
            )
        figure.update_yaxes(title_text=unit, row=row, col=1)
    for index, label in enumerate(("roll", "pitch", "yaw")):
        values = euler[:, index].copy()
        values[~valid] = np.nan
        x_plot, y_plot = decimate_series(x, values, max_points=MAX_PLOT_POINTS)
        figure.add_trace(
            go.Scattergl(x=x_plot, y=y_plot, mode="lines", name=label), row=4, col=1
        )
    figure.update_yaxes(title_text="deg", row=4, col=1)
    figure.update_xaxes(title_text="相对时间 (s)", row=4, col=1)
    return _style_figure(figure, height=900)


def _summary_cards(data: SessionData) -> list[Any]:
    metrics = compute_metrics(data)
    command_latency = _format_number(metrics.command_duration_p95_us, "µs")
    return [
        _metric_card("会话时长", _format_duration(metrics.duration_s), _format_wall_time(data.info.started_wall_time_ns)),
        _metric_card(
            "State",
            f"{metrics.state_count:,}",
            f"中位频率 {_format_number(metrics.state_frequency_hz, 'Hz')}",
        ),
        _metric_card(
            "Command",
            f"{metrics.command_count:,}",
            f"中位频率 {_format_number(metrics.command_frequency_hz, 'Hz')}",
        ),
        _metric_card(
            "写入耗时 P95",
            command_latency,
            f"P99 {_format_number(metrics.command_duration_p99_us, 'µs')}",
        ),
        _metric_card(
            "异常",
            str(metrics.invalid_state_count + metrics.failed_command_count),
            f"无效状态 {metrics.invalid_state_count} · 失败命令 {metrics.failed_command_count}",
            alert=bool(metrics.invalid_state_count or metrics.failed_command_count),
        ),
        _metric_card(
            "队列丢弃",
            str(data.dropped_state_count + data.dropped_command_count),
            f"State {data.dropped_state_count} · Command {data.dropped_command_count}",
            alert=bool(data.dropped_state_count or data.dropped_command_count),
        ),
    ]


def _quality_details(data: SessionData) -> Any:
    metrics = compute_metrics(data)
    rows = [
        ("State 间隔 P95", _format_number(metrics.state_interval_p95_ms, "ms")),
        ("Command 间隔 P95", _format_number(metrics.command_interval_p95_ms, "ms")),
        ("Command 写入 P50", _format_number(metrics.command_duration_p50_us, "µs")),
        ("Command 写入最大值", _format_number(metrics.command_duration_max_us, "µs")),
        ("累计丢弃 State", str(data.dropped_state_count)),
        ("累计丢弃 Command", str(data.dropped_command_count)),
    ]
    return html.Div(
        [
            html.Dl(
                [
                    html.Div([html.Dt(label), html.Dd(value)], className="quality-item")
                    for label, value in rows
                ],
                className="quality-grid",
            ),
            _warning_list(data.warnings, "session-warnings"),
        ]
    )


def _event_table(data: SessionData) -> Any:
    arrays = data.arrays
    events: list[tuple[int, str, str, str]] = []
    for index in np.flatnonzero(~arrays["state_valid"]):
        wall_time = int(arrays["state_wall_time_ns"][index])
        events.append((wall_time, "无效 State", "后端返回 None", "—"))
    for index in np.flatnonzero(~arrays["command_success"]):
        wall_time = int(arrays["command_wall_time_ns"][index])
        error = str(arrays["command_error_type"][index]) or "未知错误"
        message = str(arrays["command_error_message"][index]) or "—"
        duration = f"{int(arrays['command_duration_ns'][index]) / 1e3:.2f} µs"
        events.append((wall_time, f"命令失败 · {error}", message, duration))
    events.sort(key=lambda event: event[0])
    if not events:
        return _empty_message("没有记录到无效状态或失败命令")

    visible = events[:200]
    table = html.Table(
        [
            html.Thead(html.Tr([html.Th("时间"), html.Th("类型"), html.Th("信息"), html.Th("耗时")])),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(_format_wall_time(timestamp)),
                            html.Td(kind),
                            html.Td(message),
                            html.Td(duration),
                        ]
                    )
                    for timestamp, kind, message, duration in visible
                ]
            ),
        ],
        className="event-table",
    )
    if len(events) <= len(visible):
        return table
    return html.Div([table, html.P(f"仅显示前 {len(visible)} 条, 共 {len(events)} 条。")])


def _axis_names(info: SessionInfo, *, motors: bool = False) -> tuple[str, ...]:
    width = info.num_motors if motors else info.num_joints
    if width == len(G1_29_DOF_NAMES):
        return G1_29_DOF_NAMES
    prefix = "电机索引" if motors else "关节索引"
    return tuple(f"{prefix} {index}" for index in range(width))


def _can_overlay_state_and_command(info: SessionInfo) -> bool:
    return (
        info.num_joints == len(G1_29_DOF_NAMES)
        and info.num_motors == len(G1_29_DOF_NAMES)
    )


def _metric_card(label: str, value: str, detail: str, *, alert: bool = False) -> Any:
    class_name = "metric-card metric-card-alert" if alert else "metric-card"
    return html.Article(
        className=class_name,
        children=[html.Span(label), html.Strong(value), html.Small(detail)],
    )


def _warning_list(warnings: tuple[str, ...] | list[str], element_id: str) -> Any:
    if not warnings:
        return html.Div(id=element_id)
    return html.Aside(
        id=element_id,
        className="warning-box",
        children=[html.Strong("读取警告"), html.Ul([html.Li(warning) for warning in warnings])],
    )


def _empty_message(message: str) -> Any:
    return html.P(message, className="empty-message")


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _style_figure(figure, height=420)


def _style_figure(figure: go.Figure, *, height: int) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 64, "r": 24, "t": 56, "b": 52},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        uirevision="offline-log-viewer",
    )
    return figure


def _graph_config() -> dict[str, Any]:
    return {"displaylogo": False, "responsive": True, "scrollZoom": True}


def _format_wall_time(wall_time_ns: int) -> str:
    return datetime.fromtimestamp(wall_time_ns / 1e9).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _format_number(value: float | None, unit: str) -> str:
    return "—" if value is None else f"{value:,.2f} {unit}"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining:.1f}s"


__all__ = ["create_app", "make_base_figure", "make_joint_figure", "make_timing_figure"]
