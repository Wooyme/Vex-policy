from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vex_policy.config.config_types import (
    ActionMaskConfig,
    InferenceConfig,
    InputParameter,
    ObservationConfig,
    SliderInput,
    SonicTaskConfig,
    TaskConfig,
    WaistLocomotionGuardConfig,
    WaistLocomotionTaskConfig,
    WbtTaskConfig,
)
from vex_policy.policies import sonic, waist_locomotion, wbt
from vex_policy.policies.locomotion import LocomotionPolicy
from vex_policy.policies.sonic_planner import HW_TO_POLICY, MotionSequence
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState


def _state(q=None):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=np.asarray(G1_29DOF.default_dof_angles if q is None else q).reshape(1, 29),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.zeros((1, 3)),
        joint_vel=np.zeros((1, 29)),
    )


def _inputs(values):
    return tuple(
        SliderInput(type="slider", parameter=InputParameter(name=name, min=-2, max=2, default=value))
        for name, value in values.items()
    )


class FakeSession:
    def __init__(self, width, *, output=None, wbt_model=False):
        self.width = width
        self.output = np.arange(29, dtype=np.float32).reshape(1, 29) / 10 if output is None else output
        self.feeds = []
        self.wbt_model = wbt_model

    def get_inputs(self):
        return [SimpleNamespace(name="actor_obs", shape=[1, self.width])]

    def get_outputs(self):
        return [SimpleNamespace(name="action", shape=list(self.output.shape))]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={"action_scale": "0.5"})

    def run(self, names, feed):
        self.feeds.append({key: value.copy() for key, value in feed.items()})
        if self.wbt_model:
            values = {
                "actions": self.output.copy(),
                "joint_pos": np.asarray(G1_29DOF.default_dof_angles).reshape(1, 29) + 0.01,
                "joint_vel": np.zeros((1, 29)),
                "ref_quat_xyzw": np.array([[0, 0, 0, 1.0]]),
            }
            return [values[name] for name in names]
        return [self.output.copy()]


def _patch_onnx(monkeypatch, session, metadata=None):
    metadata = metadata or {"kp": [5.0] * 29, "kd": [0.5] * 29}
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(
        "onnx.load",
        lambda *args, **kwargs: SimpleNamespace(
            metadata_props=[SimpleNamespace(key=key, value=json.dumps(value)) for key, value in metadata.items()]
        ),
    )


def test_locomotion_golden_observation_command_and_fresh_episode(monkeypatch):
    dims = {
        "actions": 29,
        "dof_pos": 29,
        "dof_vel": 29,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "command_stand": 1,
        "sin_phase": 2,
        "cos_phase": 2,
        "projected_gravity": 3,
    }
    observations = ObservationConfig(
        {"actor_obs": list(reversed(dims))}, dims, dict.fromkeys(dims, 1.0), {"actor_obs": 2}
    )
    session = FakeSession(196)
    _patch_onnx(monkeypatch, session)
    config = InferenceConfig(
        robot=replace(G1_29DOF, motor_kp=None, motor_kd=None),
        inputs=_inputs({"vx": 0, "vy": 0, "yaw": 0}),
        observation=observations,
        task=TaskConfig(model_path="fake.onnx"),
        action_mask=ActionMaskConfig(masked_joints=(G1_29DOF.dof_names[0],)),
    )
    policy = LocomotionPolicy(config)
    robot_state = _state(np.asarray(G1_29DOF.default_dof_angles) + 0.1)
    control = {"vx": 0.2, "vy": 0.4, "yaw": 0.1}
    policy.activate(robot_state)
    policy.apply_control(control)
    command = policy.step(robot_state)
    phase = np.array([2 * np.pi / 50, -np.pi + 2 * np.pi / 50])
    terms = {
        "actions": np.zeros(29),
        "dof_pos": np.full(29, 0.1),
        "dof_vel": np.zeros(29),
        "command_lin_vel": [0.4, -0.2],
        "command_ang_vel": [-0.1],
        "command_stand": [1],
        "sin_phase": np.sin(phase),
        "cos_phase": np.cos(phase),
        "projected_gravity": [0, 0, -1],
    }
    expected = np.concatenate([np.concatenate([np.zeros(dims[name]), terms[name]]) for name in sorted(dims)])
    np.testing.assert_allclose(session.feeds[0]["actor_obs"][0], expected, atol=1e-7)
    action = session.output.copy()
    action[0, 0] = 0
    np.testing.assert_allclose(command.q, config.robot.default_dof_angles + action[0] * 0.25)
    np.testing.assert_array_equal(command.kp, 5)
    assert not command.controlled_joints[0]
    policy.step(robot_state)
    policy.deactivate()
    policy.activate(robot_state)
    np.testing.assert_array_equal(policy.actions.last, 0)
    np.testing.assert_array_equal(policy.lin_vel_command, 0)
    assert not policy.is_standing
    assert policy.stand_command[0, 0] == 0
    policy.apply_control(control)
    policy.step(robot_state)
    np.testing.assert_array_equal(session.feeds[0]["actor_obs"], session.feeds[-1]["actor_obs"])
    policy.close()


def _waist_urdf():
    links = ["base", *[f"link_{index}" for index in range(28)], "right_ankle_roll_link"]
    pieces = ['<robot name="test">', *[f'<link name="{name}"/>' for name in links]]
    for index, name in enumerate(G1_29DOF.dof_names):
        pieces.append(
            f'<joint name="{name}" type="revolute"><parent link="{links[index]}"/>'
            f'<child link="{links[index + 1]}"/><origin xyz="0 0 -0.01"/><axis xyz="0 1 0"/>'
            '<limit lower="-4" upper="4" effort="100" velocity="10"/></joint>'
        )
    return "".join(pieces) + "</robot>"


def test_waist_constructor_reference_capture_limits_and_restart(tmp_path, monkeypatch):
    motion = tmp_path / "waist.npz"
    pose = np.asarray(G1_29DOF.default_dof_angles)
    np.savez(
        motion,
        joint_names=np.asarray(G1_29DOF.dof_names[::-1]),
        joint_pos=np.concatenate(([0, 0, 0, 1, 0, 0, 0], pose[::-1])).reshape(1, -1),
    )
    session = FakeSession(105)
    _patch_onnx(
        monkeypatch,
        session,
        {
            "kp": [5.0] * 29,
            "kd": [0.5] * 29,
            "dof_names": G1_29DOF.dof_names,
            "action_scale": [0.25] * 29,
            "robot_urdf": _waist_urdf(),
        },
    )
    defaults = {"amplitude": 0.125, "frequency": 1.1, "height_delta": 0, "x": 1, "y": 0, "z": 0}
    inputs = tuple(
        SliderInput(
            type="slider",
            parameter=InputParameter(
                name=name, min=0.01 if name in {"amplitude", "frequency"} else -2, max=2, default=value
            ),
        )
        for name, value in defaults.items()
    )
    dims = waist_locomotion.WaistLocomotionPolicy._OBS_DIMS
    config = InferenceConfig(
        robot=G1_29DOF,
        inputs=inputs,
        observation=ObservationConfig(
            {"actor_obs": list(dims)}, dims, waist_locomotion.WaistLocomotionPolicy._OBS_SCALES, {"actor_obs": 1}
        ),
        task=WaistLocomotionTaskConfig(model_path="fake.onnx", motion_data_path=str(motion)),
        guard=WaistLocomotionGuardConfig(),
    )
    policy = waist_locomotion.WaistLocomotionPolicy(config)
    robot_state = _state()
    assert policy.activate(robot_state) is None
    np.testing.assert_allclose(policy.default_dof_angles, pose)
    command = policy.step(robot_state)
    np.testing.assert_allclose(command.q, np.clip(pose + session.output[0] * 0.25, G1_JOINT_LOWER, G1_JOINT_UPPER))
    assert session.feeds[-1]["actor_obs"].shape == (1, 105)
    first = session.feeds[-1]["actor_obs"].copy()
    policy.apply_control({**defaults, "amplitude": 0.2, "frequency": 1.8, "height_delta": 0.1})
    policy.step(robot_state)
    policy.deactivate()
    assert policy.pelvis_orientation_reference_quat is None
    assert policy.activate(robot_state) is None
    assert policy.pelvis_sine_phase == 0
    policy.step(robot_state)
    np.testing.assert_allclose(session.feeds[-1]["actor_obs"], first)
    policy.close()


@pytest.mark.parametrize("startup_mode", ["immediate", "interpolate"])
def test_wbt_real_constructor_preserves_startup_inference_and_restart(monkeypatch, startup_mode):
    session = FakeSession(58, wbt_model=True)
    _patch_onnx(monkeypatch, session, {"kp": [5.0] * 29, "kd": [0.5] * 29, "robot_urdf": "unused"})
    monkeypatch.setattr(
        wbt,
        "PinocchioRobot",
        lambda *args: SimpleNamespace(
            real2pinocchio_index=np.arange(29),
            fk_and_get_ref_body_orientation_in_world=lambda q: np.array([[0.0, 0, 0, 1]]),
        ),
    )
    config = InferenceConfig(
        robot=G1_29DOF,
        inputs=(),
        observation=ObservationConfig(
            {"actor_obs": ["dof_pos", "actions"]},
            {"dof_pos": 29, "actions": 29},
            {"dof_pos": 1.0, "actions": 1.0},
            {"actor_obs": 1},
        ),
        task=WbtTaskConfig(
            model_path="fake.onnx",
            startup_mode=startup_mode,
            init_duration_s=0.02,
            motion_start_timestep=3,
            motion_end_timestep=5,
        ),
    )
    policy = wbt.WholeBodyTrackingPolicy(config)
    robot_state = _state()
    policy.activate(robot_state)
    if startup_mode == "interpolate":
        command = policy.step(robot_state)
        assert len(session.feeds) == 1  # Only model target lookup at construction.
        assert policy._stage is wbt.WbtStage.TRACKING
        np.testing.assert_allclose(command.q, np.asarray(config.robot.default_dof_angles) + 0.01)
    command = policy.step(robot_state)
    first = session.feeds[-1]
    np.testing.assert_array_equal(first["time_step"], [[3]])
    np.testing.assert_array_equal(first["obs"], np.zeros((1, 58)))
    np.testing.assert_allclose(command.q, config.robot.default_dof_angles + session.output[0] * 0.5)
    policy.step(robot_state)
    policy.deactivate()
    policy.activate(robot_state)
    if startup_mode == "interpolate":
        policy.step(robot_state)
    policy.step(robot_state)
    for name in first:
        np.testing.assert_array_equal(session.feeds[-1][name], first[name])
    policy.close()


def _sonic_config(source):
    dims = dict.fromkeys((*sonic._ACTOR_TERMS, *sonic._ENCODER_TERMS), 1)
    return InferenceConfig(
        robot=G1_29DOF,
        inputs=_inputs({"vx": 0, "vy": 0, "yaw": 0, "height": 0}) if source == "planner" else (),
        observation=ObservationConfig(
            {"actor_obs": list(sonic._ACTOR_TERMS), "encoder_obs": list(sonic._ENCODER_TERMS)},
            dims,
            dict.fromkeys(dims, 1.0),
            {"actor_obs": 1, "encoder_obs": 1},
        ),
        task=SonicTaskConfig(
            model_path="decoder",
            encoder_model_path="encoder",
            planner_model_path="planner",
            motion_source=source,
            motion_data_path="unused",
            inference_provider="cpu",
        ),
    )


def _motion():
    return MotionSequence(np.zeros((3, 3)), np.tile([1.0, 0, 0, 0], (3, 1)), np.zeros((3, 29)), np.zeros((3, 29)))


def _patch_sonic(monkeypatch):
    decoder = FakeSession(994)
    encoder = FakeSession(1762, output=np.ones((1, 64), dtype=np.float32))
    monkeypatch.setattr(sonic, "shared_session", lambda path, providers: encoder if path == "encoder" else decoder)
    monkeypatch.setattr(sonic, "load_motion_directory", lambda *args, **kwargs: ("test", _motion()))
    return decoder, encoder


def test_sonic_directory_order_history_and_restart(monkeypatch):
    decoder, encoder = _patch_sonic(monkeypatch)
    config = _sonic_config("directory")
    policy = sonic.SonicPolicy(config)
    robot_state = _state()
    policy.activate(robot_state)
    command = policy.step(robot_state)
    expected = decoder.output[0, HW_TO_POLICY] * np.asarray(config.robot.default_per_joint_action_scale)
    np.testing.assert_allclose(command.q, config.robot.default_dof_angles + expected, atol=1e-7)
    assert encoder.feeds[0]["actor_obs"].shape == (1, 1762)
    first = decoder.feeds[0]["actor_obs"]
    assert first.shape == (1, 994)
    np.testing.assert_array_equal(first[0, :64], 1)
    np.testing.assert_array_equal(first[0, 64:-3], 0)
    np.testing.assert_array_equal(first[0, -3:], [0, 0, -1])
    policy.step(robot_state)
    policy.deactivate()
    policy.activate(robot_state)
    assert policy._motion_frame == 0
    assert not policy._state_history
    policy.step(robot_state)
    np.testing.assert_array_equal(decoder.feeds[-1]["actor_obs"], first)
    policy.close()


def test_sonic_deactivate_joins_inflight_planner_and_discards_result(monkeypatch):
    _patch_sonic(monkeypatch)
    entered, finish, stopped = (threading.Event() for _ in range(3))

    def infer(*args):
        entered.set()
        assert finish.wait(2)
        return _motion()

    monkeypatch.setattr(
        sonic, "SonicPlanner", lambda *args, **kwargs: SimpleNamespace(initial_context=lambda q: None, infer=infer)
    )
    policy = sonic.SonicPolicy(_sonic_config("planner"))
    policy.activate(_state())
    policy.apply_control({"vx": 0.3, "vy": 0.4, "yaw": 0.5, "height": 0.6})
    policy.step(_state())
    assert entered.wait(2)
    worker = policy._planner_thread

    def stop():
        policy.deactivate()
        stopped.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(stop)
        try:
            assert not stopped.wait(0.05)
        finally:
            finish.set()
        future.result()
    assert not worker.is_alive()
    assert policy._pending_motion is None
    assert policy._motion is None
    policy.activate(_state())
    assert policy._desired_heading == 0
    assert policy._command_snapshot().speed == 0
    assert policy._command_snapshot().height == -1
    policy.close()
    assert policy._planner_thread is None


def test_wbt_constructor_closes_partially_started_clock(monkeypatch):
    session = FakeSession(29, wbt_model=True)
    _patch_onnx(monkeypatch, session, {"kp": [5.0] * 29, "kd": [0.5] * 29, "robot_urdf": "unused"})
    monkeypatch.setattr(wbt, "PinocchioRobot", lambda *args: object())
    events = []

    class BrokenClock:
        def start(self):
            events.append("start")
            raise ValueError("clock start failed")

        def close(self):
            events.append("close")

    monkeypatch.setattr(wbt, "ClockSub", BrokenClock)
    config = InferenceConfig(
        robot=G1_29DOF,
        inputs=(),
        observation=ObservationConfig({"actor_obs": ["dof_pos"]}, {"dof_pos": 29}, {"dof_pos": 1.0}, {"actor_obs": 1}),
        task=WbtTaskConfig(model_path="fake.onnx", use_sim_time=True),
    )
    with pytest.raises(ValueError, match="clock start failed"):
        wbt.WholeBodyTrackingPolicy(config)
    assert events == ["start", "close"]
