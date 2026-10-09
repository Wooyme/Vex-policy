"""BFM's two-rate inference, sensor contract and runtime integration without training weights."""

from __future__ import annotations

import copy
import json
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import BfmTaskConfig, DebugConfig, InferenceConfig, PolicySpec
from vex_policy.policies import bfm
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.bfm import BfmKneelingPolicy, BfmWalkPolicy
from vex_policy.policies.utils.bfm import DEFAULT_DOF_ANGLES, JOINT_NAMES, pack_actor_observation, project_latent
from vex_policy.policy_state_machine import PolicyState, PolicyStateMachine, _policy_class
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse, rpy_to_quat

ROOT = Path(__file__).resolve().parents[1]


def _onnx(path, inputs, output, metadata):
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["action"], value=numpy_helper.from_array(output))],
        "bfm-test",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, width]) for name, width in inputs.items()],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, list(output.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    helper.set_model_props(model, {key: json.dumps(value) for key, value in metadata.items()})
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _metadata(spec):
    functions = {
        "frame": "bfm:proprioceptive_frame",
        "command": "bfm:velocity_command",
        "latent": "bfm:executed_latent",
        "cos_phase": "locomotion:cos_phase",
        "sin_phase": "locomotion:sin_phase",
        "height_command": "base_height:base_height_command",
    }
    groups = {
        group: {
            "history_length": 1,
            "concatenate": True,
            "terms": {
                term: {"func": "holosoma.managers.observation.terms." + functions[term], "scale": 1.0} for term in terms
            },
        }
        for group, terms in spec.observation.obs_dict.items()
    }
    return {
        "dof_names": JOINT_NAMES,
        "kp": [20.0] * 29,
        "kd": [2.0] * 29,
        "action_scale": np.linspace(0.02, 0.04, 29).tolist(),
        "experiment_config": {
            "env_class": "holosoma.envs.bfm_controller.environment.BFMControllerEnvironment",
            "control_decimation": 4,
            "simulator": {"config": {"sim": {"fps": 200, "control_decimation": 4}}},
            "algo": {"config": {"module_dict": {"actor": {"input_dim": list(groups)}}}},
            "robot": {
                "init_state": {"default_joint_angles": dict(zip(JOINT_NAMES, DEFAULT_DOF_ANGLES.tolist(), strict=True))}
            },
            "observation": {"groups": groups},
        },
    }


def _state(q=None, rpy=(0.0, 0.0, 0.0)):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=rpy_to_quat(rpy)[None],
        joint_pos=np.asarray(DEFAULT_DOF_ANGLES if q is None else q)[None].copy(),
        joint_vel=np.arange(29, dtype=np.float32)[None] / 10,
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.array([[1.0, 2.0, 3.0]]),
    )


class RecordingSession:
    def __init__(self, session):
        self.session = session
        self.feeds = []
        self.override = None

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def run(self, outputs, feed):
        self.feeds.append({key: value.copy() for key, value in feed.items()})
        if isinstance(self.override, Exception):
            raise self.override
        return [self.override] if self.override is not None else self.session.run(outputs, feed)


@pytest.fixture
def make_policy(tmp_path, monkeypatch):
    instances = {}

    def session(path, providers, **kwargs):
        if path not in instances:
            options = ort.SessionOptions()
            options.intra_op_num_threads = kwargs["intra_op_num_threads"]
            instances[path] = RecordingSession(ort.InferenceSession(path, sess_options=options, providers=providers))
        return instances[path]

    monkeypatch.setattr(bfm, "shared_session", session)

    def make(kneeling=False, metadata_change=None, actor_change=None, observation_change=None):
        variant = "kneeling" if kneeling else "walk"
        doc = yaml.safe_load((ROOT / f"configs/g1/bfm_{variant}.yaml").read_text())
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        doc["task"]["model_path"] = str(folder / "controller.onnx")
        doc["task"]["actor_model_path"] = str(folder / "actor.onnx")
        startup = DEFAULT_DOF_ANGLES.copy()
        if kneeling:
            startup[[0, 6]] = -0.8
            startup[[3, 9]] = 1.8
            doc["task"]["motion_data_path"] = str(folder / "kneeling.npz")
            poses = np.zeros((2, 36))
            poses[:, 3] = 1.0
            poses[:, 7:] = np.stack([DEFAULT_DOF_ANGLES, startup])
            np.savez(folder / "kneeling.npz", joint_pos=poses, joint_names=np.asarray(JOINT_NAMES))
        spec = PolicySpec.model_validate(doc)
        controller_metadata = _metadata(spec)
        actor_metadata = {
            "bfm_actor_contract": 1,
            "dof_names": JOINT_NAMES,
            "default_dof_angles": DEFAULT_DOF_ANGLES.tolist(),
        }
        if metadata_change:
            metadata_change(controller_metadata)
        if actor_change:
            actor_change(actor_metadata)
        raw_latent = np.arange(1, 257, dtype=np.float32)[None]
        motor = np.linspace(-1, 1, 29, dtype=np.float32)[None]
        _onnx(doc["task"]["model_path"], {"actor_obs": 357 if kneeling else 356}, raw_latent, controller_metadata)
        _onnx(doc["task"]["actor_model_path"], {"actor_obs": 465, "latent": 256}, motor, actor_metadata)
        if observation_change:
            observation_change(doc["observation"])
        spec = PolicySpec.model_validate(doc)
        config = InferenceConfig(G1_29DOF, spec.inputs, spec.observation, spec.task, spec.guard)
        policy = (BfmKneelingPolicy if kneeling else BfmWalkPolicy)(config)
        return policy, config, doc, startup, raw_latent, motor

    return make


@pytest.mark.parametrize("kneeling", [False, True])
def test_real_onnx_cadence_observation_order_and_motor_feedback(make_policy, kneeling):
    policy, _, _, startup, raw, motor = make_policy(kneeling)
    state = _state(startup, rpy=(0.1, 0.0, 0.0))
    assert policy.activate(state) is None
    control = {"vx": 0.2, "vy": 0.3, "yaw": -0.4}
    if kneeling:
        control["height"] = 0.45
    policy.apply_control(control)
    frames = []
    gravity = quat_rotate_inverse(state.base_quat, np.array([[0.0, 0.0, -1.0]]))
    for tick in range(8):
        state = replace(state, joint_pos=startup[None] + tick * 0.001)
        previous_motor = np.zeros((1, 29)) if tick == 0 else motor
        frame = np.concatenate(
            (state.joint_pos - DEFAULT_DOF_ANGLES, state.joint_vel, gravity, 0.25 * state.base_ang_vel, previous_motor),
            axis=1,
        )
        frames.append(frame)
        command = policy.step(state)
        expected_q = np.clip(DEFAULT_DOF_ANGLES + motor[0] * policy.action_scales, G1_JOINT_LOWER, G1_JOINT_UPPER)
        np.testing.assert_allclose(command.q, expected_q)
        assert command.controlled_joints.all()
        assert command.kp.tolist() == [20.0] * 29
        np.testing.assert_array_equal(policy.actor.feeds[-1]["actor_obs"][:, :93], frame.astype(np.float32))
    assert len(policy.actor.feeds) == 8
    assert len(policy.controller.feeds) == 2
    np.testing.assert_allclose(np.linalg.norm(policy.latent), 16.0)
    np.testing.assert_allclose(policy.latent, raw / np.linalg.norm(raw) * 16, rtol=1e-6)
    first = policy.controller.feeds[0]["actor_obs"]
    np.testing.assert_array_equal(first[:, :93], frames[0].astype(np.float32))
    np.testing.assert_allclose(first[0, 93:98], [0.3, -0.2, 0.4, 1.0, -1.0])
    if kneeling:
        assert first[0, 98] == pytest.approx(0.45)
        np.testing.assert_allclose(first[0, 99:101], [0.0, 0.0], atol=1e-7)
        assert first.shape == (1, 357)
        assert np.max(np.abs(first[0, :29])) > 1  # Kneeling joints minus standing motor zeros.
    else:
        np.testing.assert_allclose(first[0, 98:100], [0.0, 0.0], atol=1e-7)
        assert first.shape == (1, 356)
    assert not first[:, -256:].any()
    np.testing.assert_array_equal(policy.controller.feeds[1]["actor_obs"][:, -256:], policy.latent)
    # After four updates the low-level history is oldest-first; packing must reverse each prior field.
    feed = policy.actor.feeds[4]["actor_obs"]
    np.testing.assert_array_equal(feed[0, 93:122], frames[3][0, 64:93].astype(np.float32))
    np.testing.assert_array_equal(feed[0, 209:212], frames[3][0, 61:64].astype(np.float32))
    np.testing.assert_array_equal(feed[0, 221:250], frames[3][0, :29].astype(np.float32))
    np.testing.assert_array_equal(feed[0, 337:366], frames[3][0, 29:58].astype(np.float32))
    np.testing.assert_array_equal(feed[0, 453:456], frames[3][0, 58:61].astype(np.float32))


def test_zero_padding_sphere_projection_and_reset(make_policy):
    policy, _, _, _, _, _ = make_policy()
    state = _state()
    assert policy.activate(state) is None
    policy.step(state)
    first = copy.deepcopy(policy.actor.feeds[-1])
    assert not first["actor_obs"][:, 93:].any()
    policy.deactivate()
    assert not policy.actor_history.any() and not policy.latent.any() and not policy.motor_actions.any()
    assert policy._episode_step == 0
    assert policy.activate(state) is None
    policy.step(state)
    for key in first:
        np.testing.assert_array_equal(first[key], policy.actor.feeds[-1][key])
    np.testing.assert_array_equal(project_latent(np.zeros((1, 256))), np.zeros((1, 256)))
    tiny = np.full((1, 256), 1e-15, dtype=np.float32)
    np.testing.assert_allclose(project_latent(tiny), tiny / 1e-12 * 16, rtol=1e-6)
    huge = np.full((1, 256), np.finfo(np.float32).max, dtype=np.float32)
    np.testing.assert_allclose(project_latent(huge), np.ones((1, 256)), rtol=1e-6)


def test_packing_reverses_frames_within_fields_not_terms():
    frames = np.arange(2 * 5 * 93, dtype=np.float32).reshape(2, 5, 93)
    packed = pack_actor_observation(frames)
    assert packed.shape == (2, 465)
    np.testing.assert_array_equal(packed[:, :93], frames[:, 4])
    for index, start in enumerate((93, 122, 151, 180)):
        np.testing.assert_array_equal(packed[:, start : start + 29], frames[:, 3 - index, 64:93])
    with pytest.raises(ValueError, match="history"):
        pack_actor_observation(np.zeros((1, 4, 93)))


def test_standing_phase_clock_and_control_latency(make_policy):
    policy, _, _, _, _, _ = make_policy()
    state = _state()
    policy.activate(state)
    for _ in range(4):
        policy.step(state)
    np.testing.assert_allclose(policy.controller.feeds[0]["actor_obs"][0, 96:100], [-1, -1, 0, 0], atol=1e-7)
    policy.apply_control({"vx": 0.0, "vy": 0.5, "yaw": 0.0})
    policy.step(state)
    feed = policy.controller.feeds[-1]["actor_obs"]
    phase = np.array([0.0, -np.pi]) + 2 * np.pi * 4 / 50
    np.testing.assert_allclose(feed[0, 96:98], np.cos(phase), atol=1e-7)
    np.testing.assert_allclose(feed[0, 98:100], np.sin(phase), atol=1e-7)
    calls = len(policy.controller.feeds)
    policy.apply_control({"vx": 0.0, "vy": 0.4, "yaw": 0.0})
    for _ in range(3):
        policy.step(state)
    assert len(policy.controller.feeds) == calls
    policy.step(state)
    assert policy.controller.feeds[-1]["actor_obs"][0, 93] == pytest.approx(0.4)


@pytest.mark.parametrize("failure", ["state", "quaternion", "controller_nan", "actor_nan", "actor_shape", "inference"])
def test_faults_are_reported_through_policy_runtime_fault(make_policy, failure):
    policy, _, _, _, _, _ = make_policy()
    state = _state()
    policy.activate(state)
    if failure == "state":
        state.joint_vel[0, 1] = np.nan
    elif failure == "quaternion":
        state.base_quat.fill(0)
    elif failure == "controller_nan":
        policy.controller.override = np.full((1, 256), np.nan)
    elif failure == "actor_nan":
        policy.actor.override = np.full((1, 29), np.nan)
    elif failure == "actor_shape":
        policy.actor.override = np.zeros((1, 30))
    else:
        policy.actor.override = RuntimeError("inference test failure")
    with pytest.raises(PolicyRuntimeFault):
        policy.step(state)


def test_clipping_feedback_and_debug_motor_zero(make_policy):
    policy, config, _, _, _, _ = make_policy()
    state = _state()
    policy.activate(state)
    policy.actor.override = np.full((1, 29), 9.0, dtype=np.float32)
    policy.action_scales.fill(1.0)
    result = policy.step(state)
    np.testing.assert_array_equal(result.q, G1_JOINT_UPPER)
    policy.step(state)
    np.testing.assert_array_equal(policy.actor.feeds[-1]["actor_obs"][0, 64:93], np.full(29, 5))
    debug = BfmWalkPolicy(replace(config, task=replace(config.task, debug=DebugConfig(force_zero_action=True))))
    debug.activate(state)
    np.testing.assert_allclose(debug.step(state).q, DEFAULT_DOF_ANGLES)
    assert not debug.motor_actions.any()


@pytest.mark.parametrize(
    "kind", ["cadence", "actor_rate", "order", "zeros", "terms", "scale", "legacy", "clip", "gains", "actor_contract"]
)
def test_incompatible_training_contracts_fail_at_load(make_policy, kind):
    def change(metadata):
        experiment = metadata["experiment_config"]
        if kind == "cadence":
            experiment["control_decimation"] = 1
        elif kind == "actor_rate":
            experiment["simulator"]["config"]["sim"]["fps"] = 100
        elif kind == "order":
            experiment["algo"]["config"]["module_dict"]["actor"]["input_dim"].reverse()
        elif kind == "zeros":
            experiment["robot"]["init_state"]["default_joint_angles"][JOINT_NAMES[0]] = -0.8
        elif kind == "gains":
            metadata["kp"][0] = -1
        elif kind == "terms":
            del experiment["observation"]["groups"]["command_obs"]["terms"]["sin_phase"]
        elif kind in {"scale", "legacy", "clip"}:
            term = experiment["observation"]["groups"]["actor_obs"]["terms"]["frame"]
            term[{"scale": "scale", "legacy": "func", "clip": "clip"}[kind]] = {
                "scale": 0.1,
                "legacy": "holosoma.managers.observation.terms.bfm_waypoint:proprioceptive_frame",
                "clip": [-1, 1],
            }[kind]

    with pytest.raises(ValueError, match="BFM"):
        make_policy(
            metadata_change=change,
            actor_change=(lambda metadata: metadata.pop("bfm_actor_contract")) if kind == "actor_contract" else None,
        )


def test_invalid_observation_config_and_control(make_policy):
    with pytest.raises(ValueError, match="BFM observation"):
        make_policy(observation_change=lambda cfg: cfg["obs_scales"].update(frame=0.25))
    policy, _, _, _, _, _ = make_policy(True)
    assert policy.activate(_state()) is not None  # Standing must not pass the kneeling startup guard.
    assert not policy.is_active
    state = _state(policy.initial_pose.dof_pos)
    assert policy.activate(state) is None
    for control in (
        {"vx": 0.0},
        {"vx": 0, "vy": 0, "yaw": 0, "height": np.nan},
        {"vx": 0, "vy": 0, "yaw": 0, "height": 1.0},
    ):
        with pytest.raises(PolicyRuntimeFault):
            policy.apply_control(control)


@pytest.mark.parametrize("kneeling", [False, True])
def test_yaml_resolution_registry_and_state_machine_fault(make_policy, tmp_path, kneeling):
    _, _, doc, startup, _, _ = make_policy(kneeling)
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(doc))
    runtime, path = load_runtime_config(policy_file)
    resolved = resolve_policies(runtime, path)
    assert isinstance(resolved[0].config.task, BfmTaskConfig)
    kind = "bfm_kneeling" if kneeling else "bfm_walk"
    cls = BfmKneelingPolicy if kneeling else BfmWalkPolicy
    assert _policy_class(kind) is cls
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["entry-points"]["vex_policy.policies"][kind].endswith(":" + cls.__name__)
    instance = cls(resolved[0].config)
    state = _state(startup)
    writes = []
    transport = SimpleNamespace(
        publish_status=lambda value: None,
        publish_state=lambda value: None,
        publish_reference_state=lambda value: None,
        close=lambda: None,
    )
    interface = SimpleNamespace(
        get_low_state=lambda: state, send_low_command=lambda *args, **kwargs: writes.append(args)
    )
    machine = PolicyStateMachine(
        runtime,
        resolved,
        instances={doc["name"]: instance},
        transport=transport,
        interface_manager=interface,
        clock=lambda: 0.0,
    )
    try:
        for seq, names in enumerate([[], [doc["name"]]], 1):
            inputs = {name: {p.name: p.default for p in resolved[0].spec.input_parameters} for name in names}
            assert machine.inbox.accept(
                json.dumps(
                    {
                        "seq": seq,
                        "timestamp": 0,
                        "control": {"policy": names, "inputs": inputs, "estop": False},
                    }
                )
            )
            machine.tick()
        machine.tick()
        assert machine.state == PolicyState.RUNNING and writes
        count = len(writes)
        instance.actor.override = RuntimeError("test actor unavailable")
        machine.tick()
        assert machine.state == PolicyState.LATCHED
        assert len(writes) == count and not instance.is_active
        assert not instance.latent.any() and not instance.actor_history.any()
    finally:
        machine._policy_executor.shutdown(wait=True)
        instance.close()


@pytest.mark.parametrize(
    "overrides",
    [{"rl_rate": 12.5}, {"controller_decimation": 1}, {"inference_threads": 0}, {"action_mask_path": "mask.yaml"}],
)
def test_task_rejects_unsupported_clock_and_masks(overrides):
    with pytest.raises(ValueError):
        BfmTaskConfig(model_path="controller.onnx", actor_model_path="actor.onnx", **overrides)
