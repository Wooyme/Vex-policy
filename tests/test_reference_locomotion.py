from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

from vex_policy.config import ResolvedPolicy, load_runtime_config, resolve_policies
from vex_policy.config.config_types import (
    ActionMaskConfig,
    DebugConfig,
    InferenceConfig,
    PolicySpec,
    ReferenceLocomotionTaskConfig,
    RobotRuntimeConfig,
    RuntimeConfig,
)
from vex_policy.policies import reference_locomotion
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.reference_locomotion import ReferenceLocomotionPolicy
from vex_policy.policies.utils.inference import OnnxActor
from vex_policy.policy_state_machine import PolicyState, PolicyStateMachine, _policy_class
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse, rpy_to_quat

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "configs/examples/holosoma"


def _state(q, rpy=(0.2, -0.3, 0.4)):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=rpy_to_quat(rpy)[None],
        joint_pos=np.asarray(q)[None].copy(),
        joint_vel=np.full((1, 29), 2.0),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.array([[1.0, 2.0, 3.0]]),
    )


def _training_group():
    scales = {
        "actions": 1.0,
        "base_ang_vel": 0.25,
        "command_ang_vel": 1.0,
        "command_lin_vel": 1.0,
        "cos_phase": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "projected_gravity": 1.0,
        "sin_phase": 1.0,
    }
    return {
        "concatenate": True,
        "history_length": 1,
        "terms": {
            name: {"func": f"holosoma.managers.observation.terms.locomotion:{name}", "scale": scale, "clip": None}
            for name, scale in scales.items()
        },
    }


class FakeActor:
    def __init__(self):
        self.session = self
        self.metadata = {
            "dof_names": G1_29DOF.dof_names,
            "kp": [10.0] * 29,
            "kd": [2.0] * 29,
            "action_scale": [0.25] * 29,
            "experiment_config": {"observation": {"groups": {"actor_obs": _training_group()}}},
        }
        self.inputs = [SimpleNamespace(name="actor_obs", shape=[1, 100])]
        self.outputs = [SimpleNamespace(name="action", shape=[1, 29])]
        self.action = np.linspace(-0.2, 0.2, 29, dtype=np.float32)[None]
        self.feeds = []

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def __call__(self, observations):
        self.feeds.append(observations["actor_obs"].copy())
        return self.action.copy()


@pytest.fixture
def setup_policy(tmp_path, monkeypatch):
    pose = np.asarray(G1_29DOF.default_dof_angles)
    motion = tmp_path / "reference.npz"
    joints = np.stack([pose - 0.01, pose, pose + 0.01])[:, ::-1]
    root = np.tile(np.concatenate(([99, 98, 97], rpy_to_quat((0.2, -0.3, 0.4)))), (3, 1))
    np.savez(motion, joint_names=G1_29DOF.dof_names[::-1], joint_pos=np.concatenate((root, joints), axis=1))
    spec = PolicySpec.model_validate(yaml.safe_load((EXAMPLES / "g1_29dof_kneeling.yaml").read_text()))
    config = InferenceConfig(
        robot=replace(G1_29DOF, motor_kp=None, motor_kd=None),
        inputs=spec.inputs,
        observation=spec.observation,
        task=replace(spec.task, model_path="fake.onnx", motion_data_path=str(motion), reference_pose_frame=1),
        guard=spec.guard,
    )
    actor = FakeActor()
    monkeypatch.setattr(reference_locomotion, "OnnxActor", lambda path: actor)
    return config, actor, pose


def test_golden_observation_command_and_restart(setup_policy):
    config, actor, pose = setup_policy
    policy = ReferenceLocomotionPolicy(config)
    state = _state(pose + 0.01)
    assert policy.activate(state) is None
    policy.apply_control({"vx": 0.1, "vy": 0.2, "yaw": -0.3})
    command = policy.step(state)
    gravity = quat_rotate_inverse(state.base_quat, np.array([[0.0, 0.0, -1.0]]))[0]
    expected = np.concatenate(
        (
            np.zeros(29),
            [0.25, 0.5, 0.75],
            [0.3],
            [0.2, -0.1],
            [1, -1],
            np.full(29, 0.01),
            np.full(29, 0.1),
            gravity,
            [0, 0],
        )
    )
    np.testing.assert_allclose(actor.feeds[0][0], expected, atol=1e-7)
    assert actor.feeds[0].dtype == np.float32
    np.testing.assert_allclose(command.q, pose + actor.action[0] * 0.25)
    np.testing.assert_array_equal(command.kp, 10)
    np.testing.assert_array_equal(command.kd, 2)
    np.testing.assert_array_equal(command.dq, 0)
    np.testing.assert_array_equal(command.tau, 0)
    assert command.controlled_joints.all()
    policy.step(state)
    np.testing.assert_array_equal(actor.feeds[-1][0, :29], actor.action[0])
    np.testing.assert_allclose(actor.feeds[-1][0, 98:], [np.sin(2 * np.pi / 50), -np.sin(2 * np.pi / 50)])
    policy.deactivate()
    policy.activate(state)
    policy.apply_control({"vx": 0.1, "vy": 0.2, "yaw": -0.3})
    policy.step(state)
    np.testing.assert_array_equal(actor.feeds[-1], actor.feeds[0])
    policy.close()


def test_standing_preserves_episode_clock_and_wraps_phase(setup_policy):
    config, actor, pose = setup_policy
    policy = ReferenceLocomotionPolicy(replace(config, task=replace(config.task, gait_period=0.5)))
    state = _state(pose)
    policy.activate(state)
    policy.step(state)
    np.testing.assert_allclose(actor.feeds[-1][0, 35:37], [-1, -1])
    np.testing.assert_allclose(actor.feeds[-1][0, 98:], [0, 0], atol=1e-7)
    for step in range(1, 35):
        moving = step < 3 or step >= 8
        # Exactly 0.01 counts as moving, including a pure yaw command.
        policy.apply_control({"vx": 0, "vy": 0, "yaw": 0.01 if moving else 0.009})
        policy.step(state)
        expected = (
            np.array([2 * np.pi * step / 25, 2 * np.pi * step / 25 - np.pi]) if moving else np.array([np.pi, np.pi])
        )
        np.testing.assert_allclose(actor.feeds[-1][0, 35:37], np.cos(expected), atol=1e-7)
        np.testing.assert_allclose(actor.feeds[-1][0, 98:], np.sin(expected), atol=1e-7)


def test_activation_restores_input_defaults_and_repeated_activation_is_noop(setup_policy):
    config, actor, pose = setup_policy
    joystick, slider = config.inputs
    joystick = joystick.model_copy(update={"y": joystick.y.model_copy(update={"default": 0.12})})
    policy = ReferenceLocomotionPolicy(replace(config, inputs=(joystick, slider)))
    state = _state(pose)
    policy.activate(state)
    policy.step(state)
    np.testing.assert_allclose(actor.feeds[-1][0, 33:35], [0.12, 0])
    policy.apply_control({"vx": 0, "vy": -0.2, "yaw": 0})
    policy.activate(state)
    policy.step(state)
    np.testing.assert_allclose(actor.feeds[-1][0, 33:35], [-0.2, 0])
    assert abs(actor.feeds[-1][0, 98]) > 0.1
    policy.deactivate()
    policy.deactivate()
    policy.activate(state)
    policy.step(state)
    np.testing.assert_array_equal(actor.feeds[-1], actor.feeds[0])


@pytest.mark.parametrize("rpy", [(0, np.pi / 2, 0), (0, -np.pi / 2, 1.0), (np.pi, 0, 2.0)])
def test_angular_velocity_stays_in_body_frame_without_world_state(setup_policy, rpy):
    config, actor, pose = setup_policy
    policy = ReferenceLocomotionPolicy(replace(config, guard=None))
    state = _state(pose, rpy)
    state.base_pos[:] = np.nan
    state.base_lin_vel[:] = np.nan
    state.base_quat[:] *= 2
    original_quat = state.base_quat.copy()
    policy.activate(state)
    policy.step(state)
    np.testing.assert_array_equal(actor.feeds[-1][0, 29:32], [0.25, 0.5, 0.75])
    gravity = quat_rotate_inverse(rpy_to_quat(rpy)[None], np.array([[0.0, 0.0, -1.0]]))
    np.testing.assert_allclose(actor.feeds[-1][0, 95:98], gravity[0], atol=1e-7)
    np.testing.assert_array_equal(state.base_quat, original_quat)


def test_raw_action_feedback_mask_limits_and_gain_override(setup_policy):
    config, actor, pose = setup_policy
    config = replace(
        config,
        robot=replace(config.robot, motor_kp=(7.0,) * 29, motor_kd=(0.7,) * 29),
        action_mask=ActionMaskConfig(masked_joints=(G1_29DOF.dof_names[0],)),
    )
    actor.action[0, :3] = [120, -150, 200]
    policy = ReferenceLocomotionPolicy(config)
    policy.activate(_state(pose))
    command = policy.step(_state(pose))
    scaled = np.clip(actor.action[0], -100, 100) * 0.25
    scaled[0] = 0
    np.testing.assert_allclose(command.q, np.clip(pose + scaled, G1_JOINT_LOWER, G1_JOINT_UPPER))
    assert not command.controlled_joints[0]
    np.testing.assert_array_equal(command.kp, 7)
    np.testing.assert_array_equal(command.kd, 0.7)
    policy.step(_state(pose))
    np.testing.assert_array_equal(actor.feeds[-1][0, :29], actor.action[0])


def test_debug_overrides(setup_policy):
    config, actor, pose = setup_policy
    debug = DebugConfig(force_zero_action=True, force_zero_angular_velocity=True, force_upright_imu=True)
    policy = ReferenceLocomotionPolicy(replace(config, task=replace(config.task, debug=debug)))
    policy.activate(_state(pose))
    np.testing.assert_allclose(policy.step(_state(pose)).q, pose)
    policy.step(_state(pose))
    np.testing.assert_array_equal(actor.feeds[-1][0, :32], 0)
    np.testing.assert_array_equal(actor.feeds[-1][0, 95:98], [0, 0, -1])


def test_guard_rejection_and_yaw_invariance(setup_policy):
    config, actor, pose = setup_policy
    policy = ReferenceLocomotionPolicy(config)
    assert "reference_locomotion_start_check_failed" in policy.activate(_state(pose + 0.3))
    assert not policy.is_active and not actor.feeds
    assert "projected_gravity" in policy.activate(_state(pose, rpy=(0.9, -0.3, 0.4)))
    assert policy.activate(_state(pose)) is None
    policy.step(_state(pose))
    policy.deactivate()
    yaw_state = _state(pose, rpy=(0.2, -0.3, 2.2))
    assert policy.activate(yaw_state) is None
    policy.step(yaw_state)
    np.testing.assert_allclose(actor.feeds[-1], actor.feeds[0], atol=1e-7)


@pytest.mark.parametrize("index,offset", [(0, -0.01), (1, 0), (-1, 0.01), (-2, 0)])
def test_reference_frame_selection_and_joint_reordering(setup_policy, index, offset):
    config, _, pose = setup_policy
    policy = ReferenceLocomotionPolicy(replace(config, task=replace(config.task, reference_pose_frame=index)))
    np.testing.assert_allclose(policy.default_dof_angles, pose + offset, atol=1e-8)


@pytest.mark.parametrize("bad", ["names", "quaternion", "nonfinite", "limits", "frame"])
def test_invalid_reference_pose(setup_policy, bad):
    config, _, _ = setup_policy
    path = config.task.motion_data_path
    with np.load(path) as data:
        names, positions = data["joint_names"].copy(), data["joint_pos"].copy()
    if bad == "names":
        names[0] = names[1]
    elif bad == "quaternion":
        positions[1, 3:7] = 0
    elif bad == "nonfinite":
        positions[1, 8] = np.nan
    elif bad == "limits":
        positions[1, 8] = 100
    else:
        config = replace(config, task=replace(config.task, reference_pose_frame=3))
    np.savez(path, joint_names=names, joint_pos=positions)
    with pytest.raises(ValueError):
        ReferenceLocomotionPolicy(config)


@pytest.mark.parametrize("bad", ["input", "output", "order", "scale", "gains", "gain_shape"])
def test_model_contract_validation(setup_policy, bad):
    config, actor, _ = setup_policy
    if bad == "input":
        actor.inputs[0].shape = [1, 99]
    elif bad == "output":
        actor.outputs[0].name = "actions"
    elif bad == "order":
        actor.metadata["dof_names"] = G1_29DOF.dof_names[::-1]
    elif bad == "scale":
        actor.metadata["action_scale"][0] = 0.5
    elif bad == "gains":
        actor.metadata["kp"][0] = np.nan
    else:
        actor.metadata["kd"] = [[2.0]] * 29
    with pytest.raises(ValueError):
        ReferenceLocomotionPolicy(config)


@pytest.mark.parametrize("bad", ["legacy", "scale", "clip", "history", "terms", "concatenate", "malformed"])
def test_training_semantics_validation(setup_policy, bad):
    config, actor, _ = setup_policy
    group = actor.metadata["experiment_config"]["observation"]["groups"]["actor_obs"]
    expected = "Reference locomotion"
    if bad == "legacy":
        group["terms"]["base_ang_vel"]["func"] = "holosoma.managers.observation.terms.crawling:base_ang_vel"
        expected = "Legacy crawling heading-frame actors are not supported"
    elif bad == "scale":
        group["terms"]["dof_vel"]["scale"] = 1
    elif bad == "clip":
        group["terms"]["actions"]["clip"] = [-1, 1]
    elif bad == "history":
        group["history_length"] = 2
    elif bad == "terms":
        del group["terms"]["sin_phase"]
    elif bad == "concatenate":
        group["concatenate"] = False
    else:
        actor.metadata["experiment_config"] = None
        expected = "Invalid reference locomotion"
    with pytest.raises(ValueError, match=expected):
        ReferenceLocomotionPolicy(config)


def test_critic_semantics_and_absent_optional_metadata_do_not_affect_deployment(setup_policy):
    config, actor, _ = setup_policy
    groups = actor.metadata["experiment_config"]["observation"]["groups"]
    groups["critic_obs"] = {
        "terms": {"base_ang_vel": {"func": "holosoma.managers.observation.terms.crawling:base_ang_vel"}}
    }
    policy = ReferenceLocomotionPolicy(config)
    policy.close()
    del actor.metadata["experiment_config"]
    assert "robot_urdf" not in actor.metadata
    ReferenceLocomotionPolicy(config).close()


@pytest.mark.parametrize("bad", ["term", "duplicate", "scale", "dimension", "history"])
def test_observation_contract_validation(setup_policy, bad):
    config, _, _ = setup_policy
    obs = config.observation
    if bad in {"term", "duplicate"}:
        extra = "actions" if bad == "duplicate" else "base_lin_vel"
        obs = replace(obs, obs_dict={"actor_obs": [*obs.obs_dict["actor_obs"], extra]})
    elif bad == "scale":
        obs = replace(obs, obs_scales={**obs.obs_scales, "dof_vel": 1})
    elif bad == "dimension":
        obs = replace(obs, obs_dims={**obs.obs_dims, "actions": 28})
    else:
        obs = replace(obs, history_length_dict={"actor_obs": 2})
    with pytest.raises(ValueError, match="Reference locomotion"):
        ReferenceLocomotionPolicy(replace(config, observation=obs))


@pytest.mark.parametrize(
    "bad", ["joint_pos", "joint_vel", "base_ang_vel", "base_quat", "zero_quat", "shape", "action", "action_shape"]
)
def test_invalid_runtime_data(setup_policy, bad):
    config, actor, pose = setup_policy
    policy = ReferenceLocomotionPolicy(config)
    state = _state(pose)
    policy.activate(state)
    if bad == "action":
        actor.action[0, 0] = np.nan
    elif bad == "action_shape":
        actor.action = np.zeros((1, 28))
    elif bad == "zero_quat":
        state.base_quat[:] = 0
    elif bad == "shape":
        state = replace(state, joint_pos=np.zeros((1, 28)), joint_vel=np.zeros((1, 28)))
    else:
        getattr(state, bad)[0, 0] = np.nan
    with pytest.raises(PolicyRuntimeFault):
        policy.step(state)


@pytest.mark.parametrize("control", [{}, {"vx": 0, "vy": np.nan, "yaw": 0}, {"vx": 0, "vy": 1, "yaw": 0}])
def test_invalid_control(setup_policy, control):
    config, _, pose = setup_policy
    policy = ReferenceLocomotionPolicy(config)
    policy.activate(_state(pose))
    with pytest.raises(PolicyRuntimeFault):
        policy.apply_control(control)


def test_missing_input_declarations(setup_policy):
    config, _, _ = setup_policy
    with pytest.raises(ValueError, match="exactly vx, vy and yaw"):
        ReferenceLocomotionPolicy(replace(config, inputs=()))


@pytest.mark.parametrize("kind", ["kneeling", "crawling"])
def test_example_loading_registration_and_real_onnx_inference(setup_policy, tmp_path, monkeypatch, kind):
    config, actor, pose = setup_policy
    model_path = tmp_path / "model.onnx"
    weights = np.zeros((100, 29), dtype=np.float32)
    weights[37:66] = np.eye(29, dtype=np.float32) * 0.5
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["actor_obs", "weights"], ["action"])],
        "reference_locomotion_fixture",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 100])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])],
        initializer=[numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    helper.set_model_props(model, {key: json.dumps(value) for key, value in actor.metadata.items()})
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    monkeypatch.setattr(reference_locomotion, "OnnxActor", OnnxActor)
    data = yaml.safe_load((EXAMPLES / f"g1_29dof_{kind}.yaml").read_text())
    data["task"].update(
        model_path=str(model_path), motion_data_path=config.task.motion_data_path, reference_pose_frame=1
    )
    path = tmp_path / f"{kind}.yaml"
    path.write_text(yaml.safe_dump(data))
    runtime, config_path = load_runtime_config(path, ROOT / "configs/mqtt.yaml")
    resolved = resolve_policies(runtime, config_path)[0]
    assert isinstance(resolved.config.task, ReferenceLocomotionTaskConfig)
    assert resolved.kind == "reference_locomotion"
    assert _policy_class(resolved.kind) is ReferenceLocomotionPolicy
    policy = _policy_class(resolved.kind)(resolved.config)
    policy.activate(_state(pose + 0.01))
    np.testing.assert_allclose(policy.step(_state(pose + 0.01)).q, pose + 0.00125, atol=1e-7)
    policy.close()
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert manifest["project"]["entry-points"]["vex_policy.policies"][resolved.kind].endswith(
        ":ReferenceLocomotionPolicy"
    )
    Path(config.task.motion_data_path).unlink()
    with pytest.raises(ValueError, match="does not exist"):
        resolve_policies(runtime, config_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_pose_frame", True),
        ("reference_pose_frame", 1.5),
        ("rl_rate", 0),
        ("gait_period", 0),
        ("gait_period", float("inf")),
        ("policy_action_scale", float("nan")),
        ("motion_data_path", " "),
        ("model_path", " "),
        ("use_phase", True),
    ],
)
def test_strict_task_config(field, value):
    data = {"model_path": "actor.onnx", "motion_data_path": "pose.npz", field: value}
    with pytest.raises(ValueError):
        ReferenceLocomotionTaskConfig(**data)


def test_runtime_latches_without_sending_invalid_actor_output(setup_policy):
    config, actor, pose = setup_policy
    spec = PolicySpec(
        name="reference",
        implementation="reference_locomotion",
        inputs=config.inputs,
        observation=config.observation,
        task=config.task,
        guard=config.guard,
    )
    runtime = RuntimeConfig(robot=RobotRuntimeConfig(config=config.robot), policies=(spec,))
    policy = ReferenceLocomotionPolicy(config)
    writes = []
    transport = SimpleNamespace(
        publish_status=lambda value: None,
        publish_state=lambda value: None,
        publish_reference_state=lambda value: None,
        close=lambda: None,
    )
    machine = PolicyStateMachine(
        runtime,
        (ResolvedPolicy(spec, config, "reference_locomotion"),),
        instances={"reference": policy},
        clock=lambda: 0.0,
        transport=transport,
        interface_manager=SimpleNamespace(
            get_low_state=lambda: _state(pose), send_low_command=lambda *args, **kwargs: writes.append(args)
        ),
    )
    try:
        for seq, names in enumerate([[], ["reference"]], 1):
            assert machine.inbox.accept(
                json.dumps(
                    {
                        "seq": seq,
                        "timestamp": 0,
                        "control": {
                            "policy": names,
                            "inputs": {name: {"vx": 0, "vy": 0.1, "yaw": 0} for name in names},
                            "estop": False,
                        },
                    }
                )
            )
            machine.tick()
        machine.tick()
        assert machine.state == PolicyState.RUNNING and writes
        count = len(writes)
        actor.action[:] = np.nan
        machine.tick()
        assert machine.state == PolicyState.LATCHED
        assert len(writes) == count and not policy.is_active
        np.testing.assert_array_equal(policy.last_action, 0)
        np.testing.assert_array_equal(policy.lin_vel_command, 0)
    finally:
        machine._policy_executor.shutdown(wait=True)
        policy.close()
