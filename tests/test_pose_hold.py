from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from vex_policy.config import ResolvedPolicy, load_runtime_config, resolve_policies
from vex_policy.config.config_types import (
    ActionMaskConfig,
    DebugConfig,
    InferenceConfig,
    PolicySpec,
    PoseHoldTaskConfig,
    RobotRuntimeConfig,
    RuntimeConfig,
)
from vex_policy.policies import pose_hold
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.pose_hold import PoseHoldPolicy
from vex_policy.policies.utils.locomotion_utils import load_motion_last_pose, load_motion_pose
from vex_policy.policy_state_machine import PolicyState, PolicyStateMachine, _policy_class
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse, rpy_to_quat

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/examples/holosoma/g1_29dof_pose_hold.yaml"


def _state(q, rpy=(0.2, -0.3, 0.4)):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=rpy_to_quat(rpy)[None],
        joint_pos=np.asarray(q)[None].copy(),
        joint_vel=np.full((1, 29), 2.0),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.array([[1.0, 2.0, 3.0]]),
    )


def _urdf():
    # The right ankle's fixed translation makes the expected height independent
    # of the implementation's FK result, while exercising real Pinocchio.
    pieces = ['<robot name="fixture"><link name="pelvis"/>']
    for index, name in enumerate(G1_29DOF.dof_names):
        link = "right_ankle_roll_link" if name == "right_ankle_roll_joint" else f"link_{index}"
        pieces.append(
            f'<link name="{link}"/><joint name="{name}" type="revolute">'
            f'<parent link="pelvis"/><child link="{link}"/><origin xyz="0 0 -0.1"/>'
            '<axis xyz="0 1 0"/><limit lower="-4" upper="4" effort="100" velocity="10"/></joint>'
        )
    return "".join(pieces) + "</robot>"


class FakeActor:
    def __init__(self):
        self.session = self
        self.metadata = {
            "dof_names": G1_29DOF.dof_names,
            "kp": [10.0] * 29,
            "kd": [2.0] * 29,
            "action_scale": [0.25] * 29,
            "robot_urdf": _urdf(),
        }
        self.inputs = [SimpleNamespace(name="actor_obs", shape=[1, 94])]
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
    spec = PolicySpec.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    config = InferenceConfig(
        robot=replace(G1_29DOF, motor_kp=None, motor_kd=None),
        inputs=spec.inputs,
        observation=spec.observation,
        task=replace(spec.task, model_path="fake.onnx", motion_data_path=str(motion)),
        guard=spec.guard,
    )
    actor = FakeActor()
    monkeypatch.setattr(pose_hold, "OnnxActor", lambda path: actor)
    return config, actor, pose


def test_golden_observation_command_and_restart(setup_policy):
    config, actor, pose = setup_policy
    policy = PoseHoldPolicy(config)
    state = _state(pose + 0.01)
    assert policy.activate(state) is None
    policy.apply_control({})
    command = policy.step(state)
    gravity = quat_rotate_inverse(state.base_quat, np.array([[0.0, 0.0, -1.0]]))[0]
    expected = np.concatenate(
        (np.zeros(29), [0.25, 0.5, 0.75], [-0.1 * gravity[2]], np.full(29, 0.01), np.full(29, 0.1), gravity)
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
    policy.deactivate()
    assert policy.activate(state) is None
    policy.step(state)
    np.testing.assert_array_equal(actor.feeds[-1], actor.feeds[0])
    policy.close()


def test_raw_action_feedback_mask_limits_and_gain_override(setup_policy):
    config, actor, pose = setup_policy
    config = replace(
        config,
        robot=replace(config.robot, motor_kp=(7.0,) * 29, motor_kd=(0.7,) * 29),
        action_mask=ActionMaskConfig(masked_joints=(G1_29DOF.dof_names[0],)),
    )
    actor.action[0, :3] = [120.0, -150.0, 200.0]
    policy = PoseHoldPolicy(config)
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


def test_zero_action_debug_keeps_reference_and_zero_feedback(setup_policy):
    config, actor, pose = setup_policy
    policy = PoseHoldPolicy(replace(config, task=replace(config.task, debug=DebugConfig(force_zero_action=True))))
    policy.activate(_state(pose))
    np.testing.assert_allclose(policy.step(_state(pose)).q, pose)
    policy.step(_state(pose))
    np.testing.assert_array_equal(actor.feeds[-1][0, :29], 0)


def test_guard_rejection_and_yaw_invariance(setup_policy):
    config, actor, pose = setup_policy
    policy = PoseHoldPolicy(config)
    assert "pose_hold_start_check_failed" in policy.activate(_state(pose + 0.3))
    assert not policy.is_active and not actor.feeds
    assert "projected_gravity" in policy.activate(_state(pose, rpy=(0.9, -0.3, 0.4)))
    assert policy.activate(_state(pose)) is None
    policy.step(_state(pose))
    policy.deactivate()
    yaw_state = _state(pose, rpy=(0.2, -0.3, 2.2))
    yaw_state.base_pos[:] = [5, 6, 7]
    assert policy.activate(yaw_state) is None
    policy.step(yaw_state)
    np.testing.assert_allclose(actor.feeds[-1], actor.feeds[0], atol=1e-7)


@pytest.mark.parametrize("index,offset", [(0, -0.01), (1, 0), (-1, 0.01), (-2, 0)])
def test_reference_frame_selection(setup_policy, index, offset):
    config, _, pose = setup_policy
    policy = PoseHoldPolicy(replace(config, task=replace(config.task, reference_pose_frame=index)))
    np.testing.assert_allclose(policy.default_dof_angles, pose + offset, atol=1e-8)
    assert load_motion_last_pose(config.task.motion_data_path) == load_motion_pose(config.task.motion_data_path, -1)


@pytest.mark.parametrize("index", [3, -4, True, 1.5])
def test_invalid_reference_frame(setup_policy, index):
    config, _, _ = setup_policy
    with pytest.raises(ValueError, match="frame_index"):
        load_motion_pose(config.task.motion_data_path, index)


@pytest.mark.parametrize("bad", ["names", "quaternion", "nonfinite", "limits", "width"])
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
        positions = positions[:, :-1]
    np.savez(path, joint_names=names, joint_pos=positions)
    with pytest.raises(ValueError):
        PoseHoldPolicy(config)


@pytest.mark.parametrize("bad", ["input", "output", "order", "scale", "gains", "urdf"])
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
        actor.metadata.pop("robot_urdf")
    with pytest.raises(ValueError):
        PoseHoldPolicy(config)


@pytest.mark.parametrize("bad", ["term", "scale", "dimension", "history"])
def test_observation_contract_validation(setup_policy, bad):
    config, _, _ = setup_policy
    obs = config.observation
    if bad == "term":
        obs = replace(obs, obs_dict={"actor_obs": [*obs.obs_dict["actor_obs"], "pelvis_orientation_error"]})
    elif bad == "scale":
        obs = replace(obs, obs_scales={**obs.obs_scales, "dof_vel": 1})
    elif bad == "dimension":
        obs = replace(obs, obs_dims={**obs.obs_dims, "actions": 28})
    else:
        obs = replace(obs, history_length_dict={"actor_obs": 2})
    with pytest.raises(ValueError, match="Pose hold"):
        PoseHoldPolicy(replace(config, observation=obs))


@pytest.mark.parametrize("bad", ["joint_pos", "joint_vel", "base_ang_vel", "base_quat", "zero_quat", "action"])
def test_invalid_runtime_data(setup_policy, bad):
    config, actor, pose = setup_policy
    policy = PoseHoldPolicy(config)
    state = _state(pose)
    policy.activate(state)
    if bad == "action":
        actor.action[0, 0] = np.nan
    elif bad == "zero_quat":
        state.base_quat[:] = 0
    else:
        getattr(state, bad)[0, 0] = np.nan
    with pytest.raises(PolicyRuntimeFault):
        policy.step(state)


def test_example_loading_registration_and_path_validation(setup_policy, tmp_path):
    config, _, _ = setup_policy
    data = yaml.safe_load(EXAMPLE.read_text())
    model = tmp_path / "model.onnx"
    model.touch()
    data["task"]["model_path"] = str(model)
    data["task"]["motion_data_path"] = config.task.motion_data_path
    path = tmp_path / "pose_hold.yaml"
    path.write_text(yaml.safe_dump(data))
    runtime, config_path = load_runtime_config(path, ROOT / "configs/mqtt.yaml")
    resolved = resolve_policies(runtime, config_path)[0]
    assert isinstance(resolved.config.task, PoseHoldTaskConfig)
    assert resolved.config.task.reference_pose_frame == 1
    assert resolved.config.guard is not None
    assert _policy_class(resolved.kind) is PoseHoldPolicy
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert manifest["project"]["entry-points"]["vex_policy.policies"]["pose_hold"].endswith(":PoseHoldPolicy")
    Path(config.task.motion_data_path).unlink()
    with pytest.raises(ValueError, match="does not exist"):
        resolve_policies(runtime, config_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_pose_frame", True),
        ("reference_pose_frame", 1.5),
        ("rl_rate", 0),
        ("policy_action_scale", float("nan")),
        ("motion_data_path", " "),
        ("use_phase", True),
    ],
)
def test_strict_task_config(field, value):
    data = {"model_path": "actor.onnx", "motion_data_path": "pose.npz", field: value}
    with pytest.raises(ValueError):
        PoseHoldTaskConfig(**data)


def test_runtime_latches_without_sending_invalid_actor_output(setup_policy):
    config, actor, pose = setup_policy
    spec = PolicySpec(
        name="pose", implementation="pose_hold", observation=config.observation, task=config.task, guard=config.guard
    )
    runtime = RuntimeConfig(robot=RobotRuntimeConfig(config=config.robot), policies=(spec,))
    policy = PoseHoldPolicy(config)
    writes = []
    transport = SimpleNamespace(
        publish_status=lambda value: None,
        publish_state=lambda value: None,
        publish_reference_state=lambda value: None,
        close=lambda: None,
    )
    machine = PolicyStateMachine(
        runtime,
        (ResolvedPolicy(spec, config, "pose_hold"),),
        instances={"pose": policy},
        clock=lambda: 0.0,
        transport=transport,
        interface_manager=SimpleNamespace(
            get_low_state=lambda: _state(pose), send_low_command=lambda *args, **kwargs: writes.append(args)
        ),
    )
    try:
        for seq, names in enumerate([[], ["pose"]], 1):
            assert machine.inbox.accept(
                json.dumps(
                    {
                        "seq": seq,
                        "timestamp": 0,
                        "control": {"policy": names, "inputs": {name: {} for name in names}, "estop": False},
                    }
                )
            )
            machine.tick()
        machine.tick()
        assert machine.state == PolicyState.RUNNING and writes
        count = len(writes)
        actor.action = np.zeros((1, 28))
        machine.tick()
        assert machine.state == PolicyState.LATCHED
        assert len(writes) == count and not policy.is_active
        np.testing.assert_array_equal(policy.last_action, 0)
    finally:
        machine._policy_executor.shutdown(wait=True)
        policy.close()
