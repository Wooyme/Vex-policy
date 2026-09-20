from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from vex_policy.config.config_types import GuardConfig, InferenceConfig, PelvisRecoveryTaskConfig, PolicySpec
from vex_policy.config.loader import load_runtime_config, resolve_policies
from vex_policy.policies import pelvis_recovery
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.pelvis_recovery import PelvisRecoveryPolicy
from vex_policy.policy_state_machine import _policy_class
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse, quat_to_rpy, rpy_to_quat

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/g1/ppo_recovery.yaml"


def _state(q, rpy=(0.2, -0.3, 1.1)):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=rpy_to_quat(rpy)[None],
        joint_pos=np.array(q, dtype=float)[None],
        joint_vel=np.full((1, 29), 2.0),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.array([[1.0, 2.0, 3.0]]),
    )


def _urdf():
    # Independent branches make the ankle translation exactly 0.1 m below the
    # root regardless of joint angles, while still exercising real Pinocchio FK.
    parts = ['<robot name="fixture"><link name="pelvis"/>']
    for index, name in enumerate(G1_29DOF.dof_names):
        link = "right_ankle_roll_link" if name == "right_ankle_roll_joint" else f"link_{index}"
        parts.append(
            f'<link name="{link}"/><joint name="{name}" type="revolute">'
            f'<parent link="pelvis"/><child link="{link}"/><origin xyz="0 0 -0.1"/>'
            '<axis xyz="0 1 0"/><limit lower="-4" upper="4" effort="100" velocity="10"/></joint>'
        )
    return "".join(parts) + "</robot>"


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
        self.inputs = [SimpleNamespace(name="actor_obs", shape=[1, 99])]
        self.outputs = [SimpleNamespace(name="action", shape=[1, 29])]
        self.action = np.arange(29, dtype=np.float32)[None] / 10
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
    np.savez(
        motion,
        joint_names=G1_29DOF.dof_names[::-1],
        joint_pos=np.concatenate(([99, 98, 97], rpy_to_quat((0.2, -0.3, 0.4)), pose[::-1]))[None],
    )
    spec = PolicySpec.model_validate(yaml.safe_load(CONFIG.read_text()))
    config = InferenceConfig(
        robot=G1_29DOF,
        inputs=spec.inputs,
        observation=spec.observation,
        task=replace(spec.task, model_path="fake.onnx", motion_data_path=str(motion)),
    )
    actor = FakeActor()
    monkeypatch.setattr(pelvis_recovery, "OnnxActor", lambda path: actor)
    return config, actor, pose


def test_golden_observation_and_pd_targets(setup_policy):
    config, actor, pose = setup_policy
    policy = PelvisRecoveryPolicy(config)
    state = _state(pose + 0.01)
    assert policy.activate(state) is None
    command = policy.step(state)
    gravity = quat_rotate_inverse(state.base_quat, np.array([[0.0, 0.0, -1.0]]))
    height_difference = -0.1 * gravity[0, 2]
    # Explicit training order, independent of the adapter's term dictionary.
    expected = np.concatenate(
        (
            np.zeros(29),
            [0.25, 0.5, 0.75],
            [height_difference],
            np.full(29, 0.01),
            np.full(29, 0.1),
            np.zeros(3),
            [0.0, 0.2],
            gravity[0],
        )
    )
    np.testing.assert_allclose(actor.feeds[0][0], expected, atol=1e-7)
    assert actor.feeds[0].dtype == np.float32
    np.testing.assert_allclose(policy.default_dof_angles, pose)
    np.testing.assert_allclose(command.q, np.clip(pose + actor.action[0] * 0.25, G1_JOINT_LOWER, G1_JOINT_UPPER))
    ids = [G1_29DOF.dof_names.index(name) for name in policy.BRIDGE_JOINTS]
    kp, kd = np.full(29, 10.0), np.full(29, 2.0)
    kp[ids], kd[ids] = 3.5, 1.2
    np.testing.assert_allclose(command.kp, kp)
    np.testing.assert_allclose(command.kd, kd)
    np.testing.assert_array_equal(command.dq, 0.0)
    np.testing.assert_array_equal(command.tau, 0.0)
    np.testing.assert_array_equal(command.controlled_joints, True)
    # Neither metadata nor the common robot gains are changed.
    assert actor.metadata["kp"] == [10.0] * 29
    assert config.robot.motor_kp is None
    policy.step(state)
    np.testing.assert_allclose(actor.feeds[1][0, :29], actor.action[0])
    assert actor.feeds[1][0, 94] == pytest.approx(0.01)
    policy.close()


def test_height_feedback_slew_input_changes_and_restart(setup_policy):
    config, actor, pose = setup_policy
    policy = PelvisRecoveryPolicy(config)
    state = _state(pose)
    policy.activate(state)
    velocities = []
    for _ in range(30):
        policy.step(state)
        velocities.append(float(policy.recovery_command[0, 0]))
    height = 0.1 * np.cos(0.2) * np.cos(-0.3) + 0.035
    desired = 0.175 * np.tanh((0.2 - height) / 0.03)
    assert velocities[0] == 0.0
    assert velocities[-1] == pytest.approx(desired)
    assert np.max(np.abs(np.diff(velocities))) <= 0.010000001
    policy.apply_control({"target_height": 0.15, "peak_speed": 0.05})
    assert policy.recovery_command[0, 0] == velocities[-1]
    policy.step(state)
    assert policy.recovery_command[0, 0] == pytest.approx(velocities[-1] - 0.01)
    # Raise the estimated root above the target; the command never goes down.
    policy._ankle_kinematics.height_difference = lambda *args: np.array([[0.3]])
    for _ in range(30):
        policy.step(state)
    assert policy.recovery_command[0, 0] == 0.0
    policy.deactivate()
    np.testing.assert_array_equal(policy.last_action, 0.0)
    assert policy.pelvis_orientation_reference_quat is None
    policy.activate(state)
    policy.step(state)
    np.testing.assert_allclose(actor.feeds[-1][0, 94:96], [0.0, 0.2])
    np.testing.assert_array_equal(actor.feeds[-1][0, :29], 0.0)
    policy.close()


def test_reference_tilt_is_preserved_and_yaw_recaptured(setup_policy):
    config, actor, pose = setup_policy
    policy = PelvisRecoveryPolicy(config)
    state = _state(pose, rpy=(0.4, -0.3, 1.1))
    policy.activate(state)
    np.testing.assert_allclose(quat_to_rpy(policy.pelvis_orientation_reference_quat[0]), [0.2, -0.3, 1.1])
    policy.step(state)
    np.testing.assert_allclose(actor.feeds[-1][0, 91:94], [0.2, 0.0, 0.0], atol=1e-7)
    # Equivalent quaternion signs and nonunit sensor inputs normalize identically.
    policy.step(replace(state, base_quat=-2 * state.base_quat))
    np.testing.assert_allclose(actor.feeds[-1][0, 91:94], [0.2, 0.0, 0.0], atol=1e-7)
    policy.deactivate()
    policy.activate(_state(pose, rpy=(0.2, -0.3, -0.7)))
    np.testing.assert_allclose(quat_to_rpy(policy.pelvis_orientation_reference_quat[0]), [0.2, -0.3, -0.7])


def test_raw_action_history_is_independent_of_clipped_targets(setup_policy):
    config, actor, pose = setup_policy
    policy = PelvisRecoveryPolicy(config)
    policy.activate(_state(pose))
    actor.action.fill(150.0)
    command = policy.step(_state(pose))
    np.testing.assert_array_equal(command.q, G1_JOINT_UPPER)
    policy.step(_state(pose))
    np.testing.assert_array_equal(actor.feeds[-1][0, :29], 150.0)
    debug_policy = PelvisRecoveryPolicy(
        replace(config, task=replace(config.task, debug=replace(config.task.debug, force_zero_action=True)))
    )
    debug_policy.activate(_state(pose))
    np.testing.assert_allclose(debug_policy.step(_state(pose)).q, pose)
    np.testing.assert_array_equal(debug_policy.last_action, 0.0)


def test_optional_guard_and_reference_clipping(setup_policy):
    config, _, pose = setup_policy
    state = _state(pose + 0.4)
    assert PelvisRecoveryPolicy(config).activate(state) is None
    guarded = PelvisRecoveryPolicy(replace(config, guard=GuardConfig()))
    assert "pelvis_recovery_start_check_failed" in guarded.activate(state)
    assert not guarded.is_active
    assert guarded.activate(_state(pose)) is None
    # Training clips the final reference frame to hardware limits.
    with np.load(config.task.motion_data_path) as motion:
        positions = motion["joint_pos"].copy()
        names = motion["joint_names"].copy()
    positions[0, 7:] = 100.0
    np.savez(config.task.motion_data_path, joint_pos=positions, joint_names=names)
    clipped = PelvisRecoveryPolicy(config)
    np.testing.assert_array_equal(clipped.default_dof_angles, G1_JOINT_UPPER)


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_quat", np.zeros((1, 4))),
        ("base_quat", np.full((1, 4), np.nan)),
        ("joint_pos", np.full((1, 29), np.inf)),
        ("joint_vel", np.full((1, 29), np.nan)),
        ("base_ang_vel", np.full((1, 3), np.inf)),
    ],
)
def test_invalid_state_rejected_without_guard(setup_policy, field, value):
    config, actor, pose = setup_policy
    policy = PelvisRecoveryPolicy(config)
    bad_state = replace(_state(pose), **{field: value})
    with pytest.raises(PolicyRuntimeFault):
        policy.activate(bad_state)
    assert not policy.is_active
    policy.activate(_state(pose))
    with pytest.raises(PolicyRuntimeFault):
        policy.step(bad_state)
    assert not actor.feeds


@pytest.mark.parametrize("action", [np.full((1, 29), np.nan), np.full((1, 29), np.inf), np.zeros((1, 28))])
def test_invalid_actor_output_faults(setup_policy, action):
    config, actor, pose = setup_policy
    policy = PelvisRecoveryPolicy(config)
    policy.activate(_state(pose))
    actor.action = action
    with pytest.raises(PolicyRuntimeFault, match="29 finite actions"):
        policy.step(_state(pose))


@pytest.mark.parametrize(
    "key,value",
    [
        ("dof_names", G1_29DOF.dof_names[::-1]),
        ("action_scale", [0.5] * 29),
        ("action_scale", [np.nan] * 29),
        ("action_scale", 0.25),
        ("kp", [np.nan] * 29),
        ("kd", [-1.0] * 29),
        ("kp", [1.0]),
        ("robot_urdf", ""),
    ],
)
def test_bad_model_metadata_rejected(setup_policy, key, value):
    config, actor, _ = setup_policy
    actor.metadata[key] = value
    with pytest.raises(ValueError):
        PelvisRecoveryPolicy(config)


@pytest.mark.parametrize(
    "side,name,shape",
    [
        ("inputs", "actor_obs", [1, 105]),
        ("inputs", "observations", [1, 99]),
        ("outputs", "action", [1, 28]),
        ("outputs", "actions", [1, 29]),
    ],
)
def test_bad_model_interface_rejected(setup_policy, side, name, shape):
    config, actor, _ = setup_policy
    setattr(actor, side, [SimpleNamespace(name=name, shape=shape)])
    with pytest.raises(ValueError, match="ONNX must expose"):
        PelvisRecoveryPolicy(config)


@pytest.mark.parametrize(
    "changes",
    [
        {"history_length_dict": {"actor_obs": 2}},
        {"obs_dims": {**PelvisRecoveryPolicy._OBS_DIMS, "pelvis_recovery_command": 8}},
        {"obs_scales": {**PelvisRecoveryPolicy._OBS_SCALES, "base_ang_vel": 1.0}},
        {"obs_dict": {"actor_obs": [name for name in PelvisRecoveryPolicy._OBS_DIMS if name != "actions"]}},
    ],
)
def test_bad_observation_configuration_rejected(setup_policy, changes):
    config, _, _ = setup_policy
    with pytest.raises(ValueError, match="Pelvis recovery"):
        PelvisRecoveryPolicy(replace(config, observation=replace(config.observation, **changes)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("slowdown_height_m", 0.0),
        ("max_acceleration_m_s2", -1.0),
        ("right_ankle_height_m", -0.01),
        ("bridge_kp_scale", 1.1),
        ("bridge_kd_scale", np.inf),
        ("rl_rate", 0.0),
        ("motion_data_path", " "),
        ("policy_action_scale", np.nan),
        ("use_phase", True),
    ],
)
def test_invalid_task_values_rejected(setup_policy, field, value):
    config, _, _ = setup_policy
    with pytest.raises(ValueError):
        replace(config.task, **{field: value})


def test_yaml_resolution_registration_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    runtime, path = load_runtime_config(CONFIG)
    assert isinstance(runtime.policies[0].task, PelvisRecoveryTaskConfig)
    assert _policy_class("pelvis_recovery") is PelvisRecoveryPolicy
    # Resolve fixtures rather than depending on ignored local model assets.
    model, motion = tmp_path / "actor.onnx", tmp_path / "pose.npz"
    model.touch()
    motion.touch()
    spec = runtime.policies[0]
    spec = spec.model_copy(update={"task": replace(spec.task, model_path=str(model), motion_data_path=str(motion))})
    runtime = runtime.model_copy(update={"policies": (spec,)})
    resolved = resolve_policies(runtime, path)[0]
    assert resolved.config.task.motion_data_path == str(motion)
    motion.unlink()
    with pytest.raises(ValueError, match="motion file does not exist"):
        resolve_policies(runtime, path)


def test_real_model_offline_lifecycle(monkeypatch):
    monkeypatch.chdir(ROOT)
    assets = [ROOT / "models/loco/g1_29dof/ppo_recovery.onnx", ROOT / "reference/ridding1.npz"]
    if not all(path.is_file() for path in assets):
        pytest.skip("Local recovery ONNX and reference NPZ are not installed")
    runtime, path = load_runtime_config(CONFIG)
    policy = PelvisRecoveryPolicy(resolve_policies(runtime, path)[0].config)
    state = replace(_state(policy.default_dof_angles), base_quat=np.asarray(policy.initial_pose.root_quat_wxyz)[None])
    try:
        assert policy.activate(state) is None
        policy.apply_control({"target_height": 0.2, "peak_speed": 0.175})
        first = policy.step(state)
        for _ in range(5):
            command = policy.step(state)
            for values in (command.q, command.dq, command.tau, command.kp, command.kd):
                assert values.shape == (29,) and np.isfinite(values).all()
            assert np.all(command.q >= G1_JOINT_LOWER) and np.all(command.q <= G1_JOINT_UPPER)
        policy.deactivate()
        assert policy.activate(state) is None
        np.testing.assert_array_equal(policy.step(state).q, first.q)
    finally:
        policy.close()
