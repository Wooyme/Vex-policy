from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import yaml

from vex_policy.config.config_types import (
    ActionMaskConfig,
    InferenceConfig,
    InterpolationTaskConfig,
    PolicySpec,
)
from vex_policy.config.loader import load_runtime_config, resolve_policies
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.interpolation import InterpolationPolicy, load_motion_pose
from vex_policy.policies.policy_state_machine import _policy_class
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER, G1_JOINT_VELOCITY
from vex_policy.sdk.base.base_interface import LowState


def _motion(tmp_path, *, frames=None, root=False):
    if frames is None:
        frames = np.array([G1_29DOF.default_dof_angles])
    positions = np.asarray(frames)[:, ::-1]
    if root:
        positions = np.column_stack((np.zeros((len(positions), 7)), positions))
    path = tmp_path / "target.npz"
    np.savez(path, joint_names=np.asarray(G1_29DOF.dof_names[::-1]), joint_pos=positions)
    return path


def _spec(path, **task):
    return {
        "name": "interpolation-test",
        "implementation": "interpolation",
        "inputs": [],
        "observation": {"obs_dict": {}, "obs_dims": {}, "obs_scales": {}, "history_length_dict": {}},
        "task": {"motion_data_path": str(path), "duration_s": 1.0, **task},
    }


def _policy(path, *, robot=G1_29DOF, mask=None, **task):
    spec = PolicySpec.model_validate(_spec(path, **task))
    return InterpolationPolicy(
        InferenceConfig(robot=robot, inputs=spec.inputs, observation=spec.observation, task=spec.task, action_mask=mask)
    )


def _state(q):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=np.asarray(q, dtype=float).reshape(1, -1),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.zeros((1, 3)),
        joint_vel=np.zeros((1, len(q))),
    )


@pytest.mark.parametrize("root", [False, True])
@pytest.mark.parametrize("endpoint,index", [("first", 0), ("last", -1)])
@pytest.mark.parametrize("count", [1, 3])
def test_motion_endpoint_and_reordering(tmp_path, root, endpoint, index, count):
    frames = np.arange(count * 29, dtype=float).reshape(count, 29) / 100
    path = _motion(tmp_path, frames=frames, root=root)
    np.testing.assert_array_equal(load_motion_pose(path, G1_29DOF.dof_names, endpoint), frames[index])


@pytest.mark.parametrize(
    "arrays,reason",
    [
        ({"joint_pos": np.zeros((1, 29))}, "missing arrays"),
        ({"joint_names": np.array(["a", "a"]), "joint_pos": np.zeros((1, 2))}, "duplicates"),
        ({"joint_names": np.array(["a"]), "joint_pos": np.zeros((1, 1))}, "missing robot joints"),
        ({"joint_pos": np.zeros((0, 29))}, "shape"),
        ({"joint_pos": np.zeros((1, 30))}, "shape"),
        ({"joint_pos": np.zeros(29)}, "shape"),
        ({"joint_pos": np.full((1, 29), np.nan)}, "non-finite"),
        ({"joint_names": np.array(G1_29DOF.dof_names, dtype=object)}, "allow_pickle=False"),
    ],
)
def test_invalid_motion(tmp_path, arrays, reason):
    data = {"joint_names": np.array(G1_29DOF.dof_names), "joint_pos": np.zeros((1, 29))}
    if reason == "missing arrays":
        data = {}
    data.update(arrays)
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match=reason):
        load_motion_pose(path, G1_29DOF.dof_names, "first")


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_s", 0),
        ("duration_s", -1),
        ("duration_s", np.inf),
        ("duration_s", np.nan),
        ("rl_rate", 0),
        ("rl_rate", np.inf),
        ("target_frame", "middle"),
        ("motion_data_path", " "),
    ],
)
def test_invalid_task(field, value):
    with pytest.raises(ValueError, match=field):
        PolicySpec.model_validate(_spec("unused.npz", **{field: value}))


def test_required_duration_and_no_model_field():
    spec = _spec("unused.npz")
    del spec["task"]["duration_s"]
    with pytest.raises(ValueError, match="duration_s"):
        PolicySpec.model_validate(spec)
    with pytest.raises(ValueError, match="model_path"):
        PolicySpec.model_validate(_spec("unused.npz", model_path="unused.onnx"))


def test_config_resolution_and_model_free_activation(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("interpolation attempted to load a neural network")

    monkeypatch.setattr("onnx.load", no_network)
    monkeypatch.setattr("onnxruntime.InferenceSession", no_network)
    motion = _motion(tmp_path)
    config = tmp_path / "policy.yaml"
    config.write_text(yaml.safe_dump(_spec(motion.name)))
    mqtt = tmp_path / "mqtt.yaml"
    mqtt.write_text("{}")
    monkeypatch.chdir(tmp_path)
    runtime, config_path = load_runtime_config(config, mqtt)
    (resolved,) = resolve_policies(runtime, config_path)
    assert isinstance(resolved.config.task, InterpolationTaskConfig)
    assert resolved.config.task.motion_data_path == str(motion)
    policy = _policy_class(resolved.kind)(resolved.config)
    assert isinstance(policy, InterpolationPolicy)
    state = _state(G1_29DOF.default_dof_angles)
    assert policy.activate(state) is None
    command = policy.step(state)
    np.testing.assert_allclose(command.kp, G1_29DOF.stiff_startup_kp)
    np.testing.assert_allclose(command.kd, G1_29DOF.stiff_startup_kd)
    np.testing.assert_array_equal(command.dq, 0)
    np.testing.assert_array_equal(command.tau, 0)
    policy.apply_control({})
    policy.deactivate()
    policy.close()
    motion.unlink()
    with pytest.raises(ValueError, match="does not exist"):
        resolve_policies(runtime, config_path)


def test_fixed_start_offsets_mask_gains_and_reactivation(tmp_path):
    target = np.asarray(G1_29DOF.default_dof_angles)
    robot = replace(G1_29DOF, joint_offsets_deg=(1.0,) * 29, motor_kp=(5.0,) * 29, motor_kd=(0.5,) * 29)
    policy = _policy(
        _motion(tmp_path), robot=robot, mask=ActionMaskConfig(masked_joints=(robot.dof_names[0],)), duration_s=0.04
    )
    physical_target = target + np.deg2rad(1.0)
    start = physical_target - 0.02
    assert policy.activate(_state(start)) is None
    first = policy.step(_state(start + 0.2))
    np.testing.assert_allclose(first.q + policy.joint_offsets, start + 0.01)
    final = policy.step(_state(start - 0.2))
    np.testing.assert_allclose(final.q, target)
    np.testing.assert_allclose(policy.step(_state(start)).q, target)
    assert not final.controlled_joints[0]
    assert final.controlled_joints[1:].all()
    np.testing.assert_array_equal(final.kp, 5)
    np.testing.assert_array_equal(final.kd, 0.5)
    policy.deactivate()
    with pytest.raises(RuntimeError, match="not active"):
        policy.step(_state(start))
    assert policy.activate(_state(physical_target - 0.04)) is None
    np.testing.assert_allclose(policy.step(_state(start)).q, target - 0.02)


def test_velocity_limited_continuation_then_hold(tmp_path):
    target = np.asarray(G1_29DOF.default_dof_angles)
    start = target - 0.2
    robot = replace(G1_29DOF, joint_interpolation_slew_safety_factor=0.05)
    policy = _policy(_motion(tmp_path), robot=robot, duration_s=0.02)
    assert policy.activate(_state(start)) is None
    previous = start
    delta_limit = np.asarray(G1_JOINT_VELOCITY) / 50 * robot.joint_interpolation_slew_safety_factor
    commands = []
    for _ in range(20):
        command = policy.step(_state(start)).q + policy.joint_offsets
        assert np.all(np.abs(command - previous) <= delta_limit + 1e-12)
        assert np.all(command >= np.asarray(G1_JOINT_LOWER))
        assert np.all(command <= np.asarray(G1_JOINT_UPPER))
        commands.append(command)
        previous = command
    assert np.any(commands[0] < target - 1e-8)
    np.testing.assert_allclose(commands[-2], target)
    np.testing.assert_allclose(commands[-1], target)


def test_position_limits_include_calibration(tmp_path):
    frames = np.full((1, 29), 100.0)
    frames[0, ::2] = -100.0
    robot = replace(G1_29DOF, joint_offsets_deg=(2.0,) * 29)
    policy = _policy(_motion(tmp_path, frames=frames), robot=robot, duration_s=0.02)
    state = _state(G1_29DOF.default_dof_angles)
    assert policy.activate(state) is None
    for _ in range(100):
        command = policy.step(state)
    expected = np.clip(frames[0] + np.deg2rad(2), G1_JOINT_LOWER, G1_JOINT_UPPER)
    np.testing.assert_allclose(command.q + policy.joint_offsets, expected)


@pytest.mark.parametrize("q", [np.full(29, np.nan), np.zeros(28)])
def test_invalid_state_activation_and_runtime(tmp_path, q):
    policy = _policy(_motion(tmp_path))
    assert policy.activate(_state(q)).startswith("interpolation_start_failed:")
    assert policy.activate(_state(G1_29DOF.default_dof_angles)) is None
    with pytest.raises(PolicyRuntimeFault, match="interpolation_failed"):
        policy.step(_state(q))
