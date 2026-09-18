from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import GuardConfig, PassiveLocomotionTaskConfig
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.passive_locomotion import PassiveLocomotionPolicy
from vex_policy.policies.policy_state_machine import PolicyStateMachine
from vex_policy.robots import G1_29DOF, G1_JOINT_LOWER, G1_JOINT_UPPER
from vex_policy.sdk.base.base_interface import LowState

EXAMPLE = Path(__file__).resolve().parents[1] / "configs/g1/g1_passive_locomotion.yaml"


def _urdf():
    # Each actuated joint is a separate branch: ankle translation is analytically known.
    pieces = ['<robot name="passive_test"><link name="pelvis"/>']
    names = [*G1_29DOF.dof_names, "hip_bar_yaw_joint", "hip_bar_pitch_joint"]
    for i, name in enumerate(names):
        link = "right_ankle_roll_link" if name == "right_ankle_roll_joint" else f"link_{i}"
        pieces.append(
            f'<link name="{link}"/><joint name="{name}" type="revolute">'
            f'<parent link="pelvis"/><child link="{link}"/>'
            '<origin xyz="0.2 0 -0.3"/><axis xyz="0 1 0"/>'
            '<limit lower="-4" upper="4" effort="100" velocity="10"/></joint>'
        )
    return "".join(pieces) + "</robot>"


def _model(path, *, width=1960, metadata_update=None):
    # Last dof_pos frame + bias, so real inference also verifies the input layout.
    bias = np.linspace(0.01, 0.29, 29, dtype=np.float32)
    bias[0] = 150.0
    indices = np.arange(660 + 19 * 29, 660 + 20 * 29, dtype=np.int64)
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["actor_obs", "indices"], ["latest"], axis=1),
            helper.make_node("Add", ["latest", "bias"], ["action"]),
        ],
        "passive_test",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])],
        [numpy_helper.from_array(indices, "indices"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    metadata = dict(
        dof_names=G1_29DOF.dof_names, kp=[5.0] * 29, kd=[0.5] * 29, action_scale=[0.25] * 29, robot_urdf=_urdf()
    )
    metadata.update(metadata_update or {})
    helper.set_model_props(model, {key: json.dumps(value) for key, value in metadata.items()})
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return bias


def _state(q=None, quat=None):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=np.array([[1.0, 0, 0, 0]]) if quat is None else quat,
        joint_pos=np.zeros((1, 29)) if q is None else q,
        joint_vel=np.full((1, 29), 0.4),
        base_ang_vel=np.array([[0.1, 0.2, 0.3]]),
        base_lin_vel=np.zeros((1, 3)),
    )


@pytest.fixture
def deployment(tmp_path):
    model_path = tmp_path / "passive.onnx"
    bias = _model(model_path)
    motion_path = tmp_path / "pose.npz"
    pose = np.linspace(0, 0.1, 29)
    np.savez(
        motion_path,
        joint_names=np.array(G1_29DOF.dof_names[::-1]),
        joint_pos=np.stack([np.r_[[0, 0, 0, 1, 0, 0, 0], np.zeros(29)], np.r_[[0, 0, 0, 1, 0, 0, 0], pose[::-1]]]),
    )
    data = yaml.safe_load(EXAMPLE.read_text())
    data["guard"] = {"startup_joint_tolerance_rad": 0.2, "startup_gravity_tolerance": 0.2}
    data["task"].update(model_path=str(model_path), motion_data_path=str(motion_path))
    path = tmp_path / "passive.yaml"
    path.write_text(yaml.safe_dump(data))
    runtime, config_path = load_runtime_config(path, EXAMPLE.parents[1] / "mqtt.yaml")
    resolved = resolve_policies(runtime, config_path)
    return SimpleNamespace(
        path=path,
        data=data,
        runtime=runtime,
        resolved=resolved,
        config=resolved[0].config,
        bias=bias,
        pose=pose,
        model=model_path,
        motion=motion_path,
    )


def test_real_onnx_golden_history_actions_and_restart(deployment):
    policy = PassiveLocomotionPolicy(deployment.config)
    assert isinstance(deployment.config.task, PassiveLocomotionTaskConfig)
    assert isinstance(deployment.config.guard, GuardConfig)
    np.testing.assert_allclose(policy.default_dof_angles, deployment.pose)
    state = _state(q=deployment.pose.reshape(1, -1))
    assert policy.activate(state) is None
    expected_history = {name: [] for name in policy._OBS_DIMS}
    previous_action = np.zeros(29)
    controls = {"down_vel": 0.04, "up_vel": 0.08, "target_height": 0.3, "min_height": 0.12}
    policy.apply_control(controls)
    for step in range(23):
        offset = step * 0.001
        state = replace(state, joint_pos=deployment.pose.reshape(1, -1) + offset)
        command = policy.step(state)
        terms = dict(
            actions=previous_action,
            base_ang_vel=np.array([0.025, 0.05, 0.075]),
            base_right_foot_height_difference=[0.3],
            dof_pos=np.full(29, offset),
            dof_vel=np.full(29, 0.02),
            passive_command=[0.04, 0.08, 0.3, 0.12],
            projected_gravity=[0, 0, -1],
        )
        flattened = []
        for name, dim in policy._OBS_DIMS.items():
            expected_history[name].append(np.array(terms[name]))
            history = expected_history[name][-20:]
            flattened.extend([np.zeros(dim)] * (20 - len(history)) + history)
        np.testing.assert_allclose(
            policy.observations.obs_buf_dict["actor_obs"][0], np.concatenate(flattened), atol=1e-6
        )
        previous_action = deployment.bias + offset
        np.testing.assert_allclose(policy.actions.last[0], previous_action, atol=1e-6)
        np.testing.assert_allclose(
            command.q,
            np.clip(deployment.pose + np.clip(previous_action, -100, 100) * 0.25, G1_JOINT_LOWER, G1_JOINT_UPPER),
            atol=1e-7,
        )
        np.testing.assert_array_equal(command.kp, 5.0)
        assert command.controlled_joints.all()
    policy.deactivate()
    assert policy.activate(_state(q=deployment.pose.reshape(1, -1))) is None
    np.testing.assert_array_equal(policy.actions.last, 0)
    command = policy.step(_state(q=deployment.pose.reshape(1, -1)))
    buffer = policy.observations.obs_buf_dict["actor_obs"][0]
    np.testing.assert_array_equal(buffer[:580], 0)
    np.testing.assert_allclose(
        policy.passive_command, [[policy.parameters[name].default for name in policy._COMMAND_NAMES]]
    )
    policy.close()


def test_height_with_tilt_and_auxiliary_joints(deployment):
    policy = PassiveLocomotionPolicy(deployment.config)
    assert policy.kinematics._kinematics_model.nq == 31
    angle = 0.6
    quat = np.array([[np.cos(angle / 2), 0, np.sin(angle / 2), 0]])
    state = _state(quat=quat)
    terms = policy.get_current_obs_buffer_dict(state)
    expected = 0.2 * np.sin(angle) + 0.3 * np.cos(angle)
    np.testing.assert_allclose(terms["base_right_foot_height_difference"], [[expected]])
    # Neither unavailable world position nor a scaled quaternion changes this observation.
    changed = replace(state, base_pos=np.full((1, 3), 999.0), base_quat=quat * 2)
    np.testing.assert_allclose(
        policy.get_current_obs_buffer_dict(changed)["base_right_foot_height_difference"], [[expected]]
    )


@pytest.mark.parametrize("change", ["joint", "gravity", "nan_velocity", "zero_quaternion"])
def test_startup_rejected(deployment, change):
    policy = PassiveLocomotionPolicy(deployment.config)
    state = _state(q=deployment.pose.reshape(1, -1).copy())
    if change == "joint":
        state.joint_pos[0, 0] += 0.3
    elif change == "gravity":
        state.base_quat[:] = [0, 1, 0, 0]
    elif change == "nan_velocity":
        state.joint_vel[0, 0] = np.nan
    else:
        state.base_quat.fill(0)
    assert "passive_locomotion_start_check_failed" in policy.activate(state)
    assert not policy.is_active


@pytest.mark.parametrize(
    "change,match",
    [
        ("width", "actor_obs"),
        ("order", "dof_names"),
        ("scale", "action_scale"),
        ("urdf", "robot_urdf"),
        ("missing_joint", "missing joint"),
        ("gains", "motor_kp"),
    ],
)
def test_model_rejection(deployment, change, match):
    updates = {
        "order": {"dof_names": G1_29DOF.dof_names[::-1]},
        "scale": {"action_scale": 0.5},
        "urdf": {"robot_urdf": ""},
        "missing_joint": {"robot_urdf": _urdf().replace("left_hip_pitch_joint", "missing_joint")},
        "gains": {"kp": [float("nan")] * 29},
    }
    _model(deployment.model, width=1980 if change == "width" else 1960, metadata_update=updates.get(change))
    with pytest.raises(ValueError, match=match):
        PassiveLocomotionPolicy(deployment.config)


@pytest.mark.parametrize(
    "change", ["history", "terms", "scale", "command", "negative_height_range", "pose_names", "pose_limit"]
)
def test_configuration_rejection(deployment, change):
    config = deployment.config
    if change == "history":
        config = replace(config, observation=replace(config.observation, history_length_dict={"actor_obs": 1}))
    elif change == "terms":
        config = replace(config, observation=replace(config.observation, obs_dict={"actor_obs": ["actions"]}))
    elif change == "scale":
        config = replace(
            config,
            observation=replace(config.observation, obs_scales={**config.observation.obs_scales, "dof_vel": 1.0}),
        )
    elif change == "command":
        config = replace(config, inputs=config.inputs[:-1])
    elif change == "negative_height_range":
        last = config.inputs[-1].model_copy(
            update={"parameter": config.inputs[-1].parameter.model_copy(update={"min": -0.01})}
        )
        config = replace(config, inputs=(*config.inputs[:-1], last))
    else:
        names = list(G1_29DOF.dof_names)
        pose = deployment.pose.copy()
        if change == "pose_names":
            names[0] = "incorrect_joint"
        else:
            pose[0] = 10.0
        np.savez(deployment.motion, joint_names=names, joint_pos=np.r_[[0, 0, 0, 1, 0, 0, 0], pose][None])
    with pytest.raises(ValueError):
        PassiveLocomotionPolicy(config)


@pytest.mark.parametrize(
    "change", ["body", "mask", "missing_model", "missing_motion", "missing_guard", "rate", "unknown"]
)
def test_yaml_and_path_rejection(deployment, change):
    data = copy.deepcopy(deployment.data)
    if change == "body":
        data["type"] = "lower_body"
    elif change == "mask":
        data["task"]["action_mask_path"] = "mask.yaml"
    elif change in {"missing_model", "missing_motion"}:
        data["task"]["model_path" if change == "missing_model" else "motion_data_path"] = "missing.file"
    elif change == "missing_guard":
        data.pop("guard")
    elif change == "rate":
        data["task"]["rl_rate"] = 100.0
    else:
        data["task"]["unexpected"] = True
    deployment.path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        runtime, path = load_runtime_config(deployment.path, EXAMPLE.parents[1] / "mqtt.yaml")
        resolve_policies(runtime, path)


@pytest.mark.parametrize("fault", ["state", "command", "nan_action", "shape", "inference", "observation"])
def test_runtime_faults(deployment, fault):
    policy = PassiveLocomotionPolicy(deployment.config)
    state = _state(q=deployment.pose.reshape(1, -1))
    assert policy.activate(state) is None
    if fault == "command":
        with pytest.raises(PolicyRuntimeFault, match="invalid_command"):
            policy.apply_control({"down_vel": np.nan, "up_vel": 0.05, "target_height": 0.3, "min_height": 0.1})
        return
    if fault == "state":
        state.joint_vel[0, 0] = np.inf
    elif fault == "nan_action":
        policy.actor = lambda obs: np.full((1, 29), np.nan)
    elif fault == "shape":
        policy.actor = lambda obs: np.zeros((1, 28))
    elif fault == "inference":

        def fail(obs):
            raise RuntimeError("session failed")

        policy.actor = fail
    else:
        policy.kinematics.height_difference = lambda *args: np.array([[np.nan]])
    with pytest.raises(PolicyRuntimeFault):
        policy.step(state)


def test_runtime_registration_and_fault_latches_without_write(deployment):
    state = _state(q=deployment.pose.reshape(1, -1))
    writes = []
    transport = SimpleNamespace(
        close=lambda: None,
        publish_status=lambda *args: None,
        publish_state=lambda *args: None,
        publish_reference_state=lambda *args: None,
    )
    manager = SimpleNamespace(get_low_state=lambda: state, send_low_command=lambda *args, **kwargs: writes.append(args))
    machine = PolicyStateMachine(
        deployment.runtime, deployment.resolved, transport=transport, interface_manager=manager, clock=lambda: 0.0
    )
    machine._maybe_publish_state = lambda *args: None
    try:
        policy = machine.policies["g1-passive-locomotion"]
        assert isinstance(policy, PassiveLocomotionPolicy)
        assert policy.activate(state) is None
        machine.state = type(machine.state).RUNNING
        machine.active_policy = ("g1-passive-locomotion",)
        control = SimpleNamespace(
            policy=machine.active_policy,
            estop=False,
            inputs={"g1-passive-locomotion": dict(down_vel=0.03, up_vel=0.055, target_height=0.275, min_height=0.1)},
        )
        machine.inbox = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(received_at=0.0, packet=SimpleNamespace(seq=1, control=control))
        )
        policy.actor = lambda obs: np.full((1, 29), np.nan)
        machine.tick(0.0)
        assert machine.state == "latched"
        assert "passive_invalid_action" in machine.reason
        assert not policy.is_active
        assert not writes
    finally:
        machine._policy_executor.shutdown(wait=True)
        machine._close_policies(machine.policies.values())
        transport.close()
