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
from vex_policy.config.config_types import GuardConfig, PelvisRecoveryTaskConfig
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.pelvis_recovery import BRIDGE_JOINTS, PelvisRecoveryPolicy
from vex_policy.policy_state_machine import PolicyStateMachine
from vex_policy.robots import G1_29DOF
from vex_policy.sdk.base.base_interface import LowState

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/g1/g1_pelvis_recovery.yaml"


def _urdf():
    pieces = ['<robot name="recovery_test"><link name="pelvis"/>']
    for i, name in enumerate((*G1_29DOF.dof_names, "hip_bar_yaw_joint", "hip_bar_pitch_joint")):
        link = "right_ankle_roll_link" if name == "right_ankle_roll_joint" else f"link_{i}"
        pieces.append(
            f'<link name="{link}"/><joint name="{name}" type="revolute">'
            f'<parent link="pelvis"/><child link="{link}"/>'
            '<origin xyz="0.2 0 -0.03"/><axis xyz="0 1 0"/>'
            '<limit lower="-4" upper="4" effort="100" velocity="10"/></joint>'
        )
    return "".join(pieces) + "</robot>"


def _model(path, *, width=99, output_width=29, metadata_update=None):
    # Reading the joint-position slice makes inference sensitive to term ordering.
    bias = np.linspace(0.01, 0.29, output_width, dtype=np.float32)
    for name, value in (("left_knee_joint", -150), ("waist_pitch_joint", 150), ("left_shoulder_pitch_joint", 150)):
        index = G1_29DOF.dof_names.index(name)
        if index < output_width:
            bias[index] = value
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["actor_obs", "indices"], ["positions"], axis=1),
            helper.make_node("Add", ["positions", "bias"], ["action"]),
        ],
        "recovery_test",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, output_width])],
        [
            numpy_helper.from_array(np.arange(12, 12 + output_width, dtype=np.int64), "indices"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    metadata = dict(
        dof_names=G1_29DOF.dof_names, kp=[20.0] * 29, kd=[2.0] * 29, action_scale=[0.25] * 29, robot_urdf=_urdf()
    )
    metadata.update(metadata_update or {})
    helper.set_model_props(model, {key: json.dumps(value) for key, value in metadata.items()})
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return bias


def _motion(path, pose, quat=(1.0, 0, 0, 0)):
    np.savez(
        path,
        joint_names=np.array(G1_29DOF.dof_names[::-1]),
        joint_pos=np.stack([np.r_[[0, 0, 0, 1, 0, 0, 0], np.zeros(29)], np.r_[[5, 6, 7], quat, pose[::-1]]]),
    )


def _state(pose, quat=(1.0, 0, 0, 0)):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=np.asarray([quat], dtype=float),
        joint_pos=np.asarray(pose, dtype=float).reshape(1, 29),
        joint_vel=np.full((1, 29), 0.4),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.array([[0.1, 0.2, 0.3]]),
    )


@pytest.fixture
def deployment(tmp_path):
    model = tmp_path / "recovery.onnx"
    bias = _model(model)
    pose = np.linspace(0, 0.1, 29)
    motion = tmp_path / "reference.npz"
    _motion(motion, pose)
    data = yaml.safe_load(EXAMPLE.read_text())
    data["task"].update(model_path=str(model), motion_data_path=str(motion))
    path = tmp_path / "recovery.yaml"
    path.write_text(yaml.safe_dump(data))
    runtime, config_path = load_runtime_config(path, ROOT / "configs/mqtt.yaml")
    resolved = resolve_policies(runtime, config_path)
    return SimpleNamespace(
        model=model,
        motion=motion,
        pose=pose,
        bias=bias,
        path=path,
        data=data,
        config=resolved[0].config,
        runtime=runtime,
        resolved=resolved,
    )


def test_real_onnx_observations_pd_targets_and_restart(deployment):
    policy = PelvisRecoveryPolicy(deployment.config)
    assert isinstance(policy.config.task, PelvisRecoveryTaskConfig)
    np.testing.assert_allclose(policy.default_dof_angles, deployment.pose)
    state = _state(deployment.pose)
    assert policy.activate(state) is None
    previous = np.zeros(29)
    ids = policy.bridge_ids
    kp, kd = np.full(29, 20.0), np.full(29, 2.0)
    kp[ids], kd[ids] = 7.0, 1.2
    for step in range(25):
        offset = step * 0.001
        state = replace(state, joint_pos=(deployment.pose + offset)[None])
        command = policy.step(state)
        expected = np.r_[
            [0.025, 0.05, 0.075],
            [0, 0, 0, 1],
            [0.17, min(step * 0.01, 0.175 * np.tanh(0.17 / 0.03)), 0, 1, 0.2],
            np.full(29, offset),
            np.full(29, 0.02),
            previous,
        ]
        actual = policy.observations.obs_buf_dict["actor_obs"]
        assert actual.shape == (1, 99) and actual.dtype == np.float32
        np.testing.assert_allclose(actual[0], np.clip(expected, -100, 100), atol=1e-6)
        previous = deployment.bias + offset
        np.testing.assert_allclose(policy.actions.last[0], previous, atol=1e-6)
        delta = np.clip(previous, -100, 100) * 0.25
        target = deployment.pose + delta
        np.testing.assert_allclose(command.q, target, atol=1e-6)
        np.testing.assert_allclose(command.kp, kp)
        np.testing.assert_allclose(command.kd, kd)
        np.testing.assert_array_equal(command.tau, 0)
        np.testing.assert_array_equal(command.dq, 0)
        assert command.controlled_joints.all()
    # Reading observations does not advance command slew.
    before = policy.recovery_command.copy()
    policy.get_current_obs_buffer_dict(state)
    np.testing.assert_array_equal(policy.recovery_command, before)
    policy.deactivate()
    assert policy.reference_quat is None
    assert policy.activate(_state(deployment.pose)) is None
    policy.step(_state(deployment.pose))
    np.testing.assert_array_equal(policy.observations.obs_buf_dict["actor_obs"][0, 70:], 0)
    assert policy.target_velocity == 0
    np.testing.assert_allclose(policy.robot_config.motor_kp, kp)  # No second weakening on restart.
    policy.close()


def _yaw_pitch(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


def test_orientation_preserves_reference_tilt_across_vertical_and_quaternion_sign(deployment):
    _motion(deployment.motion, deployment.pose, _yaw_pitch(0, -1.7))
    config = replace(deployment.config, task=replace(deployment.config.task, right_ankle_height_m=0.12))
    policy = PelvisRecoveryPolicy(config)
    yaw = 1.2
    state = _state(deployment.pose, _yaw_pitch(yaw, -1.4) * 2)
    assert policy.activate(state) is None
    np.testing.assert_allclose(policy.reference_quat[0], _yaw_pitch(yaw, -1.7), atol=1e-7)
    expected = [[0, np.sin(0.15), 0, np.cos(0.15)]]
    np.testing.assert_allclose(policy.get_current_obs_buffer_dict(state)["base_orientation"], expected, atol=1e-7)
    changed = replace(state, base_quat=-state.base_quat)
    np.testing.assert_allclose(policy.get_current_obs_buffer_dict(changed)["base_orientation"], expected, atol=1e-7)
    # FK uses normalized IMU gravity and includes neutral auxiliary joints.
    assert policy.kinematics._kinematics_model.nq == 31
    expected_height = 0.2 * np.sin(-1.4) + 0.03 * np.cos(-1.4) + 0.12
    assert policy.current_height == pytest.approx(expected_height)
    changed = replace(changed, base_pos=np.full((1, 3), 999.0))
    policy.step(changed)
    assert policy.current_height == pytest.approx(expected_height)


def test_height_feedback_load_release_and_input_changes(deployment):
    config = replace(deployment.config, task=replace(deployment.config.task, right_ankle_height_m=0.1))
    policy = PelvisRecoveryPolicy(config)
    state = _state(deployment.pose)
    height = [0.03]
    policy.kinematics.height_difference = lambda *args: np.array([[height[0]]])
    assert policy.activate(state) is None
    assert policy.start_height == pytest.approx(0.13)
    policy.apply_control({"peak_speed": 0.2, "target_height": 0.2})
    policy.step(state)
    np.testing.assert_allclose(policy.recovery_command, [[0.07, 0, 0, 1, 0.2]], atol=1e-7)
    height[0] = 0.065
    policy.step(state)
    np.testing.assert_allclose(policy.recovery_command, [[0.035, 0.01, 1, 0, 0.2]], atol=1e-7)
    height[0] = 0.03  # Load drives the robot back to its initial height.
    policy.step(state)
    np.testing.assert_allclose(policy.recovery_command, [[0.07, 0.02, 0, 1, 0.2]], atol=1e-7)
    height[0] = 0.065  # Unloading resumes from actual progress.
    policy.step(state)
    np.testing.assert_allclose(policy.recovery_command, [[0.035, 0.03, 1, 0, 0.2]], atol=1e-7)
    policy.apply_control({"peak_speed": 0.05, "target_height": 0.2})
    assert policy.target_velocity == pytest.approx(0.03)
    assert policy.start_height == pytest.approx(0.13)
    height[0] = 0.12
    for _ in range(5):
        policy.step(state)
    np.testing.assert_allclose(policy.recovery_command, [[-0.02, 0, 0, -1, 0.2]], atol=1e-7)
    # A changed target rebases from the most recent measurement and restarts slew.
    policy.apply_control({"peak_speed": 0.2, "target_height": 0.25})
    assert policy.start_height == pytest.approx(0.22)
    policy.step(state)
    np.testing.assert_allclose(policy.recovery_command, [[0.03, 0, 0, 1, 0.25]], atol=1e-7)
    policy.step(state)
    assert policy.target_velocity == pytest.approx(0.01)
    policy.apply_control({"peak_speed": 0.2, "target_height": 0.15})
    policy.step(state)
    np.testing.assert_allclose(policy.recovery_command, [[-0.07, 0, 0, -1, 0.15]], atol=1e-7)


@pytest.mark.parametrize("start", [0.2, 0.19995, 0.21])
def test_zero_or_negative_stroke_uses_slowdown_height(deployment, start):
    policy = PelvisRecoveryPolicy(deployment.config)
    state = _state(deployment.pose)
    height = [start]
    policy.kinematics.height_difference = lambda *args: np.array([[height[0]]])
    assert policy.activate(state) is None
    policy.step(state)
    assert policy.target_velocity == 0
    height[0] = 0.185
    policy.step(state)
    np.testing.assert_allclose(policy.recovery_command, [[0.015, 0.01, 1, 0, 0.2]], atol=1e-7)


def test_gain_overrides_are_raw_and_debug_zero_action(deployment):
    config = replace(
        deployment.config,
        robot=replace(deployment.config.robot, motor_kp=(10.0,) * 29, motor_kd=(1.0,) * 29),
        task=replace(deployment.config.task, debug=replace(deployment.config.task.debug, force_zero_action=True)),
    )
    policy = PelvisRecoveryPolicy(config)
    state = _state(deployment.pose)
    assert policy.activate(state) is None
    command = policy.step(state)
    np.testing.assert_allclose(command.q, deployment.pose)
    np.testing.assert_array_equal(policy.actions.last, 0)
    np.testing.assert_allclose(command.kp[policy.bridge_ids], 3.5)
    np.testing.assert_allclose(command.kd[policy.bridge_ids], 0.6)


@pytest.mark.parametrize("change", ["width", "output", "order", "scale", "kp", "kd", "urdf", "missing_joint"])
def test_model_rejection(deployment, change):
    updates = {
        "order": {"dof_names": G1_29DOF.dof_names[::-1]},
        "scale": {"action_scale": 0.5},
        "kp": {"kp": [np.nan] * 29},
        "kd": {"kd": []},
        "urdf": {"robot_urdf": ""},
        "missing_joint": {"robot_urdf": _urdf().replace("left_hip_pitch_joint", "missing_joint")},
    }
    _model(
        deployment.model,
        width=98 if change == "width" else 99,
        output_width=28 if change == "output" else 29,
        metadata_update=updates.get(change),
    )
    with pytest.raises(ValueError):
        PelvisRecoveryPolicy(deployment.config)


@pytest.mark.parametrize(
    "change",
    ["body", "mask", "missing_model", "missing_motion", "guard", "rate", "unknown", "ankle_nan", "ankle_negative"],
)
def test_config_rejection(deployment, change):
    data = copy.deepcopy(deployment.data)
    if change == "body":
        data["type"] = "lower_body"
    elif change == "mask":
        data["task"]["action_mask_path"] = "mask.yaml"
    elif change in {"missing_model", "missing_motion"}:
        data["task"]["model_path" if change == "missing_model" else "motion_data_path"] = "missing.file"
    elif change == "guard":
        data.pop("guard")
    elif change == "rate":
        data["task"]["rl_rate"] = 100
    elif change.startswith("ankle_"):
        data["task"]["right_ankle_height_m"] = np.nan if change == "ankle_nan" else -0.1
    else:
        data["task"]["unexpected"] = True
    deployment.path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        runtime, path = load_runtime_config(deployment.path, ROOT / "configs/mqtt.yaml")
        resolve_policies(runtime, path)


@pytest.mark.parametrize(
    "change", ["history", "terms", "scale", "legacy_command", "sliders", "speed", "height", "pose_names", "pose_limit"]
)
def test_contract_rejection(deployment, change):
    config = deployment.config
    if change == "history":
        config = replace(config, observation=replace(config.observation, history_length_dict={"actor_obs": 2}))
    elif change == "legacy_command":
        config = replace(
            config,
            observation=replace(config.observation, obs_dims={**config.observation.obs_dims, "command": 4}),
        )
    elif change == "terms":
        config = replace(config, observation=replace(config.observation, obs_dict={"actor_obs": ["actions"]}))
    elif change == "scale":
        config = replace(
            config,
            observation=replace(
                config.observation, obs_scales={**config.observation.obs_scales, "joint_velocity": 1.0}
            ),
        )
    elif change == "sliders":
        config = replace(config, inputs=config.inputs[:-1])
    elif change == "speed":
        slider = config.inputs[0]
        slider = slider.model_copy(update={"parameter": slider.parameter.model_copy(update={"max": 0.5})})
        config = replace(config, inputs=(slider, config.inputs[1]))
    elif change == "height":
        slider = config.inputs[1]
        slider = slider.model_copy(update={"parameter": slider.parameter.model_copy(update={"min": 0.0})})
        config = replace(config, inputs=(config.inputs[0], slider))
    else:
        names = list(G1_29DOF.dof_names)
        pose = deployment.pose.copy()
        if change == "pose_names":
            names[0] = "wrong_joint"
        else:
            pose[0] = 10
        np.savez(deployment.motion, joint_names=names, joint_pos=np.r_[[0, 0, 0, 1, 0, 0, 0], pose][None])
    with pytest.raises(ValueError):
        PelvisRecoveryPolicy(config)


@pytest.mark.parametrize("change", ["joint", "gravity", "nan_velocity", "zero_quaternion"])
def test_startup_rejection(deployment, change):
    policy = PelvisRecoveryPolicy(deployment.config)
    state = _state(deployment.pose.copy())
    if change == "joint":
        state.joint_pos[0, 0] += 2
    elif change == "gravity":
        state.base_quat[:] = [0, 1, 0, 0]
    elif change == "nan_velocity":
        state.joint_vel[0, 0] = np.nan
    else:
        state.base_quat.fill(0)
    assert "pelvis_recovery_start_check_failed" in policy.activate(state)
    assert not policy.is_active


@pytest.mark.parametrize("fault", ["state", "command", "height", "nan_action", "shape", "inference"])
def test_runtime_faults(deployment, fault):
    policy = PelvisRecoveryPolicy(deployment.config)
    state = _state(deployment.pose)
    assert policy.activate(state) is None
    if fault == "command":
        with pytest.raises(PolicyRuntimeFault, match="invalid_command"):
            policy.apply_control({"peak_speed": np.nan, "target_height": 0.1})
        return
    if fault == "state":
        state.joint_vel[0, 0] = np.inf
    elif fault == "height":
        policy.kinematics.height_difference = lambda *args: np.array([[np.nan]])
    elif fault == "nan_action":
        policy.actor = lambda obs: np.full((1, 29), np.nan)
    elif fault == "shape":
        policy.actor = lambda obs: np.zeros((1, 28))
    else:

        def fail(obs):
            raise RuntimeError("inference failed")

        policy.actor = fail
    with pytest.raises(PolicyRuntimeFault):
        policy.step(state)


def test_runtime_registration_and_fault_without_write(deployment):
    state = _state(deployment.pose)
    writes = []
    transport = SimpleNamespace(
        close=lambda: None,
        publish_status=lambda *a: None,
        publish_state=lambda *a: None,
        publish_reference_state=lambda *a: None,
    )
    manager = SimpleNamespace(get_low_state=lambda: state, send_low_command=lambda *a, **kw: writes.append(a))
    machine = PolicyStateMachine(
        deployment.runtime, deployment.resolved, transport=transport, interface_manager=manager, clock=lambda: 0.0
    )
    machine._maybe_publish_state = lambda *args: None
    try:
        policy = machine.policies["g1-pelvis-recovery"]
        assert isinstance(policy, PelvisRecoveryPolicy)
        assert policy.activate(state) is None
        machine.state = type(machine.state).RUNNING
        machine.active_policy = ("g1-pelvis-recovery",)
        control = SimpleNamespace(
            policy=machine.active_policy,
            estop=False,
            inputs={"g1-pelvis-recovery": dict(peak_speed=0.175, target_height=0.2)},
        )
        machine.inbox = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(received_at=0.0, packet=SimpleNamespace(seq=1, control=control))
        )
        machine.tick(0.0)
        assert len(writes) == 1
        policy.actor = lambda obs: np.full((1, 29), np.nan)
        machine.tick(0.0)
        assert machine.state == "latched"
        assert "recovery_invalid_action" in machine.reason
        assert not policy.is_active
        assert len(writes) == 1
    finally:
        machine._policy_executor.shutdown(wait=True)
        machine._close_policies(machine.policies.values())
        transport.close()


def test_example_guard_covers_upstream_low_endpoint():
    # Optional local integration check; unit fixtures remain self-contained.
    from vex_policy.policies.guard.initial_pose import InitialPoseGuard
    from vex_policy.policies.utils.locomotion_utils import load_motion_last_pose

    data = yaml.safe_load(EXAMPLE.read_text())
    ref_path = (ROOT / data["task"]["motion_data_path"]).resolve()
    low_path = ref_path.with_name("ridding1_low.npz")
    if not ref_path.is_file() or not low_path.is_file():
        pytest.skip("Holosoma reference assets not installed")
    reference, low = load_motion_last_pose(ref_path), load_motion_last_pose(low_path)
    from loguru import logger

    guard = InitialPoseGuard(GuardConfig(**data["guard"]), reference, G1_29DOF.dof_names, logger)
    state = _state([low.dof_pos[low.dof_names.index(n)] for n in G1_29DOF.dof_names], low.root_quat_wxyz)
    assert guard.start_check(state) == (True, None)
    assert set(BRIDGE_JOINTS).issubset(G1_29DOF.dof_names)
