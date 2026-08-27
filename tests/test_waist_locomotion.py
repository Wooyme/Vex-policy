import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import (
    PolicySpec,
    WaistLocomotionGuardConfig,
    WaistLocomotionTaskConfig,
)
from vex_policy.policies.guard.waist_locomotion import WaistLocomotionGuard
from vex_policy.policies.switch_mode import _policy_class
from vex_policy.policies.waist_locomotion import WaistLocomotionPolicy, load_waist_motion_last_pose
from vex_policy.robots import G1_29DOF

EXAMPLE_PATH = Path("configs/examples/g1_waist_locomotion.yaml")
PRODUCTION_MODEL_PATH = Path("models/loco/g1_29dof/ppo_g1_waist.onnx")


class FakeInterface:
    def __init__(self, config, state):
        self.robot_config = config
        self.state = state
        self.commands = []

    def update_config(self, config):
        self.robot_config = config

    def get_low_state(self):
        return None if self.state is None else self.state.copy()

    def send_low_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))


def _make_model(path: Path, *, input_dim: int = 105, dof_names=G1_29DOF.dof_names) -> None:
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, input_dim])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    weights = numpy_helper.from_array(np.zeros((input_dim, 29), dtype=np.float32), name="weights")
    bias = numpy_helper.from_array(np.zeros(29, dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["actor_obs", "weights", "bias"], ["action"])],
        "waist-locomotion-test",
        [actor_obs],
        [action],
        [weights, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    production_model = onnx.load(PRODUCTION_MODEL_PATH, load_external_data=False)
    production_metadata = {prop.key: json.loads(prop.value) for prop in production_model.metadata_props}
    helper.set_model_props(
        model,
        {
            "dof_names": json.dumps(list(dof_names)),
            "action_scale": json.dumps([0.25] * 29),
            "kp": json.dumps([40.0] * 29),
            "kd": json.dumps([2.0] * 29),
            "robot_urdf": json.dumps(production_metadata["robot_urdf"]),
        },
    )
    onnx.save(model, path)


def _make_motion(path: Path, *, dof_names=G1_29DOF.dof_names) -> np.ndarray:
    hardware_values = np.arange(29, dtype=np.float64) * 0.01
    value_by_name = dict(zip(G1_29DOF.dof_names, hardware_values, strict=True))
    final_values = np.asarray([value_by_name[name] for name in dof_names], dtype=np.float64)
    joint_pos = np.zeros((2, 36), dtype=np.float64)
    joint_pos[0, 4] = 1.0  # first frame: 180 degrees about X in WXYZ order
    joint_pos[0, 7:] = -1.0
    joint_pos[1, 3] = 1.0  # final frame: identity quaternion in WXYZ order
    joint_pos[1, 7:] = final_values
    np.savez(path, joint_names=np.asarray(dof_names), joint_pos=joint_pos)
    return hardware_values


def _resolved_config(
    tmp_path: Path,
    *,
    input_dim: int = 105,
    model_dof_names=G1_29DOF.dof_names,
    motion_dof_names=G1_29DOF.dof_names,
):
    model_path = tmp_path / "waist.onnx"
    motion_path = tmp_path / "knee_down.npz"
    _make_model(model_path, input_dim=input_dim, dof_names=model_dof_names)
    _make_motion(motion_path, dof_names=motion_dof_names)
    data = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    data["task"]["model_path"] = str(model_path)
    data["task"]["motion_data_path"] = str(motion_path)
    config_path = tmp_path / "waist.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    runtime, loaded_path = load_runtime_config(config_path)
    return resolve_policies(runtime, loaded_path)[0].config


def _matching_state(config):
    pose = load_waist_motion_last_pose(config.task.motion_data_path)
    source_indices = {name: index for index, name in enumerate(pose.dof_names)}
    hardware_order = [source_indices[name] for name in config.robot.dof_names]
    state = np.zeros((1, 74), dtype=np.float32)
    state[0, 3] = 1.0
    state[0, 7:36] = np.asarray(pose.dof_pos)[hardware_order]
    state[0, 71:74] = pose.projected_gravity
    return state


def _build_policy(tmp_path: Path):
    config = _resolved_config(tmp_path)
    interface = FakeInterface(config.robot, _matching_state(config))
    policy = object.__new__(WaistLocomotionPolicy)
    policy._injected_interface = interface
    WaistLocomotionPolicy.__init__(policy, config)
    return policy, interface


def test_example_uses_specialized_parallel_policy_config():
    data = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    spec = PolicySpec.model_validate(data)

    assert spec.implementation == "waist_locomotion"
    assert spec.type == "full_body"
    assert [component.type for component in spec.inputs] == ["slider"] * 6
    assert [parameter.name for parameter in spec.input_parameters] == [
        "amplitude",
        "frequency",
        "x",
        "y",
        "z",
        "height_delta",
    ]
    assert isinstance(spec.task, WaistLocomotionTaskConfig)
    assert isinstance(spec.guard, WaistLocomotionGuardConfig)
    assert not hasattr(spec.task, "startup_joint_tolerance_rad")
    assert _policy_class(spec.implementation) is WaistLocomotionPolicy


def test_motion_loader_uses_last_frame_and_policy_reorders_joints(tmp_path):
    reversed_names = tuple(reversed(G1_29DOF.dof_names))
    config = _resolved_config(tmp_path, motion_dof_names=reversed_names)
    pose = load_waist_motion_last_pose(config.task.motion_data_path)
    interface = FakeInterface(config.robot, _matching_state(config))
    policy = object.__new__(WaistLocomotionPolicy)
    policy._injected_interface = interface
    WaistLocomotionPolicy.__init__(policy, config)

    assert pose.dof_names == reversed_names
    assert len(pose.dof_pos) == 29
    assert np.allclose(pose.root_quat_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(pose.projected_gravity, [0.0, 0.0, -1.0])
    assert np.allclose(policy.default_dof_angles, np.arange(29) * 0.01)


def test_observation_contract_and_control_mapping(tmp_path):
    policy, _ = _build_policy(tmp_path)
    state = _matching_state(policy.config)
    assert policy.activate() is None
    state[0, 39:42] = [0.4, -0.8, 1.2]
    state[0, 42:71] = np.arange(29, dtype=np.float32)
    policy.apply_control(
        {
            "amplitude": 0.2,
            "frequency": 2.0,
            "x": 0.4,
            "y": -0.3,
            "z": 0.5,
            "height_delta": 0.03,
        }
    )

    actor_obs = policy.prepare_obs_for_rl(state)["actor_obs"]

    expected_direction = np.asarray([0.4, -0.3, 0.5], dtype=np.float32)
    expected_direction /= np.linalg.norm(expected_direction)
    initial_height = policy.initial_base_right_foot_height_difference
    assert initial_height is not None
    assert actor_obs.shape == (1, 105)
    assert np.array_equal(actor_obs[0, :29], np.zeros(29))
    assert np.allclose(actor_obs[0, 29:32], [0.1, -0.2, 0.3])
    assert np.isclose(actor_obs[0, 32], initial_height)
    assert np.allclose(actor_obs[0, 33:62], np.zeros(29), atol=1e-6)
    assert np.allclose(actor_obs[0, 62:91], np.arange(29) * 0.05)
    assert np.allclose(actor_obs[0, 91:94], np.zeros(3))
    assert np.allclose(
        actor_obs[0, 94:102],
        [0.0, 1.0, 0.2, 2.0, *expected_direction, initial_height + 0.03],
    )
    assert np.allclose(actor_obs[0, 102:105], state[0, 71:74])


def test_base_right_foot_height_uses_ankle_forward_kinematics_and_gravity_projection(tmp_path):
    policy, _ = _build_policy(tmp_path)
    state = _matching_state(policy.config)
    state[0, 7:36] = 0.0

    upright_height = policy._base_right_foot_height_difference(state, np.asarray([[0.0, 0.0, -1.0]]))
    rolled_height = policy._base_right_foot_height_difference(state, np.asarray([[0.0, -1.0, 0.0]]))

    assert np.allclose(upright_height, [[0.7568637524]], atol=1e-9)
    assert np.allclose(rolled_height, [[0.118506455]], atol=1e-9)


def test_pelvis_orientation_error_uses_each_activation_quaternion_as_reference(tmp_path):
    policy, interface = _build_policy(tmp_path)
    state = _matching_state(policy.config)
    startup_angle = 0.5
    state[0, 3:7] = [np.cos(startup_angle / 2.0), 0.0, np.sin(startup_angle / 2.0), 0.0]
    interface.state = state
    assert policy.activate() is None

    actor_obs = policy.prepare_obs_for_rl(state)["actor_obs"]
    assert np.allclose(actor_obs[0, 91:94], np.zeros(3), atol=1e-6)

    current_angle = 0.9
    state[0, 3:7] = [np.cos(current_angle / 2.0), 0.0, np.sin(current_angle / 2.0), 0.0]

    actor_obs = policy.prepare_obs_for_rl(state)["actor_obs"]
    assert np.allclose(actor_obs[0, 91:94], [0.0, current_angle - startup_angle, 0.0], atol=1e-6)

    policy.deactivate()
    restarted_angle = -0.3
    state[0, 3:7] = [np.cos(restarted_angle / 2.0), 0.0, np.sin(restarted_angle / 2.0), 0.0]
    interface.state = state
    assert policy.activate() is None
    actor_obs = policy.prepare_obs_for_rl(state)["actor_obs"]

    assert np.allclose(actor_obs[0, 91:94], np.zeros(3), atol=1e-6)


def test_reactivation_resets_action_observation_and_history(tmp_path):
    policy, _ = _build_policy(tmp_path)
    state = _matching_state(policy.config)
    assert policy.activate() is None

    policy.last_policy_action.fill(1.25)
    policy.scaled_policy_action.fill(-0.5)
    for term, buffer in policy.obs_history_buffers["actor_obs"].items():
        buffer.append(np.ones((1, policy.obs_dims[term]), dtype=np.float32))
    policy.obs_buf_dict["actor_obs"].fill(2.0)

    policy.deactivate()
    assert policy.activate() is None

    assert np.array_equal(policy.last_policy_action, np.zeros((1, 29)))
    assert np.array_equal(policy.scaled_policy_action, np.zeros((1, 29)))
    assert all(not buffer for buffer in policy.obs_history_buffers["actor_obs"].values())
    assert np.count_nonzero(policy.obs_buf_dict["actor_obs"]) == 0
    actor_obs = policy.prepare_obs_for_rl(state)["actor_obs"]
    assert np.array_equal(actor_obs[0, :29], np.zeros(29))


def test_zero_direction_uses_default_command_and_one_step_is_finite(tmp_path):
    policy, interface = _build_policy(tmp_path)

    assert policy.activate() is None
    policy.apply_control(
        {
            "amplitude": 0.125,
            "frequency": 1.1,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "height_delta": 0.06,
        }
    )
    policy.step()

    expected_phase = 2.0 * np.pi * 1.1 / 50.0
    initial_height = policy.initial_base_right_foot_height_difference
    assert initial_height is not None
    assert np.allclose(
        policy.pelvis_sine_command,
        [[np.sin(expected_phase), np.cos(expected_phase), 0.125, 1.1, 1.0, 0.0, 0.0, initial_height + 0.06]],
    )
    assert len(interface.commands) == 1
    assert np.allclose(policy.cmd_q, policy.default_dof_angles)
    assert np.isfinite(policy.cmd_q).all()


def test_startup_guard_rejects_joint_and_orientation_mismatches(tmp_path):
    policy, interface = _build_policy(tmp_path)
    assert isinstance(policy.guard, WaistLocomotionGuard)
    interface.state[0, 7] += 0.21

    reason = policy.activate()

    assert "left_hip_pitch_joint" in reason
    assert not policy.use_policy_action

    interface.state = _matching_state(policy.config)
    interface.state[0, 71:74] = [0.0, 0.0, 1.0]
    reason = policy.activate()

    assert "projected_gravity" in reason
    assert not policy.use_policy_action


def test_model_contract_rejects_wrong_observation_size(tmp_path):
    config = _resolved_config(tmp_path, input_dim=104)
    interface = FakeInterface(config.robot, _matching_state(config))
    policy = object.__new__(WaistLocomotionPolicy)
    policy._injected_interface = interface

    with pytest.raises(ValueError, match=r"actor_obs\[1, 105\]"):
        WaistLocomotionPolicy.__init__(policy, config)


def test_model_contract_rejects_wrong_joint_order(tmp_path):
    config = _resolved_config(tmp_path, model_dof_names=tuple(reversed(G1_29DOF.dof_names)))
    interface = FakeInterface(config.robot, _matching_state(config))
    policy = object.__new__(WaistLocomotionPolicy)
    policy._injected_interface = interface

    with pytest.raises(ValueError, match="ONNX dof_names"):
        WaistLocomotionPolicy.__init__(policy, config)
