from pathlib import Path

import numpy as np
import yaml

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import ActionMaskConfig, HoldPositionTaskConfig
from vex_policy.policies.hold_position import HoldPositionPolicy
from vex_policy.policies.switch_mode import _policy_class

EXAMPLE_PATH = Path("configs/examples/g1_hold_position.yaml")


class FakeInterface:
    def __init__(self, config, dof_pos):
        self.robot_config = config
        self.dof_pos = np.asarray(dof_pos, dtype=np.float64)
        self.commands = []
        self.writer_started = 0
        self.writer_stopped = 0
        self.no_action = 1

    def update_config(self, config):
        self.robot_config = config

    def configure_writer(self, publish_rate, timeout):
        self.writer_config = (publish_rate, timeout)

    def start_command_writer(self):
        self.writer_started += 1

    def stop_command_writer(self):
        self.writer_stopped += 1

    def get_low_state(self):
        state = np.zeros((1, 7 + self.robot_config.num_joints + 6 + self.robot_config.num_joints))
        state[0, 3] = 1.0
        state[0, 7 : 7 + self.robot_config.num_joints] = self.dof_pos
        return state

    def send_low_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))


def build_policy(config, interface):
    policy = object.__new__(HoldPositionPolicy)
    policy._injected_interface = interface
    HoldPositionPolicy.__init__(policy, config)
    return policy


def test_example_resolves_without_a_model_and_registers_the_policy():
    runtime, path = load_runtime_config(EXAMPLE_PATH)
    resolved = resolve_policies(runtime, path)[0]

    assert isinstance(resolved.config.task, HoldPositionTaskConfig)
    assert not hasattr(resolved.config.task, "model_path")
    assert resolved.spec.inputs == ()
    assert _policy_class("hold_position") is HoldPositionPolicy


def test_policy_holds_activation_pose_and_recaptures_after_reactivation():
    runtime, path = load_runtime_config(EXAMPLE_PATH)
    config = resolve_policies(runtime, path)[0].config
    first_pose = np.linspace(-0.4, 0.4, config.robot.num_joints)
    interface = FakeInterface(config.robot, first_pose)
    policy = build_policy(config, interface)

    assert not hasattr(policy, "onnx_policy_session")
    assert policy.activate() is None
    interface.dof_pos = np.linspace(0.7, 1.0, config.robot.num_joints)
    policy.step()

    sent_q = interface.commands[-1][0][0]
    assert np.allclose(sent_q, first_pose)
    assert np.allclose(policy.held_dof_pos, first_pose)
    assert interface.writer_started == 1

    policy.deactivate()
    second_pose = np.linspace(-1.0, -0.7, config.robot.num_joints)
    interface.dof_pos = second_pose
    assert policy.activate() is None
    command = policy.compute_joint_command(interface.get_low_state())
    assert np.allclose(command.q + policy.joint_offsets, second_pose)
    assert np.array_equal(command.dq, np.zeros(config.robot.num_joints))
    assert np.array_equal(command.tau, np.zeros(config.robot.num_joints))


def test_action_mask_controls_held_joints(tmp_path):
    data = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    masked_joint = "left_shoulder_pitch_joint"
    mask_path = tmp_path / "mask.yaml"
    mask_path.write_text(yaml.safe_dump({"masked_joints": [masked_joint]}), encoding="utf-8")
    data["task"]["action_mask_path"] = str(mask_path)
    config_path = tmp_path / "hold.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    runtime, path = load_runtime_config(config_path)
    resolved = resolve_policies(runtime, path)[0]
    interface = FakeInterface(resolved.config.robot, np.zeros(resolved.config.robot.num_joints))
    policy = build_policy(resolved.config, interface)

    assert resolved.config.action_mask == ActionMaskConfig(masked_joints=(masked_joint,))
    assert not policy.controlled_joint_mask[policy.dof_names.index(masked_joint)]
    assert policy.controlled_joint_mask.sum() == policy.num_dofs - 1


def test_activation_rejects_unavailable_or_invalid_state():
    runtime, path = load_runtime_config(EXAMPLE_PATH)
    config = resolve_policies(runtime, path)[0].config
    interface = FakeInterface(config.robot, np.zeros(config.robot.num_joints))
    policy = build_policy(config, interface)

    interface.get_low_state = lambda: None
    assert policy.activate() == "hold_position_start_failed: low_state_unavailable"
    interface.get_low_state = lambda: np.zeros((1, 7))
    assert policy.activate().startswith("hold_position_start_failed: invalid_low_state_shape=")

    invalid_state = np.zeros((1, 7 + config.robot.num_joints))
    invalid_state[0, 7] = np.nan
    interface.get_low_state = lambda: invalid_state
    assert policy.activate() == "hold_position_start_failed: invalid_dof_pos"
