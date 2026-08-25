import numpy as np
import pytest

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.mqtt import CommandInbox
from vex_policy.policies.locomotion import LocomotionPolicy
from vex_policy.policies.sonic import SonicPolicy
from vex_policy.policies.switch_mode import SwitchModePolicy
from vex_policy.policies.waist_locomotion import WaistLocomotionPolicy
from vex_policy.policies.wbt import WholeBodyTrackingPolicy


class FakeInterface:
    def __init__(self, config):
        self.robot_config = config
        self.commands = []
        self.kp_level = 1.0
        self.kd_level = 1.0
        self.joint_positions = None
        self.projected_gravity = None

    def update_config(self, config):
        self.robot_config = config

    def get_low_state(self):
        count = self.robot_config.num_joints
        state = np.zeros((1, 3 + 4 + count + 3 + 3 + count), dtype=np.float32)
        state[0, 3] = 1.0
        joint_positions = self.robot_config.default_dof_angles if self.joint_positions is None else self.joint_positions
        state[0, 7 : 7 + count] = joint_positions
        if self.projected_gravity is not None:
            state = np.concatenate([state, np.asarray(self.projected_gravity, dtype=np.float32).reshape(1, 3)], axis=1)
        return state

    def send_low_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))


class NullTransport:
    def publish_status(self, payload):
        del payload

    def publish_state(self, payload):
        del payload

    def publish_reference_state(self, payload):
        del payload


@pytest.mark.parametrize("index", range(7))
def test_every_packaged_model_initializes_and_runs_one_step(index):
    runtime, path = load_runtime_config()
    resolved = resolve_policies(runtime, path)[index]
    config = resolved.config
    cls = {
        "locomotion": LocomotionPolicy,
        "sonic": SonicPolicy,
        "waist_locomotion": WaistLocomotionPolicy,
        "wbt": WholeBodyTrackingPolicy,
    }[resolved.kind]
    interface = FakeInterface(config.robot)
    policy = object.__new__(cls)
    policy._injected_interface = interface
    cls.__init__(policy, config)
    policy._on_command_sent = lambda *_: None
    if resolved.kind == "waist_locomotion":
        interface.joint_positions = policy.default_dof_angles
        interface.projected_gravity = policy.initial_pose.projected_gravity
    policy.activate()
    control = {parameter.name: parameter.default for parameter in resolved.spec.input_parameters}
    if "vx" in control:
        control["vx"] = 0.1
    policy.apply_control(control)
    policy.step()
    assert len(interface.commands) == 1
    assert np.isfinite(policy.cmd_q).all()
    if resolved.kind == "locomotion":
        assert np.allclose(policy.last_policy_action[0, policy.upper_dof_indices], 0.0)
        expected_upper = (
            policy.default_dof_angles[policy.upper_dof_indices] + policy.joint_offsets[policy.upper_dof_indices]
        )
        assert np.allclose(policy.cmd_q[policy.upper_dof_indices], expected_upper)
    if resolved.kind == "wbt":
        reference_state = policy.get_reference_state()
        assert reference_state.shape == (1, 36)
        assert np.allclose(reference_state[0, 7:], policy.motion_command_t[0, :29])
        ref_xyzw = np.asarray(policy.ref_quat_xyzw_t).reshape(4)
        assert np.allclose(reference_state[0, 3:7], ref_xyzw[[3, 0, 1, 2]])
        assert np.isfinite(reference_state).all()
    policy.close()


def test_switch_manager_preloads_all_models_on_one_interface():
    runtime, path = load_runtime_config()
    resolved = resolve_policies(runtime, path)
    interface = FakeInterface(resolved[0].config.robot)
    inbox = CommandInbox({item.spec.name: item.spec for item in resolved})
    manager = SwitchModePolicy(
        runtime,
        resolved,
        interface=interface,
        inbox=inbox,
        transport=NullTransport(),
    )
    try:
        assert len(manager.policies) == len(resolved)
        assert all(policy.interface is interface for policy in manager.policies.values())
        assert all(policy.robot_config.motor_kp is not None for policy in manager.policies.values())
    finally:
        for policy in manager.policies.values():
            policy.close()
