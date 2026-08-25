import json
from types import SimpleNamespace

import numpy as np

from vex_policy.config import ResolvedPolicy, load_runtime_config, resolve_policies
from vex_policy.mqtt import CommandInbox
from vex_policy.policies.base import PolicyJointCommand
from vex_policy.policies.switch_mode import SwitchModePolicy


class FakeInterface:
    def __init__(self):
        self.reads = 0
        self.commands = []
        self.no_action = 1

    def get_low_state(self):
        self.reads += 1
        return np.zeros((1, 7 + 29 + 6 + 29))

    def send_low_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))


class FakeLatency:
    def start_cycle(self):
        pass

    def end_cycle(self):
        pass


class FakePolicy:
    def __init__(self, interface, *, has_reference=False, controlled=None, value=0.0, gain=1.0):
        self.interface = interface
        self.dof_names = tuple(f"joint_{index}" for index in range(29))
        self.num_dofs = 29
        self.default_dof_angles = np.zeros(29)
        self.joint_offsets = np.zeros(29)
        self.controlled = np.ones(29, dtype=bool) if controlled is None else controlled
        self.value = value
        self.gain = gain
        self.latency_tracker = FakeLatency()
        self.use_phase = False
        self.config = SimpleNamespace(task=SimpleNamespace())
        self.events = []
        self.steps = 0
        self.control = {}
        self.has_reference = has_reference
        self._on_command_sent = lambda *_: None

    def _resolve_control_gains(self):
        self.events.append("gains")

    def activate(self):
        self.events.append("activate")

    def deactivate(self):
        self.events.append("deactivate")

    def apply_control(self, control):
        self.control = control

    def compute_joint_command(self, robot_state):
        del robot_state
        self.steps += 1
        return PolicyJointCommand(
            q=np.full(29, self.value),
            dq=np.zeros(29),
            tau=np.zeros(29),
            kp=np.full(29, self.gain),
            kd=np.full(29, self.gain / 10),
            controlled_joints=self.controlled.copy(),
        )

    def get_reference_state(self):
        if not self.has_reference:
            return None
        state = np.zeros((1, 7 + 29))
        state[0, 3] = 1.0
        state[0, 7:] = np.arange(29)
        return state


class FakeTransport:
    def __init__(self):
        self.statuses = []
        self.states = []
        self.reference_states = []

    def publish_status(self, payload):
        self.statuses.append(payload.copy())

    def publish_state(self, payload):
        self.states.append(json.loads(payload))

    def publish_reference_state(self, payload):
        self.reference_states.append(json.loads(payload))


def payload(seq, policy, specs, *, estop=False):
    inputs = {
        name: {parameter.name: parameter.default for parameter in specs[name].input_parameters} for name in policy
    }
    for values in inputs.values():
        values.update({name: value for name, value in {"vx": 0.4, "vy": 0.2, "yaw": -0.1}.items() if name in values})
    return json.dumps(
        {
            "seq": seq,
            "timestamp": 1,
            "control": {
                "policy": policy,
                "inputs": inputs,
                "estop": estop,
            },
        }
    )


def make_runtime():
    runtime, path = load_runtime_config()
    all_resolved = resolve_policies(runtime, path)
    resolved = (
        next(item for item in all_resolved if item.kind == "locomotion"),
        next(item for item in all_resolved if item.kind == "wbt"),
    )
    runtime = runtime.model_copy(update={"policies": tuple(item.spec for item in resolved)})
    clock_value = [0.0]

    def clock():
        return clock_value[0]

    inbox = CommandInbox({item.spec.name: item.spec for item in resolved}, clock=clock)
    interface = FakeInterface()
    instances = {item.spec.name: FakePolicy(interface, has_reference=item.kind == "wbt") for item in resolved}
    transport = FakeTransport()
    manager = SwitchModePolicy(
        runtime,
        resolved,
        instances=instances,
        inbox=inbox,
        transport=transport,
        clock=clock,
    )
    return manager, inbox, instances, clock_value


def test_startup_rearm_switch_and_estop_latch_never_step_while_latched():
    manager, inbox, policies, now = make_runtime()
    first, second = policies
    manager.tick()
    assert manager.state == "startup_latched"
    assert sum(policy.steps for policy in policies.values()) == 0

    assert inbox.accept(payload(1, [], manager._specs))
    manager.tick()
    assert manager.state == "idle"

    now[0] = 0.01
    assert inbox.accept(payload(2, [first], manager._specs))
    manager.tick()
    assert manager.active_policy == (first,)
    assert policies[first].steps == 0
    manager.tick(now=0.02)
    assert policies[first].steps == 1
    assert policies[first].control == {"vx": 0.4, "vy": 0.2, "yaw": -0.1}
    assert manager.transport.reference_states == []

    now[0] = 0.03
    assert inbox.accept(payload(3, [second], manager._specs))
    manager.tick()
    assert policies[first].events[-1] == "deactivate"
    assert policies[second].steps == 0
    manager.tick(now=0.04)
    assert policies[second].steps == 1
    assert len(manager.transport.reference_states) == 1
    assert manager.transport.reference_states[0]["joint_values"] == list(range(29))
    assert manager.transport.reference_states[0]["timestamp"] == manager.transport.states[-1]["timestamp"]

    now[0] = 0.05
    assert inbox.accept(payload(4, [second], manager._specs, estop=True))
    manager.tick()
    assert manager.state == "latched"
    steps = policies[second].steps
    now[0] = 0.06
    assert inbox.accept(payload(5, [second], manager._specs, estop=False))
    manager.tick()
    assert manager.state == "latched"
    assert policies[second].steps == steps

    now[0] = 0.07
    assert inbox.accept(payload(6, [], manager._specs))
    manager.tick()
    assert manager.state == "idle"
    now[0] = 0.08
    assert inbox.accept(payload(7, [first], manager._specs))
    manager.tick()
    assert manager.active_policy == (first,)


def test_timeout_latches_and_requires_empty_rearm():
    manager, inbox, policies, now = make_runtime()
    first = next(iter(policies))
    assert inbox.accept(payload(1, [], manager._specs))
    manager.tick()
    now[0] = 0.1
    assert inbox.accept(payload(2, [first], manager._specs))
    manager.tick()
    manager.tick(now=0.2)
    steps = policies[first].steps
    manager.tick(now=1.2)
    assert manager.state == "latched"
    assert manager.reason == "command_timeout"
    assert policies[first].steps == steps
    manager.tick(now=2.0)
    assert policies[first].steps == steps


def test_lower_and_upper_policies_share_state_and_publish_one_merged_command():
    runtime, path = load_runtime_config()
    resolved_all = resolve_policies(runtime, path)
    lower = next(item for item in resolved_all if item.spec.type == "lower_body")
    upper_source = next(item for item in resolved_all if item.kind == "wbt")
    upper_spec = upper_source.spec.model_copy(update={"name": "arms", "type": "upper_body"})
    upper = ResolvedPolicy(upper_spec, upper_source.config, upper_source.kind)
    resolved = (lower, upper)
    runtime = runtime.model_copy(update={"policies": (lower.spec, upper.spec)})
    interface = FakeInterface()
    lower_joints = np.zeros(29, dtype=bool)
    lower_joints[:15] = True
    upper_joints = ~lower_joints
    instances = {
        lower.spec.name: FakePolicy(interface, controlled=lower_joints, value=1.0, gain=10.0),
        upper.spec.name: FakePolicy(interface, controlled=upper_joints, value=2.0, gain=20.0),
    }
    inbox = CommandInbox({item.spec.name: item.spec for item in resolved}, clock=lambda: 0.0)
    transport = FakeTransport()
    manager = SwitchModePolicy(
        runtime,
        resolved,
        instances=instances,
        inbox=inbox,
        transport=transport,
        clock=lambda: 0.0,
    )

    assert inbox.accept(payload(1, [], manager._specs))
    manager.tick()
    assert inbox.accept(payload(2, [upper.spec.name, lower.spec.name], manager._specs))
    manager.tick()
    reads_before = interface.reads
    manager.tick()

    assert manager.active_policy == (lower.spec.name, upper.spec.name)
    assert interface.reads == reads_before + 1
    assert len(interface.commands) == 1
    args, kwargs = interface.commands[0]
    assert np.array_equal(args[0][:15], np.ones(15))
    assert np.array_equal(args[0][15:], np.full(14, 2.0))
    assert np.array_equal(kwargs["kp_override"][:15], np.full(15, 10.0))
    assert np.array_equal(kwargs["kp_override"][15:], np.full(14, 20.0))
    assert transport.reference_states == []
    assert transport.statuses[-1]["active_policy"] == [lower.spec.name, upper.spec.name]
