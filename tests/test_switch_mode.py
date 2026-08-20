import json

import numpy as np

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.inputs import ControlValues
from vex_policy.mqtt import CommandInbox
from vex_policy.policies.switch_mode import SwitchModePolicy


class FakeInterface:
    def __init__(self):
        self.reads = 0

    def get_low_state(self):
        self.reads += 1
        return np.zeros((1, 7 + 29 + 6 + 29))


class FakePolicy:
    def __init__(self, interface, *, has_reference=False):
        self.interface = interface
        self.dof_names = tuple(f"joint_{index}" for index in range(29))
        self.events = []
        self.steps = 0
        self.control = ControlValues()
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

    def step(self):
        self.steps += 1
        self._on_command_sent(None, self.interface.get_low_state())

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


def payload(seq, policy, *, estop=False):
    return json.dumps(
        {
            "seq": seq,
            "timestamp": 1,
            "control": {
                "vx": 0.4,
                "vy": 0.2,
                "yaw": -0.1,
                "pitch": 0.3,
                "height": 0.0,
                "policy": policy,
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

    inbox = CommandInbox([item.spec.name for item in resolved], clock=clock)
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

    assert inbox.accept(payload(1, []))
    manager.tick()
    assert manager.state == "idle"

    now[0] = 0.01
    assert inbox.accept(payload(2, [first]))
    manager.tick()
    assert manager.active_policy == first
    assert policies[first].steps == 0
    manager.tick(now=0.02)
    assert policies[first].steps == 1
    assert policies[first].control == ControlValues(vx=0.4, vy=0.2, yaw=-0.1)
    assert manager.transport.reference_states == []

    now[0] = 0.03
    assert inbox.accept(payload(3, [second]))
    manager.tick()
    assert policies[first].events[-1] == "deactivate"
    assert policies[second].steps == 0
    manager.tick(now=0.04)
    assert policies[second].steps == 1
    assert len(manager.transport.reference_states) == 1
    assert manager.transport.reference_states[0]["joint_values"] == list(range(29))
    assert manager.transport.reference_states[0]["timestamp"] == manager.transport.states[-1]["timestamp"]

    now[0] = 0.05
    assert inbox.accept(payload(4, [second], estop=True))
    manager.tick()
    assert manager.state == "latched"
    steps = policies[second].steps
    now[0] = 0.06
    assert inbox.accept(payload(5, [second], estop=False))
    manager.tick()
    assert manager.state == "latched"
    assert policies[second].steps == steps

    now[0] = 0.07
    assert inbox.accept(payload(6, []))
    manager.tick()
    assert manager.state == "idle"
    now[0] = 0.08
    assert inbox.accept(payload(7, [first]))
    manager.tick()
    assert manager.active_policy == first


def test_timeout_latches_and_requires_empty_rearm():
    manager, inbox, policies, now = make_runtime()
    first = next(iter(policies))
    assert inbox.accept(payload(1, []))
    manager.tick()
    now[0] = 0.1
    assert inbox.accept(payload(2, [first]))
    manager.tick()
    manager.tick(now=0.2)
    steps = policies[first].steps
    manager.tick(now=1.2)
    assert manager.state == "latched"
    assert manager.reason == "command_timeout"
    assert policies[first].steps == steps
    manager.tick(now=2.0)
    assert policies[first].steps == steps
