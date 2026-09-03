from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from vex_policy.policies.base import PolicyJointCommand, PolicyRuntimeFault
from vex_policy.policies.policy_state_machine import PolicyStateMachine
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.sdk.interface_manager import InterfaceManager


def _low_state() -> LowState:
    joint_pos = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=joint_pos,
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.zeros((1, 3)),
        joint_vel=np.zeros((1, 3)),
    )


class _FakeInterface:
    def __init__(self, state: LowState):
        self.state = state
        self.reads = 0
        self.commands = []
        self.closed = 0

    def get_low_state(self):
        self.reads += 1
        return self.state

    def send_low_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))

    def close(self):
        self.closed += 1


def test_interface_manager_is_the_process_singleton():
    InterfaceManager.close()
    state = _low_state()
    backend = _FakeInterface(state)
    other_backend = _FakeInterface(state)
    robot_config = SimpleNamespace()

    manager = InterfaceManager.initialize(robot_config, interface=backend)
    assert InterfaceManager.get() is manager
    assert InterfaceManager.initialize(robot_config, interface=other_backend) is manager
    assert manager.get_low_state() is state
    manager.send_low_command(np.zeros(3), np.zeros(3), np.zeros(3))

    assert backend.reads == 1
    assert len(backend.commands) == 1
    assert other_backend.reads == 0

    InterfaceManager.close()
    assert backend.closed == 1
    with pytest.raises(RuntimeError, match="has not been initialized"):
        InterfaceManager.get()


class _ParallelPolicy:
    def __init__(self, barrier: threading.Barrier, command: PolicyJointCommand):
        self.barrier = barrier
        self.command = command
        self.seen_states: list[LowState] = []
        self.applied = []
        self.activated: list[LowState] = []
        self.num_dofs = 3
        self.dof_names = ("a", "b", "c")
        self.default_dof_angles = np.zeros(3)
        self.joint_offsets = np.asarray([0.1, 0.1, 0.1])

    def step(self, robot_state: LowState) -> PolicyJointCommand:
        self.seen_states.append(robot_state)
        self.barrier.wait(timeout=2.0)
        return self.command

    def activate(self, robot_state: LowState):
        self.activated.append(robot_state)

    def deactivate(self):
        pass

    def apply_control(self, values):
        self.applied.append(values)


def _command(q, controlled, gain) -> PolicyJointCommand:
    return PolicyJointCommand(
        q=np.asarray(q, dtype=np.float64),
        dq=np.asarray(q, dtype=np.float64) + 10,
        tau=np.asarray(q, dtype=np.float64) + 20,
        kp=np.full(3, gain, dtype=np.float64),
        kd=np.full(3, gain + 1, dtype=np.float64),
        controlled_joints=np.asarray(controlled, dtype=bool),
    )


def _state_machine(manager, lower, upper) -> PolicyStateMachine:
    machine = object.__new__(PolicyStateMachine)
    machine.runtime = SimpleNamespace(mqtt=SimpleNamespace(command_timeout_s=1.0))
    machine._clock = lambda: 0.0
    machine._specs = {
        "lower": SimpleNamespace(type="lower_body"),
        "upper": SimpleNamespace(type="upper_body"),
    }
    machine.interface_manager = manager
    machine.policies = {"lower": lower, "upper": upper}
    machine.dof_names = lower.dof_names
    machine._policy_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-policy")
    machine._started_at = 0.0
    machine._state_period = 0.02
    machine._next_state_publish = 0.0
    machine.state = "running"
    machine.active_policy = ("lower", "upper")
    machine.requested_policy = ()
    machine.reason = None
    machine.last_command_seq = None
    machine._last_status = None
    machine._publish_status = lambda **kwargs: None
    machine._maybe_publish_state = lambda *args, **kwargs: None
    machine._publish_idle_state = lambda *args, **kwargs: None
    return machine


def test_two_policies_compute_in_parallel_and_write_one_merged_command():
    state = _low_state()
    backend = _FakeInterface(state)
    manager = InterfaceManager(backend)
    barrier = threading.Barrier(2)
    lower = _ParallelPolicy(barrier, _command([1, 99, 99], [True, False, False], 10))
    upper = _ParallelPolicy(barrier, _command([88, 2, 3], [False, True, True], 20))
    machine = _state_machine(manager, lower, upper)

    try:
        machine._step_active_policies(state)
    finally:
        machine._policy_executor.shutdown(wait=True)

    assert lower.seen_states == [state]
    assert upper.seen_states == [state]
    assert len(backend.commands) == 1
    args, kwargs = backend.commands[0]
    np.testing.assert_allclose(args[0], [1.1, 2.1, 3.1])
    np.testing.assert_allclose(args[1], [11, 12, 13])
    np.testing.assert_allclose(args[2], [21, 22, 23])
    np.testing.assert_allclose(kwargs["kp_override"], [10, 20, 20])
    np.testing.assert_allclose(kwargs["kd_override"], [11, 21, 21])


def test_running_tick_reads_low_state_once_for_both_policies():
    state = _low_state()
    backend = _FakeInterface(state)
    manager = InterfaceManager(backend)
    barrier = threading.Barrier(2)
    lower = _ParallelPolicy(barrier, _command([1, 99, 99], [True, False, False], 10))
    upper = _ParallelPolicy(barrier, _command([88, 2, 3], [False, True, True], 20))
    machine = _state_machine(manager, lower, upper)
    control = SimpleNamespace(
        policy=("upper", "lower"),
        estop=False,
        inputs={"lower": {"value": 1}, "upper": {"value": 2}},
    )
    machine.inbox = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            received_at=0.0,
            packet=SimpleNamespace(seq=7, control=control),
        )
    )

    try:
        machine.tick(now=0.0)
    finally:
        machine._policy_executor.shutdown(wait=True)

    assert backend.reads == 1
    assert len(backend.commands) == 1
    assert lower.seen_states == [state]
    assert upper.seen_states == [state]


def test_activation_reuses_the_tick_state_snapshot():
    state = _low_state()
    backend = _FakeInterface(state)
    manager = InterfaceManager(backend)
    barrier = threading.Barrier(2)
    lower = _ParallelPolicy(barrier, _command([1, 99, 99], [True, False, False], 10))
    upper = _ParallelPolicy(barrier, _command([88, 2, 3], [False, True, True], 20))
    machine = _state_machine(manager, lower, upper)
    machine.state = "idle"
    machine.active_policy = ()
    control = SimpleNamespace(
        policy=("upper", "lower"),
        estop=False,
        inputs={"lower": {}, "upper": {}},
    )
    machine.inbox = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            received_at=0.0,
            packet=SimpleNamespace(seq=8, control=control),
        )
    )

    try:
        machine.tick(now=0.0)
    finally:
        machine._policy_executor.shutdown(wait=True)

    assert backend.reads == 1
    assert not backend.commands
    assert lower.activated == [state]
    assert upper.activated == [state]


def test_policy_runtime_fault_latches_without_writing_command():
    state = _low_state()
    backend = _FakeInterface(state)
    manager = InterfaceManager(backend)
    policy = _ParallelPolicy(threading.Barrier(1), _command([1, 2, 3], [True, True, True], 10))
    policy.step = lambda robot_state: (_ for _ in ()).throw(PolicyRuntimeFault("invalid_action"))
    machine = _state_machine(manager, policy, policy)
    machine.policies = {"lower": policy}
    machine.active_policy = ("lower",)
    control = SimpleNamespace(policy=("lower",), estop=False, inputs={"lower": {}})
    machine.inbox = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            received_at=0.0,
            packet=SimpleNamespace(seq=9, control=control),
        )
    )

    try:
        machine.tick(now=0.0)
    finally:
        machine._policy_executor.shutdown(wait=True)

    assert machine.state == "latched"
    assert machine.active_policy == ()
    assert machine.reason == "policy_fault:lower:invalid_action"
    assert not backend.commands
