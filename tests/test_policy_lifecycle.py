from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vex_policy.config.config_types import HoldPositionTaskConfig, InferenceConfig, ObservationConfig
from vex_policy.policies.base import BasePolicy, PolicyLifecycleState, PolicyRuntimeFault
from vex_policy.policies.hold_position import HoldPositionPolicy
from vex_policy.policies.utils.inference import resolve_control_gains, shared_session
from vex_policy.policies.utils.joint_command import PositionAction, position_command
from vex_policy.policies.observations import ObservationHistory
from vex_policy.robots import G1_29DOF
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.latency import LatencyStage


def config():
    return InferenceConfig(
        robot=G1_29DOF, inputs=(), observation=ObservationConfig({}, {}, {}, {}), task=HoldPositionTaskConfig()
    )


def state(q=None):
    q = np.asarray(G1_29DOF.default_dof_angles if q is None else q, dtype=np.float64)
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=q.reshape(1, -1),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.zeros((1, 3)),
        joint_vel=np.zeros((1, q.size)),
    )


class ProbePolicy(BasePolicy):
    def __init__(self):
        super().__init__(config())
        self.events = []
        self.failure = None

    def _on_activate(self, robot_state_data):
        self.events.append("activate")
        if isinstance(self.failure, BaseException):
            raise self.failure
        return self.failure

    def _apply_control(self, control):
        self.events.append("control")

    def _compute_command(self, robot_state_data):
        self.events.append("step")
        return position_command(np.zeros(29), np.ones(29), np.ones(29), self.controlled_joint_mask)

    def _on_deactivate(self):
        self.events.append("deactivate")

    def _on_close(self):
        self.events.append("close")


def test_common_constructor_needs_no_model_or_observations():
    policy = ProbePolicy()
    assert not policy.is_active
    assert policy._lifecycle_state is PolicyLifecycleState.INACTIVE
    assert not hasattr(policy, "actor")
    assert not hasattr(policy, "phase")
    assert not hasattr(policy, "observations")
    assert policy.events == []


def test_lifecycle_idempotency_fresh_activation_and_terminal_close():
    policy = ProbePolicy()
    assert policy.activate(state()) is None
    policy.activate(state())
    assert policy._lifecycle_state is PolicyLifecycleState.ACTIVE
    policy.apply_control({})
    policy.step(state())
    policy.deactivate()
    policy.deactivate()
    assert policy._lifecycle_state is PolicyLifecycleState.INACTIVE
    policy.activate(state())
    policy.close()
    policy.close()
    policy.deactivate()
    assert policy._lifecycle_state is PolicyLifecycleState.CLOSED
    assert policy.events == ["activate", "control", "step", "deactivate", "activate", "deactivate", "close"]
    with pytest.raises(RuntimeError, match="closed"):
        policy.activate(state())
    for action in (lambda: policy.step(state()), lambda: policy.apply_control({})):
        with pytest.raises(RuntimeError, match="not active"):
            action()


@pytest.mark.parametrize("failure", ["rejected", PolicyRuntimeFault("bad data"), ValueError("bug")])
def test_activation_failure_rolls_back_and_allows_retry(failure):
    policy = ProbePolicy()
    policy.failure = failure
    if isinstance(failure, Exception):
        with pytest.raises(type(failure), match=str(failure)):
            policy.activate(state())
    else:
        assert policy.activate(state()) == failure
    assert not policy.is_active
    assert policy.events == ["activate", "deactivate"]
    policy.failure = None
    assert policy.activate(state()) is None
    policy.close()


def test_guard_rejection_does_not_enter_episode():
    policy = ProbePolicy()
    policy.guard = SimpleNamespace(start_check=lambda state: (False, "unsafe"))
    assert policy.activate(state()) == "unsafe"
    assert not policy.is_active
    assert policy.events == []
    policy.guard = None
    assert policy.activate(state()) is None
    policy.close()


def test_close_releases_instance_even_if_deactivation_fails():
    policy = ProbePolicy()
    policy.activate(state())

    def fail():
        raise ValueError("cleanup error")

    policy._on_deactivate = fail
    with pytest.raises(ValueError, match="cleanup error"):
        policy.close()
    policy.close()
    assert policy.events == ["activate", "close"]
    assert not policy.is_active


def test_step_failure_finishes_timing_and_reactivation_resets_timing():
    policy = ProbePolicy()
    policy.activate(state())

    def fail(state):
        raise PolicyRuntimeFault("invalid output")

    policy._compute_command = fail
    with pytest.raises(PolicyRuntimeFault):
        policy.step(state())
    assert policy.latency_tracker.get_stats()[LatencyStage.TOTAL].count == 1
    policy.deactivate()
    policy.activate(state())
    assert policy.latency_tracker.last_cycle_start_time is None
    assert not policy.latency_tracker.measurements
    policy.close()


def test_deactivation_waits_for_step_to_finish():
    policy = ProbePolicy()
    policy.activate(state())
    entered, finish, stopping, stopped = (threading.Event() for _ in range(4))

    def compute(state):
        entered.set()
        assert finish.wait(2)
        policy.events.append("finished_step")

    def deactivate():
        stopping.set()
        policy.deactivate()
        stopped.set()

    policy._compute_command = compute
    with ThreadPoolExecutor(max_workers=2) as pool:
        step = pool.submit(policy.step, state())
        assert entered.wait(2)
        stop = pool.submit(deactivate)
        try:
            assert stopping.wait(2)
            assert not stopped.wait(0.05)
        finally:
            finish.set()
        step.result()
        stop.result()
    assert policy.events == ["activate", "finished_step", "deactivate"]
    policy.close()


def test_hold_uses_fresh_physical_pose_without_double_offsets(monkeypatch):
    def no_model(*args, **kwargs):
        pytest.fail("model-free policy loaded ONNX")

    monkeypatch.setattr("onnxruntime.InferenceSession", no_model)
    robot = replace(G1_29DOF, joint_offsets_deg=(1.0,) * 29)
    policy = HoldPositionPolicy(replace(config(), robot=robot))
    first = state(np.arange(29) / 100)
    policy.activate(first)
    policy.activate(state(np.zeros(29)))
    np.testing.assert_allclose(policy.step(first).q + policy.joint_offsets, first.joint_pos[0])
    policy.deactivate()
    assert policy.held_dof_pos is None
    second = state(first.joint_pos[0] + 0.1)
    policy.activate(second)
    np.testing.assert_allclose(policy.step(second).q + policy.joint_offsets, second.joint_pos[0])
    policy.close()


def test_term_history_is_sorted_scaled_zero_padded_and_copies_inputs():
    history = ObservationHistory(
        ObservationConfig({"actor_obs": ["z", "a"]}, {"a": 2, "z": 1}, {"a": 2.0, "z": 3.0}, {"actor_obs": 2})
    )
    terms = {"a": np.array([[1.0, 2.0]]), "z": np.array([[3.0]])}
    first = history.prepare(terms)["actor_obs"]
    np.testing.assert_allclose(first, [[0, 0, 2, 4, 0, 9]])
    terms["a"][:] = 5
    second = history.prepare(terms)["actor_obs"]
    np.testing.assert_allclose(second, [[2, 4, 10, 10, 9, 9]])
    history.reset()
    np.testing.assert_allclose(history.prepare(terms)["actor_obs"], [[0, 0, 10, 10, 0, 9]])
    np.testing.assert_allclose(first, [[0, 0, 2, 4, 0, 9]])


def test_action_processing_preserves_mask_clip_scale_and_partial_padding():
    actions = PositionAction(3, np.array([[1, 0, 1]]), require_full_body=True)
    result = actions.process(np.array([[200, 5, -200]]), np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(actions.last, [[100, 0, -100]])
    np.testing.assert_allclose(result, [[10, 0, -30]])
    actions.reset()
    np.testing.assert_array_equal(actions.last, 0)
    with pytest.raises(ValueError, match="full-body"):
        actions.process(np.ones((1, 2)), 1)
    partial = PositionAction(3, np.ones((1, 3)))
    partial.process(np.array([[1, 2]]), 0.5)
    np.testing.assert_allclose(partial.target(np.ones(3)), [[1, 1.5, 2]])


def test_gain_precedence_and_shared_session_ownership(monkeypatch):
    kp, kd = [3.0] * 29, [0.3] * 29
    resolved = resolve_control_gains(replace(G1_29DOF, motor_kp=None, motor_kd=None), kp, kd)
    assert resolved.motor_kp == tuple(kp)
    override = replace(G1_29DOF, motor_kp=(4.0,) * 29, motor_kd=(0.4,) * 29)
    assert resolve_control_gains(override, kp, kd).motor_kp == override.motor_kp
    calls = []

    def session(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr("onnxruntime.InferenceSession", session)
    a = shared_session("lifecycle-test.onnx", ["CPUExecutionProvider"])
    assert a is shared_session("lifecycle-test.onnx", ["CPUExecutionProvider"])
    assert a is not shared_session("lifecycle-test.onnx", ["CUDAExecutionProvider"])
    assert len(calls) == 2
