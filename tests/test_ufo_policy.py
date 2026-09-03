from __future__ import annotations

from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from vex_policy.config.config_types import InferenceConfig, PolicySpec
from vex_policy.policies import ufo
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.robots import G1_29DOF
from vex_policy.sdk.base.base_interface import LowState


class _FakeSession:
    def __init__(self, output: np.ndarray | None = None):
        self.output = np.zeros((1, 29), dtype=np.float32) if output is None else output
        self.feeds: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="actor_obs", shape=[1, 721], type="tensor(float)")]

    def get_outputs(self):
        return [SimpleNamespace(name="action", shape=[1, 29], type="tensor(float)")]

    def run(self, output_names, feed):
        assert output_names == ["action"]
        self.feeds.append(feed["actor_obs"].copy())
        return [self.output.copy()]


def _observation_config():
    return {
        "obs_dict": {"actor_obs": list(ufo._ACTOR_TERMS)},
        "obs_dims": dict(ufo._OBS_DIMS),
        "obs_scales": {
            "dof_pos_minus_default": 1.0,
            "dof_vel": 1.0,
            "projected_gravity": 1.0,
            "base_ang_vel": 0.25,
            "prev_actions": 1.0,
            "prev_actions_history": 1.0,
            "base_ang_vel_history": 0.25,
            "dof_pos_minus_default_history": 1.0,
            "dof_vel_history": 1.0,
            "projected_gravity_history": 1.0,
        },
        "history_length_dict": {"actor_obs": 1},
    }


def _config(
    context: dict,
    *,
    init_duration_s: float = 0.02,
    slew_factor: float = 100.0,
    startup_mode: str = "interpolate",
) -> InferenceConfig:
    spec = PolicySpec.model_validate(
        {
            "name": "ufo-test",
            "implementation": "ufo",
            "type": "full_body",
            "inputs": [],
            "observation": _observation_config(),
            "guard": {
                "startup_joint_tolerance_rad": 0.2,
                "startup_gravity_tolerance": 0.2,
            },
            "task": {
                "model_path": "unused.onnx",
                "context": context,
                "startup_mode": startup_mode,
                "init_duration_s": init_duration_s,
                "q_target_slew_safety_factor": slew_factor,
            },
        }
    )
    return InferenceConfig(
        robot=G1_29DOF,
        inputs=spec.inputs,
        observation=spec.observation,
        task=spec.task,
        guard=spec.guard,
    )


def _state(joint_pos: np.ndarray | None = None, base_quat: np.ndarray | None = None) -> LowState:
    joint_pos = ufo._DEFAULT_DOF_ANGLES if joint_pos is None else np.asarray(joint_pos, dtype=np.float64)
    base_quat = np.asarray((1.0, 0.0, 0.0, 0.0)) if base_quat is None else np.asarray(base_quat, dtype=np.float64)
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=base_quat.reshape(1, 4),
        joint_pos=joint_pos.reshape(1, 29),
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.zeros((1, 3)),
        joint_vel=np.zeros((1, 29)),
    )


def test_ufo_observation_history_and_action_contract(tmp_path, monkeypatch):
    context_path = tmp_path / "reward_numpy.pkl"
    joblib.dump({"forward": [np.full((1, 256), 2.0, dtype=np.float32)]}, context_path)
    session = _FakeSession(np.full((1, 29), 2.0, dtype=np.float32))
    monkeypatch.setattr(ufo, "_shared_session", lambda path, provider: session)
    policy = ufo.UfoPolicy(_config({"type": "reward", "path": str(context_path), "name": "forward", "z_id": 0}))

    assert policy.activate(_state()) is None
    initialization_command = policy.step(_state())
    policy_command = policy.step(_state())

    assert session.feeds[0].shape == (1, 721)
    np.testing.assert_allclose(session.feeds[0][0, 64:209], 0.0)
    np.testing.assert_allclose(session.feeds[0][0, 465:], 2.0)
    np.testing.assert_allclose(session.feeds[1][0, 64:93], 5.0)
    np.testing.assert_allclose(session.feeds[1][0, 93:122], 5.0)
    np.testing.assert_allclose(session.feeds[1][0, 122:209], 0.0)
    np.testing.assert_allclose(initialization_command.q, ufo._DEFAULT_DOF_ANGLES)
    expected = np.clip(ufo._DEFAULT_DOF_ANGLES + 5.0 * ufo._ACTION_SCALE, ufo._JOINT_LOWER, ufo._JOINT_UPPER)
    np.testing.assert_allclose(policy_command.q, expected)
    np.testing.assert_allclose(policy_command.kp, ufo._KP)
    np.testing.assert_allclose(policy_command.kd, ufo._KD)


def test_tracking_plays_once_then_uses_stop_frame(tmp_path, monkeypatch):
    context = np.zeros((4, 256), dtype=np.float32)
    for index in range(4):
        context[index, index] = 1.0
    context_path = tmp_path / "tracking.pkl"
    joblib.dump(context, context_path)
    session = _FakeSession()
    monkeypatch.setattr(ufo, "_shared_session", lambda path, provider: session)
    policy = ufo.UfoPolicy(
        _config(
            {
                "type": "tracking",
                "path": str(context_path),
                "start_frame": 1,
                "end_frame": 3,
                "stop_frame": 0,
                "window_size": 1,
            }
        )
    )

    assert policy.activate(_state()) is None
    for _ in range(4):
        policy.step(_state())

    np.testing.assert_allclose(session.feeds[0][0, 465:], context[0])
    np.testing.assert_allclose(session.feeds[1][0, 465:], context[1])
    np.testing.assert_allclose(session.feeds[2][0, 465:], context[2])
    np.testing.assert_allclose(session.feeds[3][0, 465:], context[0])


def test_non_finite_ufo_action_becomes_runtime_fault(tmp_path, monkeypatch):
    context_path = tmp_path / "goal.pkl"
    joblib.dump({"stand": np.zeros((1, 256), dtype=np.float32)}, context_path)
    output = np.zeros((1, 29), dtype=np.float32)
    output[0, 0] = np.nan
    monkeypatch.setattr(ufo, "_shared_session", lambda path, provider: _FakeSession(output))
    policy = ufo.UfoPolicy(_config({"type": "goal", "path": str(context_path), "name": "stand"}))

    assert policy.activate(_state()) is None
    with pytest.raises(PolicyRuntimeFault, match="ufo_non_finite_action"):
        policy.step(_state())


def test_reward_context_rejects_missing_selection(tmp_path, monkeypatch):
    context_path = tmp_path / "reward.pkl"
    joblib.dump({"known": [np.zeros((1, 256), dtype=np.float32)]}, context_path)
    monkeypatch.setattr(ufo, "_shared_session", lambda path, provider: _FakeSession())

    with pytest.raises(ValueError, match="does not contain 'missing'"):
        ufo.UfoPolicy(_config({"type": "reward", "path": str(context_path), "name": "missing"}))


def test_prefill_starts_immediately_with_actual_pose_history(tmp_path, monkeypatch):
    context_path = tmp_path / "goal.pkl"
    joblib.dump({"stand": np.zeros((1, 256), dtype=np.float32)}, context_path)
    session = _FakeSession()
    monkeypatch.setattr(ufo, "_shared_session", lambda path, provider: session)
    policy = ufo.UfoPolicy(
        _config(
            {"type": "goal", "path": str(context_path), "name": "stand"},
            startup_mode="prefill",
        )
    )
    startup_action = np.full(29, 0.25, dtype=np.float64)
    startup_q = ufo._DEFAULT_DOF_ANGLES + startup_action * ufo._ACTION_SCALE
    state = _state(startup_q)

    assert policy.activate(state) is None
    command = policy.step(state)

    assert not policy._initializing
    np.testing.assert_allclose(session.feeds[0][0, 64:93], startup_action)
    np.testing.assert_allclose(session.feeds[0][0, 93:209], np.tile(startup_action, 4))
    np.testing.assert_allclose(command.q, ufo._DEFAULT_DOF_ANGLES)


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (_state(np.zeros(29)), "left_knee_joint error=0.300rad"),
        (_state(ufo._DEFAULT_DOF_ANGLES, np.asarray((0.0, 1.0, 0.0, 0.0))), "projected_gravity error=2.000"),
    ],
)
@pytest.mark.parametrize("startup_mode", ["prefill", "interpolate"])
def test_ufo_guard_rejects_unsafe_startup_pose(tmp_path, monkeypatch, state, reason, startup_mode):
    context_path = tmp_path / "goal.pkl"
    joblib.dump({"stand": np.zeros((1, 256), dtype=np.float32)}, context_path)
    monkeypatch.setattr(ufo, "_shared_session", lambda path, provider: _FakeSession())
    policy = ufo.UfoPolicy(
        _config(
            {"type": "goal", "path": str(context_path), "name": "stand"},
            startup_mode=startup_mode,
        )
    )

    assert reason in policy.activate(state)


def test_ufo_startup_mode_rejects_old_guarded_prefill_name():
    with pytest.raises(ValueError, match="startup_mode"):
        _config(
            {"type": "goal", "path": "unused.pkl", "name": "stand"},
            startup_mode="guarded_prefill",
        )
