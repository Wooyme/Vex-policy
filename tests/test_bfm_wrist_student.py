"""BFM wrist student: goal-free codebook control from six frames, the gait clock and the previous latent."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest
import yaml
from bfm_helpers import CODEBOOK, TOKEN, bfm_frame, build_policy, low_state, patch_sessions

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import BfmTaskConfig
from vex_policy.policies.bfm import BfmWristStudentPolicy
from vex_policy.policy_state_machine import _policy_class

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/examples/bfm/bfm_wrist_student.yaml"
WIDTH = 818


@pytest.fixture
def make_policy(tmp_path, monkeypatch):
    patch_sessions(monkeypatch)

    def make(metadata_change=None, doc_change=None):
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        return build_policy(BfmWristStudentPolicy, EXAMPLE, WIDTH, folder, metadata_change, doc_change)

    return make


def test_frame_history_and_running_clock(make_policy):
    policy, _ = make_policy()
    states = [low_state(rpy=(0.01 * step, 0.0, 0.0)) for step in range(9)]
    policy.activate(states[0])
    for state in states:
        policy.step(state)
    first, second, third = (feed["actor_obs"] for feed in policy.controller.feeds)
    assert first.shape == (1, WIDTH)
    # Holosoma zero-pads the oldest-first history after a reset and appends once per controller step.
    frames = [feed[:, :558].reshape(6, 93) for feed in (first, second, third)]
    assert not frames[0][:5].any()
    np.testing.assert_array_equal(frames[0][5], bfm_frame(states[0], np.zeros((1, 29)))[0])
    motor = np.linspace(-1, 1, 29, dtype=np.float32)[None]
    np.testing.assert_array_equal(frames[1][5], bfm_frame(states[4], motor)[0])
    np.testing.assert_array_equal(frames[2][3:5], frames[1][4:6])
    assert not frames[2][:3].any()
    # Without inputs the gait clock always runs: cos then sin of offsets [0, -pi] at 12.5 Hz.
    for index, feed in enumerate((first, second, third)):
        phase = index * 0.08 * 2 * np.pi + np.array([0.0, -np.pi])
        np.testing.assert_allclose(feed[0, 558:562], np.concatenate((np.cos(phase), np.sin(phase))), atol=1e-6)
    assert not first[:, -256:].any()
    np.testing.assert_array_equal(third[:, -256:], policy.latent)
    assert policy.token == TOKEN
    np.testing.assert_allclose(policy.latent[0], CODEBOOK[TOKEN] / np.linalg.norm(CODEBOOK[TOKEN]) * 16, rtol=1e-5)
    policy.deactivate()
    assert not policy.controller_frames.any() and policy.token == -1


def test_history_and_inputs_are_checked(make_policy):
    def change(metadata):
        metadata["experiment_config"]["observation"]["groups"]["student_obs"]["history_length"] = 4

    with pytest.raises(ValueError, match="student_obs is incompatible"):
        make_policy(change)
    with pytest.raises(ValueError, match="history length 6"):
        make_policy(doc_change=lambda doc: doc["observation"]["history_length_dict"].update(student_obs=4))
    slider = {"type": "slider", "parameter": {"name": "walk", "min": 0.0, "max": 1.0, "default": 0.0}}
    with pytest.raises(ValueError, match="inputs"):
        make_policy(doc_change=lambda doc: doc.update(inputs=[slider]))


def test_example_resolution_and_registry(make_policy, tmp_path):
    _, doc = make_policy()
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(doc))
    runtime, path = load_runtime_config(policy_file)
    assert isinstance(resolve_policies(runtime, path)[0].config.task, BfmTaskConfig)
    assert _policy_class("bfm_wrist_student") is BfmWristStudentPolicy
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    entry_point = project["project"]["entry-points"]["vex_policy.policies"]["bfm_wrist_student"]
    assert entry_point.endswith(":BfmWristStudentPolicy")
    doc["inputs"] = [{"type": "slider", "parameter": {"name": "walk", "min": 0.0, "max": 1.0, "default": 0.0}}]
    policy_file.write_text(yaml.safe_dump(doc))
    runtime, path = load_runtime_config(policy_file)
    with pytest.raises(ValueError, match="hand tether"):
        resolve_policies(runtime, path)


@pytest.mark.skipif(
    not all((ROOT / "models/bfm" / name).is_file() for name in ("controller/wrist_student.onnx", "actor.onnx")),
    reason="requires the local Holosoma wrist student export and converted actor",
)
def test_real_export_runs():
    runtime, path = load_runtime_config(ROOT / "configs/g1/bfm_wrist_student.yaml")
    resolved = resolve_policies(runtime, path)[0]
    policy = BfmWristStudentPolicy(resolved.config)
    state = low_state()
    policy.activate(state)
    for _ in range(12):
        policy.step(state)
    assert 0 <= policy.token < policy.num_choices
    assert np.isfinite(policy.motor_actions).all()
