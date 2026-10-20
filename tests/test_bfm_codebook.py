"""BFM codebook controllers: [token, codeword] exports with learned codewords baked into the graph."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest
import yaml
from bfm_helpers import CHOICES, CODEBOOK, TOKEN, bfm_frame, build_policy, low_state, patch_sessions

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import BfmTaskConfig
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.bfm import BfmWalkCodebookPolicy
from vex_policy.policies.bfm.contract import pack_actor_observation
from vex_policy.policy_state_machine import _policy_class

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/examples/bfm/bfm_walk_codebook.yaml"
WIDTH = 356


@pytest.fixture
def make_policy(tmp_path, monkeypatch):
    patch_sessions(monkeypatch)

    def make(metadata_change=None, doc_change=None):
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        return build_policy(BfmWalkCodebookPolicy, EXAMPLE, WIDTH, folder, metadata_change, doc_change)

    return make


def test_codebook_executes_projected_codeword(make_policy):
    policy, _ = make_policy()
    assert policy.num_choices == CHOICES
    state = low_state()
    policy.activate(state)
    policy.apply_control({"vx": 0.2, "vy": 0.3, "yaw": -0.4})
    for _ in range(5):
        policy.step(state)
    assert policy.token == TOKEN
    expected = CODEBOOK[TOKEN] / np.linalg.norm(CODEBOOK[TOKEN]) * 16
    np.testing.assert_allclose(policy.latent[0], expected, rtol=1e-5)
    first, second = (feed["actor_obs"] for feed in policy.controller.feeds)
    assert first.shape == (1, 356)
    np.testing.assert_allclose(first[0, 93:98], [0.3, -0.2, 0.4, 1.0, -1.0])
    assert not first[:, -256:].any()
    np.testing.assert_array_equal(second[:, -256:], policy.latent)
    # As BFMLatentAction, the frozen actor history starts filled with the first frame.
    frame = bfm_frame(state, np.zeros((1, 29)))
    actor_obs = policy.actor.feeds[0]["actor_obs"]
    np.testing.assert_array_equal(actor_obs[0, 221:337], np.tile(frame[0, :29], 4))
    policy.deactivate()
    assert policy.token == -1 and not policy.latent.any()


@pytest.mark.parametrize("kind", ["action", "distribution"])
def test_incompatible_codebook_contracts_fail_at_load(make_policy, kind):
    def change(metadata):
        experiment = metadata["experiment_config"]
        if kind == "action":
            experiment["action"]["terms"]["joint_control"]["func"] = "holosoma.managers.action.terms.bfm:BFMTokenAction"
        else:
            experiment["algo"]["config"]["module_dict"]["actor"]["distribution"] = "categorical"

    with pytest.raises(ValueError, match="BFM codebook"):
        make_policy(change)


def test_history_and_inputs_are_checked(make_policy):
    def change(metadata):
        metadata["experiment_config"]["observation"]["groups"]["actor_obs"]["history_length"] = 4

    with pytest.raises(ValueError, match="BFM"):
        make_policy(change)
    with pytest.raises(ValueError, match="inputs"):
        make_policy(doc_change=lambda doc: doc.update(inputs=doc["inputs"][:1]))


def test_packing_reverses_frames_within_fields_not_terms():
    frames = np.arange(2 * 5 * 93, dtype=np.float32).reshape(2, 5, 93)
    packed = pack_actor_observation(frames)
    assert packed.shape == (2, 465)
    np.testing.assert_array_equal(packed[:, :93], frames[:, 4])
    for index, start in enumerate((93, 122, 151, 180)):
        np.testing.assert_array_equal(packed[:, start : start + 29], frames[:, 3 - index, 64:93])
    with pytest.raises(ValueError, match="history"):
        pack_actor_observation(np.zeros((1, 4, 93)))


@pytest.mark.parametrize(
    "action",
    [
        np.concatenate(([[float(CHOICES)]], CODEBOOK[:1]), axis=1),
        np.concatenate(([[1.5]], CODEBOOK[:1]), axis=1),
        np.concatenate(([[1.0]], np.full((1, 256), np.nan)), axis=1),
        np.concatenate(([[1.0]], np.zeros((1, 256))), axis=1),
    ],
)
def test_invalid_codebook_outputs_fault(make_policy, action):
    policy, _ = make_policy()
    policy.activate(low_state())
    policy.controller.override = action.astype(np.float32)
    with pytest.raises(PolicyRuntimeFault, match=r"token|codeword|finite"):
        policy.step(low_state())


def test_example_resolution_and_registry(make_policy, tmp_path):
    _, doc = make_policy()
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(doc))
    runtime, path = load_runtime_config(policy_file)
    resolved = resolve_policies(runtime, path)
    assert isinstance(resolved[0].config.task, BfmTaskConfig)
    assert _policy_class("bfm_walk_codebook") is BfmWalkCodebookPolicy
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    entry_point = project["project"]["entry-points"]["vex_policy.policies"]["bfm_walk_codebook"]
    assert entry_point.endswith(":BfmWalkCodebookPolicy")
    doc["task"]["motion_data_path"] = doc["task"]["model_path"]
    policy_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="motion_data_path"):
        load_runtime_config(policy_file)
