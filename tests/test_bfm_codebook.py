"""BFM codebook controllers: [token, codeword] exports with learned codewords baked into the graph."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper
from test_bfm_policy import RecordingSession, _state
from test_bfm_token import _actor_onnx, _frame, _metadata

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import BfmTaskConfig, InferenceConfig, PolicySpec
from vex_policy.policies import bfm
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.bfm import BfmTokenPassiveWalkPolicy, BfmWalkCodebookPolicy
from vex_policy.policies.utils.bfm import DEFAULT_DOF_ANGLES
from vex_policy.policy_state_machine import _policy_class
from vex_policy.robots import G1_29DOF

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "walk": ("bfm_walk_codebook", BfmWalkCodebookPolicy, 356),
    "passive": ("bfm_token_passive_walk", BfmTokenPassiveWalkPolicy, 632),
}
CHOICES = 64
TOKEN = 17
CODEBOOK = np.random.default_rng(0).normal(size=(CHOICES, 256)).astype(np.float32)


def _codebook_onnx(path, width, metadata):
    """Holosoma's codebook export structure without Gumbel noise: index, then its codeword."""
    bias = np.zeros((1, CHOICES), dtype=np.float32)
    bias[0, TOKEN] = 1.0
    nodes = [
        helper.make_node("MatMul", ["actor_obs", "weight"], ["product"]),
        helper.make_node("Add", ["product", "bias"], ["logits"]),
        helper.make_node("TopK", ["logits", "k"], ["values", "indices"], axis=-1),
        helper.make_node("ArgMax", ["values"], ["choice"], axis=-1, keepdims=1),
        helper.make_node("GatherElements", ["indices", "choice"], ["selected"], axis=-1),
        helper.make_node("Cast", ["selected"], ["index"], to=TensorProto.FLOAT),
        helper.make_node("Squeeze", ["selected", "axis"], ["row"]),
        helper.make_node("Gather", ["codebook", "row"], ["codeword"], axis=0),
        helper.make_node("Concat", ["index", "codeword"], ["action"], axis=-1),
    ]
    initializers = [
        numpy_helper.from_array(np.zeros((width, CHOICES), dtype=np.float32), "weight"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(np.asarray([5], dtype=np.int64), "k"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis"),
        numpy_helper.from_array(CODEBOOK, "codebook"),
    ]
    graph = helper.make_graph(
        nodes,
        "bfm-codebook-test",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 257])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    helper.set_model_props(model, {key: json.dumps(value) for key, value in metadata.items()})
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _codebook_metadata(spec):
    metadata = _metadata(spec)
    experiment = metadata["experiment_config"]
    experiment["action"]["terms"]["joint_control"]["func"] = "holosoma.managers.action.terms.bfm:BFMCodebookAction"
    experiment["algo"]["config"]["module_dict"]["actor"]["distribution"] = "codebook"
    return metadata


@pytest.fixture
def make_policy(tmp_path, monkeypatch):
    def session(path, providers, **kwargs):
        return RecordingSession(ort.InferenceSession(path, providers=providers))

    monkeypatch.setattr(bfm, "shared_session", session)

    def make(variant="walk", metadata_change=None, doc_change=None):
        implementation, cls, width = VARIANTS[variant]
        doc = yaml.safe_load((ROOT / f"configs/examples/{implementation}.yaml").read_text())
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        doc["task"]["model_path"] = str(folder / "controller.onnx")
        doc["task"]["actor_model_path"] = str(folder / "actor.onnx")
        if doc_change:
            doc_change(doc)
        spec = PolicySpec.model_validate(doc)
        metadata = _codebook_metadata(spec)
        if metadata_change:
            metadata_change(metadata)
        _codebook_onnx(doc["task"]["model_path"], width, metadata)
        _actor_onnx(doc["task"]["actor_model_path"], np.linspace(-1, 1, 29, dtype=np.float32)[None])
        config = InferenceConfig(G1_29DOF, spec.inputs, spec.observation, spec.task, spec.guard)
        return cls(config), doc

    return make


def test_codebook_executes_projected_codeword(make_policy):
    policy, _ = make_policy()
    assert policy.num_choices == CHOICES
    state = _state()
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
    frame = _frame(state, np.zeros((1, 29)))
    actor_obs = policy.actor.feeds[0]["actor_obs"]
    np.testing.assert_array_equal(actor_obs[0, 221:337], np.tile(frame[0, :29], 4))
    policy.deactivate()
    assert policy.token == -1 and not policy.latent.any()


def test_passive_walk_frame_history_and_ungated_clock(make_policy):
    policy, _ = make_policy("passive")
    state = _state()
    policy.activate(state)
    policy.apply_control({})
    frames = []
    motor = np.zeros((1, 29))
    for tick in range(9):
        state = _state(DEFAULT_DOF_ANGLES + tick * 0.001)
        if tick % 4 == 0:
            frames.append(_frame(state, motor))
        policy.step(state)
        motor = policy.motor_actions
    feeds = [feed["actor_obs"] for feed in policy.controller.feeds]
    assert len(feeds) == 3 and feeds[0].shape == (1, 632)
    # Oldest-first, zero-padded controller-step frames.
    assert not feeds[0][:, : 3 * 93].any()
    np.testing.assert_array_equal(feeds[0][:, 3 * 93 : 4 * 93], frames[0])
    np.testing.assert_array_equal(feeds[2][:, 93:372], np.concatenate(frames, axis=1))
    # Zero command does not freeze the passive gait clock: cos, cos, sin, sin.
    np.testing.assert_allclose(feeds[0][0, 372:376], [1, -1, 0, 0], atol=1e-7)
    phase = np.array([0.0, -np.pi]) + 2 * np.pi * 4 / 50
    np.testing.assert_allclose(feeds[1][0, 372:376], np.concatenate((np.cos(phase), np.sin(phase))), atol=1e-6)
    np.testing.assert_array_equal(feeds[1][:, -256:], policy.latent)
    assert policy.token == TOKEN
    np.testing.assert_allclose(policy.latent[0], CODEBOOK[TOKEN] / np.linalg.norm(CODEBOOK[TOKEN]) * 16, rtol=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("kind", ["action", "distribution"])
def test_incompatible_codebook_contracts_fail_at_load(make_policy, variant, kind):
    def change(metadata):
        experiment = metadata["experiment_config"]
        if kind == "action":
            experiment["action"]["terms"]["joint_control"]["func"] = "holosoma.managers.action.terms.bfm:BFMTokenAction"
        else:
            experiment["algo"]["config"]["module_dict"]["actor"]["distribution"] = "categorical"

    with pytest.raises(ValueError, match="BFM codebook"):
        make_policy(variant, change)


def test_passive_walk_history_and_inputs_are_checked(make_policy):
    def change(metadata):
        metadata["experiment_config"]["observation"]["groups"]["actor_obs"]["history_length"] = 1

    with pytest.raises(ValueError, match="BFM"):
        make_policy("passive", change)
    with pytest.raises(ValueError, match="inputs"):
        make_policy(
            "passive",
            doc_change=lambda doc: doc.update(
                inputs=[{"type": "slider", "parameter": {"name": "yaw", "min": -1, "max": 1, "default": 0}}]
            ),
        )


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
    policy.activate(_state())
    policy.controller.override = action.astype(np.float32)
    with pytest.raises(PolicyRuntimeFault, match=r"token|codeword|finite"):
        policy.step(_state())


@pytest.mark.parametrize("variant", VARIANTS)
def test_example_resolution_and_registry(make_policy, tmp_path, variant):
    implementation, cls, _ = VARIANTS[variant]
    _, doc = make_policy(variant)
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(doc))
    runtime, path = load_runtime_config(policy_file)
    resolved = resolve_policies(runtime, path)
    assert isinstance(resolved[0].config.task, BfmTaskConfig)
    assert _policy_class(implementation) is cls
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["entry-points"]["vex_policy.policies"][implementation].endswith(":" + cls.__name__)
    doc["task"]["motion_data_path"] = doc["task"]["model_path"]
    policy_file.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="motion reference"):
        resolve_policies(*load_runtime_config(policy_file))
