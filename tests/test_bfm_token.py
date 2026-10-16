"""BFM vocabulary-token controllers: categorical exports, local vocabulary and observation layouts."""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper
from test_bfm_policy import RecordingSession, _state

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import BfmTokenTaskConfig, InferenceConfig, PolicySpec
from vex_policy.policies import bfm
from vex_policy.policies.base import PolicyRuntimeFault
from vex_policy.policies.bfm import BfmWalkTokenPolicy
from vex_policy.policies.utils.bfm import DEFAULT_DOF_ANGLES, JOINT_NAMES, load_latent_vocab
from vex_policy.policy_state_machine import _policy_class
from vex_policy.robots import G1_29DOF
from vex_policy.utils.math.quat import quat_rotate_inverse

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "data/bfm_vocab/walk"
TOKEN = 1234
FUNCTIONS = {
    "frame": "bfm:proprioceptive_frame",
    "command": "bfm:velocity_command",
    "latent": "bfm:executed_latent",
    "cos_phase": "locomotion:cos_phase",
    "sin_phase": "locomotion:sin_phase",
}
VARIANTS = {"walk": ("bfm_walk_token", BfmWalkTokenPolicy, 356)}


def _categorical_onnx(path, width, choices, token, metadata):
    """Holosoma's top-k export structure without Gumbel noise: the largest bias wins."""
    bias = np.zeros((1, choices), dtype=np.float32)
    bias[0, token] = 1.0
    nodes = [
        helper.make_node("MatMul", ["actor_obs", "weight"], ["product"]),
        helper.make_node("Add", ["product", "bias"], ["logits"]),
        helper.make_node("TopK", ["logits", "k"], ["values", "indices"], axis=-1),
        helper.make_node("ArgMax", ["values"], ["choice"], axis=-1, keepdims=1),
        helper.make_node("GatherElements", ["indices", "choice"], ["selected"], axis=-1),
        helper.make_node("Cast", ["selected"], ["action"], to=TensorProto.FLOAT),
    ]
    initializers = [
        numpy_helper.from_array(np.zeros((width, choices), dtype=np.float32), "weight"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(np.asarray([5], dtype=np.int64), "k"),
    ]
    graph = helper.make_graph(
        nodes,
        "bfm-token-test",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 1])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    helper.set_model_props(model, {key: json.dumps(value) for key, value in metadata.items()})
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _actor_onnx(path, motor):
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["action"], value=numpy_helper.from_array(motor))],
        "bfm-actor-test",
        [
            helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 465]),
            helper.make_tensor_value_info("latent", TensorProto.FLOAT, [1, 256]),
        ],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    metadata = {"bfm_actor_contract": 1, "dof_names": JOINT_NAMES, "default_dof_angles": DEFAULT_DOF_ANGLES.tolist()}
    helper.set_model_props(model, {key: json.dumps(value) for key, value in metadata.items()})
    onnx.save(model, path)


def _metadata(spec, vocab_name="walk"):
    observation = spec.observation
    groups = {
        group: {
            "history_length": observation.history_length_dict.get(group, 1),
            "concatenate": True,
            "terms": {term: {"func": "holosoma.managers.observation.terms." + FUNCTIONS[term]} for term in terms},
        }
        for group, terms in observation.obs_dict.items()
    }
    return {
        "dof_names": JOINT_NAMES,
        "kp": [20.0] * 29,
        "kd": [2.0] * 29,
        "action_scale": [0.03] * 29,
        "experiment_config": {
            "env_class": "holosoma.envs.bfm_controller.environment.BFMControllerEnvironment",
            "control_decimation": 4,
            "simulator": {"config": {"sim": {"fps": 200, "control_decimation": 4}}},
            "action": {
                "terms": {
                    "joint_control": {
                        "func": "holosoma.managers.action.terms.bfm:BFMTokenAction",
                        "params": {
                            "checkpoint_path": "@holosoma/data/thirdparty/bfm_zero",
                            "vocab_path": f"@holosoma/data/bfm_vocab/{vocab_name}",
                        },
                    }
                }
            },
            "algo": {
                "config": {
                    "module_dict": {
                        "actor": {"input_dim": list(groups), "distribution": "categorical", "inference_top_k": 5}
                    }
                }
            },
            "robot": {
                "init_state": {"default_joint_angles": dict(zip(JOINT_NAMES, DEFAULT_DOF_ANGLES.tolist(), strict=True))}
            },
            "observation": {"groups": groups},
        },
    }


@pytest.fixture
def make_policy(tmp_path, monkeypatch):
    def session(path, providers, **kwargs):
        return RecordingSession(ort.InferenceSession(path, providers=providers))

    monkeypatch.setattr(bfm, "shared_session", session)

    def make(variant, metadata_change=None, choices=2048, doc_change=None):
        implementation, cls, width = VARIANTS[variant]
        doc = yaml.safe_load((ROOT / f"configs/examples/{implementation}.yaml").read_text())
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        doc["task"]["model_path"] = str(folder / "controller.onnx")
        doc["task"]["actor_model_path"] = str(folder / "actor.onnx")
        doc["task"]["vocab_path"] = str(VOCAB)
        if doc_change:
            doc_change(doc)
        spec = PolicySpec.model_validate(doc)
        metadata = _metadata(spec)
        if metadata_change:
            metadata_change(metadata)
        _categorical_onnx(doc["task"]["model_path"], width, choices, TOKEN, metadata)
        _actor_onnx(doc["task"]["actor_model_path"], np.linspace(-1, 1, 29, dtype=np.float32)[None])
        config = InferenceConfig(G1_29DOF, spec.inputs, spec.observation, spec.task, spec.guard)
        return cls(config), doc

    return make


def _frame(state, previous_motor):
    gravity = quat_rotate_inverse(state.base_quat, np.array([[0.0, 0.0, -1.0]]))
    return np.concatenate(
        (state.joint_pos - DEFAULT_DOF_ANGLES, state.joint_vel, gravity, 0.25 * state.base_ang_vel, previous_motor),
        axis=1,
    ).astype(np.float32)


def test_local_vocabulary_is_a_projected_copy_of_holosoma_walk():
    z, metadata = load_latent_vocab(VOCAB)
    assert z.shape == (2048, 256) and metadata["size"] == 2048
    np.testing.assert_allclose(np.linalg.norm(z, axis=-1), 16.0, rtol=1e-6)


def test_walk_token_executes_vocabulary_row(make_policy):
    policy, _ = make_policy("walk")
    vocab, _ = load_latent_vocab(VOCAB)
    state = _state()
    policy.activate(state)
    policy.apply_control({"vx": 0.2, "vy": 0.3, "yaw": -0.4})
    for _ in range(5):
        policy.step(state)
    assert policy.token == TOKEN
    np.testing.assert_array_equal(policy.latent, vocab[TOKEN : TOKEN + 1])
    first, second = (feed["actor_obs"] for feed in policy.controller.feeds)
    assert first.shape == (1, 356)
    np.testing.assert_allclose(first[0, 93:98], [0.3, -0.2, 0.4, 1.0, -1.0])
    assert not first[:, -256:].any()
    np.testing.assert_array_equal(second[:, -256:], policy.latent)
    # Holosoma fills the frozen actor history with the first frame, not zeros.
    frame = _frame(state, np.zeros((1, 29)))
    actor_obs = policy.actor.feeds[0]["actor_obs"]
    np.testing.assert_array_equal(actor_obs[0, 93:209], np.tile(frame[0, 64:93], 4))
    np.testing.assert_array_equal(actor_obs[0, 221:337], np.tile(frame[0, :29], 4))
    policy.deactivate()
    assert policy.token == -1 and not policy.latent.any()


@pytest.mark.parametrize("kind", ["size", "vocab_name", "action", "distribution"])
def test_incompatible_token_contracts_fail_at_load(make_policy, kind):
    def change(metadata):
        experiment = metadata["experiment_config"]
        term = experiment["action"]["terms"]["joint_control"]
        if kind == "vocab_name":
            term["params"]["vocab_path"] = "@holosoma/data/bfm_vocab/walk_query4k"
        elif kind == "action":
            term["func"] = "holosoma.managers.action.terms.bfm:BFMLatentAction"
        elif kind == "distribution":
            experiment["algo"]["config"]["module_dict"]["actor"]["distribution"] = "gaussian"

    with pytest.raises(ValueError, match="BFM"):
        make_policy("walk", metadata_change=change, choices=4000 if kind == "size" else 2048)


def test_invalid_token_is_rejected(make_policy):
    policy, _ = make_policy("walk")
    policy.activate(_state())
    policy.controller.override = np.full((1, 1), 2048.0, dtype=np.float32)
    with pytest.raises(PolicyRuntimeFault, match="token"):
        policy.step(_state())


@pytest.mark.parametrize("variant", VARIANTS)
def test_example_resolution_and_registry(make_policy, tmp_path, variant):
    _, doc = make_policy(variant)
    implementation, cls, _ = VARIANTS[variant]
    vocab = tmp_path / "vocab" / "walk"
    shutil.copytree(VOCAB, vocab)
    doc["task"]["vocab_path"] = str(vocab)
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(doc))
    runtime, path = load_runtime_config(policy_file)
    resolved = resolve_policies(runtime, path)
    assert isinstance(resolved[0].config.task, BfmTokenTaskConfig)
    assert resolved[0].config.task.vocab_path == str(vocab.resolve())
    assert _policy_class(implementation) is cls
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["entry-points"]["vex_policy.policies"][implementation].endswith(":" + cls.__name__)
    (vocab / "vocab.safetensors").unlink()
    with pytest.raises(ValueError, match="vocab_path"):
        resolve_policies(runtime, path)
