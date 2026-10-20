"""Synthetic BFM controller/actor exports and Holosoma metadata shared by the BFM policy tests."""

from __future__ import annotations

import json

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper, numpy_helper

from vex_policy.config.config_types import InferenceConfig, PolicySpec
from vex_policy.policies.bfm import base
from vex_policy.policies.bfm.contract import DEFAULT_DOF_ANGLES, JOINT_NAMES
from vex_policy.robots import G1_29DOF
from vex_policy.sdk.base.base_interface import LowState
from vex_policy.utils.math.quat import quat_rotate_inverse, rpy_to_quat

CHOICES = 64
TOKEN = 17
CODEBOOK = np.random.default_rng(0).normal(size=(CHOICES, 256)).astype(np.float32)
FUNCTIONS = {
    "frame": "bfm:proprioceptive_frame",
    "command": "bfm:velocity_command",
    "latent": "bfm:executed_latent",
    "cos_phase": "locomotion:cos_phase",
    "sin_phase": "locomotion:sin_phase",
}


def low_state(q=None, rpy=(0.0, 0.0, 0.0)):
    return LowState(
        base_pos=np.zeros((1, 3)),
        base_quat=rpy_to_quat(rpy)[None],
        joint_pos=np.asarray(DEFAULT_DOF_ANGLES if q is None else q)[None].copy(),
        joint_vel=np.arange(29, dtype=np.float32)[None] / 10,
        base_lin_vel=np.zeros((1, 3)),
        base_ang_vel=np.array([[1.0, 2.0, 3.0]]),
    )


def bfm_frame(state, previous_motor):
    gravity = quat_rotate_inverse(state.base_quat, np.array([[0.0, 0.0, -1.0]]))
    return np.concatenate(
        (state.joint_pos - DEFAULT_DOF_ANGLES, state.joint_vel, gravity, 0.25 * state.base_ang_vel, previous_motor),
        axis=1,
    ).astype(np.float32)


class RecordingSession:
    def __init__(self, session):
        self.session = session
        self.feeds = []
        self.override = None

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def run(self, outputs, feed):
        self.feeds.append({key: value.copy() for key, value in feed.items()})
        if isinstance(self.override, Exception):
            raise self.override
        return [self.override] if self.override is not None else self.session.run(outputs, feed)


def actor_onnx(path, motor):
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


def codebook_onnx(path, width, metadata):
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


def metadata_for(spec):
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
            "action": {"terms": {"joint_control": {"func": "holosoma.managers.action.terms.bfm:BFMCodebookAction"}}},
            "algo": {
                "config": {
                    "module_dict": {
                        "actor": {"input_dim": list(groups), "distribution": "codebook", "inference_top_k": 5}
                    }
                }
            },
            "robot": {
                "init_state": {"default_joint_angles": dict(zip(JOINT_NAMES, DEFAULT_DOF_ANGLES.tolist(), strict=True))}
            },
            "observation": {"groups": groups},
        },
    }


def patch_sessions(monkeypatch):
    """Record every controller/actor feed instead of sharing real sessions."""

    def session(path, providers, **kwargs):
        return RecordingSession(ort.InferenceSession(path, providers=providers))

    monkeypatch.setattr(base, "shared_session", session)


def build_policy(policy_class, example, width, folder, metadata_change=None, doc_change=None):
    """Instantiate ``policy_class`` from an example YAML with synthetic exports written to ``folder``."""
    doc = yaml.safe_load(example.read_text())
    folder.mkdir()
    doc["task"]["model_path"] = str(folder / "controller.onnx")
    doc["task"]["actor_model_path"] = str(folder / "actor.onnx")
    if doc_change:
        doc_change(doc)
    spec = PolicySpec.model_validate(doc)
    metadata = metadata_for(spec)
    if metadata_change:
        metadata_change(metadata)
    codebook_onnx(doc["task"]["model_path"], width, metadata)
    actor_onnx(doc["task"]["actor_model_path"], np.linspace(-1, 1, 29, dtype=np.float32)[None])
    config = InferenceConfig(G1_29DOF, spec.inputs, spec.observation, spec.task, spec.guard)
    return policy_class(config), doc
