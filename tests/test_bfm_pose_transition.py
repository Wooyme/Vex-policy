"""BFM pose transition: offline bank, observation layout and the 50 Hz waypoint schedule."""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import yaml
from test_bfm_policy import RecordingSession, _onnx, _state

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import BfmPoseTransitionTaskConfig, InferenceConfig, PolicySpec
from vex_policy.policies import bfm
from vex_policy.policies.bfm import BfmPoseTransitionPolicy
from vex_policy.policies.utils.bfm import DEFAULT_DOF_ANGLES, JOINT_NAMES, residual_waypoint, slerp
from vex_policy.policy_state_machine import _policy_class
from vex_policy.robots import G1_29DOF

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bfm_pose_bank", ROOT / "scripts/bfm_pose_bank.py")
bfm_pose_bank = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bfm_pose_bank)

FUNCTIONS = {
    "frame": "bfm:proprioceptive_frame",
    "latent": "bfm:executed_latent",
    "goal_latents": "bfm_waypoint:goal_latents",
    "target": "pose_transition:target_pose",
    "timing": "bfm_waypoint:timing",
}


def _sphere(rng, count):
    value = rng.normal(size=(count, 256))
    return (value / np.linalg.norm(value, axis=-1, keepdims=True) * 16).astype(np.float32)


def _holosoma_data(folder, sha="abc"):
    rng = np.random.default_rng(0)
    names = np.asarray(["sit", "kneel", "crawl"])
    np.savez(
        folder / "goal_bank.npz",
        pose_names=names,
        joint_names=np.asarray(JOINT_NAMES),
        latents=_sphere(rng, 3),
        manifest=np.asarray(json.dumps({"checkpoint_sha256": "abc"})),
    )
    np.savez(
        folder / "endpoint.npz",
        pose_names=names,
        joints=rng.normal(size=(3, 29)).astype(np.float32),
        gravity=rng.normal(size=(3, 3)).astype(np.float32),
        height=np.asarray([0.3, 0.4, 0.5], dtype=np.float32),
        checkpoint_sha256=np.asarray(sha),
    )
    return folder / "goal_bank.npz", folder / "endpoint.npz"


def test_offline_bank_matches_holosoma_target_pose(tmp_path):
    bank_path, reference_path = _holosoma_data(tmp_path)
    data = bfm_pose_bank.build(bank_path, reference_path)
    with np.load(reference_path) as reference:
        expected = np.concatenate(
            (reference["joints"] - DEFAULT_DOF_ANGLES, reference["gravity"], reference["height"][:, None]), axis=1
        )
    np.testing.assert_allclose(data["targets"], expected)
    assert data["targets"].shape == (3, 33) and data["latents"].shape == (3, 256)
    _holosoma_data(tmp_path, sha="other")
    with pytest.raises(ValueError, match="different frozen actor"):
        bfm_pose_bank.build(bank_path, reference_path)


@pytest.fixture
def make_policy(tmp_path, monkeypatch):
    def session(path, providers, **kwargs):
        return RecordingSession(ort.InferenceSession(path, providers=providers))

    monkeypatch.setattr(bfm, "shared_session", session)
    doc = yaml.safe_load((ROOT / "configs/g1/bfm_pose_transition.yaml").read_text())
    bank_path, reference_path = _holosoma_data(tmp_path)
    np.savez(tmp_path / "pose_bank.npz", **bfm_pose_bank.build(bank_path, reference_path))
    doc["task"].update(
        model_path=str(tmp_path / "controller.onnx"),
        actor_model_path=str(tmp_path / "actor.onnx"),
        pose_bank_path=str(tmp_path / "pose_bank.npz"),
        source_pose="sit",
        target_pose="crawl",
        settle_s=0.08,
        transition_duration_s=4.0,
    )
    spec = PolicySpec.model_validate(doc)
    groups = {
        group: {
            "history_length": 1,
            "concatenate": True,
            "terms": {t: {"func": "holosoma.managers.observation.terms." + FUNCTIONS[t]} for t in terms},
        }
        for group, terms in spec.observation.obs_dict.items()
    }
    metadata = {
        "dof_names": JOINT_NAMES,
        "kp": [20.0] * 29,
        "kd": [2.0] * 29,
        "action_scale": [0.03] * 29,
        "experiment_config": {
            "env_class": "holosoma.envs.bfm_controller.environment.BFMControllerEnvironment",
            "control_decimation": 4,
            "simulator": {"config": {"sim": {"fps": 200, "control_decimation": 4}}},
            "algo": {"config": {"module_dict": {"actor": {"input_dim": list(groups)}}}},
            "robot": {
                "init_state": {"default_joint_angles": dict(zip(JOINT_NAMES, DEFAULT_DOF_ANGLES.tolist(), strict=True))}
            },
            "observation": {"groups": groups},
        },
    }
    residual = np.linspace(-3, 3, 256, dtype=np.float32)[None]
    _onnx(doc["task"]["model_path"], {"actor_obs": 896}, residual, metadata)
    actor_metadata = {
        "bfm_actor_contract": 1,
        "dof_names": JOINT_NAMES,
        "default_dof_angles": DEFAULT_DOF_ANGLES.tolist(),
    }
    _onnx(
        doc["task"]["actor_model_path"],
        {"actor_obs": 465, "latent": 256},
        np.zeros((1, 29), np.float32),
        actor_metadata,
    )
    config = InferenceConfig(G1_29DOF, spec.inputs, spec.observation, spec.task, spec.guard)
    return BfmPoseTransitionPolicy(config), doc, residual


def test_observation_layout_and_waypoint_schedule(make_policy):
    policy, _, residual = make_policy
    with np.load(policy.task.pose_bank_path) as bank:
        source, target, target_features = bank["latents"][0:1], bank["latents"][2:3], bank["targets"][2]
    state = _state()
    assert policy.activate(state) is None
    latents = []
    for _ in range(4 * 53):  # 0.08 s settle + 4 s transition, then four held ticks.
        policy.step(state)
        latents.append(policy.actor.feeds[-1]["latent"])
    first = policy.controller.feeds[0]["actor_obs"][0]
    assert first.shape == (896,)
    np.testing.assert_allclose(first[93:349], source[0])
    np.testing.assert_allclose(first[349:605], target[0])
    np.testing.assert_allclose(first[605:638], target_features)
    np.testing.assert_allclose(first[638:640], [-0.02, 4.0], rtol=1e-6)
    np.testing.assert_allclose(first[640:], source[0])
    # The settle interval executes the source latent; the controller stops after the deadline.
    for latent in latents[:4]:
        np.testing.assert_allclose(latent, source, rtol=1e-6)
    assert len(policy.controller.feeds) == 51  # Steps 0..200; step 204 is the 4.08 s deadline.
    np.testing.assert_allclose(latents[-1], target, rtol=1e-6)
    # Inside the transition each actor tick slerps toward the next residual waypoint.
    # t=0.1 s: a quarter of the way from the phase-0 waypoint (the source) to the phase-0.02 waypoint.
    waypoint = residual_waypoint(source, target, (0.08 + 0.08 - 0.08) / 4.0, residual)
    np.testing.assert_allclose(latents[5], slerp(source, waypoint, 0.25), atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(np.concatenate(latents), axis=-1), 16, rtol=1e-5)
    np.testing.assert_allclose(policy.controller.feeds[1]["actor_obs"][:, 640:], latents[3])
    policy.deactivate()
    np.testing.assert_array_equal(policy.latent, source)


def test_waypoint_endpoints_and_antipodes():
    rng = np.random.default_rng(1)
    a, b = _sphere(rng, 1), _sphere(rng, 1)
    big = rng.normal(size=(1, 256)) * 100
    np.testing.assert_allclose(residual_waypoint(a, b, 0.0, big), a, rtol=1e-6)
    np.testing.assert_allclose(residual_waypoint(a, b, 1.0, big), b, rtol=1e-6)
    middle = slerp(a, -a, 0.5)
    assert abs(float((middle * a).sum())) < 1e-3 and np.linalg.norm(middle) == pytest.approx(16)


def test_yaml_resolution_and_registry(make_policy, tmp_path):
    _, doc, _ = make_policy
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(doc))
    runtime, path = load_runtime_config(policy_file)
    resolved = resolve_policies(runtime, path)
    assert isinstance(resolved[0].config.task, BfmPoseTransitionTaskConfig)
    assert _policy_class("bfm_pose_transition") is BfmPoseTransitionPolicy
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["entry-points"]["vex_policy.policies"]["bfm_pose_transition"].endswith(
        ":BfmPoseTransitionPolicy"
    )
    doc["task"]["target_pose"] = "missing"
    policy_file.write_text(yaml.safe_dump(doc))
    runtime, path = load_runtime_config(policy_file)
    with pytest.raises(ValueError, match="not in the bank"):
        BfmPoseTransitionPolicy(resolve_policies(runtime, path)[0].config)
