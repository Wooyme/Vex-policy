from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from vex_policy.config.config_types import PolicySpec, SonicTaskConfig
from vex_policy.policies.sonic import SonicPolicy
from vex_policy.policies.sonic_planner import MODE_NAMES, POLICY_TO_HW, MotionSequence, SonicPlanner


def test_all_sonic_mode_configs_keep_current_yaml_shape():
    paths = sorted(Path("configs/g1/sonic").glob("*.yaml"))
    assert len(paths) == 27
    for expected_mode, path in enumerate(paths):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        spec = PolicySpec.model_validate(data)
        assert spec.implementation == "sonic"
        assert isinstance(spec.task, SonicTaskConfig)
        assert spec.task.planner_mode == expected_mode
        assert spec.name == f"g1-sonic-{MODE_NAMES[expected_mode].replace('_', '-')}"
        assert sum(spec.observation.obs_dims[name] for name in spec.observation.obs_dict["actor_obs"]) == 994
        assert sum(spec.observation.obs_dims[name] for name in spec.observation.obs_dict["encoder_obs"]) == 1762


def test_planner_resamples_30hz_output_and_maps_to_policy_order():
    qpos = np.zeros((3, 36), dtype=np.float32)
    qpos[:, 2] = np.arange(3)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = np.arange(29, dtype=np.float32)
    motion = SonicPlanner._resample_50hz(qpos)
    assert motion.frames == 5
    assert motion.root_positions[1, 2] == np.float32(0.6)
    assert np.array_equal(motion.joint_positions[0], np.arange(29, dtype=np.float32)[POLICY_TO_HW])
    assert np.isfinite(motion.joint_velocities).all()


def test_sonic_observation_contracts_are_exact():
    policy = object.__new__(SonicPolicy)
    policy.sonic_task = SimpleNamespace(planner_encoder_mode=0)
    policy._heading_robot_initial = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    policy._heading_reference_initial = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    policy._state_history = deque(maxlen=10)
    motion = MotionSequence(
        np.zeros((60, 3), dtype=np.float32),
        np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (60, 1)),
        np.arange(60 * 29, dtype=np.float32).reshape(60, 29),
        np.ones((60, 29), dtype=np.float32),
    )
    encoder = policy._encoder_observation(motion, 0, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    decoder = policy._decoder_observation(np.zeros((1, 64), dtype=np.float32))
    assert encoder.shape == (1, 1762)
    assert decoder.shape == (1, 994)
    assert np.array_equal(encoder[0, 4:294], motion.joint_positions[np.arange(10) * 5].reshape(-1))
    assert np.allclose(encoder[0, 601:661].reshape(10, 6), [1, 0, 0, 1, 0, 0])


def test_sonic_task_replace_preserves_specialized_type():
    task = SonicTaskConfig(model_path="decoder.onnx")
    updated = replace(task, model_path="/tmp/decoder.onnx")
    assert isinstance(updated, SonicTaskConfig)
    assert updated.lowcmd_publish_rate == 500.0
