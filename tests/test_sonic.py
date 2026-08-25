from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from vex_policy.config import load_runtime_config, resolve_policies
from vex_policy.config.config_types import PolicySpec, SonicTaskConfig
from vex_policy.policies.sonic import SonicPolicy
from vex_policy.policies.sonic_motion import load_motion_directory
from vex_policy.policies.sonic_planner import HW_TO_POLICY, MODE_NAMES, POLICY_TO_HW, MotionSequence, SonicPlanner


class _FakeInterface:
    def __init__(self, config):
        self.robot_config = config
        self.commands = []

    def update_config(self, config):
        self.robot_config = config

    def get_low_state(self):
        count = self.robot_config.num_joints
        state = np.zeros((1, 3 + 4 + count + 3 + 3 + count), dtype=np.float32)
        state[0, 3] = 1.0
        state[0, 7 : 7 + count] = self.robot_config.default_dof_angles
        return state

    def send_low_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))


def _write_reference_motion(motion_directory: Path, frames: int = 6) -> tuple[np.ndarray, ...]:
    motion_directory.mkdir(parents=True)
    joint_positions = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) * 0.01
    joint_velocities = joint_positions + 0.005
    body_positions = np.zeros((frames, 6), dtype=np.float32)
    body_positions[:, :3] = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3) * 0.01
    body_quaternions = np.zeros((frames, 8), dtype=np.float32)
    body_quaternions[:, 0] = 2.0
    body_quaternions[:, 4] = 1.0
    for filename, values in (
        ("joint_pos.csv", joint_positions),
        ("joint_vel.csv", joint_velocities),
        ("body_pos.csv", body_positions),
        ("body_quat.csv", body_quaternions),
    ):
        header = ",".join(f"value_{index}" for index in range(values.shape[1]))
        np.savetxt(motion_directory / filename, values, delimiter=",", header=header, comments="")
    return joint_positions, joint_velocities, body_positions, body_quaternions


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


def test_planner_motion_context_interpolates_and_maps_to_hardware_order():
    frames = 20
    joint_positions = np.arange(frames, dtype=np.float32)[:, None] * 100 + np.arange(29, dtype=np.float32)[None, :]
    motion = MotionSequence(
        np.zeros((frames, 3), dtype=np.float32),
        np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1)),
        joint_positions,
        np.zeros_like(joint_positions),
    )
    planner = object.__new__(SonicPlanner)
    planner.look_ahead_steps = 1

    context = planner.motion_context(motion, current_frame=2)

    sample_frames = 3 + np.arange(4, dtype=np.float64) * (50.0 / 30.0)
    expected_policy = sample_frames[:, None] * 100 + np.arange(29, dtype=np.float64)[None, :]
    assert context.shape == (1, 4, 36)
    assert np.allclose(context[0, :, 7:][:, POLICY_TO_HW], expected_policy)


def test_motion_directory_loads_reference_csv_in_policy_order(tmp_path):
    motion_directory = tmp_path / "motions" / "wave"
    joint_positions, joint_velocities, body_positions, _ = _write_reference_motion(motion_directory)

    name, motion = load_motion_directory(tmp_path / "motions", start_frame=1, end_frame=5)

    assert name == "wave"
    assert motion.frames == 4
    assert np.array_equal(motion.joint_positions, joint_positions[1:5])
    assert np.array_equal(motion.joint_velocities, joint_velocities[1:5])
    assert np.array_equal(motion.root_positions, body_positions[1:5, :3])
    assert np.array_equal(motion.root_quaternions, np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)))


def test_directory_motion_initializes_and_runs_without_planner_model(tmp_path):
    motion_directory = tmp_path / "motions" / "wave"
    _write_reference_motion(motion_directory, frames=60)
    example = Path("configs/examples/g1_sonic_motion_directory.yaml")
    data = yaml.safe_load(example.read_text(encoding="utf-8"))
    data["task"]["motion_data_path"] = str(motion_directory)
    data["task"]["planner_model_path"] = str(tmp_path / "missing-planner.onnx")
    config_path = tmp_path / "sonic-motion.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    runtime, loaded_path = load_runtime_config(config_path)
    resolved = resolve_policies(runtime, loaded_path)

    assert resolved[0].config.task.motion_source == "directory"
    assert resolved[0].config.task.motion_data_path == str(motion_directory.resolve())
    interface = _FakeInterface(resolved[0].config.robot)
    policy = object.__new__(SonicPolicy)
    policy._injected_interface = interface
    SonicPolicy.__init__(policy, resolved[0].config)
    assert policy.planner is None
    policy.activate()
    policy.step()
    assert policy._planner_thread is None
    assert policy._motion_frame == 1
    assert len(interface.commands) == 1
    policy.close()


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


def test_sonic_action_mask_is_converted_from_hardware_to_policy_order():
    policy = object.__new__(SonicPolicy)
    policy.action_mask = np.ones((1, 29), dtype=np.float32)
    policy.action_mask[:, 15:] = 0.0
    policy.config = SimpleNamespace(task=SimpleNamespace(debug=SimpleNamespace(force_zero_action=False)))

    action_policy = np.arange(1, 30, dtype=np.float32).reshape(1, -1)
    masked_policy = policy._mask_policy_order_action(action_policy)
    masked_hardware = masked_policy[:, HW_TO_POLICY]

    assert np.array_equal(masked_hardware[0, :15], action_policy[:, HW_TO_POLICY][0, :15])
    assert np.array_equal(masked_hardware[0, 15:], np.zeros(14))
