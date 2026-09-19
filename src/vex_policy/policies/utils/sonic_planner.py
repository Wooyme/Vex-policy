"""ONNX Runtime implementation of the GEAR-SONIC local motion planner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime

# Hardware/MuJoCo index -> policy/IsaacLab index.
HW_TO_POLICY = np.asarray(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
# Policy/IsaacLab index -> hardware/MuJoCo index.
POLICY_TO_HW = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

MODE_NAMES = (
    "idle",
    "slow_walk",
    "walk",
    "run",
    "idle_squat",
    "idle_kneel_two_legs",
    "idle_kneel",
    "idle_lying_face_down",
    "crawling",
    "idle_boxing",
    "walk_boxing",
    "left_punch",
    "right_punch",
    "random_punch",
    "elbow_crawling",
    "left_hook",
    "right_hook",
    "forward_jump",
    "stealth_walk",
    "injured_walk",
    "ledge_walking",
    "object_carrying",
    "stealth_walk_2",
    "happy_dance_walk",
    "zombie_walk",
    "gun_walk",
    "scare_walk",
)
STATIC_MODES = frozenset({0, 4, 5, 6, 7, 9})
ONE_SHOT_MODES = frozenset({11, 12, 13, 15, 16, 17})


def ort_providers(choice: str) -> list[str]:
    available = set(onnxruntime.get_available_providers())
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider was requested but is not available")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if choice != "auto":
        raise ValueError(f"Unknown ONNX Runtime provider choice: {choice}")
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-8)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q).copy()
    result[..., 1:] *= -1
    return result


def heading_quaternion(q: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(np.asarray(q, dtype=np.float64))
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    result = np.zeros(q.shape, dtype=np.float64)
    result[..., 0] = np.cos(yaw / 2)
    result[..., 3] = np.sin(yaw / 2)
    return result


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(np.asarray(q, dtype=np.float64))
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((*q.shape[:-1], 3, 3))


def quaternion_slerp(a: np.ndarray, b: np.ndarray, amount: np.ndarray | float) -> np.ndarray:
    a = _normalize_quaternion(np.asarray(a, dtype=np.float64))
    b = _normalize_quaternion(np.asarray(b, dtype=np.float64))
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    amount_array = np.asarray(amount, dtype=np.float64)
    while amount_array.ndim < a.ndim:
        amount_array = amount_array[..., None]
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    linear = (1 - amount_array) * a + amount_array * b
    spherical = (
        np.sin((1 - amount_array) * angle) / np.maximum(sin_angle, 1e-8) * a
        + np.sin(amount_array * angle) / np.maximum(sin_angle, 1e-8) * b
    )
    return _normalize_quaternion(np.where(sin_angle < 1e-6, linear, spherical)).astype(np.float32)


@dataclass(frozen=True)
class MovementCommand:
    mode: int
    speed: float
    height: float
    movement_direction: np.ndarray
    facing_direction: np.ndarray


@dataclass(frozen=True)
class MotionSequence:
    root_positions: np.ndarray
    root_quaternions: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.joint_positions.shape[0])


class SonicPlanner:
    """Runs the 30 Hz planner model and resamples its output to 50 Hz."""

    def __init__(
        self,
        model_path: str,
        *,
        provider: str,
        version: int,
        look_ahead_steps: int,
        default_height: float,
        seed: int,
        session=None,
    ):
        self.session = session or onnxruntime.InferenceSession(model_path, providers=ort_providers(provider))
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        expected = 6 if version == 0 else 11
        if len(self.input_names) != expected:
            raise ValueError(f"Planner v{version} expects {expected} inputs, model exposes {len(self.input_names)}")
        self.version = version
        self.look_ahead_steps = look_ahead_steps
        self.default_height = default_height
        self.seed = seed

    def initial_context(self, joint_positions_hw: np.ndarray) -> np.ndarray:
        context = np.zeros((1, 4, 36), dtype=np.float32)
        context[0, :, 2] = self.default_height
        context[0, :, 3] = 1.0
        context[0, :, 7:] = np.asarray(joint_positions_hw, dtype=np.float32)
        return context

    def motion_context(self, motion: MotionSequence, current_frame: int) -> np.ndarray:
        context = np.zeros((1, 4, 36), dtype=np.float32)
        generation_frame = current_frame + self.look_ahead_steps
        sample_frames = generation_frame + np.arange(4, dtype=np.float64) * (50.0 / 30.0)
        f0 = np.clip(np.floor(sample_frames).astype(np.int64), 0, motion.frames - 1)
        f1 = np.clip(f0 + 1, 0, motion.frames - 1)
        weights = sample_frames - f0
        context[0, :, :3] = (
            motion.root_positions[f0] * (1 - weights[:, None]) + motion.root_positions[f1] * weights[:, None]
        )
        context[0, :, 3:7] = quaternion_slerp(motion.root_quaternions[f0], motion.root_quaternions[f1], weights)
        interpolated = (
            motion.joint_positions[f0] * (1 - weights[:, None]) + motion.joint_positions[f1] * weights[:, None]
        )
        context_joints = context[0, :, 7:]
        context_joints[:, POLICY_TO_HW] = interpolated
        return context

    def infer(self, context: np.ndarray, command: MovementCommand) -> MotionSequence:
        if not 0 <= command.mode < (4 if self.version == 0 else 20 if self.version == 1 else 27):
            raise ValueError(f"Mode {command.mode} is unsupported by planner v{self.version}")
        feed: dict[str, np.ndarray] = {
            "context_mujoco_qpos": np.asarray(context, dtype=np.float32),
            "target_vel": np.asarray([command.speed], dtype=np.float32),
            "mode": np.asarray([command.mode], dtype=np.int64),
            "movement_direction": np.asarray(command.movement_direction, dtype=np.float32).reshape(1, 3),
            "facing_direction": np.asarray(command.facing_direction, dtype=np.float32).reshape(1, 3),
            "random_seed": np.asarray([self.seed], dtype=np.int64),
        }
        if self.version:
            feed.update(
                {
                    "height": np.asarray([command.height], dtype=np.float32),
                    "has_specific_target": np.zeros((1, 1), dtype=np.int64),
                    "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
                    "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
                    "allowed_pred_num_tokens": np.asarray(
                        [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64
                    ),
                }
            )
        missing = self.input_names - feed.keys()
        if missing:
            raise ValueError(f"Planner model has unsupported inputs: {sorted(missing)}")
        outputs = self.session.run(self.output_names, {name: feed[name] for name in self.input_names})
        by_name = dict(zip(self.output_names, outputs, strict=True))
        count = int(np.asarray(by_name["num_pred_frames"]).reshape(-1)[0])
        qpos = np.asarray(by_name["mujoco_qpos"], dtype=np.float32).reshape(-1, 36)[:count]
        if count < 2 or count > 64 or not np.isfinite(qpos).all():
            raise RuntimeError(f"Planner returned an invalid trajectory with {count} frames")
        return self._resample_50hz(qpos)

    @staticmethod
    def _resample_50hz(qpos_30hz: np.ndarray) -> MotionSequence:
        frame_count = int(np.floor(len(qpos_30hz) / 30.0 * 50.0))
        sample = np.arange(frame_count, dtype=np.float64) * (30.0 / 50.0)
        f0 = np.floor(sample).astype(np.int64)
        f1 = np.minimum(f0 + 1, len(qpos_30hz) - 1)
        weight = sample - f0
        positions = qpos_30hz[f0, :3] * (1 - weight[:, None]) + qpos_30hz[f1, :3] * weight[:, None]
        quaternions = quaternion_slerp(qpos_30hz[f0, 3:7], qpos_30hz[f1, 3:7], weight)
        joints_hw = qpos_30hz[f0, 7:] * (1 - weight[:, None]) + qpos_30hz[f1, 7:] * weight[:, None]
        joints_policy = joints_hw[:, POLICY_TO_HW]
        velocities = np.zeros_like(joints_policy)
        velocities[:-1] = np.diff(joints_policy, axis=0) * 50.0
        velocities[-1] = velocities[-2]
        return MotionSequence(
            positions.astype(np.float32),
            quaternions.astype(np.float32),
            joints_policy.astype(np.float32),
            velocities.astype(np.float32),
        )
