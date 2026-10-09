"""BFM-Zero's G1 sensor layout and field-major low-level actor history."""

from __future__ import annotations

import numpy as np

from vex_policy.robots.g1 import DOF_NAMES

JOINT_NAMES = tuple(DOF_NAMES)
LATENT_DIM = 256
FRAME_DIM = 93
ACTOR_HISTORY_LENGTH = 5
ACTOR_OBS_DIM = FRAME_DIM * ACTOR_HISTORY_LENGTH
ACTOR_CONTRACT_VERSION = 1
# Frozen checkpoint motor zeros; the kneeling reset pose does not redefine them.
DEFAULT_DOF_ANGLES = np.asarray([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 17, dtype=np.float32)


def pack_actor_observation(frames: np.ndarray) -> np.ndarray:
    """Pack five oldest-first frames as current state plus newest-first field histories."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (ACTOR_HISTORY_LENGTH, FRAME_DIM):
        raise ValueError("BFM actor history must have shape (N, 5, 93)")
    previous = frames[:, -2::-1]
    # Source history order: last_action, omega, q, dq, projected_gravity.
    history = [
        previous[:, :, start:end].reshape(len(frames), -1)
        for start, end in ((64, 93), (61, 64), (0, 29), (29, 58), (58, 61))
    ]
    return np.concatenate([frames[:, -1], *history], axis=-1)


def project_latent(latent: np.ndarray) -> np.ndarray:
    """Match torch F.normalize's epsilon, including finite zero vectors, with radius sqrt(256)."""
    latent = np.asarray(latent, dtype=np.float32)
    # Calculate the norm in float64 to avoid overflowing on finite controller outputs.
    norm = np.linalg.norm(latent.astype(np.float64), axis=-1, keepdims=True)
    return (latent / np.maximum(norm, 1e-12) * LATENT_DIM**0.5).astype(np.float32)
