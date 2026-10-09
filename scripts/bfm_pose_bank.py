#!/usr/bin/env python3
"""Precompute the bfm_pose_transition deployment bank from Holosoma's offline data.

Holosoma's controller observes, per goal, a cached BFM latent (backward-encoded
offline) and a fixed 33-D target feature: endpoint joints relative to the frozen
actor's motor zeros, endpoint projected gravity and endpoint root height. Both
are constant per pose, so this script resolves them once; the runtime policy
only loads the result. Requires numpy only.

    python scripts/bfm_pose_bank.py \\
        --goal-bank ~/robot/holosoma/src/holosoma/holosoma/data/bfm_goal_bank.npz \\
        --endpoint-reference ~/robot/holosoma/src/holosoma/holosoma/data/bfm_endpoint_reference.npz \\
        --output models/bfm/pose_bank.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Frozen BFM checkpoint motor zeros (Holosoma g1 bfm init_state default_joint_angles).
DEFAULT_DOF_ANGLES = np.asarray([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 17, dtype=np.float32)


def build(goal_bank: Path, endpoint_reference: Path) -> dict[str, np.ndarray]:
    with np.load(goal_bank, allow_pickle=False) as bank:
        names = bank["pose_names"]
        joint_names = bank["joint_names"]
        latents = bank["latents"].astype(np.float32)
        manifest = json.loads(str(bank["manifest"]))
    with np.load(endpoint_reference, allow_pickle=False) as reference:
        if reference["pose_names"].tolist() != names.tolist():
            raise ValueError("Endpoint reference goal ordering differs from the latent bank")
        if str(reference["checkpoint_sha256"]) != manifest["checkpoint_sha256"]:
            raise ValueError("Endpoint reference was measured with a different frozen actor")
        joints, gravity, height = reference["joints"], reference["gravity"], reference["height"]
    n = len(names)
    if len(joint_names) != 29 or latents.shape != (n, 256) or joints.shape != (n, 29):
        raise ValueError("Unexpected goal bank shapes")
    if not np.allclose(np.linalg.norm(latents, axis=-1), 16, atol=1e-4):
        raise ValueError("Goal bank latents must have source radius 16")
    # Holosoma target_pose term: target_dof_pos - default_dof_pos, target_gravity, target_height.
    targets = np.concatenate((joints - DEFAULT_DOF_ANGLES, gravity, height[:, None]), axis=1).astype(np.float32)
    if not np.isfinite(targets).all() or not np.isfinite(latents).all():
        raise ValueError("Goal bank contains nonfinite values")
    return {
        "pose_names": names,
        "joint_names": joint_names,
        "latents": latents,
        "targets": targets,
        "checkpoint_sha256": np.asarray(manifest["checkpoint_sha256"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goal-bank", type=Path, required=True)
    parser.add_argument("--endpoint-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/bfm/pose_bank.npz"))
    args = parser.parse_args()
    data = build(args.goal_bank.expanduser(), args.endpoint_reference.expanduser())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    print(f"Wrote {args.output}: poses {data['pose_names'].tolist()}")


if __name__ == "__main__":
    main()
