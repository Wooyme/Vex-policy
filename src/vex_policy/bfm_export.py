"""Offline conversion of Holosoma's frozen BFM-Zero actor (CC BY-NC 4.0).

The inference architecture is adapted from holosoma/utils/bfm_models.py,
following BFM-Zero/humanoidverse (Meta Platforms et al.). See NOTICE and
LICENSE-BFM-Zero.txt. Runtime policies do not import this PyTorch module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from vex_policy.policies.utils.bfm import (
    ACTOR_CONTRACT_VERSION,
    ACTOR_OBS_DIM,
    DEFAULT_DOF_ANGLES,
    JOINT_NAMES,
    LATENT_DIM,
    project_latent,
)


class Block(nn.Module):
    """Checkpoint-compatible layer normalization, linear projection and optional Mish."""

    def __init__(self, input_dim, output_dim, activation=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), *([nn.Mish()] if activation else [])
        )

    def forward(self, value):
        return self.mlp(value)


class ResidualBlock(Block):
    """A same-width checkpoint block with its residual connection."""

    def __init__(self, size):
        super().__init__(size, size)

    def forward(self, value):
        return value + self.mlp(value)


class ResidualDecoder(nn.Module):
    """The supported checkpoint uses width 2048, six residual blocks and two-layer embeddings."""

    def __init__(self, hidden=2048, layers=6):
        super().__init__()
        self.embed_s = nn.Sequential(Block(ACTOR_OBS_DIM, hidden), Block(hidden, hidden // 2))
        self.embed_z = nn.Sequential(Block(ACTOR_OBS_DIM + LATENT_DIM, hidden), Block(hidden, hidden // 2))
        self.policy = nn.Sequential(*[ResidualBlock(hidden) for _ in range(layers)], Block(hidden, 29, False))

    def forward(self, obs, latent):
        state = self.embed_s(obs)
        conditioned = self.embed_z(torch.cat((obs, latent), dim=-1))
        return torch.tanh(self.policy(torch.cat((state, conditioned), dim=-1)))


class FrozenActor(nn.Module):
    """Embed immutable normalization and tanh * 5 so runtime consumes motor actions directly."""

    def __init__(self, decoder: ResidualDecoder | None = None):
        super().__init__()
        self.decoder = decoder if decoder is not None else ResidualDecoder()
        self.register_buffer("mean", torch.zeros(ACTOR_OBS_DIM))
        self.register_buffer("variance", torch.ones(ACTOR_OBS_DIM))
        self.requires_grad_(False)
        self.eval()

    def load_source(self, checkpoint_path: str | Path) -> None:
        """Materialize only actor weights and the three actor observation normalizers on CPU."""
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("BFM export requires the 'bfm-export' extra: uv sync --extra bfm-export") from error
        path = Path(checkpoint_path).expanduser() / "model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as source:
            keys = source.keys()
            weights = {key[len("_actor.") :]: source.get_tensor(key) for key in keys if key.startswith("_actor.")}
            self.decoder.load_state_dict(weights, strict=True)
            for name, target in (("running_mean", self.mean), ("running_var", self.variance)):
                values = [
                    source.get_tensor(f"_obs_normalizer._normalizers.{field}._normalizer.{name}")
                    for field in ("state", "last_action", "history_actor")
                ]
                target.copy_(torch.cat(values))
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.variance).all()
            or (self.variance < 0).any()
            or any(not torch.isfinite(p).all() for p in self.decoder.parameters())
        ):
            raise ValueError("BFM checkpoint contains invalid weights or normalization statistics")

    def forward(self, actor_obs, latent):
        # Source BatchNorm uses epsilon 1e-5; action rescale is five, before per-joint PD scaling.
        normalized = (actor_obs - self.mean) * torch.rsqrt(self.variance + 1e-5)
        return self.decoder(normalized, latent) * 5.0


def export_actor(actor: FrozenActor, output_path: str | Path, *, overwrite: bool = False) -> float:
    """Write a verified batch-one ONNX atomically; return the maximum CPU parity error."""
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".onnx":
        raise ValueError("BFM actor output must have an .onnx extension")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}; use --overwrite to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    actor = actor.cpu().eval()
    with TemporaryDirectory(prefix=".bfm-export-", dir=destination.parent) as staging:
        temporary = Path(staging) / "actor.onnx"
        with torch.no_grad():
            torch.onnx.export(
                actor,
                (torch.zeros(1, ACTOR_OBS_DIM), torch.zeros(1, LATENT_DIM)),
                str(temporary),
                input_names=["actor_obs", "latent"],
                output_names=["action"],
                opset_version=17,
                dynamo=False,
            )
        model = onnx.load(temporary)
        onnx.helper.set_model_props(
            model,
            {
                key: json.dumps(value)
                for key, value in {
                    "bfm_actor_contract": ACTOR_CONTRACT_VERSION,
                    "dof_names": JOINT_NAMES,
                    "default_dof_angles": DEFAULT_DOF_ANGLES.tolist(),
                }.items()
            },
        )
        onnx.checker.check_model(model)
        onnx.save(model, temporary)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        maximum_error = 0.0
        # Test zero latent and two independent normalized latents with nonzero observations.
        for index in range(3):
            obs = rng.normal(size=(1, ACTOR_OBS_DIM)).astype(np.float32)
            latent = (
                np.zeros((1, LATENT_DIM), dtype=np.float32)
                if index == 0
                else project_latent(rng.normal(size=(1, LATENT_DIM)).astype(np.float32))
            )
            with torch.no_grad():
                expected = actor(torch.from_numpy(obs), torch.from_numpy(latent)).numpy()
            actual = session.run(["action"], {"actor_obs": obs, "latent": latent})[0]
            if not np.isfinite(actual).all():
                raise ValueError("Exported BFM actor produced nonfinite actions")
            # LayerNorm/Mish can accumulate small float32 differences across backends.
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        del session, model
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during export: {destination}")
        temporary.replace(destination)
    return maximum_error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Directory containing model.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="Destination actor ONNX file")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    # Export is offline and serial; avoid CPU oversubscription during parity checks.
    torch.set_num_threads(1)
    actor = FrozenActor()
    actor.load_source(args.checkpoint_path)
    error = export_actor(actor, args.output, overwrite=args.overwrite)
    print(f"Exported BFM actor: {args.output} (maximum absolute parity error: {error:.3g})")


if __name__ == "__main__":
    main()
