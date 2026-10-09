"""Verify selective checkpoint loading and the normalizer/action convention inside actor ONNX."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch", reason="BFM conversion requires the bfm-export extra")

from vex_policy.bfm_export import FrozenActor, ResidualDecoder, export_actor  # noqa: E402
from vex_policy.policies.utils.inference import load_metadata  # noqa: E402


@pytest.fixture
def checkpoint(tmp_path):
    # Smaller widths exercise identical normalization, residual, LayerNorm and Mish operations.
    generator = torch.Generator().manual_seed(0)
    actor = FrozenActor(ResidualDecoder(hidden=16, layers=2))
    fields = ("state", "last_action", "history_actor")
    sizes = (64, 29, 372)
    statistics = {}
    means, variances = [], []
    for field, size in zip(fields, sizes, strict=True):
        mean = torch.randn(size, generator=generator)
        variance = torch.rand(size, generator=generator) + 0.2
        statistics[f"_obs_normalizer._normalizers.{field}._normalizer.running_mean"] = mean
        statistics[f"_obs_normalizer._normalizers.{field}._normalizer.running_var"] = variance
        means.append(mean)
        variances.append(variance)
    tensors = {"_actor." + key: value for key, value in actor.decoder.state_dict().items()}
    tensors.update(statistics)
    # Unrelated networks may be present and must never be loaded or validated by this tool.
    tensors["_discriminator.trunk.0.weight"] = torch.full((1, 1), float("nan"))
    safetensors.save_file(tensors, tmp_path / "model.safetensors")
    return actor, tensors, torch.cat(means), torch.cat(variances)


def test_checkpoint_normalization_residual_decoder_and_export_parity(checkpoint, tmp_path):
    actor, _, mean, variance = checkpoint
    actor.load_source(tmp_path)
    torch.testing.assert_close(actor.mean, mean)
    torch.testing.assert_close(actor.variance, variance)
    obs = torch.randn(1, 465)
    latent = torch.randn(1, 256)
    with torch.no_grad():
        expected = actor.decoder((obs - mean) / torch.sqrt(variance + 1e-5), latent) * 5.0
        torch.testing.assert_close(actor(obs, latent), expected)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        output = tmp_path / "actor.onnx"
        error = export_actor(actor, output)
        assert error < 1e-4
        metadata = load_metadata(str(output))
        assert metadata["bfm_actor_contract"] == 1
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
        actual = session.run(["action"], {"actor_obs": obs.numpy(), "latent": latent.numpy()})[0]
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)
        assert np.max(np.abs(actual)) <= 5.0
        with pytest.raises(FileExistsError):
            export_actor(actor, output)
        previous = output.read_bytes()
        # A failed numerical check must leave the existing export untouched.
        actor.variance.fill_(-1.0)
        with pytest.raises(ValueError, match="nonfinite"):
            export_actor(actor, output, overwrite=True)
        assert output.read_bytes() == previous
    finally:
        torch.set_num_threads(old_threads)


def test_checkpoint_requires_exact_actor_weights_and_valid_statistics(checkpoint, tmp_path):
    actor, tensors, _, _ = checkpoint
    invalid = dict(tensors)
    invalid["_obs_normalizer._normalizers.state._normalizer.running_var"] = torch.full((64,), -1.0)
    safetensors.save_file(invalid, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="invalid"):
        actor.load_source(tmp_path)
    invalid = dict(tensors)
    invalid.pop(next(key for key in invalid if key.startswith("_actor.")))
    safetensors.save_file(invalid, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        actor.load_source(tmp_path)
