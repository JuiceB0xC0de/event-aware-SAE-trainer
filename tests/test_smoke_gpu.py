"""CUDA-only smoke test: one forward+backward of the SAE on device.

Skipped automatically when no GPU is present so the suite stays green on CPU."""
import pytest
import torch

import sae_trainer_rolling as t

pytestmark = pytest.mark.gpu

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device")


@requires_cuda
def test_sae_forward_backward_on_cuda():
    device = torch.device("cuda")
    sae = t._make_sae(d_in=8, n_features=16, seed=0).to(device)
    x = torch.randn(4, 8, device=device)
    x_hat, acts = sae(x)
    assert x_hat.shape == (4, 8)
    assert acts.shape == (4, 16)
    loss = (x - x_hat).pow(2).mean()
    loss.backward()
    assert sae.W_enc.weight.grad is not None


@requires_cuda
def test_provider_yields_batches_on_cuda(tmp_path):
    # RollingActivationProvider uses pin_memory(), which needs CUDA.
    from pathlib import Path
    d = Path(tmp_path) / "pool"
    n_seqs, seq, dim = 2, 4, 8
    for i in range(3):
        t._write_shard(d, i, torch.randn(n_seqs, seq, dim))
    provider = t.RollingActivationProvider(d, torch.device("cuda"), seed=0)
    try:
        batch = provider.next_batch()
        assert batch.shape == (n_seqs * seq, dim)
        assert batch.dtype == torch.float32
        assert batch.is_cuda
    finally:
        provider.close()
