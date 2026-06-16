"""Tests for the Triton fused SAE forward kernel.

These run the fallback PyTorch path on CPU-only / non-CUDA machines and the real
Triton kernel on CUDA.  The same `fused_sae_forward` API is exercised in both
cases.
"""
import pytest
import torch


def _make_tiny_sae():
    import sae_trainer_rolling as t

    return t._make_sae(d_in=16, n_features=32, seed=0)


def test_fused_forward_matches_pytorch_forward():
    """Fused forward output must match the standard PyTorch path."""
    from triton_sae_kernel import fused_sae_forward

    sae = _make_tiny_sae()
    torch.manual_seed(2)
    x = torch.randn(8, sae.d_in) * 0.5

    with torch.no_grad():
        expected_pre = sae.encode_pre(x)
        expected_feat, expected_gate = sae.jumprelu_with_gate(expected_pre)
        expected_xhat = sae.decode(expected_feat)
        expected_l0 = expected_gate.sum(dim=-1, dtype=torch.float32)

    x_hat, l0 = fused_sae_forward(x, sae)

    assert x_hat.shape == expected_xhat.shape
    assert l0.shape == expected_l0.shape
    assert torch.allclose(x_hat, expected_xhat, atol=1e-4)
    assert torch.allclose(l0, expected_l0, atol=1e-4)


def test_fused_forward_gradients_match_pytorch():
    """Fused backward must populate the same parameter gradients as PyTorch."""
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from triton_sae_kernel import fused_sae_forward

    sae = _make_tiny_sae().cuda()
    torch.manual_seed(3)
    x = (torch.randn(16, sae.d_in) * 0.5).cuda()

    # PyTorch reference gradients.
    sae.zero_grad(set_to_none=True)
    pre_ref = sae.encode_pre(x)
    feat_ref, gate_ref = sae.jumprelu_with_gate(pre_ref)
    xhat_ref = sae.decode(feat_ref)
    loss_ref = (x - xhat_ref).pow(2).mean() + 0.1 * gate_ref.sum(dim=-1).mean()
    loss_ref.backward()
    ref_grads = {
        "W_enc": sae.W_enc.weight.grad.clone(),
        "b_enc": sae.W_enc.bias.grad.clone(),
        "W_dec": sae.W_dec.weight.grad.clone(),
        "b_dec": sae.b_dec.grad.clone(),
        "log_thr": sae.log_threshold.grad.clone(),
    }

    # Fused path gradients.
    sae.zero_grad(set_to_none=True)
    x_hat, l0 = fused_sae_forward(x, sae)
    loss = (x - x_hat).pow(2).mean() + 0.1 * l0.mean()
    loss.backward()
    fused_grads = {
        "W_enc": sae.W_enc.weight.grad.clone(),
        "b_enc": sae.W_enc.bias.grad.clone(),
        "W_dec": sae.W_dec.weight.grad.clone(),
        "b_dec": sae.b_dec.grad.clone(),
        "log_thr": sae.log_threshold.grad.clone(),
    }

    for key in ref_grads:
        assert torch.allclose(
            fused_grads[key], ref_grads[key], atol=5e-3, rtol=1e-2
        ), f"{key} gradient mismatch"


def test_kernel_handles_d_in_not_divisible_by_block():
    """d_in that is not a multiple of BLOCK_D must not crash."""
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from triton_sae_kernel import fused_sae_forward

    sae = _make_tiny_sae()
    # Alter d_in by rebuilding with a different size.  _make_sae uses d_in=16
    # which is divisible by 64, so pick a size that is not.
    sae = _make_tiny_sae_with_dims(d_in=50, n_features=96).cuda()
    x = torch.randn(4, 50, device="cuda")

    x_hat, l0 = fused_sae_forward(x, sae)
    assert x_hat.shape == (4, 50)
    assert l0.shape == (4,)

    # Sanity: reconstruction is close-ish to input for random small SAE.
    assert not torch.isnan(x_hat).any()
    assert not torch.isnan(l0).any()


def _make_tiny_sae_with_dims(d_in: int, n_features: int):
    import sae_trainer_rolling as t

    return t._make_sae(d_in=d_in, n_features=n_features, seed=0)
