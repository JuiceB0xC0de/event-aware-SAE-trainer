"""Tests for the JumpReLU SAE architecture (`_make_sae`)."""
import math

import torch

import sae_trainer_rolling as t


def test_parameter_shapes(tiny_sae):
    sae = tiny_sae
    assert sae.W_enc.weight.shape == (8, 4)      # [n_features, d_in]
    assert sae.W_dec.weight.shape == (4, 8)      # [d_in, n_features]
    assert sae.b_dec.shape == (4,)
    assert sae.log_threshold.shape == (8,)


def test_decoder_columns_unit_norm_at_init(tiny_sae):
    # Each FEATURE's decoder direction is a column of W_dec.weight ([d_in, n_features]).
    col_norms = tiny_sae.W_dec.weight.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones(8), atol=1e-5)


def test_encoder_tied_to_decoder_transpose_at_init(tiny_sae):
    assert torch.allclose(tiny_sae.W_enc.weight, tiny_sae.W_dec.weight.t(), atol=1e-6)
    assert torch.allclose(tiny_sae.W_enc.bias, torch.zeros(8))


def test_log_threshold_init_value(tiny_sae):
    expected = math.log(t.INIT_THRESHOLD)
    assert torch.allclose(tiny_sae.log_threshold, torch.full((8,), expected), atol=1e-6)


def test_seed_determinism():
    a = t._make_sae(4, 8, seed=0)
    b = t._make_sae(4, 8, seed=0)
    c = t._make_sae(4, 8, seed=1)
    assert torch.equal(a.W_dec.weight, b.W_dec.weight)
    assert not torch.equal(a.W_dec.weight, c.W_dec.weight)


def test_jumprelu_gating_threshold():
    sae = t._make_sae(4, 4, seed=0)
    # threshold = exp(log(0.1)) = 0.1 for every feature
    pre = torch.tensor([[0.05, 0.2, -1.0, 0.5]])
    out = sae.apply_jumprelu(pre)
    expected = torch.tensor([[0.0, 0.2, 0.0, 0.5]])  # <=0.1 -> 0, else passthrough
    assert torch.allclose(out, expected, atol=1e-6)


def test_l0_indicator_is_binary_gate():
    sae = t._make_sae(4, 4, seed=0)
    pre = torch.tensor([[0.05, 0.2, -1.0, 0.5]])
    ind = sae.l0_indicator(pre)
    assert torch.equal(ind, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))


def test_forward_shapes_and_roundtrip(tiny_sae):
    x = torch.randn(3, 4)
    x_hat, acts = tiny_sae(x)
    assert x_hat.shape == (3, 4)
    assert acts.shape == (3, 8)


def test_encode_pre_equals_linear_when_bdec_zero(tiny_sae):
    x = torch.randn(5, 4)
    # b_dec is zeros at init, so encode_pre(x) == W_enc(x)
    assert torch.allclose(tiny_sae.encode_pre(x), tiny_sae.W_enc(x), atol=1e-6)


def test_backward_populates_gradients(tiny_sae):
    x = torch.randn(6, 4)
    x_hat, acts = tiny_sae(x)
    loss = (x - x_hat).pow(2).mean()
    loss.backward()
    assert tiny_sae.W_enc.weight.grad is not None
    assert tiny_sae.W_dec.weight.grad is not None
    assert tiny_sae.b_dec.grad is not None
    # threshold participates in the forward via the STE, so it gets a grad tensor
    assert tiny_sae.log_threshold.grad is not None


def test_normalize_decoder_restores_unit_columns(tiny_sae):
    with torch.no_grad():
        tiny_sae.W_dec.weight.mul_(7.3)   # blow up the columns
    tiny_sae._normalize_decoder()
    col_norms = tiny_sae.W_dec.weight.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones(8), atol=1e-5)


def test_fused_jumprelu_matches_separate(tiny_sae):
    """The fused JumpReLU+L0 path must produce the same forward values and
    stable gradients as a manual STE reference."""
    sae = tiny_sae
    torch.manual_seed(1)
    x = torch.randn(32, sae.d_in) * 0.5

    # Fused path.
    sae.zero_grad(set_to_none=True)
    pre = sae.encode_pre(x)
    feat, gate = sae.jumprelu_with_gate(pre)
    loss = feat.pow(2).sum() + 0.37 * gate.sum()
    loss.backward()
    g_thr = sae.log_threshold.grad.clone()
    g_wenc = sae.W_enc.weight.grad.clone()

    # Forward correctness.
    threshold = sae.log_threshold.exp()
    expected_gate = (pre > threshold).to(pre.dtype)
    expected_feat = pre * expected_gate
    assert torch.allclose(feat, expected_feat)
    assert torch.allclose(gate, expected_gate)

    # W_enc gradient: grad_pre = grad_feat * gate.
    grad_feat = 2 * feat  # from feat.pow(2).sum()
    expected_grad_pre = grad_feat * expected_gate
    # W_enc.weight is [n_features, d_in]; grad is expected_grad_pre.T @ x_centered
    x_centered = x - sae.b_dec
    expected_g_wenc = expected_grad_pre.t() @ x_centered
    assert torch.allclose(g_wenc, expected_g_wenc, atol=1e-5)

    # Threshold STE gradient: combined = pre * grad_feat + 0.37 * grad_gate,
    # where grad_gate is broadcast 0.37.
    grad_gate = torch.full_like(pre, 0.37)
    combined = pre * grad_feat + grad_gate
    eps = sae.ste_bandwidth
    in_band = (pre - threshold).abs() < eps
    expected_g_thr = -((in_band * combined).sum(dim=0) / (2 * eps)) * threshold
    assert torch.allclose(g_thr, expected_g_thr, atol=1e-5)
