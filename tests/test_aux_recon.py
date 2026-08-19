"""The sparse dead-feature aux reconstruction must equal the dense one.

The aux loss revives dead features by asking them to explain the current
reconstruction residual. It is top-k sparse by construction (AUX_K entries per
token), but was computed densely: a [B, n_dead] zeros buffer, a scatter, a
[n_dead, d] decoder-row gather per microbatch, and a matmul over a mostly-zero
matrix.

`_dead_feature_aux_recon` replaces that with a gather-sum. These tests pin that
the swap changes nothing that matters -- values and gradients -- because the aux
path drives dead-feature recovery and a silent change there would show up as a
worse dead% trajectory rather than as a failure.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sae_trainer_rolling as T  # noqa: E402

B = 24
D_IN = 12
N_FEATURES = 40
SEED = 7


def _dense_reference(pre, dead_indices, eff_k, w_dec_weight):
    """The original formulation, kept here as the thing under comparison."""
    pre_dead = pre[:, dead_indices].relu()
    topk_vals, topk_idx = pre_dead.topk(eff_k, dim=-1)
    aux_acts = torch.zeros_like(pre_dead)
    aux_acts.scatter_(-1, topk_idx, topk_vals)
    w_dec_dead = w_dec_weight.t()[dead_indices]
    return aux_acts @ w_dec_dead


def _setup(dtype=torch.float64, n_dead=17, eff_k=5):
    g = torch.Generator().manual_seed(SEED)
    pre = torch.randn(B, N_FEATURES, generator=g, dtype=dtype)
    w_dec = torch.randn(D_IN, N_FEATURES, generator=g, dtype=dtype)
    dead_indices = torch.randperm(N_FEATURES, generator=g)[:n_dead].sort().values
    return pre, w_dec, dead_indices, eff_k


@pytest.mark.parametrize("n_dead,eff_k", [(17, 5), (40, 1), (3, 3), (25, 25)])
def test_values_match_dense(n_dead, eff_k):
    pre, w_dec, dead_indices, eff_k = _setup(n_dead=n_dead, eff_k=eff_k)
    expected = _dense_reference(pre.clone(), dead_indices, eff_k, w_dec)
    got = T._dead_feature_aux_recon(pre.clone(), dead_indices, eff_k, w_dec)
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-10)


def test_gradients_match_dense():
    pre, w_dec, dead_indices, eff_k = _setup()

    pre_a = pre.clone().requires_grad_(True)
    w_a = w_dec.clone().requires_grad_(True)
    _dense_reference(pre_a, dead_indices, eff_k, w_a).pow(2).sum().backward()

    pre_b = pre.clone().requires_grad_(True)
    w_b = w_dec.clone().requires_grad_(True)
    T._dead_feature_aux_recon(pre_b, dead_indices, eff_k, w_b).pow(2).sum().backward()

    torch.testing.assert_close(pre_b.grad, pre_a.grad, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(w_b.grad, w_a.grad, rtol=1e-9, atol=1e-9)


def test_gradient_reaches_only_dead_features():
    """Live features must receive no gradient from the aux term."""
    pre, w_dec, dead_indices, eff_k = _setup()
    pre_b = pre.clone().requires_grad_(True)
    T._dead_feature_aux_recon(pre_b, dead_indices, eff_k, w_dec).pow(2).sum().backward()

    live = torch.ones(N_FEATURES, dtype=torch.bool)
    live[dead_indices] = False
    assert pre_b.grad[:, live].abs().max() == 0


def test_negative_pre_activations_are_relu_gated():
    """A dead feature with an all-negative pre-activation contributes nothing."""
    pre, w_dec, dead_indices, eff_k = _setup()
    with torch.no_grad():
        pre[:, dead_indices] = -1.0
    got = T._dead_feature_aux_recon(pre.clone(), dead_indices, eff_k, w_dec)
    torch.testing.assert_close(got, torch.zeros_like(got), rtol=0, atol=0)


def test_does_not_mutate_caller_pre():
    """The helper relu_'s a gathered copy, never the caller's tensor."""
    pre, w_dec, dead_indices, eff_k = _setup()
    before = pre.clone()
    T._dead_feature_aux_recon(pre, dead_indices, eff_k, w_dec)
    torch.testing.assert_close(pre, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_runs_under_training_dtypes(dtype):
    """bf16 activations against fp32 decoder weights is the real training mix;
    embedding_bag requires per_sample_weights to match the weight dtype."""
    g = torch.Generator().manual_seed(SEED)
    pre = torch.randn(B, N_FEATURES, generator=g).to(dtype)
    w_dec = torch.randn(D_IN, N_FEATURES, generator=g)  # fp32 params
    dead_indices = torch.randperm(N_FEATURES, generator=g)[:17].sort().values
    out = T._dead_feature_aux_recon(pre, dead_indices, 5, w_dec)
    assert out.shape == (B, D_IN)
    assert torch.isfinite(out).all()
