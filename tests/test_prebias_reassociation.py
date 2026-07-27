"""The reassociated encoder pre-bias gradient must match plain autograd exactly.

The fast path computes d(loss)/d(b_dec) as `-(sum_b dpre_b) @ W_enc` instead of
letting autograd build the full [B, F] x [F, d] grad_input and then reduce it over
the batch. That removes one of the six equal-sized GEMMs in a JumpReLU step, and it
is an algebraic identity rather than an approximation -- so the gradients reaching
W_enc, b_enc and b_dec must agree with the naive graph to floating-point tolerance.

Both paths are driven through `SAE._pre`, toggled by the module-level
SAE_FAST_PREBIAS flag, so this tests the code that actually runs rather than a
private symbol. If it ever fails, SAE_FAST_PREBIAS=0 is the escape hatch.
"""
import torch

import sae_trainer_rolling as t


def _grads_through_pre(sae, x, fast: bool):
    """Run sae._pre under the chosen path and return (dW, db_enc, db_dec)."""
    prev = t.SAE_FAST_PREBIAS
    t.SAE_FAST_PREBIAS = fast
    try:
        for p in (sae.W_enc.weight, sae.W_enc.bias, sae.b_dec):
            p.grad = None
        pre = sae._pre(x)
        # Non-symmetric weighting so every gradient component is exercised.
        w = torch.linspace(0.5, 2.0, pre.numel(), dtype=pre.dtype).reshape(pre.shape)
        (pre * w).sum().backward()
        return (sae.W_enc.weight.grad.clone(),
                sae.W_enc.bias.grad.clone(),
                sae.b_dec.grad.clone())
    finally:
        t.SAE_FAST_PREBIAS = prev


def test_reassociated_gradients_match_autograd():
    torch.manual_seed(0)
    sae = t._make_sae(d_in=5, n_features=11, seed=0).double()   # non-square, odd sizes
    x = torch.randn(7, 5, dtype=torch.float64)                  # activations: no grad

    fast_w, fast_be, fast_bd = _grads_through_pre(sae, x, fast=True)
    ref_w, ref_be, ref_bd = _grads_through_pre(sae, x, fast=False)

    assert torch.allclose(fast_w, ref_w, atol=1e-10), "W_enc gradient diverged"
    assert torch.allclose(fast_be, ref_be, atol=1e-10), "b_enc gradient diverged"
    assert torch.allclose(fast_bd, ref_bd, atol=1e-10), "b_dec gradient diverged"


def test_forward_value_matches():
    torch.manual_seed(1)
    sae = t._make_sae(d_in=6, n_features=9, seed=1).double()
    x = torch.randn(4, 6, dtype=torch.float64)

    t.SAE_FAST_PREBIAS = True
    fast = sae._pre(x)
    ref = sae.W_enc(x - sae.b_dec)
    assert torch.allclose(fast, ref, atol=1e-12)


def test_fallback_when_input_needs_grad():
    """A caller that needs grad wrt x must transparently get the plain path."""
    torch.manual_seed(2)
    sae = t._make_sae(d_in=4, n_features=8, seed=2)
    xg = torch.randn(6, 4, requires_grad=True)

    t.SAE_FAST_PREBIAS = True
    sae._pre(xg).sum().backward()
    assert xg.grad is not None, "fallback path must still produce grad wrt x"
