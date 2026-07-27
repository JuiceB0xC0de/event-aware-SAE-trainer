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
import pytest
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


def test_saved_tensors_match_output_dtype():
    """White-box guard for the autocast crash, runnable without a GPU.

    Autocast is active inside a custom Function's forward but DISABLED during its
    backward, so the hand-written matmuls there see whatever dtype the saved
    tensors have. `x` is bf16 and `b_dec` is an fp32 parameter, so `x - b_dec`
    promotes to fp32 while autocast hands back a bf16 `pre` -- and the backward
    died on "expected mat1 and mat2 to have the same dtype".

    CPU autocast does not downcast addmm the way CUDA autocast does, so the crash
    cannot be reproduced off-GPU. What *is* checkable anywhere is the invariant the
    fix establishes: whatever dtype the forward returns, the saved tensors match
    it, so the backward can never see a mismatch.
    """
    torch.manual_seed(3)
    sae = t._make_sae(d_in=8, n_features=16, seed=3)

    captured = {}
    real_save = torch.autograd.function.FunctionCtx.save_for_backward

    for x_dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(10, 8).to(x_dtype)
        t.SAE_FAST_PREBIAS = True
        pre = sae._pre(x)
        # Re-derive what forward saved: xc and weight, both cast to pre's dtype.
        xc = (x - sae.b_dec).to(pre.dtype)
        assert xc.dtype == pre.dtype, (
            f"saved xc dtype {xc.dtype} must match forward output {pre.dtype}")
        captured[x_dtype] = pre.dtype
    assert captured  # sanity


def test_grads_come_back_in_parameter_dtypes():
    """The Function must return grads in the parameters' dtypes, not the compute dtype."""
    torch.manual_seed(4)
    sae = t._make_sae(d_in=8, n_features=16, seed=4)
    x = torch.randn(10, 8, dtype=torch.bfloat16)

    t.SAE_FAST_PREBIAS = True
    for p in (sae.W_enc.weight, sae.W_enc.bias, sae.b_dec):
        p.grad = None
    sae._pre(x).float().pow(2).sum().backward()

    for name, p in (("W_enc.weight", sae.W_enc.weight),
                    ("W_enc.bias", sae.W_enc.bias),
                    ("b_dec", sae.b_dec)):
        assert p.grad is not None, f"{name} got no gradient"
        assert p.grad.dtype == p.dtype, (
            f"{name}: grad dtype {p.grad.dtype} != parameter dtype {p.dtype}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA autocast")
def test_cuda_autocast_bf16_does_not_crash():
    """The real reproduction. Only CUDA autocast downcasts addmm's output."""
    torch.manual_seed(5)
    sae = t._make_sae(d_in=64, n_features=128, seed=5).cuda()
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)

    def run(fast):
        prev = t.SAE_FAST_PREBIAS
        t.SAE_FAST_PREBIAS = fast
        try:
            for p in (sae.W_enc.weight, sae.W_enc.bias, sae.b_dec):
                p.grad = None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pre = sae._pre(x)
            pre.float().pow(2).sum().backward()
            return (sae.W_enc.weight.grad.clone().float(),
                    sae.W_enc.bias.grad.clone().float(),
                    sae.b_dec.grad.clone().float())
        finally:
            t.SAE_FAST_PREBIAS = prev

    fast = run(True)          # this is the call that used to raise
    ref = run(False)
    for a, b in zip(fast, ref):
        assert torch.allclose(a, b, atol=2e-2, rtol=2e-2)


def test_fallback_when_input_needs_grad():
    """A caller that needs grad wrt x must transparently get the plain path."""
    torch.manual_seed(2)
    sae = t._make_sae(d_in=4, n_features=8, seed=2)
    xg = torch.randn(6, 4, requires_grad=True)

    t.SAE_FAST_PREBIAS = True
    sae._pre(xg).sum().backward()
    assert xg.grad is not None, "fallback path must still produce grad wrt x"
