"""The gathered decoder must be numerically identical to the dense one.

`_SparseDecode` evaluates `acts @ W_dec.T + b_dec` over the active set only, which
is exact because the skipped terms are multiplication by zero. Two of the six equal
GEMMs in a JumpReLU step disappear (decoder forward, decoder weight gradient).

`grad_acts` is deliberately left dense. The threshold STE masks on
`(pre - threshold).abs() < bandwidth` -- a band spanning BOTH sides of the
threshold -- and consumes `pre * grad_feat`. Features just below threshold are
inactive, so a gathered gradient there would zero exactly the signal that pulls
the threshold down. These tests pin that: forward, all three gradients, and the
fallback when the active set is too wide.
"""
import pytest
import torch

import sae_trainer_rolling as t


def _sparse_acts(B, F, k, dtype=torch.float64, seed=0):
    """Activations with exactly k nonzeros per row, like a JumpReLU active set."""
    g = torch.Generator().manual_seed(seed)
    acts = torch.zeros(B, F, dtype=dtype)
    for b in range(B):
        idx = torch.randperm(F, generator=g)[:k]
        acts[b, idx] = torch.rand(k, generator=g, dtype=dtype) + 0.5
    return acts


def _run(sae, acts, sparse: bool):
    prev = t.SAE_SPARSE_DECODE
    t.SAE_SPARSE_DECODE = sparse
    try:
        for p in (sae.W_dec.weight, sae.b_dec):
            p.grad = None
        a = acts.detach().clone().requires_grad_(True)
        out = sae.decode(a)
        w = torch.linspace(0.3, 1.7, out.numel(), dtype=out.dtype).reshape(out.shape)
        (out * w).sum().backward()
        return out.detach(), sae.W_dec.weight.grad.clone(), sae.b_dec.grad.clone(), a.grad.clone()
    finally:
        t.SAE_SPARSE_DECODE = prev


def test_forward_and_all_gradients_match_dense():
    sae = t._make_sae(d_in=7, n_features=64, seed=0).double()
    acts = _sparse_acts(9, 64, k=5)

    s_out, s_dw, s_db, s_da = _run(sae, acts, sparse=True)
    d_out, d_dw, d_db, d_da = _run(sae, acts, sparse=False)

    assert torch.allclose(s_out, d_out, atol=1e-12), "forward diverged"
    assert torch.allclose(s_dw, d_dw, atol=1e-12), "W_dec gradient diverged"
    assert torch.allclose(s_db, d_db, atol=1e-12), "b_dec gradient diverged"
    assert torch.allclose(s_da, d_da, atol=1e-12), "acts gradient diverged"


def test_grad_acts_is_dense_not_gathered():
    """The whole point: inactive features must still receive a gradient.

    If this ever returns zeros off the active set, the STE loses its
    below-threshold band and thresholds stop adapting downward.
    """
    sae = t._make_sae(d_in=6, n_features=32, seed=1).double()
    acts = _sparse_acts(5, 32, k=3, seed=1)

    _, _, _, s_da = _run(sae, acts, sparse=True)
    inactive = acts == 0
    assert (s_da[inactive].abs() > 0).any(), (
        "gradient wrt inactive features is identically zero -- STE band would break")


def test_chunking_does_not_change_the_result():
    sae = t._make_sae(d_in=5, n_features=48, seed=2).double()
    acts = _sparse_acts(16, 48, k=4, seed=2)

    prev = t.SAE_SPARSE_DECODE_CHUNK
    try:
        t.SAE_SPARSE_DECODE_CHUNK = 10**9      # one chunk
        big = _run(sae, acts, sparse=True)
        t.SAE_SPARSE_DECODE_CHUNK = 3          # many chunks, uneven tail
        small = _run(sae, acts, sparse=True)
    finally:
        t.SAE_SPARSE_DECODE_CHUNK = prev

    for a, b, name in zip(big, small, ("out", "dW", "db", "dacts")):
        assert torch.allclose(a, b, atol=1e-12), f"{name} changed with chunk size"


def test_falls_back_when_active_set_exceeds_kmax():
    """Too many actives must fall back to dense, never silently truncate."""
    sae = t._make_sae(d_in=5, n_features=40, seed=3).double()
    acts = _sparse_acts(6, 40, k=30, seed=3)

    prev = t.SAE_SPARSE_DECODE_KMAX
    try:
        t.SAE_SPARSE_DECODE_KMAX = 8           # below the 30 actives present
        s_out, s_dw, s_db, s_da = _run(sae, acts, sparse=True)
    finally:
        t.SAE_SPARSE_DECODE_KMAX = prev
    d_out, d_dw, d_db, d_da = _run(sae, acts, sparse=False)

    assert torch.allclose(s_out, d_out, atol=1e-12)
    assert torch.allclose(s_dw, d_dw, atol=1e-12)
    assert torch.allclose(s_da, d_da, atol=1e-12)


def test_all_zero_activations_fall_back():
    sae = t._make_sae(d_in=4, n_features=16, seed=4).double()
    acts = torch.zeros(3, 16, dtype=torch.float64)

    s_out, _, _, _ = _run(sae, acts, sparse=True)
    d_out, _, _, _ = _run(sae, acts, sparse=False)
    assert torch.allclose(s_out, d_out, atol=1e-12)


def test_end_to_end_sae_forward_matches():
    """Through SAE.forward, so encoder + JumpReLU + decode all compose."""
    torch.manual_seed(5)
    sae = t._make_sae(d_in=8, n_features=64, seed=5).double()
    x = torch.randn(12, 8, dtype=torch.float64)

    def run(sparse):
        prev = t.SAE_SPARSE_DECODE
        t.SAE_SPARSE_DECODE = sparse
        try:
            for p in sae.parameters():
                p.grad = None
            xh, a = sae(x)
            xh.pow(2).sum().backward()
            return xh.detach(), sae.log_threshold.grad.clone(), sae.W_enc.weight.grad.clone()
        finally:
            t.SAE_SPARSE_DECODE = prev

    s_xh, s_thr, s_wenc = run(True)
    d_xh, d_thr, d_wenc = run(False)

    assert torch.allclose(s_xh, d_xh, atol=1e-12), "reconstruction diverged"
    # The STE threshold gradient is the thing sparsifying grad_acts would break.
    assert torch.allclose(s_thr, d_thr, atol=1e-10), "threshold STE gradient diverged"
    assert torch.allclose(s_wenc, d_wenc, atol=1e-10), "encoder gradient diverged"
