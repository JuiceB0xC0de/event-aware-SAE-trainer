"""Closed-form check on the JumpReLU straight-through estimator.

This is the guard for any change to the gate / threshold gradient path. It does
not compare the code to itself -- it compares the code to the STE formula worked
out by hand, so a refactor that silently alters the gradient fails here even if
it is internally self-consistent.

The gradient under test (see `_log_threshold_hook` in sae_trainer_rolling.py):

    in_band   = |pre - threshold| < eps
    combined  = pre * grad_feat + grad_gate
    d/dthr    = -sum_over_tokens(where(in_band, combined, 0)) / (2 * eps)
    d/dlogthr = d/dthr * threshold          # chain rule, threshold = exp(log_threshold)

Loss used: L = sum(feat_acts^2) + c * sum(gate)
    -> grad_feat = 2 * feat_acts
    -> grad_gate = c
both of which are exact, so the expected threshold gradient is closed-form.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sae_trainer_rolling as T  # noqa: E402

D_IN = 16
N_FEATURES = 32
N_TOKENS = 64
C_GATE = 0.37
SEED = 1234
TOL = 2e-5


def _build():
    torch.manual_seed(SEED)
    sae = T._make_sae(D_IN, N_FEATURES, seed=SEED).to(torch.float64)
    # Spread thresholds so a meaningful number of pre-activations land in-band.
    with torch.no_grad():
        sae.log_threshold.copy_(
            torch.linspace(-1.5, 0.0, N_FEATURES, dtype=torch.float64)
        )
    return sae


def _inputs():
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(N_TOKENS, D_IN, generator=g, dtype=torch.float64)


def _loss(feat_acts, gate):
    return (feat_acts.pow(2).sum() + C_GATE * gate.sum())


def test_ste_threshold_gradient_matches_closed_form():
    sae = _build()
    x = _inputs()

    pre = sae.encode_pre(x)
    feat_acts, gate = sae.jumprelu_with_gate(pre)
    loss = _loss(feat_acts, gate)

    sae.zero_grad(set_to_none=True)
    loss.backward()
    got = sae.log_threshold.grad.clone()

    # -- closed form ------------------------------------------------------
    with torch.no_grad():
        pre_d = sae.encode_pre(x)
        threshold = sae.log_threshold.exp()
        eps = sae.ste_bandwidth
        gate_hard = (pre_d > threshold).to(pre_d.dtype)
        feat_d = pre_d * gate_hard

        grad_feat = 2.0 * feat_d          # dL/d feat_acts
        grad_gate = C_GATE                # dL/d gate
        combined = pre_d * grad_feat + grad_gate

        in_band = (pre_d - threshold).abs() < eps
        masked = torch.where(in_band, combined, torch.zeros_like(combined))
        expected = -(masked.sum(dim=0) / (2 * eps)) * threshold

    assert in_band.any(), "no pre-activations landed in band; test is vacuous"
    torch.testing.assert_close(got, expected, rtol=TOL, atol=TOL)


def test_grad_pre_is_gated():
    """grad wrt the encoder pre-activation must be grad_feat * gate (hard gate)."""
    sae = _build()
    x = _inputs()

    pre = sae.encode_pre(x).detach().requires_grad_(True)
    feat_acts, gate = sae.jumprelu_with_gate(pre)
    _loss(feat_acts, gate).backward()

    with torch.no_grad():
        threshold = sae.log_threshold.exp()
        gate_hard = (pre.detach() > threshold).to(pre.dtype)
        expected = 2.0 * (pre.detach() * gate_hard) * gate_hard

    torch.testing.assert_close(pre.grad, expected, rtol=TOL, atol=TOL)


def test_gate_is_exactly_the_hard_indicator():
    """The returned gate must equal (pre > threshold) in value, whatever grad
    plumbing is attached to it."""
    sae = _build()
    x = _inputs()
    pre = sae.encode_pre(x)
    _, gate = sae.jumprelu_with_gate(pre)
    with torch.no_grad():
        expected = (pre > sae.log_threshold.exp()).to(pre.dtype)
    torch.testing.assert_close(gate.detach(), expected, rtol=0, atol=0)


def test_feat_acts_are_pre_times_gate():
    sae = _build()
    x = _inputs()
    pre = sae.encode_pre(x)
    feat_acts, gate = sae.jumprelu_with_gate(pre)
    torch.testing.assert_close(feat_acts.detach(), (pre * gate).detach(),
                               rtol=0, atol=0)


@pytest.mark.parametrize("n_accum", [1, 2, 4])
def test_threshold_gradient_accumulates_across_microbatches(n_accum):
    """Gradient accumulation must sum the per-microbatch STE contributions.

    The live path stores backward state on the module (`_saved_pre` and friends),
    which is reset at the top of every forward. This test pins the behaviour so a
    change to that state handling cannot silently drop microbatch contributions.
    """
    sae = _build()
    x = _inputs()
    chunks = x.chunk(n_accum, dim=0)

    sae.zero_grad(set_to_none=True)
    for chunk in chunks:
        pre = sae.encode_pre(chunk)
        feat_acts, gate = sae.jumprelu_with_gate(pre)
        _loss(feat_acts, gate).backward()
    accumulated = sae.log_threshold.grad.clone()

    sae.zero_grad(set_to_none=True)
    pre = sae.encode_pre(x)
    feat_acts, gate = sae.jumprelu_with_gate(pre)
    _loss(feat_acts, gate).backward()
    single = sae.log_threshold.grad.clone()

    torch.testing.assert_close(accumulated, single, rtol=TOL, atol=TOL)
