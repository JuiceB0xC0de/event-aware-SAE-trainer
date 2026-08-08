"""Tests for the quasi-orthogonality diagnostic and its wiring into the trainer.

Exercises `sae_diagnostics.quasi_orthogonality_signal` against real SAE forwards
built by `sae_trainer_rolling._make_sae` (the non-new call-site module), matching
what the per-log-step quality block now computes.
"""
import math

import torch

import sae_trainer_rolling as t
from sae_diagnostics import _c_qo, quasi_orthogonality_signal


def test_signal_keys_and_types(tiny_sae):
    x = torch.randn(6, 4)
    # Same two tensors the log step captures: xh_p, z_p = sae(acts_mb).
    x_hat, z = tiny_sae(x)
    qo = quasi_orthogonality_signal(z, x_hat, x)
    assert set(qo) == {"qo_ratio", "qo_gap", "feat_l2", "recon_l2", "input_l2"}
    for v in qo.values():
        assert isinstance(v, float) and math.isfinite(v)
    assert qo["qo_gap"] >= 0.0


def test_single_active_feature_has_zero_gap(tiny_sae):
    # With unit-norm decoder atoms, one active feature per token gives
    # x_hat = z_i * d_i, so ||x_hat|| == |z_i| == ||z||: gap is exactly 0
    # regardless of how the other atoms are oriented.
    z = torch.zeros(3, 8)
    z[0, 0] = 1.5
    z[1, 3] = -2.0
    z[2, 5] = 0.7
    x_hat = tiny_sae.decode(z)  # b_dec is zeros at init
    qo = quasi_orthogonality_signal(z, x_hat, x_hat)
    assert qo["qo_gap"] < 1e-5
    assert abs(qo["qo_ratio"] - 1.0) < 1e-5


def test_redundant_features_raise_gap(tiny_sae):
    # Two co-activated features sharing a (unit-norm) decoder direction load
    # the off-diagonal Gram mass that qo_gap proxies: ||z||=sqrt(2) but
    # ||x_hat||=2, so the gap is well above zero.
    with torch.no_grad():
        tiny_sae.W_dec.weight[:, 1] = tiny_sae.W_dec.weight[:, 0]
    z = torch.zeros(1, 8)
    z[0, 0] = 1.0
    z[0, 1] = 1.0
    x_hat = tiny_sae.decode(z)
    qo = quasi_orthogonality_signal(z, x_hat, x_hat)
    assert qo["qo_gap"] > 0.2


def test_empty_batch_returns_none():
    z = torch.zeros(0, 8)
    x = torch.zeros(0, 4)
    qo = quasi_orthogonality_signal(z, x, x)
    assert qo["qo_gap"] is None and qo["qo_ratio"] is None


def test_bf16_activations_do_not_break_diagnostic(tiny_sae):
    x = torch.randn(5, 4).to(torch.bfloat16)
    x_hat, z = tiny_sae(x)
    qo = quasi_orthogonality_signal(z.to(torch.bfloat16), x_hat.to(torch.bfloat16), x)
    assert math.isfinite(qo["qo_gap"]) and math.isfinite(qo["qo_ratio"])


def test_c_qo_uses_trainer_coloring():
    # The console helper defers to the trainer's _c so the log line stays
    # consistent; green under warn, red past bad, n/a on None.
    good = _c_qo(0.05)
    bad = _c_qo(0.9)
    assert good == t._c("0.050", "1;92")
    assert bad == t._c("0.900", "1;91")
    assert "n/a" in _c_qo(None)
