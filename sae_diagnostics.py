"""Quasi-orthogonality diagnostics for JumpReLU SAEs.

Adapted from "Evaluating and Designing Sparse Autoencoders by Approximating
Quasi-Orthogonality" (arXiv:2503.24277). The paper's first contribution is a
theoretically-grounded check that a k-sparse decomposition respects the
quasi-orthogonality of its dictionary: with unit-norm, near-orthogonal decoder
atoms d_i, the reconstruction x_hat = D z satisfies

    ||x_hat||^2 = z^T (D^T D) z = sum_i z_i^2 + sum_{i!=j} z_i z_j <d_i, d_j>
                ~= ||z||^2                     (off-diagonal Gram mass ~ 0)

so the l2-norm of the *sparse feature vector* z tracks the l2-norm of the
reconstruction, and (to the extent the SAE reconstructs well) of the dense input
embedding x. That collapse holds only for *unit-norm* decoder atoms: the trainer
renormalizes the decoder columns to unit norm every optimizer step (see
`_normalize_decoder`), matching the reference, so ||d_i|| = 1 and a lone active
feature gives ||x_hat|| = |z_i| = ||z||. Were the atoms not renormalized, the gap
would track decoder-norm scale rather than dictionary geometry. The relative gap
| ||z|| - ||x_hat|| | / ||x_hat|| (qo_gap) is therefore a proxy for the
off-diagonal Gram mass -- it grows with co-activated features whose decoder
directions are not orthogonal.

This is a signal none of the trainer's existing per-log quality numbers capture:
EV measures reconstruction fidelity, dead%/fire_rate measure usage, but neither
sees whether the *active* dictionary atoms are actually quasi-orthogonal. A
growing qo_gap flags feature redundancy / dictionary degeneracy even while EV
still looks healthy.

Deliberately NOT ported: the paper's second contribution (top-AFA activation) --
as a replacement activation it would displace the trainer's augmented-Lagrangian
L0 controller; the paper's afa norm-matching loss term, afa_coeff * (||z|| -
||x||)^2, which is what actually drives ||z|| toward ||x|| during training (this
module only *observes* the ||z||/||x|| ratio, it does not add the loss, so it
never displaces the trainer's AL-L0 / JumpReLU objective); and the paper's
offline benchmark suite, which belongs downstream. This module is a read-only
diagnostic over tensors the log step already holds.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


_EMPTY = {
    "qo_ratio": None,
    "qo_gap": None,
    "feat_l2": None,
    "recon_l2": None,
    "input_l2": None,
}


def quasi_orthogonality_signal(
    feat_acts: "torch.Tensor",
    x_hat: "torch.Tensor",
    x: "torch.Tensor",
    eps: float = 1e-6,
) -> dict:
    """Per-batch quasi-orthogonality diagnostic for one SAE forward.

    Args:
        feat_acts: sparse feature activations z, shape [B, F].
        x_hat: reconstruction D z, shape [B, d].
        x: dense input embedding, shape [B, d].
        eps: floor on norm denominators to avoid divide-by-zero on silent tokens.

    Returns a dict of python floats (or ``None`` when there are no tokens):
        qo_ratio: mean over tokens of ||z||_2 / ||x||_2. The paper drives this
            toward 1 with a trained norm-matching loss, afa_coeff * (||z|| -
            ||x||)^2; this diagnostic only observes the ratio and does not add
            that loss, so nothing in the trainer's objective pins it to 1 -- it
            is a descriptive signal, not a predicted value.
        qo_gap:   mean of | ||z||_2 - ||x_hat||_2 | / ||x_hat||_2. This isolates
            the dictionary geometry from reconstruction error (it compares z to
            x_hat, not x): 0 means the active atoms are orthogonal, larger means
            co-activated features share direction (redundancy / collapse risk).
        feat_l2, recon_l2, input_l2: mean per-token l2 norms of z, x_hat, x --
            the raw components, kept for interpretability of the two ratios.
    """
    import torch

    if feat_acts.numel() == 0 or feat_acts.shape[0] == 0:
        return dict(_EMPTY)

    with torch.no_grad():
        # float(): bf16 activations lose precision in a sum-of-squares over F
        # dims; the diagnostic is cheap enough to always run in fp32.
        feat_l2 = feat_acts.float().norm(dim=-1)
        recon_l2 = x_hat.float().norm(dim=-1)
        input_l2 = x.float().norm(dim=-1)

        ratio = feat_l2 / input_l2.clamp_min(eps)
        gap = (feat_l2 - recon_l2).abs() / recon_l2.clamp_min(eps)

        return {
            "qo_ratio": ratio.mean().item(),
            "qo_gap": gap.mean().item(),
            "feat_l2": feat_l2.mean().item(),
            "recon_l2": recon_l2.mean().item(),
            "input_l2": input_l2.mean().item(),
        }


def _c_qo(gap: Optional[float], warn: float = 0.25, bad: float = 0.5) -> str:
    """Color the quasi-orthogonality gap: green near 0, red once the active
    dictionary is far from orthogonal. Mirrors the trainer's _c_ev / _c_dead
    console-coloring helpers so the log line reads consistently."""
    from sae_trainer_rolling import _c

    if gap is None:
        return _c("  n/a", "90")
    if gap >= bad:
        return _c(f"{gap:.3f}", "1;91")
    if gap >= warn:
        return _c(f"{gap:.3f}", "1;93")
    return _c(f"{gap:.3f}", "1;92")
