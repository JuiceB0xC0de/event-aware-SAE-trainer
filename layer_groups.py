"""Group contiguous decoder layers by residual-stream similarity so the atlas
run can train **one SAE per group** instead of one per layer.

Adapted from *Group-SAE: Efficient Training of Sparse Autoencoders for Large
Language Models via Layer Groups* (Balcells et al., arXiv:2410.21508). The
paper observes that the residual-stream representations of adjacent layers are
highly similar, so a single SAE trained for a group of contiguous layers
reconstructs each member nearly as well as a per-layer SAE — at a fraction of
the training cost.

Scope of this port (kept deliberately small; see the PR body):

* **Kept at full fidelity** — the core mechanism: measure residual-stream
  similarity between contiguous layers and group similar ones so only one SAE
  is trained per group.
* **Substituted (auxiliary)** — the paper builds a full angular-distance
  similarity *matrix* over a dataset and runs offline agglomerative clustering
  to hit a fixed target group count K. That needs every layer's activations in
  hand at once. This module instead uses the mean-activation *signature* that
  the rolling activation pools already make available layer-by-layer, and does
  a single-pass, threshold-driven contiguous grouping that fits the trainer's
  streaming residual-chain walk. The group's first ("anchor") layer is the one
  trained; the paper's train-on-the-group's-pooled-activations step is out of
  scope for this integration.

The functions here are parameter-free and framework-light: signatures are
plain float vectors, similarity is cosine, and the grouping decision is a
single comparison, so the same primitives drive both the online trainer wiring
and offline group planning.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

Signature = torch.Tensor


def mean_activation_signature(activations: torch.Tensor) -> Signature:
    """Compress a block of activations to a per-layer residual-stream fingerprint.

    Accepts a pool shard shaped ``[n_seqs, seq_len, d]`` or an already-flattened
    ``[tokens, d]`` batch (bf16 or otherwise) and returns the fp32 mean over all
    token positions — a length-``d`` vector. The mean direction of the residual
    stream is a cheap, stable proxy for "what this layer represents"; contiguous
    layers that encode similar information have near-parallel mean vectors.
    """
    x = activations
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    elif x.dim() == 1:
        x = x.unsqueeze(0)
    return x.float().mean(dim=0)


def cosine_similarity(a: Signature, b: Signature) -> float:
    """Cosine similarity of two 1-D signatures; 0.0 if either is degenerate."""
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def should_share_with_anchor(
    signature: Signature, anchor_signature: Signature, threshold: float
) -> bool:
    """True when ``signature`` is similar enough to the current group's anchor to
    reuse the anchor's SAE instead of training a fresh one.

    ``threshold`` is a cosine floor in ``[-1, 1]``: higher means stricter (fewer,
    tighter groups → less compute saved); lower groups more aggressively.
    """
    return cosine_similarity(signature, anchor_signature) >= threshold


def group_contiguous_layers(
    signatures: Sequence[Signature], threshold: float
) -> List[List[int]]:
    """Plan the full contiguous grouping offline from precomputed signatures.

    Walks the signatures in layer order, keeping each group anchored on its first
    layer; a layer joins the current group while it stays within ``threshold``
    cosine of that anchor, otherwise it opens a new group. Returns a list of
    groups, each a list of (contiguous) layer indices into ``signatures``.

    This mirrors exactly the online decision the trainer makes as pools stream in
    (see ``sae_trainer_rolling.run_atlas_rolling``); it exists so a run can be
    planned or inspected without executing training.
    """
    groups: List[List[int]] = []
    anchor_sig: Signature | None = None
    for i, sig in enumerate(signatures):
        if anchor_sig is not None and should_share_with_anchor(sig, anchor_sig, threshold):
            groups[-1].append(i)
        else:
            groups.append([i])
            anchor_sig = sig
    return groups


def sae_training_savings(groups: Sequence[Sequence[int]]) -> int:
    """Number of ``train_sae_on_activations`` calls avoided by grouping —
    i.e. members that reuse an anchor's SAE (total layers minus group count)."""
    total = sum(len(g) for g in groups)
    return total - len(groups)
