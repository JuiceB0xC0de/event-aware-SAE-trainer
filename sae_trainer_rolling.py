"""
Event-aware SAE trainer -- train one sparse autoencoder per decoder layer, unattended.
======================================================================================
THE POINT of this trainer is the event-aware scheduler (sae_scheduler.py): an AECS
mode controller plus an Augmented-Lagrangian L0 integrator that makes training
*self-tuning*. You set an L0 target; the scheduler overshoots it, then the dual
integrator pulls lambda back, L0 oscillates and converges as close to target as the
model allows, while dead features are held ~0 (aux-loss revival + threshold reset +
resampling + a dead-feature emergency mode). It sweeps every layer in a single run
with no per-layer hyperparameter retuning and no babysitting -- the failure mode that
forces restart-and-retune loops in fixed-schedule SAE trainers.

The scheduler and the training loop are MODEL-AGNOSTIC: they consume a stream of
activation batches from a provider and never touch model internals. How activations
are produced is pluggable (`--capture`):

  auto     (default) -- forward-hook the residual stream of any AutoModelForCausalLM.
                        Correct by construction (observes the real forward). Re-runs a
                        (truncated) forward per layer: 1 pool of disk, N x compute.
  rolling  (opt-in)  -- single-block walk: load the model once, run only block L over
                        the residual cached from block L-1. 1 pool of disk AND ~1
                        forward total -- a big VRAM/compute win, but its block-
                        invocation machinery is Gemma-3n/4-family specific (per-layer
                        embeddings, sliding/full attention types, KV sharing -> the
                        layers-0..14 HARD_STOP). Guarded by validate_rolling_cache.py,
                        which checks this single-block path bit-exact vs a full forward.

Run (plain Python, expects a CUDA GPU; H100/A100 target):
    python sae_trainer_rolling.py --model-id meta-llama/Llama-3.2-1B --end-layer 16
    python sae_trainer_rolling.py --capture rolling --end-layer 15      # Gemma fast path
    python sae_trainer_rolling.py --hub-id me/my-saes --wandb-project my-run

Config: $SAE_DATA_DIR (default ./data), $SAE_MODEL_ID, $SAE_HUB_ID, $WANDB_PROJECT,
$SAE_SCRATCH_DIR; HF_TOKEN read from the environment. d_in is auto-detected.
"""
from __future__ import annotations

__version__ = "0.1.0"

import math
import os
from pathlib import Path

try:
    from torch.utils.data import IterableDataset
except ImportError:                      # torch may be absent at import-time on a control host
    IterableDataset = object

def _slug(model_id: str) -> str:
    """Filesystem-safe tag from a HF model id ('google/gemma-4-E2B-it' ->
    'google_gemma-4-e2b-it'). Keeps outputs from different models separate."""
    return model_id.strip("/").replace("/", "_").replace(" ", "_").lower()


# Default model -- override per-run with --model-id / $SAE_MODEL_ID. Nothing below is
# hardcoded to a model family except the opt-in 'rolling' capture path.
MODEL_ID   = os.environ.get("SAE_MODEL_ID", "google/gemma-4-E2B-it")
# HF upload target. NO default org: push happens only when set explicitly (--hub-id /
# $SAE_HUB_ID), so a clone-and-run can never push somewhere unintended or fail on perms.
SAE_HUB_ID = os.environ.get("SAE_HUB_ID")
# wandb project. Off unless set (--wandb-project / $WANDB_PROJECT), independent of any key.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

# -- Local data layout (replaces the Modal /data volume) --------------------
# Everything reads/writes under $SAE_DATA_DIR (default ./data). Activation pools
# can be redirected to fast scratch via $SAE_SCRATCH_DIR (defaults under DATA_DIR).
DATA_DIR   = Path(os.environ.get("SAE_DATA_DIR", "./data")).expanduser()
SAE_DIR    = str(DATA_DIR / "saes" / _slug(MODEL_ID))        # persistent SAE outputs (per-model)
PRETOK_DIR = str(DATA_DIR / "pretok" / "fineweb-edu")        # optional pre-tokenized shards
# Activation pools are large and ephemeral -- point this at the fastest local disk
# you have. Pools are deleted incrementally during a run (we hold ~one pool at a time).
ROLLCACHE  = os.environ.get("SAE_SCRATCH_DIR", str(DATA_DIR / "rollcache"))
HARD_STOP_LAYER = 15                       # exclusive upper bound; never touch 15+

# -- Hyperparameters --------------------------------------------------------
D_IN          = 1536        # fallback residual width; auto-detected from the model config at run time
EXPANSION     = 32          # SAE dictionary size = EXPANSION * d_in (Llama-Scope-class config)
N_FEATURES    = EXPANSION * D_IN   # 49152 at d_in=1536; recomputed for the detected d_in
K             = 500         # FINAL target L0. Natural L0 settles ~550 at 32x; target<natural keeps lambda slightly positive.
K_INIT        = 500         # Curriculum disabled (== K)
K_CURRICULUM_STEPS = 1      # effectively disabled (target_l0 = K from step 1)
BATCH_TOKENS  = 32_768      # total tokens per SAE step (accumulated across microbatches if --accum-steps > 1)
MICROBATCH_TOKENS = 32_768  # tokens per microbatch for gradient accumulation; default = no accumulation
SEQ_LEN       = 2_048       # length passed to the model (attention is O(seq^2))
N_STEPS       = 15_000      # peak-EV early-stop usually fires well before this
LR            = 2e-4
LOG_EVERY     = 250
LR_WARMUP_STEPS = 300

# JumpReLU / aux-loss / resampling knobs
INIT_THRESHOLD = 0.1        # initial per-feature threshold
STE_BANDWIDTH  = 0.1        # epsilon for the straight-through estimator
BDEC_INIT_BATCHES = 50      # batches used to estimate b_dec mean
AUX_DEAD_THRESHOLD = 250    # steps without firing before a feature gets aux-loss revival
AUX_K          = 128        # top-k dead features per token to revive
AUX_COEFF      = 1 / 32     # Gao 2024 standard
RESET_EVERY    = 1_000      # sweep + theta-reset cadence for chronically-dead features
RESET_THRESHOLD = 1_500     # silent-steps required to qualify for theta reset
CHECKPOINT_EVERY = 1_000
RESAMPLE_STEPS = (K_CURRICULUM_STEPS, 12_000)   # steps where full W_enc/W_dec resample fires
ERR_BUFFER_SZ  = 16_384     # high-error activation buffer (tokens)
RESAMPLE_SCALE = 1.0        # scale of resampled encoder rows vs mean alive norm
DEFAULT_SEED   = 0

# Pool sizing: T batches of fresh activations per layer. Early-stop converges
# ~3500 steps, so T=4000 gives fresh-every-step (no epoching) with headroom.
POOL_BATCHES_DEFAULT = 4000


# ===========================================================================
#  SAE model  (JumpReLU, inlined -- this file is the single source of truth)
# ===========================================================================

import torch
import torch.nn as nn


class _JumpReLU(torch.autograd.Function):
    """JumpReLU activation with straight-through estimator for threshold gradient.
    Forward:  y = pre * H(pre - theta)
    Backward: dy/dpre passes through where active (relu-like);
              dy/dtheta uses rectangle-kernel pseudo-derivative
    """
    @staticmethod
    def forward(ctx, pre, log_threshold, bandwidth):
        threshold = log_threshold.exp()
        gate = (pre > threshold).to(pre.dtype)
        ctx.save_for_backward(pre, threshold, gate)
        ctx.bandwidth = bandwidth
        return pre * gate

    @staticmethod
    def backward(ctx, grad_output):
        pre, threshold, gate = ctx.saved_tensors
        eps = ctx.bandwidth
        grad_pre = grad_output * gate
        in_band = ((pre - threshold).abs() < eps).to(pre.dtype) / (2 * eps)
        sum_dims = tuple(range(grad_output.ndim - 1))
        grad_threshold = -(pre * in_band * grad_output).sum(dim=sum_dims)
        grad_log_threshold = grad_threshold * threshold
        return grad_pre, grad_log_threshold, None

class _L0Indicator(torch.autograd.Function):
    """Step function H(pre - theta) with STE for L0 sparsity loss."""
    @staticmethod
    def forward(ctx, pre, log_threshold, bandwidth):
        threshold = log_threshold.exp()
        ctx.save_for_backward(pre, threshold)
        ctx.bandwidth = bandwidth
        return (pre > threshold).to(pre.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        pre, threshold = ctx.saved_tensors
        eps = ctx.bandwidth
        in_band = ((pre - threshold).abs() < eps).to(pre.dtype) / (2 * eps)
        sum_dims = tuple(range(grad_output.ndim - 1))
        grad_threshold = -(in_band * grad_output).sum(dim=sum_dims)
        grad_log_threshold = grad_threshold * threshold
        return None, grad_log_threshold, None

class JumpReLUSAE(nn.Module):
    def __init__(self, d_in: int, n_features: int):
        super().__init__()
        self.d_in      = d_in
        self.n_features = n_features
        self.W_enc = nn.Linear(d_in, n_features, bias=True)
        self.W_dec = nn.Linear(n_features, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_threshold = nn.Parameter(
            torch.full((n_features,), math.log(INIT_THRESHOLD))
        )
        # W_dec orthonormal, W_enc tied to W_dec.T at init (Anthropic recipe)
        nn.init.orthogonal_(self.W_dec.weight)
        self._normalize_decoder()
        with torch.no_grad():
            self.W_enc.weight.copy_(self.W_dec.weight.t())
            self.W_enc.bias.zero_()

    def _normalize_decoder(self):
        with torch.no_grad():
            # W_dec.weight is [d_in, n_features]; each FEATURE direction is a COLUMN.
            # Normalize columns to unit norm so L0 loss is meaningful (each firing = a
            # unit-norm contribution to recon).
            norms = self.W_dec.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)  # [1, n_features]
            self.W_dec.weight.div_(norms)

    def encode(self, x: "torch.Tensor", k: int = K) -> "torch.Tensor":
        pre = self.W_enc(x - self.b_dec)
        return _JumpReLU.apply(pre, self.log_threshold, STE_BANDWIDTH)

    def encode_pre(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.W_enc(x - self.b_dec)

    def apply_jumprelu(self, pre: "torch.Tensor") -> "torch.Tensor":
        return _JumpReLU.apply(pre, self.log_threshold, STE_BANDWIDTH)

    def l0_indicator(self, pre: "torch.Tensor") -> "torch.Tensor":
        return _L0Indicator.apply(pre, self.log_threshold, STE_BANDWIDTH)

    def decode(self, acts: "torch.Tensor") -> "torch.Tensor":
        return self.W_dec(acts) + self.b_dec

    def forward(self, x: "torch.Tensor", k: int = K):
        acts = self.encode(x, k=k)
        x_hat = self.decode(acts)
        return x_hat, acts

def _make_sae(d_in: int, n_features: int, seed: int = 0):
    torch.manual_seed(seed)
    return JumpReLUSAE(d_in, n_features)




# ===========================================================================
#  Data streaming  (token datasets, inlined)
# ===========================================================================

class StreamingBatchDataset(IterableDataset):
    """IterableDataset over the full FineWeb-Edu corpus, yielding ready-to-train
    1-D LongTensors of length `batch_tokens`. Tokenizes on the fly; multiple
    DataLoader workers each get a disjoint shard subset via HF's `.shard(...)`.
    Module-level class so it stays picklable for worker processes."""

    def __init__(self, hf_token, model_id, batch_tokens,
                 max_seq_len: int = 4096, shuffle_buffer: int = 10_000, seed: int = 0):
        super().__init__()
        self.hf_token = hf_token
        self.model_id = model_id
        self.batch_tokens = batch_tokens
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self):
        import random as _random
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        _random.seed(self.seed * 7919 + worker_id)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)

        def _open_fineweb():
            ds = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                split="train", streaming=True, token=self.hf_token,
            )
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
            if num_workers > 1:
                ds = ds.shard(num_shards=num_workers, index=worker_id)
            return ds

        fw_iter = iter(_open_fineweb())

        def _next_text():
            nonlocal fw_iter
            while True:
                try:
                    row = next(fw_iter)
                    return row.get("text", "")
                except StopIteration:
                    fw_iter = iter(_open_fineweb())

        def _tokenize(text: str):
            return tokenizer(
                text, truncation=True, max_length=self.max_seq_len,
                return_tensors="pt", add_special_tokens=True,
            ).input_ids[0]

        token_buffer: list = []
        while True:
            while sum(len(t) for t in token_buffer) < self.batch_tokens:
                text = _next_text()
                if not text.strip():
                    continue
                ids = _tokenize(text)
                if len(ids) < 8:
                    continue
                token_buffer.append(ids)
            flat = torch.cat(token_buffer)
            yield flat[: self.batch_tokens].contiguous()
            leftover = flat[self.batch_tokens :]
            token_buffer = [leftover] if leftover.numel() > 0 else []


class PreTokenizedDataset(IterableDataset):
    """IterableDataset over pre-tokenized FineWeb-Edu shards on disk (memmap reads,
    no tokenization on the hot path). Build the shards with a pretokenize pass."""

    def __init__(self, pretok_dir: str, batch_tokens: int, seed: int = 0):
        super().__init__()
        self.pretok_dir = pretok_dir
        self.batch_tokens = batch_tokens
        self.seed = seed

    def __iter__(self):
        import json
        import random as _random
        import numpy as np
        import torch

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        _random.seed(self.seed * 7919 + worker_id)

        with open(f"{self.pretok_dir}/manifest.json") as f:
            manifest = json.load(f)
        all_idx = list(range(manifest["n_shards"]))
        my_idx = [i for i in all_idx if i % num_workers == worker_id]
        if not my_idx:
            my_idx = [worker_id % manifest["n_shards"]]

        shards = []
        for i in my_idx:
            path = f"{self.pretok_dir}/shard_{i:02d}.npy"
            arr = np.load(path, mmap_mode="r")
            if arr.shape[0] >= self.batch_tokens + 1:
                shards.append(arr)

        if not shards:
            raise RuntimeError(
                f"Worker {worker_id} got no usable shards. my_idx={my_idx}"
            )

        while True:
            shard = _random.choice(shards)
            max_start = shard.shape[0] - self.batch_tokens
            start = _random.randint(0, max_start)
            tokens = shard[start : start + self.batch_tokens].copy()
            yield torch.from_numpy(tokens).long()


def _build_token_dataset(hf_token, batch_tokens, seed: int = 0,
                         use_pretok: bool = False, pretok_dir: str = PRETOK_DIR):
    """Construct the streaming dataset. If `use_pretok=True`, reads pre-tokenized
    memmap shards (faster); otherwise streams + tokenizes FineWeb-Edu on the fly."""
    if use_pretok:
        return PreTokenizedDataset(pretok_dir=pretok_dir, batch_tokens=batch_tokens, seed=seed)
    return StreamingBatchDataset(
        hf_token=hf_token, model_id=MODEL_ID,
        batch_tokens=batch_tokens, max_seq_len=4096, seed=seed,
    )


# ===========================================================================
#  b2 single-block invocation  (proven bit-exact in validate_rolling_cache.py)
# ===========================================================================

def _find_text_model(model, n_layers):
    """Return (text_model, layers_attr_name, decoder_layers)."""
    import torch.nn as nn
    queue = [(model, None, None)]
    while queue:
        node, parent, attr = queue.pop(0)
        for name, child in node.named_children():
            if isinstance(child, nn.ModuleList) and len(child) == n_layers:
                return node, name, child
            queue.append((child, node, name))
    raise RuntimeError(f"Cannot find decoder ModuleList of length {n_layers}")


def _make_invariants(text_model, tcfg, ids):
    """Per-batch quantities that are CONSTANT across layers for these tokens:
    per_layer_inputs [B,S,n_layers,ple], per-type causal masks, per-type rotary,
    position_ids. Mirrors the head of Gemma4TextModel.forward."""
    import torch
    try:
        from transformers.masking_utils import (
            create_causal_mask, create_sliding_window_causal_mask)
    except Exception:
        from transformers.models.gemma4.modeling_gemma4 import (
            create_causal_mask, create_sliding_window_causal_mask)

    unique_layer_types = getattr(text_model, "unique_layer_types", None) \
        or sorted(set(tcfg.layer_types))
    with torch.no_grad():
        inputs_embeds0 = text_model.embed_tokens(ids)
        ple_raw = text_model.get_per_layer_inputs(ids, None)
        per_layer_inputs = text_model.project_per_layer_inputs(inputs_embeds0, ple_raw)
        S = inputs_embeds0.shape[1]
        position_ids = torch.arange(S, device=inputs_embeds0.device).unsqueeze(0)
        mask_kwargs = {
            "config": tcfg, "inputs_embeds": inputs_embeds0,
            "attention_mask": None, "past_key_values": None,
            "position_ids": position_ids,
        }
        masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }
        pos_emb = {lt: text_model.rotary_emb(inputs_embeds0, position_ids, lt)
                   for lt in unique_layer_types}
    return {
        "inputs_embeds0": inputs_embeds0, "per_layer_inputs": per_layer_inputs,
        "masks": masks, "pos_emb": pos_emb, "position_ids": position_ids,
    }


def _run_block(decoder_layers, tcfg, layer, hidden, inv):
    """Run ONLY block `layer` on `hidden`, exactly as Gemma4TextModel.forward's
    loop does. layers 0..14 are kv-own, so a fresh empty shared_kv dict is fine."""
    import torch
    lt = tcfg.layer_types[layer]
    with torch.no_grad():
        out = decoder_layers[layer](
            hidden,
            inv["per_layer_inputs"][:, :, layer, :],
            shared_kv_states={},  # nobody reads it in 0..14; storers 13/14 write harmlessly
            position_embeddings=inv["pos_emb"][lt],
            attention_mask=inv["masks"][lt],
            position_ids=inv["position_ids"],
            past_key_values=None,
        )
    return out[0] if isinstance(out, tuple) else out


# ===========================================================================
#  Pool I/O  (activation shards on container-local NVMe at ROLLCACHE)
# ===========================================================================

def _pool_dir(tag: str) -> Path:
    return Path(ROLLCACHE) / tag


def _write_shard(dir_path: Path, i: int, tensor):
    import torch
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.to(torch.bfloat16).cpu(), dir_path / f"shard_{i:05d}.pt")


def _read_shard(dir_path: Path, i: int):
    import torch
    return torch.load(dir_path / f"shard_{i:05d}.pt", map_location="cpu", weights_only=True)


def _shard_paths(dir_path: Path):
    return sorted(dir_path.glob("shard_*.pt"))


def _rm_pool(dir_path: Path):
    """Delete a layer's activation pool once it is no longer needed."""
    import shutil
    if dir_path.exists():
        shutil.rmtree(dir_path, ignore_errors=True)


# ===========================================================================
#  Token pool capture  (run once -- the same tokens flow through every layer)
# ===========================================================================

def _capture_token_pool(hf_token, seed, pool_batches, use_pretok, tok_dir: Path, bos_token_id):
    """Materialize T batches of BOS-prepended [n_seqs, SEQ_LEN] token shards so
    every layer trains on identical tokens (residual pool[L]=block L output == input
    to block L+1 requires this).

    Iterates the IterableDataset DIRECTLY in the main process -- no DataLoader worker
    procs. A persistent-worker spawn DataLoader crashes the interpreter on teardown
    (PyGILState_Release during finalization). With use_pretok the dataset is memmap
    reads (instant); live streaming is single-threaded but this is a one-time capture."""
    import torch

    if _shard_paths(tok_dir) and len(_shard_paths(tok_dir)) >= pool_batches:
        print(f"  [tokens] reusing {len(_shard_paths(tok_dir))} cached token shards")
        return

    dataset = _build_token_dataset(
        hf_token=hf_token, batch_tokens=BATCH_TOKENS, seed=seed, use_pretok=use_pretok)
    it = iter(dataset)                                     # main-process iteration
    n_seqs = BATCH_TOKENS // SEQ_LEN
    tok_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [tokens] capturing {pool_batches} token shards ([{n_seqs},{SEQ_LEN}]) ...")
    for i in range(pool_batches):
        batch = next(it)                                   # [BATCH_TOKENS]
        real = batch[: n_seqs * (SEQ_LEN - 1)].view(n_seqs, SEQ_LEN - 1)
        bos = torch.full((n_seqs, 1), bos_token_id, dtype=real.dtype)
        ids = torch.cat([bos, real], dim=1)                # [n_seqs, SEQ_LEN]
        torch.save(ids.cpu(), tok_dir / f"shard_{i:05d}.pt")
        if (i + 1) % 500 == 0:
            print(f"    token shard {i+1}/{pool_batches}")
    del it, dataset
    print(f"  [tokens] done -> {tok_dir}")


# ===========================================================================
#  Pool production  (capture activations -> pool[L] shards)
#  Two backends, both writing the same shard format the provider reads:
#    _produce_pool_hooked  -- model-agnostic forward-hook capture (default)
#    _produce_pool         -- Gemma-3n/4 single-block walk (opt-in, --capture rolling)
# ===========================================================================

def _produce_pool_hooked(model, decoder_layers, layer, tok_dir, dst_dir, device):
    """Model-agnostic capture: run the model forward over the token pool with a
    forward hook on decoder block `layer`, recording its residual-stream output.
    Works for any AutoModelForCausalLM (and text-only forwards of multimodal models).

    Correct by construction -- it observes the real forward, so there is no bit-exact
    risk. The hook raises _EarlyExit so layers after `layer` and the LM head are
    skipped. Each layer is captured independently (one forward per layer), so disk
    stays at one pool while compute is N_layers x a (truncated) forward."""
    import torch
    import time

    class _EarlyExit(Exception):
        """Raised inside a forward hook to abort the forward once the target layer's
        residual has been captured -- skips the remaining layers + LM head."""
        pass

    tok_paths = _shard_paths(tok_dir)
    n = len(tok_paths)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if len(_shard_paths(dst_dir)) >= n:
        print(f"  [produce L{layer}] pool already present ({n} shards) -- skip")
        return

    cap = {}

    def _hook(_module, _inp, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()
        raise _EarlyExit

    handle = decoder_layers[layer].register_forward_hook(_hook)
    print(f"  [produce L{layer}] hook capture over {n} batches ...")
    t0 = time.time()
    try:
        for i in range(n):
            ids = _read_shard(tok_dir, i).to(device)           # [n_seqs, SEQ_LEN] int
            try:
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
            except _EarlyExit:
                pass
            _write_shard(dst_dir, i, cap["h"])
            if (i + 1) % 500 == 0:
                tok_s = (i + 1) * BATCH_TOKENS / (time.time() - t0)
                print(f"    produced {i+1}/{n}  ({tok_s/1e3:.1f}k tok/s)")
    finally:
        handle.remove()
    print(f"  [produce L{layer}] done in {(time.time()-t0)/60:.1f}min -> {dst_dir}")


def _produce_pool(model, text_model, decoder_layers, tcfg, layer, tok_dir, src_dir, dst_dir, device):
    """Gemma-3n/4 single-block walk. Produce pool[layer] (block-L output for every batch).
    layer 0: embed_tokens + block 0 over the token pool.
    layer>=1: block L over src_dir (= pool[L-1]).
    Tokens are always needed for per_layer_inputs (PLE)."""
    import torch
    import time

    tok_paths = _shard_paths(tok_dir)
    n = len(tok_paths)
    dst_dir.mkdir(parents=True, exist_ok=True)
    # idempotent skip if already produced
    if len(_shard_paths(dst_dir)) >= n:
        print(f"  [produce L{layer}] pool already present ({n} shards) -- skip")
        return
    # Incremental source deletion: delete src shard i right after producing dst
    # shard i, so we never hold two full pools. Peak disk stays ~one pool (~404GB
    # at 4000 batches) instead of ~800GB -- required because the container overlay
    # fs caps near ~512GB and there is no large local scratch.
    print(f"  [produce L{layer}] running block {layer} over {n} batches "
          f"(incremental src-delete) ...")
    t0 = time.time()
    for i in range(n):
        ids = _read_shard(tok_dir, i).to(device)           # [n_seqs, SEQ_LEN] int
        inv = _make_invariants(text_model, tcfg, ids)
        if layer == 0:
            hidden = inv["inputs_embeds0"]                  # scaled word embeds
        else:
            hidden = _read_shard(src_dir, i).to(device, dtype=inv["inputs_embeds0"].dtype)
        out = _run_block(decoder_layers, tcfg, layer, hidden, inv)
        _write_shard(dst_dir, i, out)
        if src_dir is not None:                            # free the consumed src shard now
            try:
                (src_dir / f"shard_{i:05d}.pt").unlink()
            except FileNotFoundError:
                pass
        if (i + 1) % 500 == 0:
            tok_s = (i + 1) * BATCH_TOKENS / (time.time() - t0)
            print(f"    produced {i+1}/{n}  ({tok_s/1e3:.1f}k tok/s)")
    print(f"  [produce L{layer}] done in {(time.time()-t0)/60:.1f}min -> {dst_dir}")


# ===========================================================================
#  Activation provider  (reads pool[L] shards, prefetched, yields SAE batches)
# ===========================================================================

class RollingActivationProvider:
    """Yields [BATCH_TOKENS, D_IN] bf16 activations on device from pool[L] shards.
    Background thread prefetches the next shard so the SAE hot loop never waits.

    Supports checkpoint resume via get_state()/restore_state() -- tracks the current
    shard cursor and consumed shards so training can resume at the exact position.
    """

    def __init__(self, pool_dir: Path, device, seed=0, queue_size=8):
        import threading, queue as _queue
        self.paths = sorted(_shard_paths(pool_dir))
        if not self.paths:
            raise RuntimeError(f"No pool shards in {pool_dir}")
        self.device = device
        self.seed = seed
        self._q = _queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        # State for resume: current position in the shuffled order
        self._current_order = list(range(len(self.paths)))
        self._cursor = 0
        self._consumed = set()
        self._lock = threading.Lock()

    def _worker(self):
        import random as _random
        import torch
        rng = _random.Random(self.seed * 7919)
        while not self._stop.is_set():
            with self._lock:
                if self._cursor >= len(self.paths):
                    # Reshuffle for a new epoch
                    rng.shuffle(self._current_order)
                    self._cursor = 0
                    self._consumed.clear()
                idx = self._current_order[self._cursor]
                self._cursor += 1
                self._consumed.add(idx)
            path = self.paths[idx]
            # Load with mmap for faster IO, reshape, keep on CPU but pinned
            t = torch.load(path, map_location="cpu", weights_only=True, mmap=True)  # [n_seqs,SEQ_LEN,d] bf16
            t = t.reshape(-1, t.shape[-1])  # flatten to [total_tokens, d_in]
            # Pin memory for faster GPU transfer
            if not t.is_pinned():
                t = t.pin_memory()
            self._q.put(t, timeout=5)

    def next_batch(self):
        t = self._q.get(timeout=5)
        return t.to(self.device, non_blocking=True)        # [BATCH_TOKENS, D_IN] bf16

    def get_state(self):
        """Return provider state for checkpoint resume."""
        with self._lock:
            return {
                "current_order": self._current_order.copy(),
                "cursor": self._cursor,
                "consumed": list(self._consumed),
            }

    def restore_state(self, state):
        """Restore provider state from a checkpoint."""
        with self._lock:
            self._current_order = state.get("current_order", list(range(len(self.paths))))
            self._cursor = state.get("cursor", 0)
            self._consumed = set(state.get("consumed", []))

    def close(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass


# ===========================================================================
#  SAE training body  (trains one JumpReLU SAE on cached activations)
# ===========================================================================

def _save_full_checkpoint(out_dir, step, sae, optimizer, scheduler, rng_states,
                          feature_fire_counts, steps_since_fired, provider_state):
    """Save a complete checkpoint: model, optimizer, scheduler, RNG, dead-feature stats."""
    import torch
    ckpt = {
        "step": step,
        "sae_state": {k: v.cpu().clone() for k, v in sae.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {
            "lambda_l0": scheduler.lambda_l0,
            "mode": scheduler.mode,
            "mode_steps": scheduler.mode_steps,
            "total_steps": scheduler.total_steps,
            "event_counter": scheduler.event_counter,
            "transition_log": scheduler.transition_log[-100:],  # last 100 transitions
            "_lambda_history": scheduler._lambda_history,
        },
        "rng_state": rng_states,
        "feature_fire_counts": feature_fire_counts.cpu().clone(),
        "steps_since_fired": steps_since_fired.cpu().clone(),
        "provider_state": provider_state,
    }
    ckpt_path = out_dir / "checkpoint_full.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def _load_full_checkpoint(ckpt_path, sae, optimizer, scheduler, device):
    """Load a full checkpoint, returning (step, rng_states, feature_fire_counts,
    steps_since_fired, provider_state) or None if loading fails."""
    import torch
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sae.load_state_dict({k: v.to(device) for k, v in ckpt["sae_state"].items()})
        optimizer.load_state_dict(ckpt["optimizer_state"])
        sched_state = ckpt["scheduler_state"]
        scheduler.lambda_l0 = sched_state["lambda_l0"]
        scheduler.mode = sched_state["mode"]
        scheduler.mode_steps = sched_state["mode_steps"]
        scheduler.total_steps = sched_state["total_steps"]
        scheduler.event_counter = sched_state["event_counter"]
        scheduler.transition_log = sched_state["transition_log"]
        scheduler._lambda_history = sched_state.get("_lambda_history", [])
        return {
            "step": ckpt["step"],
            "rng_state": ckpt["rng_state"],
            "feature_fire_counts": ckpt["feature_fire_counts"].to(device),
            "steps_since_fired": ckpt["steps_since_fired"].to(device),
            "provider_state": ckpt.get("provider_state"),
        }
    except Exception as e:
        print(f"  WARNING: checkpoint load failed ({e})")
        return None


def train_sae_on_activations(layer, d_in, seed, provider, *, frozen_decoder=False,
                             max_steps=N_STEPS, bdec_batches=BDEC_INIT_BATCHES,
                             microbatch_tokens=MICROBATCH_TOKENS,
                             resume_from=None, push=True):
    """Train one JumpReLU SAE on activations supplied by `provider`.

    Gradient accumulation: if `microbatch_tokens < BATCH_TOKENS`, accumulate gradients
    over `BATCH_TOKENS // microbatch_tokens` microbatches before stepping.

    Resume: if `resume_from` points to a `checkpoint_full.pt`, restore model,
    optimizer, scheduler, RNG, and dead-feature stats.

    bf16 path: activations are kept in bf16 through the forward; loss/reductions in fp32.
    """
    import json, math, time, threading
    import torch
    import torch.nn as nn
    from sae_scheduler import SAEAECSConfig, SAEEventControlScheduler

    n_steps = int(min(max_steps, N_STEPS))
    accum_steps = BATCH_TOKENS // microbatch_tokens
    assert BATCH_TOKENS % microbatch_tokens == 0, \
        f"BATCH_TOKENS ({BATCH_TOKENS}) must be divisible by microbatch_tokens ({microbatch_tokens})"

    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Critical for H100
    torch.set_float32_matmul_precision("high")

    # -- LAMBDA tier (layers 0-14 only -> hot/trough/mid) ---------------------
    if layer < 10:
        tier = "hot"
    elif layer < 12:
        tier = "trough"
    else:
        tier = "mid"
    tier_lambda = {"hot": 1e-3, "trough": 3e-4, "mid": 5e-4}[tier]

    sae_cfg = SAEAECSConfig(
        base_lr=LR, warmup_steps=min(LR_WARMUP_STEPS, max(1, n_steps // 5)), total_steps=n_steps,
        event_warmup_steps=5_000, instability_z_thresh=10.0,
        loss_spike_ratio=3.0, loss_spike_min_recent=50,
        plateau_grad_norm_thresh=1e-4,
        recovery_lr_factor=0.7, recovery_momentum_factor=0.5, explore_lr_factor=1.5,
        cooldown_steps=250, event_persistence=3, mode_verbose=True,
        target_l0=float(K_INIT), l0_tolerance=0.20,
        use_augmented_lagrangian=True, lambda_l0_init=0.0, lambda_l0_min=0.0,
        lambda_l0_max=5e-3, al_mu=1e-7, al_dual_step=5e-10, al_log_every=LOG_EVERY,
        ev_floor=0.88, ev_floor_patience=3, ev_drop_thresh=-0.05,
        ev_stop_thresh=0.88, ev_stop_patience=3,
        dead_emergency_thresh=20.0, dead_emergency_cooldown=5000,
    )
    out_suffix = "_frozen" if frozen_decoder else ""

    # -- wandb ----------------------------------------------------------------
    use_wandb = False
    wandb = None
    if WANDB_PROJECT and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb as _wandb
            try:
                if _wandb.run is not None:
                    _wandb.finish(exit_code=1)
            except Exception:
                pass
            _wandb.init(project=WANDB_PROJECT, name=f"L{layer:02d}_s{seed}",
                        reinit="finish_previous",
                        config={"layer": layer, "seed": seed, "model_id": MODEL_ID,
                                "n_features": N_FEATURES, "k": K, "batch_tokens": BATCH_TOKENS,
                                "microbatch_tokens": microbatch_tokens, "accum_steps": accum_steps,
                                "lr": LR, "n_steps": N_STEPS, "tier": tier, "resume_from": resume_from})
            wandb = _wandb
            use_wandb = True
            print(f"  WandB: {wandb.run.url}")
        except Exception as e:
            print(f"  WARNING: wandb init failed ({e})")

    print(f"\n{'='*60}\nROLLING SAE  layer={layer}  seed={seed}  tier={tier}  d_in={d_in}")
    print(f"  accum_steps={accum_steps} (microbatch={microbatch_tokens//1024}k tokens)")
    if resume_from:
        print(f"  resume_from={resume_from}")
    print(f"{'='*60}")

    sae = _make_sae(d_in, N_FEATURES, seed).to(device)
    if frozen_decoder:
        sae.W_dec.weight.requires_grad_(False)

    # Compile for H100 speedup (fuse kernels, reduce launch overhead)
    sae = torch.compile(sae, mode="reduce-overhead")

    optimizer = torch.optim.Adam(
        [{"params": sae.W_enc.parameters(), "weight_decay": 0},
         {"params": sae.W_dec.parameters(), "weight_decay": 1e-4},
         {"params": [sae.b_dec], "weight_decay": 0},
         {"params": [sae.log_threshold], "weight_decay": 0}],
        lr=LR, betas=(0.9, 0.999), fused=True)
    scheduler = SAEEventControlScheduler(optimizer, sae_cfg, mode_label=f"L{layer:02d}")

    # -- Resume from checkpoint -----------------------------------------------
    start_step = 1
    feature_fire_counts = torch.zeros(N_FEATURES, device=device, dtype=torch.long)
    steps_since_fired = torch.zeros(N_FEATURES, device=device, dtype=torch.long)
    err_buffer = []

    if resume_from and Path(resume_from).exists():
        loaded = _load_full_checkpoint(resume_from, sae, optimizer, scheduler, device)
        if loaded:
            start_step = loaded["step"] + 1
            feature_fire_counts = loaded["feature_fire_counts"]
            steps_since_fired = loaded["steps_since_fired"]
            # Restore RNG states
            if "cuda" in loaded["rng_state"]:
                torch.cuda.set_rng_state(loaded["rng_state"]["cuda"])
            if "cpu" in loaded["rng_state"]:
                torch.set_rng_state(loaded["rng_state"]["cpu"])
            print(f"  Resumed from step {start_step}, lambda={scheduler.lambda_l0:.3e}, mode={scheduler.mode}")
            # Tell provider to resume at the right cursor if it supports it
            if loaded.get("provider_state"):
                provider.restore_state(loaded["provider_state"])

    # -- b_dec init from provider activations (skip if resumed) ---------------
    if start_step == 1:
        print(f"Estimating b_dec from {bdec_batches} batches ...")
        acc = torch.zeros(d_in, device=device, dtype=torch.float32)
        n_acc = 0
        for _ in range(bdec_batches):
            a = provider.next_batch()
            acc += a.sum(dim=0)
            n_acc += a.shape[0]
        if n_acc > 0:
            with torch.no_grad():
                sae.b_dec.copy_(acc / n_acc)
            print(f"  b_dec set from {n_acc} tokens, ||b_dec||={sae.b_dec.norm().item():.4f}")

    metrics = {"recon_loss": [], "mean_l0": [], "dead_pct": [], "resampled": [],
               "ev": [], "nonlinear_err": [], "linear_err": []}
    total_resampled = 0
    log_window_start = time.time()
    log_window_tokens = 0

    best_ev = -float("inf"); best_ev_step = 0; best_ev_l0 = 0.0
    best_state_in_memory = None; best_state_pending = False
    best_ev_persist_margin = 0.005
    ev_decline_margin = 0.01; ev_decline_patience = 5; ev_decline_warmup_steps = 1500
    ev_below_peak_streak = 0

    def resample_dead_neurons(dead_mask, err_buf):
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0 or len(err_buf) == 0:
            return 0
        candidates = torch.cat(err_buf, dim=0)
        with torch.no_grad():
            idx = torch.randint(0, candidates.shape[0], (n_dead,), device=candidates.device)
            samples = candidates[idx]
            norms = samples.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            alive_mask = ~dead_mask
            alive_norm = sae.W_enc.weight[alive_mask].norm(dim=1).mean() if alive_mask.any() \
                else torch.tensor(1.0, device=candidates.device)
            sae.W_enc.weight.data[dead_mask] = (samples / norms) * RESAMPLE_SCALE * alive_norm
            sae.W_enc.bias.data[dead_mask] = 0.0
            rand_dec = torch.randn(sae.d_in, n_dead, device=candidates.device)
            rand_dec = rand_dec / rand_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
            sae.W_dec.weight.data[:, dead_mask] = rand_dec
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p, {})
                    if "exp_avg" not in state:
                        continue
                    if p is sae.W_enc.weight or p is sae.W_enc.bias:
                        state["exp_avg"][dead_mask] = 0.0; state["exp_avg_sq"][dead_mask] = 0.0
                    elif p is sae.W_dec.weight:
                        state["exp_avg"][:, dead_mask] = 0.0; state["exp_avg_sq"][:, dead_mask] = 0.0
        return n_dead

    # -- training loop --------------------------------------------------------
    for step in range(1, n_steps + 1):
        if scheduler.should_stop:
            print(f"  EARLY STOP @ step {step}: {scheduler.stop_reason}")
            if use_wandb:
                wandb.log({"train/early_stop": True}, step=step)
            break

        sae_cfg.target_l0 = float(K)  # curriculum disabled (K_INIT==K)

        # Gradient accumulation: accumulate over microbatches before stepping
        optimizer.zero_grad()
        accum_recon_loss = 0.0
        accum_l0 = 0.0
        accum_sparsity_loss = 0.0
        accum_aux_loss = 0.0

        # Get full batch and split into microbatches for accumulation
        full_batch = provider.next_batch()  # [BATCH_TOKENS, d_in] bf16
        full_batch_device = full_batch.to(device)
        microbatch_size = microbatch_tokens  # tokens per microbatch

        # Compute a single full-batch L0 and sparsity penalty for the whole accumulation
        with torch.no_grad():
            full_pre = sae.encode_pre(full_batch_device)
            full_gate = sae.l0_indicator(full_pre)
            l0_full = full_gate.sum(dim=-1).float().mean()
            slack = (l0_full - sae_cfg.target_l0).clamp(min=0.0)
            sparsity_loss_full = (
                scheduler.lambda_l0 * slack
                + 0.5 * sae_cfg.al_mu * slack * slack
            ) / accum_steps

            # Pre-compute dead feature parameters for the aux loss to avoid doing this per microbatch
            dead_mask_aux = (steps_since_fired >= AUX_DEAD_THRESHOLD)
            n_dead = int(dead_mask_aux.sum().item())
            if n_dead > 0:
                dead_indices = torch.where(dead_mask_aux)[0]
                eff_k = min(AUX_K, n_dead)

        for accum_idx in range(accum_steps):
            start_idx = accum_idx * microbatch_size
            end_idx = start_idx + microbatch_size
            acts_mb = full_batch_device[start_idx:end_idx]  # bf16, [microbatch_tokens, d_in]

            # bf16 forward: keep activations in bf16, cast only for loss computation
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pre = sae.encode_pre(acts_mb)
                feat_acts = sae.apply_jumprelu(pre)
                x_hat = sae.decode(feat_acts)

                # Cast to fp32 only for loss computation (numerical stability)
                recon_loss = (acts_mb.float() - x_hat.float()).pow(2).mean() / accum_steps
                sparsity_loss = sparsity_loss_full

                # Aux loss for dead features (scaled by accum_steps)
                if n_dead > 0:
                    pre_dead = pre[:, dead_indices].relu()
                    topk_vals, topk_idx = pre_dead.topk(eff_k, dim=-1)
                    aux_acts = torch.zeros_like(pre_dead)
                    aux_acts.scatter_(-1, topk_idx, topk_vals)
                    W_dec_dead = sae.W_dec.weight.t()[dead_indices]
                    x_aux = aux_acts @ W_dec_dead
                    residual_target = acts_mb.float() - x_hat.detach().float()
                    aux_loss = ((residual_target - x_aux.float()).pow(2).mean() * AUX_COEFF) / accum_steps
                else:
                    aux_loss = torch.zeros((), device=acts_mb.device, dtype=torch.float32)

                loss = recon_loss + sparsity_loss + aux_loss

            loss.backward()

            accum_recon_loss += recon_loss.detach()
            accum_l0 += l0_full.detach()
            accum_sparsity_loss += sparsity_loss.detach()
            accum_aux_loss += aux_loss.detach()

        # Extract values for logging once after the microbatch loop
        # We also compute the average L0 directly rather than sum and divide by accum_steps for accum_l0
        # since L0 does not multiply by accum_steps like others
        accum_recon_loss = accum_recon_loss.item() * accum_steps if isinstance(accum_recon_loss, torch.Tensor) else 0.0
        accum_l0 = accum_l0.item() if isinstance(accum_l0, torch.Tensor) else 0.0
        accum_sparsity_loss = accum_sparsity_loss.item() * accum_steps if isinstance(accum_sparsity_loss, torch.Tensor) else 0.0
        accum_aux_loss = accum_aux_loss.item() * accum_steps if isinstance(accum_aux_loss, torch.Tensor) else 0.0

        # Single optimizer step after accumulation
        grad_norm_t = nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        if not frozen_decoder:
            sae._normalize_decoder()
        with torch.no_grad():
            enc_norms = sae.W_enc.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
            sae.W_enc.weight.div_(enc_norms)

        # Use accumulated values for logging
        recon_loss = torch.tensor(accum_recon_loss / accum_steps, device=device)
        l0 = torch.tensor(accum_l0 / accum_steps, device=device)  # average L0 across microbatches

        with torch.no_grad():
            # Compute fired features across the full accumulated batch
            full_pre = sae.encode_pre(full_batch.to(device))
            full_feat = sae.apply_jumprelu(full_pre)
            fired = (full_feat > 0).any(dim=0)
            steps_since_fired += 1
            steps_since_fired[fired] = 0
            feature_fire_counts += fired.long()
            log_window_tokens += BATCH_TOKENS

        if step % RESET_EVERY == 0 and step > 0:
            with torch.no_grad():
                very_dead = (steps_since_fired >= RESET_THRESHOLD)
                n_reset = int(very_dead.sum().item())
                if n_reset > 0:
                    sae.log_threshold.data[very_dead] = math.log(INIT_THRESHOLD)
                    state = optimizer.state.get(sae.log_threshold, {})
                    if "exp_avg" in state:
                        state["exp_avg"][very_dead] = 0.0; state["exp_avg_sq"][very_dead] = 0.0
                    steps_since_fired[very_dead] = 0
                    print(f"  [RESET @ {step}] theta->{INIT_THRESHOLD} for {n_reset} dead")

        next_resample_step = min((s for s in RESAMPLE_STEPS if s >= step), default=None)
        if next_resample_step is not None and (next_resample_step - step) <= ERR_BUFFER_SZ // 64:
            with torch.no_grad():
                per_token_err = (full_batch.float() - sae.decode(sae.apply_jumprelu(sae.encode_pre(full_batch.to(device)))).detach().float()).pow(2).sum(dim=-1)
                top_err_idx = per_token_err.topk(min(64, len(per_token_err))).indices
                err_buffer.append(full_batch[top_err_idx].detach().cpu())
                if sum(t.shape[0] for t in err_buffer) > ERR_BUFFER_SZ:
                    err_buffer = err_buffer[-ERR_BUFFER_SZ // 64:]

        if step in RESAMPLE_STEPS:
            dead_mask = (steps_since_fired >= 1000)
            n_dead = int(dead_mask.sum().item())
            n_res = resample_dead_neurons(dead_mask, [t.to(device) for t in err_buffer])
            if n_res > 0:
                with torch.no_grad():
                    sae.log_threshold.data[dead_mask] = math.log(INIT_THRESHOLD)
                    state = optimizer.state.get(sae.log_threshold, {})
                    if "exp_avg" in state:
                        state["exp_avg"][dead_mask] = 0.0; state["exp_avg_sq"][dead_mask] = 0.0
                steps_since_fired[dead_mask] = 0
            total_resampled += n_res
            print(f"  [RESAMPLE @ {step}] reinit {n_res}/{n_dead} dead")
            err_buffer = []

        recon_val = recon_loss.detach().item()
        grad_norm_val = grad_norm_t.item()
        l0_val = l0.detach().item()

        is_log_step = (step % LOG_EVERY == 0)
        if is_log_step:
            dead = (steps_since_fired >= LOG_EVERY).float().mean().item() * 100
            with torch.no_grad():
                total_var = full_batch.float().var()
                ev = 1.0 - (recon_val / total_var) if total_var > 0 else 0.0
        else:
            dead = None; ev = None

        scheduler.step({"loss": recon_val, "grad_norm": grad_norm_val,
                        "l0": l0_val, "ev": ev, "dead_pct": dead})

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                thr = sae.log_threshold.exp()
                fire_rate = feature_fire_counts.float() / max(LOG_EVERY, 1)
                ultra_active = (fire_rate > 0.10).float().sum().item()
            now = time.time()
            tokens_per_sec = log_window_tokens / max(now - log_window_start, 1e-6)
            log_window_start = now; log_window_tokens = 0
            feature_fire_counts.zero_()

            if ev > best_ev:
                best_ev = ev; best_ev_step = step; best_ev_l0 = l0_val
                best_state_in_memory = {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()}
                best_state_pending = True
            elif best_state_pending and ev < best_ev - best_ev_persist_margin:
                best_ckpt = Path(SAE_DIR) / f"layer_{layer:02d}_s{seed}{out_suffix}_latest" / "checkpoint_best.pt"
                best_ckpt.parent.mkdir(parents=True, exist_ok=True)
                bp = {"step": best_ev_step, "sae_state": best_state_in_memory,
                      "best_ev": best_ev, "best_ev_l0": best_ev_l0, "layer": layer,
                      "seed": seed, "tier": tier, "d_in": d_in, "n_features": N_FEATURES,
                      "target_l0_final": K, "path": "rolling"}
                def _savebest(p, path):
                    torch.save(p, path)
                threading.Thread(target=_savebest, args=(bp, best_ckpt), daemon=True).start()
                best_state_pending = False
                print(f"  [BEST PERSIST @ {step}] peak EV={best_ev:.4f}@{best_ev_step}")

            if step >= ev_decline_warmup_steps and best_ev > -float("inf"):
                if (best_ev - ev) >= ev_decline_margin:
                    ev_below_peak_streak += 1
                else:
                    ev_below_peak_streak = 0
                if ev_below_peak_streak >= ev_decline_patience:
                    scheduler.should_stop = True
                    scheduler.stop_reason = (f"Post-peak decline: EV {ev:.4f} below peak "
                                             f"{best_ev:.4f}@{best_ev_step} for {ev_decline_patience} windows")
                    print(f"  [POST-PEAK STOP @ {step}] {scheduler.stop_reason}")
            else:
                ev_below_peak_streak = 0

            metrics["recon_loss"].append(round(recon_val, 6))
            metrics["mean_l0"].append(round(l0_val, 2))
            metrics["dead_pct"].append(round(dead, 2))
            metrics["resampled"].append(total_resampled)
            metrics["ev"].append(ev)
            print(f"  step={step:>5} mode={scheduler.mode:<9s} recon={recon_val:.5f} "
                  f"L0={l0_val:.1f} dead={dead:.1f}% ev={ev:.3f} thr={thr.mean().item():.3f} "
                  f"ultra={int(ultra_active):>4d} lr={scheduler.optimizer.param_groups[0]['lr']:.2e} "
                  f"lam={scheduler.lambda_l0:.2e} tok/s={tokens_per_sec/1e3:.1f}k "
                  f"tokens={step*BATCH_TOKENS/1e6:.1f}M")
            if use_wandb:
                wandb.log({"train/recon_loss": recon_val, "train/mean_l0": l0_val,
                           "train/dead_pct": dead, "train/explained_variance": ev,
                           "train/lr": scheduler.optimizer.param_groups[0]["lr"],
                           "train/lambda_l0": scheduler.lambda_l0,
                           "train/best_ev_so_far": best_ev,
                           "features/ultra_active_count": ultra_active,
                           "timing/tokens_per_sec": tokens_per_sec,
                           "scheduler/mode": scheduler.mode}, step=step)

    # -- final best flush -----------------------------------------------------
    if best_state_pending and best_state_in_memory is not None:
        best_ckpt = Path(SAE_DIR) / f"layer_{layer:02d}_s{seed}{out_suffix}_latest" / "checkpoint_best.pt"
        best_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": best_ev_step, "sae_state": best_state_in_memory,
                    "best_ev": best_ev, "best_ev_l0": best_ev_l0, "layer": layer,
                    "seed": seed, "tier": tier, "d_in": d_in, "n_features": N_FEATURES,
                    "target_l0_final": K, "note": "final flush", "path": "rolling"}, best_ckpt)
        print(f"  [BEST FINAL FLUSH] peak EV={best_ev:.4f}@{best_ev_step}")

    # -- save final + meta + HF push ------------------------------------------
    out_dir = Path(SAE_DIR) / f"layer_{layer:02d}_s{seed}{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), out_dir / "sae.pt")
    sched_summary = scheduler.summary()
    meta = {"layer": layer, "seed": seed, "model_id": MODEL_ID, "d_in": d_in,
            "n_features": N_FEATURES, "k": K, "batch_tokens": BATCH_TOKENS,
            "n_steps": step, "lr": LR, "lambda_l0": scheduler.lambda_l0, "tier": tier,
            "scheduler_mode": scheduler.mode,
            "scheduler_transitions": sched_summary["transitions"],
            "total_tokens": step * BATCH_TOKENS, "early_stopped": scheduler.should_stop,
            "path": "rolling", "best_ev": best_ev, "best_ev_step": best_ev_step,
            "final_metrics": {k: v[-1] if v else None for k, v in metrics.items()},
            "training_curve": metrics}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save full checkpoint at layer completion (for resume if interrupted between layers)
    rng_states = {"cuda": torch.cuda.get_rng_state(), "cpu": torch.get_rng_state()}
    provider_state = provider.get_state() if hasattr(provider, "get_state") else None
    _save_full_checkpoint(out_dir, step, sae, optimizer, scheduler, rng_states,
                          feature_fire_counts.clone(), steps_since_fired.clone(), provider_state)

    print(f"Saved: {out_dir}/sae.pt + meta.json + checkpoint_full.pt")

    if push and SAE_HUB_ID:
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        api = HfApi(token=hf_token)
        try:
            api.create_repo(SAE_HUB_ID, repo_type="model", exist_ok=True, private=False)
        except Exception:
            pass
        api.upload_folder(folder_path=str(out_dir), repo_id=SAE_HUB_ID,
                          path_in_repo=f"layer_{layer:02d}_s{seed}", repo_type="model")
        print(f"Pushed layer_{layer:02d}_s{seed} -> {SAE_HUB_ID}")
    elif push and not SAE_HUB_ID:
        print(f"  [push skipped] no --hub-id / $SAE_HUB_ID set -- SAE saved locally only")
    else:
        print(f"  [push disabled] skipped HF upload for layer {layer}")

    if use_wandb:
        try:
            wandb.summary.update({f"final/{k}": v for k, v in meta["final_metrics"].items()})
        finally:
            try:
                wandb.finish()
            except Exception:
                pass
    return meta["final_metrics"]


# ===========================================================================
#  Orchestrator  (load model once, sweep layers, train one SAE each)
# ===========================================================================

def run_atlas_rolling(start_layer: int = 0, end_layer: int = 9, seed: int = DEFAULT_SEED,
                      pool_batches: int = POOL_BATCHES_DEFAULT, microbatch_tokens: int = None,
                      use_pretok: bool = True, max_steps: int = N_STEPS,
                      bdec_batches: int = BDEC_INIT_BATCHES, resume_from: str = None,
                      push: bool = True, capture: str = "auto", model_id: str = None,
                      hub_id: str = None, wandb_project: str = None, expansion: int = None):
    """Train one SAE per decoder layer in [start_layer, end_layer).

    capture: "auto" = model-agnostic forward-hook capture (any AutoModelForCausalLM);
             "rolling" = Gemma-3n/4 single-block walk (layers 0..14 only, VRAM-optimized).
    pool_batches: activation batches cached per layer (default 4000; use 500-1000 for limited disk)
    microbatch_tokens: tokens per microbatch for gradient accumulation (default = no accum)
    resume_from: path to checkpoint_full.pt to resume from
    model_id/hub_id/wandb_project/expansion override module defaults; d_in is auto-detected.
    max_steps/bdec_batches/push: cap work + skip upload for smoke tests."""
    import time
    import torch

    # -- config bus: thread runtime overrides into the module globals that the rest of
    #    the code (train_sae_on_activations, dataset builder, paths) already reads ------
    global MODEL_ID, SAE_DIR, SAE_HUB_ID, WANDB_PROJECT, EXPANSION, N_FEATURES
    if model_id:
        MODEL_ID = model_id
        SAE_DIR = str(DATA_DIR / "saes" / _slug(MODEL_ID))
    if expansion:
        EXPANSION = int(expansion)
    if hub_id is not None:
        SAE_HUB_ID = hub_id
    if wandb_project is not None:
        WANDB_PROJECT = wandb_project

    assert capture in ("auto", "rolling"), f"unknown --capture {capture!r}"
    if capture == "rolling" and end_layer > HARD_STOP_LAYER:
        print(f"  [scope] rolling clamps end_layer {end_layer} -> {HARD_STOP_LAYER} "
              f"(Gemma KV-share boundary)")
        end_layer = HARD_STOP_LAYER

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = torch.device("cuda")
    torch.manual_seed(seed)

    # -- load model ONCE (generic loader: CausalLM, multimodal fallback) -------
    print(f"Loading {MODEL_ID} ...")
    from transformers import AutoTokenizer
    # Detect if MODEL_ID is a local path
    is_local = os.path.isdir(MODEL_ID)
    tokenizer_kwargs = {"token": hf_token} if not is_local else {}
    if is_local:
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_kwargs)
    bos_token_id = tokenizer.bos_token_id or 2
    try:
        from transformers import AutoModelForCausalLM
        model_kwargs = {"token": hf_token, "dtype": torch.bfloat16, "device_map": "cpu"}
        if is_local:
            model_kwargs["local_files_only"] = True
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    except (ValueError, KeyError, OSError) as e:
        # Multimodal checkpoints (e.g. Gemma-4) aren't plain CausalLM -- fall back.
        print(f"  CausalLM load failed ({type(e).__name__}); using image-text-to-text loader")
        from transformers import AutoModelForImageTextToText
        model_kwargs = {"token": hf_token, "dtype": torch.bfloat16, "device_map": "cpu"}
        if is_local:
            model_kwargs["local_files_only"] = True
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    n_layers = int(tcfg.num_hidden_layers)
    d_in = int(getattr(tcfg, "hidden_size", D_IN))
    N_FEATURES = EXPANSION * d_in              # SAE dict scales with the detected width
    text_model, attr_name, decoder_layers = _find_text_model(model, n_layers)
    model.to(device)

    end_layer = min(end_layer, n_layers)
    assert 0 <= start_layer < end_layer <= n_layers, \
        f"bad layer range [{start_layer},{end_layer}) for a {n_layers}-layer model"
    layers = list(range(start_layer, end_layer))
    print(f"  model={type(model).__name__}  blocks={len(decoder_layers)}  d_in={d_in}  "
          f"n_features={N_FEATURES} ({EXPANSION}x)  capture={capture}")
    if capture == "rolling" and d_in != D_IN:
        print(f"  [warn] rolling capture was tuned at d_in={D_IN}; model reports {d_in}")

    print(f"\n{'#'*60}\n  SAE ATLAS  model={_slug(MODEL_ID)}  layers={layers}  seed={seed}  "
          f"capture={capture}\n{'#'*60}\n")

    # -- disk sizing (we hold ~ONE activation pool at a time) -----------------
    import shutil
    os.makedirs(ROLLCACHE, exist_ok=True)
    shard_gb = (BATCH_TOKENS // SEQ_LEN) * SEQ_LEN * d_in * 2 / 1e9   # bf16 [n_seqs,SEQ_LEN,d]
    peak_gb = pool_batches * shard_gb * 1.1 + 2                       # ~one pool + slack
    free_gb = shutil.disk_usage(ROLLCACHE).free / 1e9
    sentinel = free_gb > 1e6                                          # overlay fs -> unreliable
    print(f"  [disk] {ROLLCACHE} peak~{peak_gb:.0f}GB (one pool of {pool_batches} x "
          f"{shard_gb*1e3:.0f}MB); free={'(overlay: unknown)' if sentinel else f'{free_gb:.0f}GB'}")
    if not sentinel:
        assert free_gb > peak_gb, (
            f"insufficient disk at {ROLLCACHE}: {free_gb:.0f}GB free < {peak_gb:.0f}GB peak. "
            f"Lower --pool-batches or set $SAE_SCRATCH_DIR to a bigger disk.")

    # -- token pool (once -- same tokens flow through every layer) ------------
    tok_dir = _pool_dir(f"tokens_s{seed}")
    _capture_token_pool(hf_token, seed, pool_batches, use_pretok, tok_dir, bos_token_id)

    marker_path = Path(SAE_DIR) / f"atlas_marker_s{seed}.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    def pool_dir_for(L):
        return _pool_dir(f"pool_L{L:02d}_s{seed}")

    results = {}
    t0 = time.time()
    # rolling must walk from 0 to build the residual chain; hook capture is independent
    # per layer, so it starts at start_layer.
    walk_start = 0 if capture == "rolling" else start_layer
    for L in range(walk_start, end_layer):
        dst_dir = pool_dir_for(L)
        if capture == "rolling":
            src_dir = pool_dir_for(L - 1) if L >= 1 else None
            _produce_pool(model, text_model, decoder_layers, tcfg, L, tok_dir, src_dir, dst_dir, device)
            if src_dir is not None:                       # consumed -> free now
                _rm_pool(src_dir)
            if L < start_layer:
                print(f"  [skip-train] L{L} pool produced; below start_layer={start_layer}")
                continue
        else:
            _produce_pool_hooked(model, decoder_layers, L, tok_dir, dst_dir, device)

        provider = RollingActivationProvider(dst_dir, device, seed=seed)
        try:
            # Determine resume checkpoint for this layer
            layer_resume = None
            if resume_from:
                # Check if the checkpoint is for this specific layer
                layer_resume = resume_from
            res = train_sae_on_activations(L, d_in, seed, provider,
                                           max_steps=max_steps, bdec_batches=bdec_batches,
                                           microbatch_tokens=microbatch_tokens or MICROBATCH_TOKENS,
                                           resume_from=layer_resume, push=push)
        finally:
            provider.close()
        results[f"layer_{L:02d}"] = res

        if capture == "auto":                             # independent capture -> free now
            _rm_pool(dst_dir)

        import json
        with open(marker_path, "w") as f:
            json.dump({"last_completed_layer": L, "seed": seed,
                       "elapsed_min": (time.time() - t0) / 60}, f)
        elapsed = (time.time() - t0) / 60
        print(f"  [ok] L{L} done: {res}  | elapsed {elapsed:.1f}min")

    print(f"\n{'#'*60}\n  SAE ATLAS COMPLETE  layers={layers}  "
          f"wall={ (time.time()-t0)/60:.1f}min\n{'#'*60}")
    return results


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Event-aware SAE trainer: one self-tuning SAE per decoder layer, "
                    "model-agnostic. The scheduler converges L0 to target with no per-layer "
                    "retuning. Default capture works on any AutoModelForCausalLM.")
    p.add_argument("--model-id", default=None,
                   help=f"HF model id to train SAEs on (default {MODEL_ID})")
    p.add_argument("--capture", choices=["auto", "rolling"], default="auto",
                   help="auto=model-agnostic forward-hook (default); rolling=Gemma single-block fast path")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--end-layer", type=int, default=9,
                   help="exclusive; clamped to model depth (and to 15 under --capture rolling)")
    p.add_argument("--expansion", type=int, default=None,
                   help=f"SAE dict = expansion * d_in (default {EXPANSION})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--pool-batches", type=int, default=POOL_BATCHES_DEFAULT,
                   help="Activation batches cached per layer. Default=4000 (~400GB). Use 500-1000 for limited disk (~50-100GB).")
    p.add_argument("--microbatch-tokens", type=int, default=MICROBATCH_TOKENS,
                   help=f"Tokens per microbatch for gradient accumulation. Default={MICROBATCH_TOKENS//1024}k (no accum). Use 8192 for 4x VRAM savings.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to checkpoint_full.pt to resume training from.")
    p.add_argument("--no-pretok", dest="use_pretok", action="store_false",
                   help="stream + tokenize FineWeb-Edu live instead of using pre-tokenized shards")
    p.add_argument("--max-steps", type=int, default=N_STEPS, help="cap steps (smoke tests)")
    p.add_argument("--bdec-batches", type=int, default=BDEC_INIT_BATCHES)
    p.add_argument("--hub-id", default=None,
                   help="HF repo to upload SAEs to. No default -- upload is skipped unless set.")
    p.add_argument("--wandb-project", default=None,
                   help="wandb project name. Off unless set (also needs WANDB_API_KEY).")
    p.add_argument("--no-push", dest="push", action="store_false",
                   help="skip the HuggingFace upload even if --hub-id is set")
    p.set_defaults(use_pretok=True, push=True)
    args = p.parse_args()

    res = run_atlas_rolling(
        start_layer=args.start_layer, end_layer=args.end_layer, seed=args.seed,
        pool_batches=args.pool_batches, microbatch_tokens=args.microbatch_tokens,
        use_pretok=args.use_pretok, max_steps=args.max_steps, bdec_batches=args.bdec_batches,
        resume_from=args.resume_from, push=args.push, capture=args.capture,
        model_id=args.model_id, hub_id=args.hub_id, wandb_project=args.wandb_project,
        expansion=args.expansion)
    print(f"\nDone. {res}")


if __name__ == "__main__":
    main()
