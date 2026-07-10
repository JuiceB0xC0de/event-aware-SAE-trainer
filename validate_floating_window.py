"""Bit-exact + timing validation for `--capture rolling-hf-float`.

Run on a RunPod H100 (or any CUDA box) with the target model cached:

    export HF_TOKEN=hf_...
    python validate_floating_window.py \
        --model-id Qwen/Qwen3-0.6B \
        --layers 0,1,5,10,15,20,27 \
        --pool-batches 50

This script produces the same activation pools twice:
  1. With the existing `rolling-hf` path (full model on GPU).
  2. With the new `rolling-hf-float` path (only active blocks on GPU).

It compares the pools for bit-exactness and prints production time for both.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch

import sae_trainer_rolling as base


def _load_model(model_id: str, hf_token: str):
    """Load model to CPU, eval mode, no grad."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=hf_token, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _setup_token_pool(model_id: str, model, hf_token: str, pool_batches: int, seed: int,
                      tok_dir: Path):
    """Capture token shards once, shared by both production runs."""
    from transformers import AutoTokenizer
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    bos_token_id = tokenizer.bos_token_id or 2
    tcfg = getattr(model.config, "text_config", model.config)
    vocab_size = tokenizer.vocab_size or getattr(tcfg, "vocab_size", None)
    base._capture_token_pool(hf_token, seed, pool_batches, False, tok_dir,
                             bos_token_id, model_id=model_id, vocab_size=vocab_size)


def _produce_with_full_model(model, text_model, decoder_layers, layer: int,
                             tok_dir: Path, src_dir: Path | None, dst_dir: Path,
                             device: torch.device):
    """Produce pool[layer] with the full model on GPU (existing rolling-hf path)."""
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    base._produce_pool_hf_rolling(model, text_model, decoder_layers, layer,
                                  tok_dir, src_dir, dst_dir, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time() - t0


def _produce_with_floating_window(text_model, decoder_layers, layer: int,
                                  tok_dir: Path, src_dir: Path | None, dst_dir: Path,
                                  device: torch.device):
    """Produce pool[layer] with only the active block + shared components on GPU."""
    window = base.FloatingLayerWindow(text_model, decoder_layers, device)
    window.activate(layer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    base._produce_pool_hf_rolling(text_model, text_model, decoder_layers, layer,
                                  tok_dir, src_dir, dst_dir, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    window.deactivate_all()
    return time.time() - t0


def _compare_dirs(dir_a: Path, dir_b: Path) -> tuple[bool, float]:
    """Return (match, max_abs_diff) across all shard files in two directories."""
    shards_a = sorted(dir_a.glob("shard_*.pt"))
    shards_b = sorted(dir_b.glob("shard_*.pt"))
    if len(shards_a) != len(shards_b):
        return False, float("inf")
    max_diff = 0.0
    for a, b in zip(shards_a, shards_b):
        ta = torch.load(a, map_location="cpu", weights_only=True).float()
        tb = torch.load(b, map_location="cpu", weights_only=True).float()
        max_diff = max(max_diff, (ta - tb).abs().max().item())
    return max_diff < 1e-3, max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layers", type=str, default="0,1,5,10,15,20,27",
                        help="comma-separated layers to validate")
    parser.add_argument("--pool-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN required")

    layers = [int(x) for x in args.layers.split(",")]

    # Dedicated scratch dir so we don't collide with production runs.
    scratch = Path("/tmp") / "validate_floating_window"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    base.ROLLCACHE = str(scratch / "rollcache")
    base.DATA_DIR = scratch / "data"
    base.SAE_DIR = str(base.DATA_DIR / "saes" / base._slug(args.model_id))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA available; validation runs on CPU and timing is not meaningful.")

    print(f"Validating floating window for {args.model_id}")
    print(f"  layers: {layers}")
    print(f"  pool_batches: {args.pool_batches}")
    print(f"  device: {device}")
    print(f"  scratch: {scratch}\n")

    print("Loading model to CPU ...")
    model = _load_model(args.model_id, hf_token)
    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    n_layers = int(tcfg.num_hidden_layers)
    d_in = int(getattr(tcfg, "hidden_size", base.D_IN))
    text_model, attr_name, decoder_layers = base._find_text_model(model, n_layers)

    if not base._is_hf_rolling_supported(text_model, decoder_layers):
        print(f"rolling-hf not supported for {type(text_model).__name__}")
        sys.exit(1)

    print(f"  text_model={type(text_model).__name__}  layers={n_layers}  d_in={d_in}\n")

    # Capture tokens once.
    tok_dir = Path(base.ROLLCACHE) / f"tokens_{base._slug(args.model_id)}_s{args.seed}"
    print("Capturing token pool ...")
    _setup_token_pool(args.model_id, model, hf_token, args.pool_batches, args.seed, tok_dir)

    results = {}
    all_ok = True

    for layer in layers:
        print(f"\n{'-'*60}\nLayer {layer}\n{'-'*60}")

        src_dir = Path(base.ROLLCACHE) / f"pool_L{layer-1:02d}_s{args.seed}_hf" \
            if layer > 0 else None
        # For layer 0 we use the token dir as source for embeddings, not a pool dir.
        # _produce_pool_hf_rolling handles layer==0 internally via inputs_embeds.

        # Produce with full model on GPU
        dst_hf = Path(base.ROLLCACHE) / f"pool_L{layer:02d}_s{args.seed}_hf"
        dst_hf.mkdir(parents=True, exist_ok=True)
        t_hf = _produce_with_full_model(model, text_model, decoder_layers, layer,
                                          tok_dir, src_dir, dst_hf, device)
        print(f"  rolling-hf:       {t_hf:6.1f}s")

        # Move model back to CPU before floating run
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Produce with floating window
        dst_float = Path(base.ROLLCACHE) / f"pool_L{layer:02d}_s{args.seed}_float"
        dst_float.mkdir(parents=True, exist_ok=True)
        t_float = _produce_with_floating_window(text_model, decoder_layers, layer,
                                                  tok_dir, src_dir, dst_float, device)
        print(f"  rolling-hf-float: {t_float:6.1f}s")

        match, max_diff = _compare_dirs(dst_hf, dst_float)
        results[layer] = {"t_hf": t_hf, "t_float": t_float,
                          "match": match, "max_diff": max_diff}
        print(f"  bit-exact: {match}  max_diff={max_diff:.3e}  "
              f"speedup={t_hf/max(t_float, 1e-9):.2f}x")
        if not match:
            all_ok = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for layer, r in results.items():
        print(f"  L{layer:02d}: hf={r['t_hf']:.1f}s  float={r['t_float']:.1f}s  "
              f"speedup={r['t_hf']/max(r['t_float'], 1e-9):.2f}x  "
              f"match={r['match']}  max_diff={r['max_diff']:.3e}")

    print(f"\nAll layers bit-exact: {all_ok}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
