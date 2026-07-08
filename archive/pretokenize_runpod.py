#!/usr/bin/env python3
"""Pre-tokenize FineWeb-Edu for any HF model on RunPod and push shards to HF.

Parallelized across CPU workers. Run on the pod:
    python pretokenize_runpod.py --model-id google/gemma-4-E2B-it \
        --n-shards 32 --tokens-per-shard 12000000 --workers 12

Writes:
    /workspace/data/pretok/google_gemma-4-e2b-it/
        manifest.json
        shard_00.npy ... shard_31.npy

Pushes to HF dataset:
    juiceb0xc0de/gemma-4-e2b-pretok
"""
import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer


DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_HF_DATASET_REPO = "juiceb0xc0de/gemma-4-e2b-pretok"


def _slug(model_id: str) -> str:
    return model_id.strip("/").replace("/", "_").lower()


def _shard_path(idx: int, pretok_dir: Path) -> Path:
    return pretok_dir / f"shard_{idx:02d}.npy"


def _build_one_shard(args, tokenizer_name: str = DEFAULT_MODEL_ID):
    shard_idx, n_shards, tokens_per_shard, pretok_dir = args
    out_path = _shard_path(shard_idx, pretok_dir)
    if out_path.exists():
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape[0] >= tokens_per_shard:
            print(f"[shard {shard_idx:02d}] exists ({existing.shape[0]} tok) -- skip")
            return shard_idx, int(existing.shape[0])

    print(f"[shard {shard_idx:02d}] streaming FineWeb-Edu shard {shard_idx}/{n_shards}...")

    # Each worker builds its own tokenizer instance (thread-safe, no global state issues)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, token=token)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    ds = ds.shard(num_shards=n_shards, index=shard_idx)

    buf = []
    total = 0
    for row in ds:
        text = row.get("text", "")
        if not text.strip():
            continue
        ids = tok(text, add_special_tokens=True).input_ids
        if len(ids) < 8:
            continue
        ids_arr = np.asarray(ids, dtype=np.int32)
        buf.append(ids_arr)
        total += len(ids)
        if total >= tokens_per_shard:
            break

    if not buf:
        raise RuntimeError(f"[shard {shard_idx:02d}] no tokens collected")

    arr = np.concatenate(buf)[:tokens_per_shard]
    np.save(out_path, arr)
    print(f"[shard {shard_idx:02d}] wrote {arr.shape[0]} tokens -> {out_path}")
    return shard_idx, int(arr.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help="HF model id to tokenize for")
    parser.add_argument("--n-shards", type=int, default=32)
    parser.add_argument("--tokens-per-shard", type=int, default=12_000_000)
    parser.add_argument("--workers", type=int, default=12,
                        help="Number of parallel shard-building processes")
    parser.add_argument("--push", action="store_true", default=True,
                        help="Push shards + manifest to HF dataset (default True)")
    parser.add_argument("--no-push", action="store_false", dest="push",
                        help="Skip HF upload")
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_DATASET_REPO)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var required")

    model_id = args.model_id
    slug = _slug(model_id)
    pretok_dir = Path("/workspace/data/pretok") / slug
    pretok_dir.mkdir(parents=True, exist_ok=True)

    # Get vocab size once in the main process
    print(f"Loading tokenizer for {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    vocab_size = tok.vocab_size

    # Build task list and run in parallel
    tasks = [(i, args.n_shards, args.tokens_per_shard, pretok_dir) for i in range(args.n_shards)]
    workers = min(args.workers, os.cpu_count() or 1, args.n_shards)
    print(f"Building {args.n_shards} shards using {workers} workers for {model_id}...")

    with Pool(processes=workers) as pool:
        results = pool.map(partial(_build_one_shard, tokenizer_name=model_id), tasks)

    # Write manifest
    manifest = {
        "n_shards": args.n_shards,
        "vocab_size": vocab_size,
        "model_id": model_id,
        "shard_tokens": {f"shard_{i:02d}": c for i, c in results},
    }
    manifest_path = pretok_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"[manifest] wrote {manifest_path}: n_shards={args.n_shards} vocab_size={vocab_size}")

    # Push to HF
    if args.push:
        print(f"Pushing to HF dataset {args.hf_repo}...")
        api = HfApi(token=token)
        try:
            create_repo(args.hf_repo, repo_type="dataset", private=False, exist_ok=True, token=token)
        except Exception as e:
            print(f"  (repo creation note: {e})")

        api.upload_folder(
            folder_path=str(pretok_dir),
            repo_id=args.hf_repo,
            repo_type="dataset",
            token=token,
        )
        print(f"[HF] uploaded {pretok_dir} -> {args.hf_repo}")
    else:
        print("[HF] skipped --no-push")

    print("\nPretokenize complete.")
    total_tokens = sum(c for _, c in results)
    print(f"  Output: {pretok_dir}")
    print(f"  Total tokens written: {total_tokens:,}")


if __name__ == "__main__":
    main()
