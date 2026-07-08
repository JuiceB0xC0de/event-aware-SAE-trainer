#!/usr/bin/env python3
"""Train Qwen3-0.6B SAE atlas on RunPod.

Reproduces the SmolLM2 H100 run shape:
  --pool-batches 500 --max-steps 5000 --target-l0 50
  --microbatch-tokens 32768 --capture rolling-hf

Run:
    export HF_TOKEN=hf_...
    export WANDB_API_KEY=...
    python run_qwen3_0_6b_runpod.py \
        --layer-range 0,27 \
        --pool-batches 500 \
        --max-steps 5000

Everything lands under /workspace/data on the persistent network volume.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


MODEL_ID = "Qwen/Qwen3-0.6B"
HF_SAE_REPO = "juiceb0xc0de/qwen3-0.6b-sae"
SLUG = MODEL_ID.replace("/", "_").lower()
DATA_DIR = Path("/workspace/data")
SAE_DIR = DATA_DIR / "saes" / SLUG


def _env():
    os.environ.setdefault("SAE_DATA_DIR", str(DATA_DIR))
    os.environ.setdefault("SAE_SCRATCH_DIR", str(DATA_DIR / "rollcache"))
    os.environ.setdefault("SAE_MODEL_ID", MODEL_ID)
    os.environ.setdefault("SAE_HUB_ID", HF_SAE_REPO)
    os.environ.setdefault("SAE_BATCH_TOKENS", "32768")
    os.environ.setdefault("SAE_MICROBATCH_TOKENS", "32768")
    os.environ.setdefault("SAE_USE_TRITON", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("SAE_TIMING", "1")
    os.environ.setdefault("WANDB_PROJECT", "qwen3-0.6b-sae")


def _push_layer_to_hf(layer: int, seed: int = 0):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(f"[HF push L{layer:02d}] no HF_TOKEN, skipping")
        return False

    layer_dir = SAE_DIR / f"layer_{layer:02d}_s{seed}"
    if not layer_dir.exists():
        print(f"[HF push L{layer:02d}] {layer_dir} not found, skipping")
        return False

    api = HfApi(token=token)
    try:
        create_repo(HF_SAE_REPO, repo_type="model", private=False, exist_ok=True, token=token)
    except Exception as e:
        print(f"  [HF push L{layer:02d}] repo creation note: {e}")

    upload_folder(
        folder_path=str(layer_dir),
        repo_id=HF_SAE_REPO,
        repo_type="model",
        path_in_repo=f"layer_{layer:02d}_s{seed}",
        token=token,
    )
    print(f"[HF push L{layer:02d}] uploaded -> {HF_SAE_REPO}/tree/main/layer_{layer:02d}_s{seed}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-range", type=str, default="0,27",
                        help="Inclusive start,end e.g. 0,27")
    parser.add_argument("--capture", type=str, default="rolling-hf",
                        choices=["auto", "rolling", "rolling-hf", "rolling-hf-float"])
    parser.add_argument("--expansion", type=int, default=32)
    parser.add_argument("--pool-batches", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--target-l0", type=int, default=50)
    parser.add_argument("--microbatch-tokens", type=int, default=32768)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-pretok", action="store_true")
    parser.add_argument("--timing", action="store_true", default=True)
    parser.add_argument("--no-timing", action="store_false", dest="timing")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--no-model-evict", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="qwen3-0.6b-sae")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN env var required")

    _env()
    os.environ["WANDB_PROJECT"] = args.wandb_project

    start, end = map(int, args.layer_range.split(","))

    sys.path.insert(0, str(Path(__file__).parent))
    from sae_trainer_rolling import run_atlas_rolling

    print(f"\n{'='*60}")
    print("RunPod SAE atlas training")
    print(f"  Model: {MODEL_ID}")
    print(f"  Layers: {start}-{end} ({end - start + 1} layers)")
    print(f"  Capture: {args.capture}")
    print(f"  Expansion: {args.expansion}x")
    print(f"  Pool batches: {args.pool_batches}")
    print(f"  Max steps/layer: {args.max_steps}")
    print(f"  Target L0: {args.target_l0}")
    print(f"  Microbatch: {args.microbatch_tokens} tokens")
    print(f"  SAE dir: {SAE_DIR}")
    print(f"  HF repo: {HF_SAE_REPO}")
    print(f"{'='*60}\n")

    results = run_atlas_rolling(
        start_layer=start,
        end_layer=end,
        seed=0,
        pool_batches=args.pool_batches,
        microbatch_tokens=args.microbatch_tokens,
        use_pretok=not args.no_pretok,
        max_steps=args.max_steps,
        bdec_batches=50,
        resume_from=args.resume_from,
        push=not args.no_push,
        capture=args.capture,
        model_id=MODEL_ID,
        hub_id=HF_SAE_REPO,
        wandb_project=args.wandb_project,
        expansion=args.expansion,
        evict_model=not args.no_model_evict,
        target_l0=args.target_l0,
    )

    if not args.no_push:
        for layer in range(start, end + 1):
            _push_layer_to_hf(layer, seed=0)

        summary = {
            "model_id": MODEL_ID,
            "layers": f"{start},{end}",
            "capture": args.capture,
            "expansion": args.expansion,
            "pool_batches": args.pool_batches,
            "max_steps": args.max_steps,
            "target_l0": args.target_l0,
            "results": results,
        }
        SAE_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = SAE_DIR / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo="run_summary.json",
            repo_id=HF_SAE_REPO,
            repo_type="model",
        )
        print("[HF] uploaded run_summary.json")

    print("\nTraining complete.")
    print(f"  Results: {results}")


if __name__ == "__main__":
    main()
