#!/usr/bin/env python3
"""Train SmolLM2-135M SFT-Only SAE atlas on RunPod.

Run:
    export HF_TOKEN=hf_...
    export WANDB_API_KEY=...
    python run_smollm2_runpod.py \
        --model-id google/gemma-4-E4B-it \
        --hf-sae-repo juiceb0xc0de/gemma-4-E4B-it-SAE \
        --layer-range 0,29 \
        --capture rolling-hf \
        --pool-batches 500 \
        --max-steps 5000 \
        --target-l0 50 \
        --microbatch-tokens 32768 \
        --timing

Everything is /workspace; persistent network volume keeps outputs if the pod dies.
"""
import argparse
import os
import sys
from pathlib import Path


MODEL_ID = "google/gemma-4-E4B-it"
HF_SAE_REPO = "juiceb0xc0de/gemma-4-E4B-it-SAE"


def _env():
    os.environ.setdefault("SAE_DATA_DIR", "/workspace/data")
    os.environ.setdefault("SAE_SCRATCH_DIR", "/workspace/data/rollcache")
    os.environ.setdefault("SAE_MODEL_ID", MODEL_ID)
    os.environ.setdefault("SAE_HUB_ID", HF_SAE_REPO)
    os.environ.setdefault("SAE_BATCH_TOKENS", "32768")
    os.environ.setdefault("SAE_MICROBATCH_TOKENS", "32768")
    os.environ.setdefault("SAE_USE_TRITON", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("SAE_TIMING", "1")
    os.environ.setdefault("WANDB_PROJECT", "smollm2-sae")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--hf-sae-repo", type=str, default=HF_SAE_REPO)
    parser.add_argument("--layer-range", type=str, default="0,29")
    parser.add_argument("--capture", type=str, default="rolling-hf",
                        choices=["auto", "rolling", "rolling-hf"])
    parser.add_argument("--pool-batches", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--target-l0", type=int, default=50)
    parser.add_argument("--microbatch-tokens", type=int, default=32768)
    parser.add_argument("--timing", action="store_true", default=True)
    parser.add_argument("--no-pretok", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="smollm2-sae")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN env var required")

    _env()
    os.environ["WANDB_PROJECT"] = args.wandb_project

    start, end = map(int, args.layer_range.split(","))

    sys.path.insert(0, str(Path(__file__).parent))
    from sae_trainer_rolling import run_atlas_rolling

    run_atlas_rolling(
        start_layer=start,
        end_layer=end,
        seed=0,
        pool_batches=args.pool_batches,
        microbatch_tokens=args.microbatch_tokens,
        use_pretok=not args.no_pretok,
        max_steps=args.max_steps,
        bdec_batches=50,
        resume_from=None,
        push=not args.no_push,
        capture=args.capture,
        model_id=args.model_id,
        hub_id=args.hf_sae_repo,
        wandb_project=args.wandb_project,
        expansion=32,
        evict_model=True,
        target_l0=args.target_l0,
    )


if __name__ == "__main__":
    main()
