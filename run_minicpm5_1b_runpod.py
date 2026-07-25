#!/usr/bin/env python3
"""Train MiniCPM5-1B SAE atlas on RunPod.

Same shape as run_qwen3_0_6b_runpod.py — this is a sibling runner, not a fork.
Core trainer (sae_trainer_rolling.py / sae_scheduler.py) is untouched.

Model notes (openbmb/MiniCPM5-1B):
  - Pure LlamaForCausalLM, no remote code, no trust_remote_code needed.
    `rolling-hf` / `rolling-hf-float` apply as written.
  - 24 layers, d_in 1536 -> 49,152 features at 32x expansion.
  - GQA 16 Q / 2 KV, head_dim 128. Note 16*128 = 2048 != hidden 1536:
    attention projects UP then o_proj back DOWN. Capture site is the
    residual stream, so this does not affect the pool, but it is why the
    per-layer param count looks small relative to hidden size.
  - vocab 130,560, UNTIED embeddings (~401M params, 37% of the model).
    Expect low EV on L00/L01 the same way Qwen3-0.6B gave 0.70/0.74 there.
  - Bilingual en/zh model, English-only probe corpus -> STATED CAVEAT on
    the released weights, per the nanbeige-session decision.

Disk: pools at d_in=1536 for a 500-batch run scale to ~111GB
(74GB measured at d_in=1024). Put SAE_SCRATCH_DIR on a >=150GB volume.

Run:
    export HF_TOKEN=$(cat ~/.cache/huggingface/token)
    export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc)
    python run_minicpm5_1b_runpod.py --layer-range 0,23 \
        --capture rolling-hf-float --pool-batches 500 --max-steps 5000

Everything lands under /workspace/data on the persistent volume.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


MODEL_ID = "openbmb/MiniCPM5-1B"
HF_SAE_REPO = "juiceb0xc0de/minicpm5-1b-SAE"
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
    os.environ.setdefault("SAE_TIMING", "25")
    os.environ.setdefault("WANDB_PROJECT", "minicpm5-1b-sae")


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
    parser.add_argument("--layer-range", type=str, default="0,23",
                        help="Inclusive start,end. MiniCPM5-1B has 24 layers -> 0,23")
    parser.add_argument("--capture", type=str, default="rolling-hf-float",
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
    parser.add_argument("--wandb-project", type=str, default="minicpm5-1b-sae")
    parser.add_argument("--norm-ref", type=float, default=None,
                        help="pin activation_norm_ref (L0 probe norm) when retraining "
                             "a mid-chain layer in a fresh process")
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
    print(f"  Expansion: {args.expansion}x  (d_in 1536 -> {1536 * args.expansion} features)")
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
        norm_ref=args.norm_ref,
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
            "corpus_note": "English-only probe corpus on a bilingual en/zh model "
                           "(stated caveat, not a defect)",
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
        print(f"[HF] uploaded run_summary.json")

    print("\nTraining complete.")
    print(f"  Results: {results}")


if __name__ == "__main__":
    main()
