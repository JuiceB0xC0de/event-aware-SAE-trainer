"""
Modal app for training event-aware SAEs on Gemma-4 models from HuggingFace.

Usage:
    modal run gemma4_sae.py --layer-range 0,15 --capture rolling
    modal run gemma4_sae.py --layer-range 15,42 --capture auto

    # Or detach for long runs:
    modal deploy gemma4_sae.py
    modal run gemma4-sae-train --layer-range 0,15 --capture rolling
"""
import modal
from modal import Image, Volume

# =============================================================================
# Image definition
# =============================================================================

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "unzip")
    .pip_install(
        "torch>=2.5.0",
        "transformers>=4.51.0",
        "accelerate",
        "datasets>=2.20.0",
        "numpy",
        "tqdm",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "hf_transfer",
        "wandb",
        "bitsandbytes",
        "triton>=2.0.0",  # For fused SAE kernels
        "colorama",
    )
    .env({
        "TRITON_CACHE_DIR": "/tmp/triton-cache",  # Avoid permission issues
        "SAE_USE_TRITON": "1",  # Enable Triton fused kernel experiment
    })
    .env({})
    # Trainer modules are NOT baked into the image — they're cloned fresh from
    # GitHub at container start (see _sync_repo). Keeps this cached pip layer
    # untouched so new commits don't trigger a full reinstall.
    .env({
        "SAE_DATA_DIR": "/data",
        # Container-local NVMe, NOT a Modal Volume. The activation pool is read
        # once per step (168MB/shard at d_in=2560); on the network Volume this
        # capped throughput at ~193MB/s and starved the H100 (37k tok/s) while
        # blocking the Modal heartbeat -> container restarts. Local disk is GB/s.
        "SAE_SCRATCH_DIR": "/root/rollcache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Synchronous CUDA errors so assert/oom tracebacks point at the real line.
        "CUDA_LAUNCH_BLOCKING": "1",
    })
)
# =============================================================================
# Volumes (persistent storage)
# =============================================================================

# Persistent volume for SAE outputs, activation pools, and Kaggle model cache
data_volume = Volume.from_name("sae-training-data", create_if_missing=True)

# Scratch volume for temporary activation pools (can be ephemeral)
scratch_volume = Volume.from_name("sae-training-scratch", create_if_missing=True)

# =============================================================================
# Secrets
# =============================================================================

# HF_TOKEN read from the "huggingface" Modal secret
# Optional: WANDB_PROJECT for logging

# =============================================================================
# App definition
# =============================================================================

app = modal.App("gemma4-sae-train", image=image)

# =============================================================================
# Runtime code sync
# =============================================================================

REPO_URL = "https://github.com/JuiceB0xC0de/event-aware-SAE-trainer.git"
REPO_BRANCH = "main"
REPO_DIR = "/opt/sae-trainer"


def _sync_repo():
    """Clone the latest trainer code from GitHub on container start.

    Runtime clone (not baked into the image) so new commits land without an
    image rebuild — the cached pip layer above is never invalidated. Re-clones
    fresh on every cold start; idempotent within a warm container. Public repo,
    so no Secret/token needed.
    """
    import os
    import sys
    import subprocess

    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        subprocess.run(["rm", "-rf", REPO_DIR], check=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
             REPO_URL, REPO_DIR],
            check=True,
        )
        sha = subprocess.run(
            ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        print(f"[_sync_repo] cloned {REPO_BRANCH} @ {sha} -> {REPO_DIR}")

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)


@app.cls(
    gpu="A10G",  # 24GB VRAM - enough for SmolLM2-360M with room to profile
    volumes={"/data": data_volume},
    ephemeral_disk=524_288,  # 512 GiB container-local NVMe
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=86400,  # 24 hours
)
class SAETainer:
    """Train SAEs on any HuggingFace causal LM."""

    model_id: str = modal.parameter(default="HuggingFaceTB/SmolLM2-360M")

    @modal.enter()
    def setup(self):
        import os
        from huggingface_hub import snapshot_download

        # Pull fresh trainer code from GitHub (cached pip layer untouched).
        _sync_repo()

        hf_token = os.environ.get("HF_TOKEN")
        model_id = self.model_id
        slug = model_id.replace("/", "_")
        volume_model_path = f"/data/models/{slug}"

        if os.path.exists(volume_model_path) and os.listdir(volume_model_path):
            print(f"Using cached {model_id} from volume: {volume_model_path}")
        else:
            print(f"Downloading {model_id} from HuggingFace (first run, will cache on volume)...")
            os.makedirs(volume_model_path, exist_ok=True)
            snapshot_download(
                model_id,
                local_dir=volume_model_path,
                token=hf_token,
            )
            print(f"Cached to volume at: {volume_model_path}")

        self.model_path = volume_model_path
        self._model_id = model_id  # Keep original HF model_id for trainer

    @modal.method()
    def train(self, layer_range: str, capture: str, expansion: int = 32,
              pool_batches: int = 2000, max_steps: int = 15000,
              microbatch_tokens: int = 32768, resume_from: str | None = None,
              evict_model: bool = True, target_l0: int | None = None,
              timing: bool = False, compile: bool = False,
              scratch_dir: str = "/root/rollcache",
              model_id: str = "google/gemma-4-e4b-it") -> dict:
        """
        Train SAEs on a range of layers.

        Args:
            layer_range: "start,end" e.g. "0,15" or "15,42"
            capture: "rolling" (layers 0-14) or "auto" (any layer)
            expansion: SAE expansion factor (default 32)
            pool_batches: activation batches to cache (default 2000)
            max_steps: max training steps per layer
            microbatch_tokens: for gradient accumulation (default 32k = no accum)
            resume_from: optional /data/.../checkpoint_full.pt for the first trained layer
            evict_model: move the LLM to CPU during SAE training to free VRAM (default True)
            compile: enable torch.compile on the SAE (default False)
        """
        import os
        import subprocess

        # Trainer modules are cloned fresh from GitHub at container start;
        # setup() already ran _sync_repo(), this is a no-op safety net.
        _sync_repo()
        start, end = map(int, layer_range.split(","))

        # Import trainer module
        from sae_trainer_rolling import run_atlas_rolling

        print(f"\n{'='*60}")
        print("Training SAE on Gemma-4 E4B")
        print(f"  Model: {self.model_path}")
        print(f"  Layers: {start}-{end} ({end-start} layers)")
        print(f"  Capture: {capture}")
        print(f"  Expansion: {expansion}x")
        print(f"  Pool batches: {pool_batches}")
        if resume_from:
            print(f"  Resume: {resume_from}")
        print(f"  Evict model during training: {evict_model}")
        print(f"  Target L0: {target_l0 if target_l0 is not None else 'default'}")
        print(f"  Timing: {timing}")
        print(f"  Compile SAE: {compile}")
        print(f"  Scratch dir: {scratch_dir}")
        print(f"{'='*60}\n")

        import os as _os
        if timing:
            _os.environ["SAE_TIMING"] = "1"
        if compile:
            _os.environ["SAE_COMPILE"] = "1"
        if scratch_dir:
            _os.environ["SAE_SCRATCH_DIR"] = scratch_dir

        # Run training
        results = run_atlas_rolling(
            start_layer=start,
            end_layer=end,
            seed=0,
            pool_batches=pool_batches,
            microbatch_tokens=microbatch_tokens,
            use_pretok=True,  # Use pretokenized shards (faster). Build once with:
                              #   modal run gemma4_sae.py::pretokenize
            max_steps=max_steps,
            bdec_batches=50,
            resume_from=resume_from,
            push=False,  # don't push to HF, save locally
            capture=capture,
            model_id=self._model_id,  # Use original HF model_id (not /data/models/... path)
            hub_id=None,
            wandb_project=os.environ.get("WANDB_PROJECT"),
            expansion=expansion,
            evict_model=evict_model,
            target_l0=target_l0,
        )

        # Commit volumes
        subprocess.run(["sync"])

        return {"status": "complete", "layers": list(range(start, end)), "results": results}

    @modal.method()
    def benchmark_kernel(self) -> dict:
        """Run the Triton kernel micro-benchmark on an A10G.

        Pulls the latest trainer code via _sync_repo, then runs the isolated
        forward+backward correctness + timing benchmark.
        """
        import os
        import subprocess
        import sys

        _sync_repo()
        sys.path.insert(0, "/opt/sae-trainer")
        env = os.environ.copy()
        proc = subprocess.run(
            [sys.executable, "/opt/sae-trainer/benchmark_triton_kernel.py"],
            cwd="/opt/sae-trainer",
            env=env,
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            raise RuntimeError(f"benchmark failed with code {proc.returncode}")
        return {"status": "ok", "stdout": proc.stdout}

    @modal.method()
    def debug_kernel(self) -> dict:
        """Run the tiny debug script that prints Triton vs PyTorch intermediates."""
        import os
        import subprocess
        import sys

        _sync_repo()
        sys.path.insert(0, "/opt/sae-trainer")
        env = os.environ.copy()
        proc = subprocess.run(
            [sys.executable, "/opt/sae-trainer/debug_triton_kernel.py"],
            cwd="/opt/sae-trainer",
            env=env,
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            raise RuntimeError(f"debug failed with code {proc.returncode}")
        return {"status": "ok", "stdout": proc.stdout}


# =============================================================================
# Pre-tokenization (build /data/pretok/fineweb-edu shards -- the fast data path)
# =============================================================================

PRETOK_OUT = "/data/pretok/fineweb-edu"


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=86400,
)
def pretokenize_shard(shard_idx: int, n_shards: int, tokens_per_shard: int,
                      model_id: str, model_path: str, hf_token: str) -> int:
    """Stream a disjoint slice of FineWeb-Edu, tokenize it with the target model's
    tokenizer, and write one 1-D int32 token shard to
    /data/pretok/<model_slug>/shard_NN.npy.  Parallelized across containers via
    .starmap (one container per shard). Idempotent: skips a shard that already
    has >= tokens_per_shard tokens."""
    import os
    import numpy as np
    from pathlib import Path
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    # Import _slug from trainer to ensure path matches exactly
    # The trainer slugs the full MODEL_ID path, not just the model name.
    # _sync_repo clones the code (no @modal.enter here — this is a plain fn).
    _sync_repo()
    from sae_trainer_rolling import _slug
    # Match trainer's behavior: slug the full model path
    model_slug = _slug(f"data_models_{model_id.replace('/', '_')}")
    out_dir = f"/data/pretok/{model_slug}"
    os.makedirs(out_dir, exist_ok=True)
    shard_path = f"{out_dir}/shard_{shard_idx:02d}.npy"
    if os.path.exists(shard_path):
        existing = np.load(shard_path, mmap_mode="r")
        if existing.shape[0] >= tokens_per_shard:
            print(f"  [pretok {shard_idx:02d}] exists ({existing.shape[0]} tok) -- skip")
            return shard_idx

    # Download model to volume path if not cached
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"  [pretok {shard_idx:02d}] downloading model to {model_path}...")
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(model_id, local_dir=model_path, token=hf_token)

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    ds = ds.shard(num_shards=n_shards, index=shard_idx)

    buf, total = [], 0
    for row in ds:
        text = row.get("text", "")
        if not text.strip():
            continue
        ids = tok(text, add_special_tokens=True).input_ids
        if len(ids) < 8:
            continue
        prev = total
        buf.append(np.asarray(ids, dtype=np.int32))
        total += len(ids)
        if total // 1_000_000 != prev // 1_000_000:
            print(f"  [pretok {shard_idx:02d}] {total/1e6:.0f}M / {tokens_per_shard/1e6:.0f}M tok")
        if total >= tokens_per_shard:
            break

    arr = np.concatenate(buf)[:tokens_per_shard]
    np.save(shard_path, arr)
    data_volume.commit()
    print(f"  [pretok {shard_idx:02d}] wrote {arr.shape[0]} tok -> {shard_path}")
    return shard_idx


@app.function(image=image, volumes={"/data": data_volume})
def _write_pretok_manifest(n_shards: int, model_slug: str, vocab_size: int):
    import json
    import os
    out_dir = f"/data/pretok/{model_slug}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump({"n_shards": n_shards, "vocab_size": vocab_size}, f)
    data_volume.commit()
    print(f"  [pretok] manifest -> n_shards={n_shards} vocab_size={vocab_size}")


@app.local_entrypoint()
def pretokenize(
    n_shards: int = 16,
    tokens_per_shard: int = 12_000_000,
    model_id: str = "google/gemma-4-e4b-it",
):
    """Build the pretokenized FineWeb-Edu shards the trainer reads with use_pretok=True.
    Run once per tokenizer:

        modal run gemma4_sae.py::pretokenize
        modal run gemma4_sae.py::pretokenize --model-id meta-llama/Llama-3.2-1B

    16 shards x 12M tok = ~192M tokens (~770MB int32). Containers run in parallel,
    so wall time is ~one shard's tokenize pass, not the sum."""
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    import os

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model_slug = model_id.replace("/", "_")
    volume_model_path = f"/data/models/{model_slug}"

    # Download model locally first (to ~/.cache/huggingface), then remote will cache on volume
    print(f"Downloading {model_id} to local cache...")
    snapshot_download(model_id, token=hf_token)

    # Get vocab size locally
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    vocab_size = tok.vocab_size

    # Remote function handles volume download + pretok
    args = [(i, n_shards, tokens_per_shard, model_id, volume_model_path, hf_token) for i in range(n_shards)]
    done = list(pretokenize_shard.starmap(args))
    _write_pretok_manifest.remote(n_shards, model_slug, vocab_size)
    print(f"\nPretokenize complete: {len(done)} shards x {tokens_per_shard/1e6:.0f}M tok "
          f"-> /data/pretok/{model_slug}")


# =============================================================================
# Local entry point (for `modal run`)
# =============================================================================

@app.local_entrypoint()
def main(
    layer_range: str = "0,15",
    capture: str = "rolling",
    expansion: int = 32,
    pool_batches: int = 2000,
    max_steps: int = 15000,
    resume_from: str | None = None,
    evict_model: bool = True,
    target_l0: int | None = None,
    timing: bool = False,
    scratch_dir: str = "/root/rollcache",
    model_id: str = "google/gemma-4-e4b-it",
    compile: bool = False,
):
    """
    Train SAEs on any HuggingFace causal LM.

    Examples:
        modal run gemma4_sae.py --layer-range 0,15 --capture rolling
        modal run gemma4_sae.py --model-id meta-llama/Llama-3.2-1B --layer-range 0,16 --capture auto
        modal run gemma4_sae.py --layer-range 5,10 --capture rolling --resume-from /data/saes/data_models_gemma-4-e4b/layer_05_s0/checkpoint_full.pt
        modal run gemma4_sae.py --model-id meta-llama/Llama-3.2-1B --layer-range 0,16 --capture auto --compile
    """
    tainer = SAETainer(model_id=model_id)
    result = tainer.train.remote(
        layer_range=layer_range,
        capture=capture,
        expansion=expansion,
        pool_batches=pool_batches,
        max_steps=max_steps,
        resume_from=resume_from,
        evict_model=evict_model,
        target_l0=target_l0,
        timing=timing,
        compile=compile,
        scratch_dir=scratch_dir,
        model_id=model_id,
    )
    print(f"\nTraining complete: {result['status']}")
    print(f"Layers trained: {result['layers']}")
    return result
