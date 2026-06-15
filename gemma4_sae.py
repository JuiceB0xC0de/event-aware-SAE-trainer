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
        "colorama",  # cache-bust + harmless
        force_build=True,  # rebuild this layer every deploy; remove when stable
    )
    # Bake the local trainer modules into the image so the container
    # always sees the working tree (Modal 1.4 dropped cls-level mounts).
    .add_local_file(
        "/Users/chiggy/event-aware-SAE-trainer/sae_trainer_rolling.py",
        "/opt/sae-trainer/sae_trainer_rolling.py",
        copy=True,
    )
    .add_local_file(
        "/Users/chiggy/event-aware-SAE-trainer/sae_scheduler.py",
        "/opt/sae-trainer/sae_scheduler.py",
        copy=True,
    )
    .env({
        "SAE_DATA_DIR": "/data",
        # Container-local NVMe, NOT a Modal Volume. The activation pool is read
        # once per step (168MB/shard at d_in=2560); on the network Volume this
        # capped throughput at ~193MB/s and starved the H100 (37k tok/s) while
        # blocking the Modal heartbeat -> container restarts. Local disk is GB/s.
        "SAE_SCRATCH_DIR": "/root/rollcache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
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


@app.cls(
    gpu="H100",
    volumes={"/data": data_volume},          # /scratch removed: pool lives on local NVMe now
    ephemeral_disk=1_048_576,                 # 1 TiB container-local NVMe for the activation pool
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=86400,  # 24 hours max per Modal limits
)
class SAETainer:
    """Train SAEs on Gemma-4 layers."""

    @modal.enter()
    def setup(self):
        import os
        from huggingface_hub import snapshot_download

        hf_token = os.environ.get("HF_TOKEN")
        volume_model_path = "/data/models/gemma-4-e4b"

        if os.path.exists(volume_model_path) and os.listdir(volume_model_path):
            print(f"Using cached Gemma-4 E4B from volume: {volume_model_path}")
        else:
            print("Downloading google/gemma-4-e4b-it from HuggingFace (first run, will cache on volume)...")
            os.makedirs(volume_model_path, exist_ok=True)
            snapshot_download(
                "google/gemma-4-e4b-it",
                local_dir=volume_model_path,
                token=hf_token,
            )
            print(f"Cached to volume at: {volume_model_path}")

        self.model_path = volume_model_path

    @modal.method()
    def train(self, layer_range: str, capture: str, expansion: int = 32,
              pool_batches: int = 2000, max_steps: int = 15000,
              microbatch_tokens: int = 32768, resume_from: str | None = None,
              evict_model: bool = True, target_l0: int | None = None,
              timing: bool = False,
              scratch_dir: str = "/root/rollcache") -> dict:
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
        """
        import os
        import sys
        import subprocess

        # The trainer modules are baked into the image at /opt/sae-trainer
        # (see Image.add_local_file above). No need to clone or patch.
        sys.path.insert(0, "/opt/sae-trainer")
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
        print(f"  Scratch dir: {scratch_dir}")
        print(f"{'='*60}\n")

        import os as _os
        if timing:
            _os.environ["SAE_TIMING"] = "1"
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
            model_id=self.model_path,
            hub_id=None,
            wandb_project=os.environ.get("WANDB_PROJECT"),
            expansion=expansion,
            evict_model=evict_model,
            target_l0=target_l0,
        )

        # Commit volumes
        subprocess.run(["sync"])

        return {"status": "complete", "layers": list(range(start, end)), "results": results}


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
                      model_path: str = "/data/models/gemma-4-e4b") -> int:
    """Stream a disjoint slice of FineWeb-Edu, tokenize it, and write one 1-D int32
    token shard to /data/pretok/fineweb-edu/shard_NN.npy. Parallelized across
    containers via .starmap (one container per shard). Idempotent: skips a shard
    that already has >= tokens_per_shard tokens."""
    import os
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(PRETOK_OUT, exist_ok=True)
    shard_path = f"{PRETOK_OUT}/shard_{shard_idx:02d}.npy"
    if os.path.exists(shard_path):
        existing = np.load(shard_path, mmap_mode="r")
        if existing.shape[0] >= tokens_per_shard:
            print(f"  [pretok {shard_idx:02d}] exists ({existing.shape[0]} tok) -- skip")
            return shard_idx

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
def _write_pretok_manifest(n_shards: int):
    import json
    import os
    os.makedirs(PRETOK_OUT, exist_ok=True)
    with open(f"{PRETOK_OUT}/manifest.json", "w") as f:
        json.dump({"n_shards": n_shards}, f)
    data_volume.commit()
    print(f"  [pretok] manifest -> n_shards={n_shards}")


@app.local_entrypoint()
def pretokenize(n_shards: int = 16, tokens_per_shard: int = 12_000_000):
    """Build the pretokenized FineWeb-Edu shards the trainer reads with use_pretok=True.
    Run ONCE (the Gemma-4 model must already be cached on /data from a prior run):

        modal run gemma4_sae.py::pretokenize
        modal run gemma4_sae.py::pretokenize --n-shards 24 --tokens-per-shard 10000000

    16 shards x 12M tok = ~192M tokens (~770MB int32). Containers run in parallel,
    so wall time is ~one shard's tokenize pass, not the sum."""
    args = [(i, n_shards, tokens_per_shard) for i in range(n_shards)]
    done = list(pretokenize_shard.starmap(args))
    _write_pretok_manifest.remote(n_shards)
    print(f"\nPretokenize complete: {len(done)} shards x {tokens_per_shard/1e6:.0f}M tok "
          f"-> {PRETOK_OUT}")


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
):
    """
    Train SAEs on Gemma-4 E4B layers.

    Examples:
        modal run gemma4_sae.py --layer-range 0,15 --capture rolling
        modal run gemma4_sae.py --layer-range 15,42 --capture auto
        modal run gemma4_sae.py --layer-range 5,10 --capture rolling --resume-from /data/saes/data_models_gemma-4-e4b/layer_05_s0/checkpoint_full.pt
    """
    tainer = SAETainer()
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
        scratch_dir=scratch_dir,
    )
    print(f"\nTraining complete: {result['status']}")
    print(f"Layers trained: {result['layers']}")
    return result
