"""
Modal app for training event-aware SAEs on Gemma-4 models from Kaggle.

Usage:
    modal run gemma4_sae.py --layer-range 0,15 --capture rolling
    modal run gemma4_sae.py --layer-range 15,42 --capture auto

    # Or detach for long runs:
    modal deploy gemma4_sae.py
    modal run gemma4-sae-train --layer-range 0,15 --capture rolling
"""
import modal
from modal import Image, Volume
from modal.mount import Mount

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
        "kagglehub",
        "wandb",
    )
    .env({"SAE_DATA_DIR": "/data", "SAE_SCRATCH_DIR": "/scratch"})
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

# KAGEL_KEY should already be set in your Modal secrets
# Optional: WANDB_PROJECT for logging

# =============================================================================
# App definition
# =============================================================================

app = modal.App("gemma4-sae-train", image=image)


# Mount the local trainer so container uses your updated code (with local_files_only fix)
trainer_mount = Mount._from_local_dir(
    local_path="/Users/chiggy/event-aware-SAE-trainer",
    remote_path="/opt/sae-trainer",
    condition=lambda p: p.endswith(".py"),
)

@app.cls(
    gpu="H100",
    volumes={"/data": data_volume, "/scratch": scratch_volume},
    secrets=[modal.Secret.from_name("KAGEL_KEY")],
    timeout=86400,  # 24 hours max per Modal limits
)
class SAETainer:
    """Train SAEs on Gemma-4 layers."""

    @modal.enter()
    def setup(self):
        import os
        import kagglehub
        import shutil

        # Kaggle auth - KAGEL_KEY should be set from Modal secret
        kaggle_key = os.environ.get("KAGEL_KEY") or os.environ.get("KAGGLE_API_KEY")
        if kaggle_key:
            os.environ["KAGGLE_API_KEY"] = kaggle_key

        # Cache model on persistent volume (survives across runs)
        volume_model_path = "/data/models/gemma-4-e4b"
        kaggle_cache_path = "/root/.cache/kagglehub/google/gemma-4/transformers/gemma-4-e4b"

        if os.path.exists(volume_model_path) and os.listdir(volume_model_path):
            # Already cached on volume from previous run
            print(f"Loading cached Gemma-4 E4B from volume: {volume_model_path}")
            self.model_path = volume_model_path
        else:
            # First run - download from Kaggle to volume
            print("Downloading Gemma-4 E4B from Kaggle (first run, will cache on volume)...")
            kaggle_path = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e4b")
            # Find the versioned directory
            if os.path.isdir(kaggle_cache_path):
                versions = [d for d in os.listdir(kaggle_cache_path) if d.isdigit()]
                if versions:
                    kaggle_path = os.path.join(kaggle_cache_path, versions[0])
            # Copy to volume for persistence
            os.makedirs(os.path.dirname(volume_model_path), exist_ok=True)
            if os.path.exists(volume_model_path):
                shutil.rmtree(volume_model_path)
            shutil.copytree(kaggle_path, volume_model_path)
            print(f"Cached to volume at: {volume_model_path}")
            self.model_path = volume_model_path

    @modal.method()
    def train(self, layer_range: str, capture: str, expansion: int = 32,
              pool_batches: int = 2000, max_steps: int = 15000,
              microbatch_tokens: int = 32768) -> dict:
        """
        Train SAEs on a range of layers.

        Args:
            layer_range: "start,end" e.g. "0,15" or "15,42"
            capture: "rolling" (layers 0-14) or "auto" (any layer)
            expansion: SAE expansion factor (default 32)
            pool_batches: activation batches to cache (default 2000)
            max_steps: max training steps per layer
            microbatch_tokens: for gradient accumulation (default 32k = no accum)
        """
        import os
        import sys
        sys.path.insert(0, "/opt/sae-trainer")

        start, end = map(int, layer_range.split(","))

        # Import trainer module (mounted from local directory)
        from sae_trainer_rolling import run_atlas_rolling

        print(f"\n{'='*60}")
        print(f"Training SAE on Gemma-4 E4B")
        print(f"  Model: {self.model_path}")
        print(f"  Layers: {start}-{end} ({end-start} layers)")
        print(f"  Capture: {capture}")
        print(f"  Expansion: {expansion}x")
        print(f"  Pool batches: {pool_batches}")
        print(f"{'='*60}\n")

        # Run training
        results = run_atlas_rolling(
            start_layer=start,
            end_layer=end,
            seed=0,
            pool_batches=pool_batches,
            microbatch_tokens=microbatch_tokens,
            use_pretok=True,  # Use pretokenized shards (faster)
            max_steps=max_steps,
            bdec_batches=50,
            resume_from=None,
            push=False,  # don't push to HF, save locally
            capture=capture,
            model_id=self.model_path,
            hub_id=None,
            wandb_project=os.environ.get("WANDB_PROJECT"),
            expansion=expansion,
        )

        # Commit volumes
        import subprocess
        subprocess.run(["sync"])

        return {"status": "complete", "layers": list(range(start, end)), "results": results}


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
):
    """
    Train SAEs on Gemma-4 E4B layers.

    Examples:
        modal run gemma4_sae.py --layer-range 0,15 --capture rolling
        modal run gemma4_sae.py --layer-range 15,42 --capture auto
    """
    tainer = SAETainer()
    result = tainer.train.remote(
        layer_range=layer_range,
        capture=capture,
        expansion=expansion,
        pool_batches=pool_batches,
        max_steps=max_steps,
    )
    print(f"\nTraining complete: {result['status']}")
    print(f"Layers trained: {result['layers']}")
    return result
