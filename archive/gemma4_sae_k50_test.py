"""
Modal test app for event-aware SAE trainer with aggressive L0 target (K=50).

Clones the trainer from GitHub at image build time so it always runs the latest
pushed code. Defaults to a small, fast test scope:
    layers 0-2, pool_batches=500, max_steps=3000, target_l0=50

Usage:
    modal run gemma4_sae_k50_test.py
    modal run gemma4_sae_k50_test.py --layer-range 0,5 --pool-batches 1000 --max-steps 5000
"""
import modal
from modal import Image, Volume

# =============================================================================
# Image definition -- clones the public repo so the container sees the latest
# pushed trainer code (sae_trainer_rolling.py + sae_scheduler.py).
# =============================================================================

REPO_URL = "https://github.com/JuiceB0xC0de/event-aware-SAE-trainer.git"
REPO_REF = "main"  # pin to branch/tag/commit if you need a specific version

# Bust the Modal image cache on every push: the commit SHA is hardcoded below
# and updated locally before each push, so the run_commands layer hash changes.
_BUILD_VERSION = "d6de8c7"

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
    )
    .run_commands(
        f"echo 'trainer-build-version={_BUILD_VERSION}' && "
        f"git clone --depth 1 --branch {REPO_REF} {REPO_URL} /opt/sae-trainer",
    )
    .env({
        "SAE_DATA_DIR": "/data",
        # Container-local NVMe for the activation pool. Network Volume throughput
        # starved the H100 in earlier runs; local disk is GB/s.
        "SAE_SCRATCH_DIR": "/root/rollcache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

# =============================================================================
# Volumes (persistent storage)
# =============================================================================

data_volume = Volume.from_name("sae-training-data", create_if_missing=True)

# =============================================================================
# Secrets
# =============================================================================

# HF_TOKEN from the "huggingface" Modal secret. WANDB_PROJECT env optional.

# =============================================================================
# App definition
# =============================================================================

app = modal.App("gemma4-sae-k50-test", image=image)


@app.cls(
    gpu="H100",
    volumes={"/data": data_volume},
    ephemeral_disk=1_048_576,  # 1 TiB container-local NVMe
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=86400,
)
class SAETainerK50:
    """Train SAEs on Gemma-4 E2B with an aggressive L0 target."""

    @modal.enter()
    def setup(self):
        import os
        from huggingface_hub import snapshot_download

        hf_token = os.environ.get("HF_TOKEN")
        volume_model_path = "/data/models/gemma-4-e2b"

        if os.path.exists(volume_model_path) and os.listdir(volume_model_path):
            print(f"Using cached Gemma-4 E2B from volume: {volume_model_path}")
        else:
            print("Downloading google/gemma-4-e2b-it from HuggingFace ...")
            os.makedirs(volume_model_path, exist_ok=True)
            snapshot_download(
                "google/gemma-4-e2b-it",
                local_dir=volume_model_path,
                token=hf_token,
            )
            print(f"Cached to volume at: {volume_model_path}")

        self.model_path = volume_model_path

    @modal.method()
    def train(
        self,
        layer_range: str = "0,3",
        capture: str = "rolling",
        expansion: int = 16,
        pool_batches: int = 500,
        max_steps: int = 10_000,
        microbatch_tokens: int = 32768,
        target_l0: int = 50,
        resume_from: str | None = None,
        evict_model: bool = True,
    ) -> dict:
        """
        Run a K=50 SAE test on Gemma-4 E2B.

        Args:
            layer_range: "start,end" e.g. "0,3" or "0,15"
            capture: "rolling" (layers 0-14) or "auto" (any layer)
            expansion: SAE expansion factor (default 16 for K=50; 32 is overkill at this sparsity)
            pool_batches: activation batches per layer (default 2000)
            max_steps: training steps per layer (default 10k, gives late layers room)
            microbatch_tokens: tokens per microbatch for gradient accumulation
            target_l0: L0 target K (default 50 for this aggressive test)
            resume_from: optional checkpoint_full.pt for the first trained layer
            evict_model: move LLM to CPU during SAE training (default True)
        """
        import os
        import sys
        import subprocess

        sys.path.insert(0, "/opt/sae-trainer")
        start, end = map(int, layer_range.split(","))

        from sae_trainer_rolling import run_atlas_rolling

        print(f"\n{'='*60}")
        print("K=50 SAE TEST  Gemma-4 E2B")
        print(f"  Model: {self.model_path}")
        print(f"  Layers: {start}-{end}")
        print(f"  Capture: {capture}")
        print(f"  Expansion: {expansion}x")
        print(f"  Pool batches: {pool_batches}")
        print(f"  Max steps/layer: {max_steps}")
        print(f"  Scratch disk needed: ~{pool_batches * 101 / 1024:.1f}GB per pool")
        print(f"  Microbatch tokens: {microbatch_tokens}")
        print(f"  Target L0 (K): {target_l0}")
        print(f"  Expansion: {expansion}x  (dict size = {expansion * 2560})")
        print(f"  Evict model during training: {evict_model}")
        if resume_from:
            print(f"  Resume: {resume_from}")
        print(f"{'='*60}\n")

        results = run_atlas_rolling(
            start_layer=start,
            end_layer=end,
            seed=0,
            pool_batches=pool_batches,
            microbatch_tokens=microbatch_tokens,
            use_pretok=True,
            max_steps=max_steps,
            bdec_batches=50,
            resume_from=resume_from,
            push=False,
            capture=capture,
            model_id=self.model_path,
            hub_id=None,
            wandb_project=os.environ.get("WANDB_PROJECT"),
            expansion=expansion,
            evict_model=evict_model,
            target_l0=target_l0,
        )

        subprocess.run(["sync"])

        return {
            "status": "complete",
            "layers": list(range(start, end)),
            "target_l0": target_l0,
            "results": results,
        }


@app.local_entrypoint()
def main(
    layer_range: str = "0,3",
    capture: str = "rolling",
    expansion: int = 16,
    pool_batches: int = 2000,
    max_steps: int = 10_000,
    microbatch_tokens: int = 32768,
    target_l0: int = 50,
    resume_from: str | None = None,
    evict_model: bool = True,
):
    """
    Run the K=50 test.

    Examples:
        modal run gemma4_sae_k50_test.py
        modal run gemma4_sae_k50_test.py --layer-range 0,5 --pool-batches 1000
        modal run gemma4_sae_k50_test.py --target-l0 100 --capture auto
    """
    tainer = SAETainerK50()
    result = tainer.train.remote(
        layer_range=layer_range,
        capture=capture,
        expansion=expansion,
        pool_batches=pool_batches,
        max_steps=max_steps,
        microbatch_tokens=microbatch_tokens,
        target_l0=target_l0,
        resume_from=resume_from,
        evict_model=evict_model,
    )
    print(f"\nTraining complete: {result['status']}")
    print(f"Layers trained: {result['layers']}")
    print(f"Target L0: {result['target_l0']}")
    return result
