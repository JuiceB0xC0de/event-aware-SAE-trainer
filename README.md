# Event-Aware SAE Trainer

Train one JumpReLU sparse autoencoder per transformer layer without maintaining a
hand-tuned hyperparameter table for every depth.

[![tests](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer/actions/workflows/tests.yml/badge.svg)](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

The trainer combines a JumpReLU SAE, an augmented-Lagrangian sparsity
controller, an event-aware optimizer scheduler, feature revival, rollback, and
several activation-capture backends. The flagship validation run is a complete
35-layer atlas for `google/gemma-4-E2B-it`.

- **Published atlas:** [juiceb0xc0de/gemma-4-e2b-it-SAE](https://huggingface.co/datasets/juiceb0xc0de/gemma-4-e2b-it-SAE)
- **Known-good CUDA image:** `juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311`
- **Attention kernel:** [zzhhjjj/gemma-triton-flash-attn](https://github.com/zzhhjjj/gemma-triton-flash-attn)

## What has been demonstrated

One Gemma-4-E2B-it run trained 35 SAEs with a single model configuration:

| Setting | Value |
|---|---:|
| Decoder layers | 35 (`00..34`) |
| Residual width | 1,536 |
| Dictionary width | 49,152 (32x expansion) |
| Sparsity target | L0 = 50 |
| Corpus | `HuggingFaceFW/fineweb-edu` |
| Pool | 500 shards x 32,768 tokens |
| Training microbatch | 32,768 tokens |
| Hardware | NVIDIA H100 80GB HBM3 |
| Runtime | PyTorch 2.9.1+cu128, Transformers 5.5.4 |

After warmup, SAE optimization generally ran around 250k-273k tokens/s on that
H100. Activation-pool production varied much more with depth and shared-KV
dependencies; those numbers are workload observations, not a hardware promise.

The run was not magically intervention-free. Layer 34 exposed a real controller
boundary: the 1% dead-feature ceiling could abort during the initial collapse
before revival had time to act. Giving the revival path a 1,000-step grace period
allowed L34 to recover to EV 0.9768, L0 46.39, and 0.27% dead features. That fix is
now part of the trainer and covered by regression tests.

## SAE objective

For an activation vector `x`, the encoder and JumpReLU gate are:

```text
pre = W_enc (x - b_dec) + b_enc
theta = exp(log_threshold)
z = pre * 1[pre > theta]
x_hat = W_dec z + b_dec
```

This is **not** `ReLU(pre - theta)`. JumpReLU preserves the pre-activation
magnitude after the threshold is crossed.

Decoder feature directions are initialized orthogonally and renormalized after
optimizer steps. The reconstruction objective is combined with a hinge-form
augmented-Lagrangian penalty:

```text
slack = max(0, mean_L0 - K)
loss = reconstruction_loss + lambda * slack + (mu / 2) * slack^2
```

The projected dual update adapts `lambda`. In practice it is augmented by
activation-norm-scaled gains, threshold nudging, adaptive STE bandwidth, and a
PIN phase that freezes sparsity pressure near the target. `K` is a control target,
not a proof that every run will finish at an exact equality.

## Controller

The trainer has two interacting control layers:

1. **AECS optimizer modes** — `BASELINE`, `RECOVERY`, `EXPLORE`, and
   `STABILIZE` respond to loss, gradient, EV, and dead-feature events.
2. **Sparsity phases** — `DESCENT` drives L0 toward the target and `PIN` freezes
   the sparsity actuators while reconstruction catches up. FINETUNE readiness is
   observable but is not currently an active third phase.

Additional safeguards include:

- L0-aware lambda warmup and slingshot gain scaling;
- symmetric threshold nudging for overshoot and undershoot;
- adaptive straight-through-estimator bandwidth;
- AuxK revival, threshold reset, and dead-feature resampling;
- dead-feature rollback after the initial 1,000-step grace period;
- median-smoothed EV gates and EV-plateau early stopping;
- resumable checkpoints with SAE weights, optimizer state, core scheduler phase,
  provider position, feature-fire counters, and RNG state.

Checkpoint resume is practical, but it is not yet bit-for-bit controller replay:
some transient convergence windows and PIN tail-guard counters are not serialized.
Resume-sensitive experiments should record the interruption boundary and compare
the resumed trajectory against an uninterrupted control.

The quasi-orthogonality metrics in `sae_diagnostics.py` are descriptive
diagnostics. They measure feature/reconstruction norm geometry; they are not a
validated early-warning oracle and do not alter the loss.

## Activation capture

| Mode | Intended use | Behavior |
|---|---|---|
| `auto` | Broad compatibility | Captures the real model forward independently for each layer. Slow but the safest reference path. |
| `rolling` | Gemma-3n / Gemma-4 | Walks one block at a time through persistent residual pools. |
| `rolling-float` | Gemma-3n / Gemma-4 on limited VRAM | Keeps the model on CPU and hoists only the active block plus shared components. |
| `rolling-hf` | Standard Llama/Qwen-style decoder stacks | Uses Hugging Face rotary/position inputs for a one-block rolling walk. |
| `rolling-hf-float` | Larger standard decoder stacks | `rolling-hf` with active-block CPU/GPU hoisting. |

Generic rolling candidates are checked against a real full-model forward before
the trainer trusts them. Architecture-specific fast paths still require
representative numerical validation when the model implementation changes.

### Gemma shared KV

Gemma-4 changes regime at layer 15. Later attention layers consume KV produced by
earlier layers, so residual state alone is insufficient. The Gemma rolling path:

1. retains the required residual anchor pools;
2. materializes shared-KV sidecars once;
3. loads the matching KV shard for each downstream block;
4. keeps only the active target block resident on GPU in `rolling-float` mode.

Pool production can fuse independent shards along the batch dimension. The
Gemma configuration uses `pool_forward_fusion: 2`; output files remain in the
ordinary one-shard format.

## Recommended Gemma setup

The Docker image is the reproducible path for A100/H100 work. It contains the
CUDA toolchain, PyTorch/Transformers stack, Triton attention package, trainer
source, `tmux`, `nvitop`, Nsight Systems, `cuda-gdb`, and
`compute-sanitizer`. It contains no model weights or credentials.

```bash
docker run --rm -it --gpus all --ipc=host \
  -v gemma-workspace:/workspace \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311
```

`TRITON_CACHE_DIR` is `/workspace/.cache/triton` in the image. Because
`/workspace` is persistent, A100 (`sm_80`) and H100 (`sm_90`) compiled kernels
survive container restarts in separate cache entries.

The image contains a pinned trainer snapshot. Clone the current repository into
the persistent workspace when you want current `main`:

```bash
cd /workspace
git clone https://github.com/JuiceB0xC0de/event-aware-SAE-trainer.git
cd event-aware-SAE-trainer

export SAE_DATA_DIR=/workspace/data
export SAE_SCRATCH_DIR=/workspace/data/rollcache

python run_atlas.py --model gemma4 --dry-run
python run_atlas.py --model gemma4
```

Authenticate with `hf auth login` and `wandb login` instead of putting token
values in commands or configuration files. Pass `--no-push` for a local-only run.

The 500-shard Gemma run needs substantial persistent disk. Allow roughly 500 GB
for checkpoints, rolling pools, retained anchors, shared-KV sidecars, and safe
operational headroom; 750 GB is comfortable. Point `SAE_SCRATCH_DIR` at fast local
or network storage with predictable write capacity.

## Native installation

Python 3.11 and the published cu128 stack are the known-good Gemma environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install \
  'git+https://github.com/zzhhjjj/gemma-triton-flash-attn.git@c6a0cf860612792597382e7745d1514a52d6ca58'
```

The project accepts PyTorch 2.6 through 2.x for broader CPU and non-Gemma use,
but that range is not a claim that every Torch/Transformers/CUDA combination has
been validated with Gemma-4.

## CLI

List the configurations that actually exist:

```bash
python run_atlas.py --list
```

Run or override the Gemma configuration:

```bash
python run_atlas.py --model gemma4
python run_atlas.py --model gemma4 --layer-range 34,34 --no-push
python run_atlas.py --model gemma4 --pool-batches 50 --max-steps 1 --no-push
```

Use an arbitrary YAML file:

```bash
python run_atlas.py --config ./my_experiment.yaml
```

The direct trainer CLI remains available for model-family experiments:

```bash
python sae_trainer_rolling.py \
  --model-id meta-llama/Llama-3.2-1B \
  --capture rolling-hf \
  --start-layer 0 \
  --end-layer 15 \
  --target-l0 50 \
  --no-push
```

## Loading a trained SAE

Each atlas directory is named `layer_XX_s0` and contains `sae.pt`, `meta.json`,
and `checkpoint_full.pt`.

```python
import torch
from examples.use_trained_sae import load_sae, encode, decode, get_top_features

sae = load_sae("./data/saes/google_gemma-4-e2b-it/layer_00_s0")
activations = torch.randn(2, 16, sae["meta"]["d_in"])
features = encode(activations, sae)
reconstructed = decode(features, sae)
top_features = get_top_features(features, k=5)
```

The helper implements the same JumpReLU value rule as training. Producing the
input activations at the correct hook location remains the caller's
responsibility; an SAE is only meaningful on the activation surface it was
trained against.

Download one published layer without pulling the full atlas:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="juiceb0xc0de/gemma-4-e2b-it-SAE",
    repo_type="dataset",
    allow_patterns=["layer_22_s0/*"],
    local_dir="./gemma4-sae-layer22",
)
```

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `SAE_DATA_DIR` | Persistent SAE/checkpoint root | `./data` |
| `SAE_SCRATCH_DIR` | Activation pools and shared-KV sidecars | `$SAE_DATA_DIR/rollcache` |
| `SAE_MODEL_ID` | Direct-CLI model default | `google/gemma-4-E2B-it` |
| `SAE_HUB_ID` | Hugging Face upload target | unset |
| `SAE_HUB_REPO_TYPE` | Hugging Face target kind: `model` or `dataset` | `model`; Gemma config uses `dataset` |
| `WANDB_PROJECT` | W&B project | unset for direct CLI; config-derived in `run_atlas.py` |
| `SAE_BATCH_TOKENS` | Tokens in one training batch | `32768` |
| `SAE_MICROBATCH_TOKENS` | Tokens per gradient microbatch | batch size |
| `SAE_TIMING` | Step-timing print cadence | `25` direct; `250` in Gemma config |
| `SAE_CONTROL_FILE` | Live control/killswitch JSON | `/tmp/live_tune.json` |
| `SAE_COMPILE` | Enable `torch.compile` for the SAE | `0` |
| `SAE_USE_TRITON` | Experimental fused SAE path, not Gemma attention | `0` |
| `TRITON_CACHE_DIR` | Triton compilation cache | image: `/workspace/.cache/triton` |
| `WANDB_MODE=disabled` | Disable W&B explicitly | unset |

## Repository map

```text
sae_trainer_rolling.py       SAE, capture providers, pool pipeline, training loop
sae_scheduler.py             AECS modes and augmented-Lagrangian sparsity control
sae_diagnostics.py           Read-only quasi-orthogonality diagnostics
gemma_attention.py           Optional Gemma Triton registration/bootstrap
run_atlas.py                 YAML-driven atlas runner
configs/gemma4.yaml          Reproduced Gemma-4-E2B-it configuration
examples/use_trained_sae.py  Weight loader and JumpReLU inference helpers
bench/                       SAE-core benchmarks
tests/                       CPU contract/regression suite plus GPU-marked tests
```

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pytest -m gpu
```

The CPU suite checks architecture, controller transitions, capture contracts,
shared-KV routing, floating residency, pool IO, rollback, revival, inference,
and repository documentation/packaging contracts. GPU correctness and throughput
still require the target CUDA environment.

## Scope and caveats

- One full Gemma atlas is evidence for that model, corpus, seed, configuration,
  and hardware—not universal convergence.
- `auto` is the compatibility-first reference path; optimized rolling capture is
  architecture-specific.
- EV is depth-dependent. A lower deep-layer EV does not by itself identify a
  controller failure, but it does warrant downstream validation.
- Feature names and causal interpretations are downstream research tasks. The
  trainer produces sparse dictionaries, not semantic labels.
- Full checkpoints are large because they include optimizer and controller state.

## License and citation

MIT licensed. See [CITATION.cff](CITATION.cff), or cite:

```bibtex
@software{event_aware_sae_trainer2026,
  author = {Rick Holmberg},
  title = {Event-Aware SAE Trainer},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/JuiceB0xC0de/event-aware-SAE-trainer}
}
```
