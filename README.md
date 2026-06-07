# event-aware-SAE-trainer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![pip installable](https://img.shields.io/badge/pip-installable-green)](https://pypi.org/)

Train a sparse autoencoder (SAE) on **every decoder layer of any Hugging Face language
model — in a single, unattended run.** No per-layer hyperparameter retuning, no
restart-and-retune loop. The scheduler watches the training signal and adjusts itself.

## The point: a self-tuning scheduler

DeepMind's Gemma Scope reported the usual SAE-training pain: sparsity/LR had to be retuned
per layer and runs repeated until they landed. This trainer removes that loop.

`sae_scheduler.py` is an **event-aware controller** — an AECS mode machine plus an
**Augmented-Lagrangian L0 integrator**. You set an L0 target and walk away:

1. λ starts at 0; the SAE overshoots the target (L0 too high).
2. The dual integrator raises λ in proportion to the constraint violation.
3. L0 falls, oscillates around the target, and **converges** as close as the model allows — λ plateaus when it gets there, relaxes if it overshoots into being too sparse.
4. Throughout, dead features are held **≈0** (aux-loss revival + threshold reset + resampling + a dead-feature emergency mode).

So one process sweeps all layers, each converging on its own. That self-tuning convergence
is the product — everything else is plumbing around it.

## Model-agnostic by construction

The scheduler and training loop never touch model internals — they consume a stream of
activation batches. **How activations are captured is pluggable (`--capture`):**

| Mode | Works on | Cost | Notes |
|------|----------|------|-------|
| `auto` *(default)* | **any `AutoModelForCausalLM`** (+ multimodal text path) | 1 pool of disk, N× forward | Forward-hooks the residual stream. **Correct by construction** — observes the real forward, no reconstruction. |
| `rolling` | Gemma-3n/4 family | 1 pool of disk, ~1 forward total | Single-block walk: run only block L over the residual cached from block L−1. VRAM/compute-optimized; uses Gemma-specific internals (PLE, attention types, KV sharing → layers 0–14 only). |

`d_in` and layer count are auto-detected from the model config; the SAE dictionary is
`expansion × d_in` (default 32×).

## Files

| File | Role |
|------|------|
| `sae_trainer_rolling.py` | The trainer — scheduler-driven training loop, SAE arch, datasets, both capture backends, CLI. |
| `sae_scheduler.py`       | The event-aware scheduler (AECS modes + Augmented-Lagrangian λ control). |
| `examples/configs.py`    | Example expansion / sparsity configs. |
| `examples/use_trained_sae.py` | Load a trained SAE and inspect feature activations. |

## Install

```bash
# Development install (editable)
pip install -e .

# Or from requirements
pip install -r requirements.txt
```

Requires a CUDA GPU (H100/A100 target). Configuration is via flags or environment:

```bash
export HF_TOKEN=hf_...            # model + corpus access (and upload, if used)
export SAE_DATA_DIR=./data        # where SAEs/pools are written (default ./data)
export SAE_SCRATCH_DIR=/mnt/nvme  # optional: fast disk for ephemeral activation pools
# SAE_MODEL_ID / SAE_HUB_ID / WANDB_PROJECT also settable via env
```

No HF org, wandb project, or upload target is baked in: a clone-and-run **trains locally and
pushes nowhere** unless you pass `--hub-id` (and `--wandb-project` for logging).

## Run

```bash
# any causal LM, model-agnostic capture, layers 0–16
python sae_trainer_rolling.py --model-id meta-llama/Llama-3.2-1B --end-layer 16

# Gemma fast path (single-block, layers 0–14)
python sae_trainer_rolling.py --model-id google/gemma-4-E2B-it --capture rolling --end-layer 15

# publish + log
python sae_trainer_rolling.py --model-id Qwen/Qwen2.5-1.5B --hub-id me/qwen-saes --wandb-project run1

# fast smoke test: 2 layers, few steps, live tokenization, no upload
python sae_trainer_rolling.py --start-layer 0 --end-layer 2 --max-steps 500 --no-pretok
```

### Hobbyist presets (limited VRAM / disk)

```bash
# 24GB VRAM, 100GB disk: 4x gradient accumulation, smaller pool
python sae_trainer_rolling.py --model-id meta-llama/Llama-3.2-1B \
    --microbatch-tokens 8192 --pool-batches 1000 --end-layer 8

# Resume from checkpoint after preemption/interruption
python sae_trainer_rolling.py --model-id google/gemma-4-E2B-it \
    --resume-from ./data/saes/google_gemma-4-e2b-it/layer_03_s0/checkpoint_full.pt

# Ultra-low disk (50GB): train with 500 batches, more epoching
python sae_trainer_rolling.py --pool-batches 500 --end-layer 4
```

`python sae_trainer_rolling.py --help` lists every flag.

## Key hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `--expansion` | `32` | SAE dict size = `expansion × d_in` |
| `K` (target L0) | `500` | the constraint the scheduler converges to |
| `BATCH_TOKENS` | `32_768` | tokens per SAE step (accumulated across microbatches) |
| `--microbatch-tokens` | `32_768` | tokens per microbatch; use 8192 for 4x VRAM savings |
| `SEQ_LEN` | `2_048` | sequence length into the model |
| `AUX_K` | `128` | dead features revived per token via the aux loss |
| `--pool-batches` | `4000` | activation batches cached per layer (~400GB); use 500-1000 for ~50-100GB |

`d_in` is **not** a hyperparameter — it's read from the model config.

## Checkpoint resume

During training and at the end of each layer, training saves `checkpoint_full.pt` with:
- Model weights (`sae_state`)
- Optimizer state (`optimizer_state`)
- Scheduler state (lambda, mode, event history)
- RNG states (CUDA + CPU)
- Dead-feature stats (`steps_since_fired`, `feature_fire_counts`)

The async activation reader is deliberately optimized for throughput; after resume it restarts
from the cached activation pool rather than serializing the in-flight prefetch queue.

To resume after interruption:
```bash
python sae_trainer_rolling.py --resume-from ./data/saes/google_gemma-4-e2b-it/layer_03_s0/checkpoint_full.pt
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

CPU-only, no GPU/network/token needed. Covers the SAE arch, pool I/O, dataset dispatch, the
scheduler's λ integrator (climb/clamp/floor), the CLI, and the model-agnostic hook capture —
including an exact match against a manual forward. Two further guards run when their deps are
present and skip otherwise: a CUDA smoke test (`-m gpu`), and a **bit-exactness check** that
hooked capture equals a real model's own `output_hidden_states` (needs network). That second
test is the one that keeps the atlas trustworthy — it proves capture returns the *actual*
residual stream, not a subtly wrong copy.

For the `rolling` backend specifically, `validate_rolling_cache.py` is a standalone gate:
it imports the shipped `_make_invariants`/`_run_block` and checks, per layer, that the
single-block reconstruction matches a true full forward to tolerance — and demonstrates the
expected divergence at/above `HARD_STOP_LAYER` that scopes rolling to layers 0–14.

```bash
python validate_rolling_cache.py --model-id google/gemma-4-E2B-it
```

## License

MIT — see [LICENSE](LICENSE).
