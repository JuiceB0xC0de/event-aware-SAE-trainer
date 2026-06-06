# Event-Aware SAE Trainer — AI Agent Context

## Project Overview

This is a **self-tuning Sparse Autoencoder (SAE) trainer** for language model interpretability research. The key innovation is an **event-aware scheduler** that uses an Augmented-Lagrangian L0 integrator to automatically converge sparsity to a target value without per-layer hyperparameter tuning.

## Quick Start

```bash
# Install
pip install -e .

# Train on any model
python sae_trainer_rolling.py --model-id <model> --end-layer 16

# Train on Gemma-4 with rolling capture (layers 0-14)
python sae_trainer_rolling.py --model-id google/gemma-4-e4b --capture rolling --end-layer 15

# Train layers 15+ with auto capture
python sae_trainer_rolling.py --model-id google/gemma-4-e4b --capture auto --start-layer 15 --end-layer 42
```

## Key Files

| File | Purpose |
|------|---------|
| `sae_trainer_rolling.py` | Main trainer (~1200 lines) — training loop, SAE arch, capture backends, CLI |
| `sae_scheduler.py` | Event-aware scheduler (AECS + Augmented-Lagrangian L0 control) |
| `gemma4_sae.py` | Modal app for running on Kaggle models with GPU |
| `examples/use_trained_sae.py` | Load and inspect trained SAEs |
| `validate_rolling_cache.py` | Verify rolling capture bit-exactness |

## Architecture

### SAE Model
- JumpReLU autoencoder with learned thresholds
- Tied encoder/decoder weights (orthonormal initialization)
- Aux loss for dead feature revival
- Resampling for chronically dead features

### Scheduler (`sae_scheduler.py`)
- **AECS base**: 4-mode state machine (BASELINE, RECOVERY, EXPLORE, STABILIZE)
- **Augmented-Lagrangian L0**: Dual ascent on L0 constraint, lambda adapts automatically
- **EV floor protection**: Detects quality drops, triggers recovery
- **Dead feature emergency**: High dead % → STABILIZE + resample

### Capture Backends
- **`auto`**: Forward-hook capture, works with any `AutoModelForCausalLM`
- **`rolling`**: Gemma-4 optimized single-block walk (layers 0-14 only, ~1 forward total)

## Hobbyist Optimizations

| Feature | Flag | Impact |
|---------|------|--------|
| Gradient accumulation | `--microbatch-tokens 8192` | 4x VRAM reduction |
| Smaller activation pool | `--pool-batches 1000` | 4x disk reduction (400GB → 100GB) |
| Checkpoint resume | `--resume-from ckpt.pt` | Resume after interruption |
| bf16 activations | automatic | 2x IO/memory savings |

## Modal/Kaggle Setup

```bash
# Secret must contain KAGEL_KEY
modal secret list

# Run training
modal run gemma4_sae.py --layer-range 0,15 --capture rolling
modal run gemma4_sae.py --layer-range 15,42 --capture auto
```

## Key Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `--expansion` | 32 | SAE dict = expansion × d_in |
| `K` (target L0) | 500 | Sparsity constraint |
| `BATCH_TOKENS` | 32,768 | Tokens per step |
| `--pool-batches` | 4000 | Activation cache size |
| `--microbatch-tokens` | 32,768 | Gradient accumulation |

## Testing

```bash
pip install -r requirements-dev.txt
pytest  # CPU-only, no GPU needed
```

## Output Structure

```
$SAE_DATA_DIR/
├── saes/
│   └── <model_slug>/
│       ├── layer_00_s0/
│       │   ├── sae.pt           # Final SAE weights
│       │   ├── meta.json        # Training metadata
│       │   ├── checkpoint_best.pt  # Best EV checkpoint
│       │   └── checkpoint_full.pt  # Full state for resume
│       └── layer_01_s0/
└── rollcache/
    └── pool_L00_s0/  # Activation pools (ephemeral)
```

## Common Tasks

### Add a new capture backend
1. Add capture function in `sae_trainer_rolling.py`
2. Update `--capture` CLI choices
3. Add to `run_atlas_rolling()` dispatch

### Modify scheduler behavior
1. Edit `sae_scheduler.py` — `SAEEventControlScheduler`
2. Update `SAEAECSConfig` for new knobs
3. Test with `pytest tests/test_scheduler.py`

### Change SAE architecture
1. Edit `_make_sae()` in `sae_trainer_rolling.py`
2. Update `JumpReLUSAE` class
3. Verify with `pytest tests/test_sae_model.py`

## Design Principles

1. **Self-tuning**: No per-layer hyperparameter retuning
2. **Model-agnostic**: Default capture works on any HF model
3. **Single-file training**: `sae_trainer_rolling.py` is the single source of truth
4. **Checkpoint everything**: Full state saved per layer for resume
5. **No hardcoded secrets**: Upload targets must be explicit
