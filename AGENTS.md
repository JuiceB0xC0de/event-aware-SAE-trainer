# Event-Aware SAE Trainer — AI Agent Context

## Project Overview

This is a **self-tuning Sparse Autoencoder (SAE) trainer** for language model interpretability research. The key innovation is an **event-aware scheduler** that uses an Augmented-Lagrangian L0 integrator to automatically converge sparsity to a target value without per-layer hyperparameter tuning.

## Quick Start

```bash
# Install
pip install -e .

# Train on any model (auto capture)
python sae_trainer_rolling.py --model-id <model> --end-layer 16

# Faster generic capture for standard decoder stacks
python sae_trainer_rolling.py --model-id <model> --capture rolling-hf --start-layer 0 --end-layer 28

# Config-driven runner (one YAML per model, CLI overrides on top)
python run_atlas.py --list
python run_atlas.py --config ./my-model.yaml
```

## Key Files

| File | Purpose |
|------|---------|
| `sae_trainer_rolling.py` | Main trainer (~3100 lines) — training loop, SAE arch, capture backends, CLI |
| `sae_scheduler.py` | Event-aware scheduler (~1500 lines, AECS + Augmented-Lagrangian L0 control) |
| `run_atlas.py` | Config-driven front end: assembles trainer args from `configs/*.yaml`, handles upload |
| `configs/` | Per-model YAML configs consumed by `run_atlas.py` |
| `examples/use_trained_sae.py` | Load and inspect trained SAEs |

## Architecture

### SAE Model
- JumpReLU autoencoder with learned thresholds
- Tied encoder/decoder weights (orthonormal initialization)
- Aux loss for dead feature revival
- Resampling for chronically dead features
- Hard 1% dead-feature ceiling with rollback to the last clean window

### Scheduler (`sae_scheduler.py`)
- **AECS base**: 4-mode state machine (BASELINE, RECOVERY, EXPLORE, STABILIZE)
- **Augmented-Lagrangian L0**: Dual ascent on L0 constraint, lambda adapts automatically
- **PIN stop gate**: pins convergence when L0 holds in-band; releases if it escapes
- **EV floor protection**: Detects quality drops, triggers recovery
- **Dead feature emergency**: High dead % → STABILIZE + resample

### Capture Backends (`--capture`)
- **`auto`**: Forward-hook capture, works with any `AutoModelForCausalLM`
- **`rolling-hf`**: Generic single-block walk for standard decoder stacks; no layer cap
- **`rolling`**: architecture-scoped single-block walk (clamped to layers 0–14)
- **`rolling-float` / `rolling-hf-float`**: hoist/drop variants that keep only the active block sandwich in VRAM

## Resource Flags

| Feature | Flag | Impact |
|---------|------|--------|
| Gradient accumulation | `--microbatch-tokens 8192` | 4x VRAM reduction |
| Smaller activation pool | `--pool-batches 1000` | 4x disk reduction (400GB → 100GB) |
| Checkpoint resume | `--resume-from ckpt.pt` | Resume after interruption |
| LLM VRAM eviction | on by default (`--no-model-evict` to disable) | Base model moves to CPU while the SAE trains |
| bf16 activations | automatic | 2x IO/memory savings |

## Key Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `--expansion` | 32 | SAE dict = expansion × d_in |
| `--target-l0` | 500 (`K`) | Sparsity constraint; runtime override of `K` |
| `BATCH_TOKENS` | 32,768 | Tokens per step (`SAE_BATCH_TOKENS` env) |
| `--pool-batches` | 4000 | Activation cache size |
| `--microbatch-tokens` | 32,768 | Gradient accumulation |

## Testing

```bash
pip install -r requirements-dev.txt
pytest  # CPU-only, no GPU needed
```

CI runs the suite on every push and pull request (`.github/workflows/tests.yml`).

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
    └── pool_L00_s0/  # Activation pools (ephemeral, plus one persisted resume pool)
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

### Add a model config
1. Drop a YAML in `configs/` (see `configs/example.yaml`)
2. Verify with `python run_atlas.py --model <name> --dry-run`

## Design Principles

1. **Self-tuning**: No per-layer hyperparameter retuning
2. **Model-agnostic**: Default capture works on any HF model
3. **Single-file training**: `sae_trainer_rolling.py` is the single source of truth
4. **Checkpoint everything**: Full state saved per layer for resume
5. **No hardcoded secrets**: Upload targets must be explicit
