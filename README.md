# event-aware-SAE-trainer

**Train a sparse autoencoder on every layer of any HuggingFace causal LM in one unattended run. No per-layer retuning, no babysitting λ.**

[![tests](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer/actions/workflows/tests.yml/badge.svg)](https://github.com/JuiceB0xC0de/event-aware-SAE-trainer/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An event-aware augmented-Lagrangian scheduler auto-tunes the sparsity penalty (λ) **during** training and converges each layer to the L0 target you set. If a layer drifts off target, the controller recovers mid-run — you don't come back to a failed layer and a restart.

## Why this exists

The usual SAE workflow is: guess λ → train → check L0 → restart with a new λ → repeat for every layer. On a 28-layer model that's days of manual tuning.

This trainer replaces that loop with a controller:

- **Augmented-Lagrangian λ integrator** — treats your target L0 as a constraint and integrates the penalty multiplier until the constraint holds, independently per layer.
- **Event state machine** — watches training signals and switches modes (baseline, recovery, …) when a layer stalls or overshoots, instead of burning steps on a lost run.
- **Dead-feature suppression** — an auxiliary-loss term holds dead features near zero, so the dictionary stays usable at high sparsity.
- **Early stop on convergence** — layers stop when they've converged rather than when the step cap hits, so a full-model sweep is genuinely unattended.

## Install

```bash
pip install -e .        # or pip install -r requirements.txt
```

## Quickstart

```bash
# smoke test: 2 layers, 500 steps, no upload
python sae_trainer_rolling.py \
  --model-id <hf-model-id> \
  --start-layer 0 --end-layer 1 \
  --max-steps 500 --no-pretok --no-push

# full run: every layer, one command
python sae_trainer_rolling.py \
  --model-id <hf-model-id> \
  --capture rolling-hf \
  --start-layer 0 --end-layer 28 \
  --target-l0 50
```

Config-driven alternative (one YAML per model, CLI overrides on top):

```bash
python run_atlas.py --list                 # available configs
python run_atlas.py --config ./my-model.yaml
```

`d_in`, layer count, and vocab are auto-detected. Dictionary size is `expansion × d_in` (default 32×). Add `--hub-id <your-hf-id>` to upload each layer's SAE to the Hugging Face Hub as it finishes, and `--wandb-project` for live metrics.

## Use a trained SAE

```bash
python examples/use_trained_sae.py \
  --sae-dir ./data/saes/<model>/layer_0 \
  --text "The quick brown fox jumps over the lazy dog."
```

Prints reconstruction MSE, explained variance, sparsity, and the top-firing features for a prompt. See [examples/use_trained_sae.py](examples/use_trained_sae.py) for the programmatic API.

## Runs on your hardware

Defaults target data-center GPUs, but presets scale down to consumer cards (details and measurements in [EFFICIENCY.md](EFFICIENCY.md)):

| Your setup | Flags | Footprint |
|------------|-------|-----------|
| H100 / A100 pod | defaults | ~400GB disk/layer, fresh activations every step |
| RTX 4090 (24GB) | `--microbatch-tokens 8192 --pool-batches 1000` | ~6GB VRAM, ~100GB disk/layer |
| RTX 3080 (10GB) | `--microbatch-tokens 4096 --pool-batches 500 --expansion 16` | ~3GB VRAM, ~50GB disk/layer |

Multi-day runs are resumable: full-state checkpoints (weights, optimizer, scheduler, RNG, dead-feature stats) plus a rolling resume pool that survives crashes without regenerating from layer 0.

## How it works

- `sae_scheduler.py` — the controller: event state machine, augmented-Lagrangian λ integrator, dead-feature handling.
- `sae_trainer_rolling.py` — the trainer: JumpReLU SAE architecture, training loop, activation capture, checkpointing, CLI.
- `run_atlas.py` — optional config-driven front end.

Activation capture is pluggable via `--capture`:

| `--capture` | Scope | Notes |
|-------------|-------|-------|
| `auto` | any `AutoModelForCausalLM` | Forward-hooks the residual stream. Correct by construction. |
| `rolling-hf` | most standard decoder stacks | Generic single-block walk. Much faster than hooks. |
| `rolling`, `*-float` | architecture-scoped variants | Fast paths for specific block layouts; see the module docstring. |

Training text is streamed and tokenized on the fly, or pre-tokenized once and reused.

## Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--start-layer` / `--end-layer` | `0` / `9` | Inclusive; clamped to model depth. |
| `--capture` | `auto` | See table above. |
| `--target-l0` | model default | The L0 the scheduler converges each layer to. |
| `--expansion` | `32` | SAE dictionary = `expansion × d_in`. |
| `--pool-batches` | `4000` | Cached activation batches per layer (~400GB). Use `500–1000` for limited disk. |
| `--microbatch-tokens` | `32768` | Tokens per microbatch; `8192` for ~4× VRAM savings. |
| `--max-steps` | `15000` | Per-layer cap; early stop usually fires first. |
| `--hub-id` | off | Upload target for trained SAEs. |
| `--wandb-project` | off | Metrics logging. |

## Resume

Per-layer `checkpoint_full.pt` saves weights, optimizer, scheduler state, RNG, and dead-feature stats. Rolling capture also persists a resume pool, so a restart continues from the last completed layer instead of regenerating pools from layer 0.

```bash
python sae_trainer_rolling.py --resume-from ./data/saes/<model>/layer_12_s0/checkpoint_full.pt
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

CPU-only; covers the SAE architecture, scheduler integrator, dataset dispatch, CLI, and exact capture-vs-forward checks. A GPU smoke test runs with `-m gpu` when CUDA is available. CI runs the suite on every push and PR.

## More docs

- [EFFICIENCY.md](EFFICIENCY.md) — hardware presets, memory math, and the resume/eviction internals.
- [CHANGELOG.md](CHANGELOG.md) — what changed and why.
- [CONTRIBUTING.md](CONTRIBUTING.md) — ground rules for PRs (profile before optimizing; don't change training dynamics silently).

## Citation

If you use this in research, see [CITATION.cff](CITATION.cff) — GitHub's "Cite this repository" button reads it automatically.

## License

MIT — see [LICENSE](LICENSE).
