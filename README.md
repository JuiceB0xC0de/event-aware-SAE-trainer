# event-aware-SAE-trainer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Train a JumpReLU sparse autoencoder on every decoder layer of a causal LM in one unattended run. No per-layer retuning.

An event-aware augmented-Lagrangian scheduler auto-tunes the sparsity penalty (λ) during training and converges each layer to the L0 target you set. Dead features are held near zero by aux-loss revival, resampling, and a hard rollback ceiling. You set `--target-l0`, hit go, and come back to a full set of per-layer SAEs.

## How it works

- `sae_scheduler.py` — the controller: event state machine, augmented-Lagrangian λ integrator, dead-feature handling.
- `sae_trainer_rolling.py` — the trainer: SAE architecture, training loop, activation capture, checkpointing, CLI.
- `run_atlas.py` — optional config-driven front end: one YAML per model, CLI overrides on top.

The trainer consumes activation batches; capture is pluggable via `--capture`:

| `--capture` | Scope | Notes |
|-------------|-------|-------|
| `auto` | any `AutoModelForCausalLM` | Forward-hooks the residual stream. Correct by construction. |
| `rolling-hf` | most standard decoder stacks | Generic single-block walk. Much faster than hooks. |
| `rolling`, `*-float` | architecture-scoped variants | Fast paths for specific block layouts; see the module docstring. |

`d_in`, layer count, and vocab are auto-detected. Dictionary size is `expansion × d_in` (default 32×). Training text is streamed and tokenized on the fly, or pre-tokenized once and reused.

## Install

```bash
pip install -e .        # or pip install -r requirements.txt
```

## Usage

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

Or config-driven:

```bash
python run_atlas.py --list                 # available configs
python run_atlas.py --config ./my-model.yaml
```

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
| `--group-similarity` | off | Group-SAE: cosine floor for sharing one SAE across similar contiguous layers. See below. |

## Resume

Per-layer `checkpoint_full.pt` saves weights, optimizer, scheduler state, RNG, and dead-feature stats. Rolling capture also persists a resume pool, so a restart continues from the last completed layer instead of recapturing from layer 0.

```bash
python sae_trainer_rolling.py --resume-from ./data/saes/<model>/layer_12_s0/checkpoint_full.pt
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

CPU-only; covers the SAE architecture, scheduler integrator, dataset dispatch, CLI, and exact capture-vs-forward checks. A GPU smoke test runs with `-m gpu` when CUDA is available.

## Group-SAE — adapted from *Group-SAE: Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups*

Adjacent decoder layers carry near-identical residual-stream representations, so training a separate SAE for each is largely redundant. With `--group-similarity <cos>`, the atlas run groups **contiguous** layers whose mean-activation signature stays within that cosine floor of their group's first ("anchor") layer, trains one SAE for the anchor, and shares it verbatim with the rest of the group — cutting `train_sae_on_activations` calls roughly in proportion to the grouping.

```bash
# one SAE per group of layers that stay within cos>=0.95 of their anchor
python sae_trainer_rolling.py --model-id <hf-model-id> --end-layer 28 --group-similarity 0.95
```

Opt-in and off by default: without the flag, behavior is unchanged (one SAE per layer). Signatures are read from the activation pools the run already produces, so grouping adds no extra capture pass. Shared members are written locally with a `shared_from` tag in `meta.json`; per-group HF upload of members is out of scope. Adapted from [arXiv:2410.21508](https://arxiv.org/abs/2410.21508); the paper's offline fixed-K agglomerative clustering over a full similarity matrix is replaced here by a single-pass, threshold-driven contiguous grouping that fits the trainer's streaming residual-chain walk.

## License

MIT — see [LICENSE](LICENSE).
