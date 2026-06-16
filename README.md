# event-aware-SAE-trainer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Train a sparse autoencoder on **every decoder layer of any Hugging Face causal LM — unattended, without retuning per layer.**

The scheduler watches the run, chases the L0 target you give it, and converges. You set `target_l0`, hit go, and come back to a full atlas. No babysitting. No "this layer needs a different LR" loop.

## What is this

A one-command SAE atlas trainer with an **event-aware Augmented-Lagrangian scheduler**. It trains a JumpReLU SAE on each layer, auto-tunes the sparsity penalty (`λ`) in real time, and converges L0 to whatever target you pick. Dead features stay near zero via aux-loss revival + resampling + emergency dead-feature recovery.

It is model-agnostic for capture, fast-path for Llama/SmolLM2/Qwen-family models, and runs locally or on Modal.

## Why does this exist

SAE training usually means: pick a layer, guess LR/sparsity, run, check L0, tweak, rerun, repeat for every layer. Gemma Scope even called that out as their main pain. This removes the loop — the scheduler handles it. One process, all layers, each landing on its own L0 target.

The rough idea:
1. Start λ at 0, the SAE overshoots (L0 too high).
2. λ climbs based on how far off L0 is.
3. L0 oscillates around target, then λ plateaus when it locks in.
4. If it overshoots into too-sparse territory, λ relaxes.

You still get all the knobs if you want them, but you don't need them.

## Results

We trained a **full 29-layer atlas on `HuggingFaceTB/SmolLM2-135M-Instruct`** (layers 0–28 finished; L29 pending Modal credits) with `--capture rolling-hf`:

- **~410k tok/s sustained** SAE forward+backward on an A100-40GB.
- **Activation capture is no longer the bottleneck** — the rolling-hf fast path runs 350k–700k tok/s depending on depth.
- **0% dead features** across every completed layer.
- **EV ~0.93–0.95** on most mid-to-deep layers at `target_l0=50`.
- **No per-layer retuning.** Same command, 30 layers.

If you've trained SAEs before, you know those numbers are kind of ridiculous. The rolling window is the cheat code.

## How it works

Three pieces:

- `sae_scheduler.py` — the event-aware controller (AECS state machine + Augmented-Lagrangian λ integrator + dead-feature handling).
- `sae_trainer_rolling.py` — the full trainer: SAE arch, loop, datasets, checkpointing, capture backends, CLI.
- `gemma4_sae.py` — the Modal cloud runner if you want to train on A10/A100/H100 without managing machines.

The trainer doesn't touch model internals directly. It eats activation batches. How they get captured is pluggable:

| `--capture` | Works on | Speed | Notes |
|-------------|----------|-------|-------|
| `auto` | any `AutoModelForCausalLM` | medium | Forward-hooks the residual stream. Correct by construction, observes the real forward. |
| `rolling` | Gemma-3n/4 family | fast | Single-block walk with Gemma-specific internals. Scoped to layers 0–14 due to KV-share boundary. |
| `rolling-hf` | Llama / SmolLM2 / Qwen / etc. | fast | Generic single-block walk. No layer cap. This is the one that changed the game for us. |

`d_in`, layer count, and vocab are auto-detected. Dictionary size is `expansion × d_in` (default 32×).

## Quick start

### Local

```bash
pip install -e .      # or pip install -r requirements.txt
export HF_TOKEN=hf_...

# SmolLM2 0-28 (29 layers), 500 pool batches, 5000 steps, L0=50
python sae_trainer_rolling.py \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --capture rolling-hf \
  --layer-range 0,28 \
  --pool-batches 500 \
  --max-steps 5000 \
  --target-l0 50

# any causal LM, 8 layers, model-agnostic capture
python sae_trainer_rolling.py \
  --model-id meta-llama/Llama-3.2-1B \
  --end-layer 7

# smoke test: 2 layers, 500 steps, no upload
python sae_trainer_rolling.py \
  --start-layer 0 --end-layer 1 \
  --max-steps 500 --no-pretok
```

`--layer-range` is **inclusive** now: `0,28` means layers 0 through 28. (Not half-open. We fixed that foot-gun.)

### Modal

```bash
modal run gemma4_sae.py::SAETainer.train \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --layer-range 0,28 \
  --capture rolling-hf \
  --pool-batches 500 \
  --max-steps 5000 \
  --target-l0 50 \
  --microbatch-tokens 32768 \
  --timing
```

Default scratch is `/data/scratch` (persistent Modal volume), so rolling resume pools survive container death. If you want faster ephemeral NVMe and don't need resume, pass `--scratch-dir /root/rollcache`.

### Pretokenize once, train forever

FineWeb-Edu is streamed and tokenized on the fly by default. For repeated runs, pretokenize first:

```bash
modal run gemma4_sae.py::pretokenize \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --n-shards 16
```

Then training auto-detects the shards and skips live tokenization.

## Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--layer-range` | `0,15` | Inclusive `start,end`. Use `29,29` for a single layer. |
| `--capture` | `auto` | `auto`, `rolling`, or `rolling-hf`. |
| `--expansion` | `32` | SAE dictionary = `expansion × d_in`. |
| `--target-l0` | model default | Runtime override of the L0 target. We used `50` for the SmolLM2 atlas. |
| `--pool-batches` | `4000` | Cached activation batches per layer (~400GB). Use `500–1000` for limited disk. |
| `--microbatch-tokens` | `32768` | Per-microbatch tokens; `8192` for 4× VRAM savings. |
| `--max-steps` | `15000` | Cap per layer. Convergence usually happens earlier. |

## Hobbyist presets

24GB GPU, 100GB disk:

```bash
python sae_trainer_rolling.py \
  --model-id meta-llama/Llama-3.2-1B \
  --capture rolling-hf \
  --layer-range 0,7 \
  --microbatch-tokens 8192 \
  --pool-batches 1000
```

## Resume

Per-layer `checkpoint_full.pt` saves weights, optimizer, scheduler state, RNG, and dead-feature stats. For rolling capture, the trainer also persists one resume pool — restart the same command and it resumes the residual chain from the last completed layer instead of regenerating from 0.

```bash
python sae_trainer_rolling.py \
  --resume-from ./data/saes/huggingfacetb_smollm2-135m-instruct/layer_12_s0/checkpoint_full.pt
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

CPU-only, covers SAE arch, scheduler integrator, dataset dispatch, CLI, and exact capture-vs-forward checks. A GPU smoke test runs with `-m gpu` if CUDA is around.

## Known issues / rough edges

- `rolling` is scoped to Gemma layers 0–14 by the KV-share boundary. Use `rolling-hf` for deeper Llama-family layers.
- L29 of SmolLM2 is the final pre-head residual — training it works, but we ran out of Modal credits mid-capture. The 0–28 atlas is complete and solid.
- No baked-in wandb or HF upload target. Add `--wandb-project` / `--hub-id` if you want them.

## What's next

- Finish and publish the SmolLM2-135M-Instruct SAE atlas.
- Make `rolling-hf` the default for supported architectures.
- Persisted pool workflow so single-layer resumes don't walk from layer 0.

## License

MIT — see [LICENSE](LICENSE).

Built by [Rick / juiceb0xc0de](https://huggingface.co/juiceb0xc0de). If this helps you train SAEs without losing your mind, that's the whole point.
