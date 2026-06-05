# event-aware-sae-trainer

Train sparse autoencoders (SAEs) on the residual stream of **`google/gemma-4-E2B-it`**,
one SAE per decoder layer, with an *event-aware* adaptive scheduler that watches the
training signal and reacts to it instead of following a fixed LR/sparsity curve.

JumpReLU architecture (Rajamanoharan et al., [2407.14435](https://arxiv.org/abs/2407.14435)),
a **rolling single-block activation cache** that trains with zero transformer in the SAE hot
loop, and a dual-loop **AECS scheduler** that tunes the learning rate and the L0 sparsity
penalty (λ) online via an adaptive-Lagrangian integrator. Built for a single H100/A100.

## What makes it different

- **Rolling single-block cache.** Instead of running a full `0 → L` forward every step for
  every layer, the model is loaded once and layers `0..14` are walked in order. For layer L,
  only block L runs over the residual stream cached from block L−1 (bit-exact, validated),
  then the SAE trains on those cached activations — **no transformer in the SAE step**. This
  is the dominant cost saving over a naïve per-layer trainer.
- **Event-aware scheduler.** `sae_scheduler.py` detects events in the training signal
  (dead-feature emergencies, EV stalls, gradient-norm anomalies) and adjusts LR and λ in
  response, rather than running open-loop.
- **Adaptive-Lagrangian sparsity.** λ is driven by an integrator toward a target L0 instead
  of being hand-set, so sparsity converges to spec without manual tuning per layer.
- **Peak-EV early stop.** The best-EV SAE state is held in memory and flushed to disk once EV
  declines past the peak, so each layer ships its best checkpoint without burning the rest of
  the step budget.

## Scope

Layers **0–14** (hard stop at 15). Residual width is constant (`d_in = 1536`) across this
range; layers 15+ use cross-layer KV sharing whose interaction with the rolling cache is out
of scope here.

## Files

| File | Role |
|------|------|
| `sae_trainer_rolling.py` | The trainer — SAE arch, datasets, rolling cache, training loop, CLI. Self-contained. |
| `sae_scheduler.py`       | AECS dual-loop scheduler (LR controller + adaptive-Lagrangian λ integrator). |
| `examples/configs.py`    | Example expansion / sparsity configs. |

## Install

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU (H100/A100 target). Set credentials in the environment:

```bash
export HF_TOKEN=hf_...          # required (model + corpus access, SAE upload)
export WANDB_API_KEY=...        # optional (metrics logging; training runs fine without it)
export SAE_DATA_DIR=./data      # optional (default ./data) — where SAEs/pools are written
export SAE_SCRATCH_DIR=/mnt/nvme/rollcache  # optional — fast disk for ephemeral activation pools
```

## Run

```bash
# hot zone (layers 0–8)
python sae_trainer_rolling.py

# full 0–14
python sae_trainer_rolling.py --end-layer 15

# a slice, live tokenization, no Hub upload (smoke test)
python sae_trainer_rolling.py --start-layer 0 --end-layer 2 --max-steps 500 --no-pretok --no-push
```

`python sae_trainer_rolling.py --help` lists all flags.

## Key hyperparameters (`sae_trainer_rolling.py`)

| Param | Default | Notes |
|-------|---------|-------|
| `N_FEATURES`  | `32 * 1536 = 49152` | 32× expansion over the 1536-d residual stream |
| `K`           | `500` | target L0 (natural L0 ≈ 550 at 32× — λ stays slightly positive) |
| `BATCH_TOKENS`| `32_768` | tokens per SAE step |
| `SEQ_LEN`     | `2_048` | sequence length into the model (attention is O(seq²)) |
| `AUX_K`       | `128` | top-k dead features revived per token via the aux loss |
| `POOL_BATCHES_DEFAULT` | `4000` | activation batches cached per layer |

## Performance notes

- **Ghost-feature aux loss is sparse and guarded.** Dead-feature revival operates only on the
  dead columns (`[B, n_dead]`) and is skipped entirely when nothing is dead — instead of
  allocating two full `[B, n_features]` tensors and running a dense matmul over a
  ~0.3%-nonzero tensor every step.
- The rolling cache trades disk for compute: activation pools are large and ephemeral, deleted
  incrementally so peak disk stays at ~one pool. Point `SAE_SCRATCH_DIR` at your fastest disk.

## License

MIT — see [LICENSE](LICENSE).
