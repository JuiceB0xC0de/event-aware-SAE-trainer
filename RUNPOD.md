# Running the SAE Trainer on RunPod

Battle-tested workflow from the 2026-07-10 migration (Modal → RunPod). Everything
in here was hit for real on a fresh pod; follow it in order and you skip the
potholes we already paid for.

## Why RunPod

- $/hr is a fraction of Modal's for equivalent cards (RTX 6000 Ada 48GB ≈ $0.77/hr
  secure; RTX A5000 24GB ≈ $0.16/hr community).
- ~4× the GPU variety. Match the card to the job — a validation run does not need
  an H100.
- `/workspace` network volume persists across pod stop/start: repo, caches, and
  outputs survive. Only GPU time is billed while running.

## Pod setup

**Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
**Ports:** `22/tcp` (direct SSH) — or use the `ssh.runpod.io` proxy (see below).
**Disk:** 60GB container disk minimum; pools for a full 500-batch run at
d_in=1024 peak ~74GB, so put `SAE_SCRATCH_DIR` on `/workspace` for big runs.

### Do NOT build a venv

`setup_runpod.sh`'s venv creation fails on this image (`ensurepip` error), and it
isn't needed: the image ships Python 3.11 + torch 2.7.0+cu128 + transformers in
the **system python**. Just top up:

```bash
pip install accelerate datasets sentencepiece protobuf huggingface_hub hf_transfer wandb orjson
```

This saves ~10 minutes and ~5GB per pod versus rebuilding torch.

### Clone and verify

```bash
mkdir -p /workspace/code && cd /workspace/code
git clone https://github.com/JuiceB0xC0de/event-aware-SAE-trainer.git
cd event-aware-SAE-trainer && git rev-parse --short HEAD
```

**Check the SHA matches your pushed commit.** The classic false-debugging loop is
a pod running stale code because a local fix was never pushed.

## SSH access (the proxy gotchas)

The `ssh.runpod.io` proxy (`ssh <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519`)
is **PTY-only**:

- `ssh host 'command'` fails with *"Your SSH client doesn't support PTY"*.
- Scripted/agent use: pipe commands via stdin to `ssh -tt`:
  ```bash
  printf 'nvidia-smi\nexit\n' | ssh -tt <pod>@ssh.runpod.io -i ~/.ssh/id_ed25519
  ```
- **Always launch long jobs under `nohup ... &`** — a dropped proxy connection
  kills foreground processes.

For direct SSH instead (supports normal exec), create the pod with port `22/tcp`
exposed and `PUBLIC_KEY` env set to your pubkey, then connect to the mapped
public port.

## Tokens (never in files, never in the repo)

- **HF:** run `hf auth login` once on the pod. That stores the token at
  `~/.cache/huggingface/token` — but the trainer scripts require the **env var**:
  ```bash
  export HF_TOKEN=$(cat ~/.cache/huggingface/token)
  ```
- **W&B:** run `wandb login` once on the pod (stores to `~/.netrc`), then:
  ```bash
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc)
  ```
  The trainer logs to W&B only when `WANDB_API_KEY` is set; it runs fine without.

Note: `HF_HUB_ENABLE_HF_TRANSFER` is deprecated (hub uses Xet now); harmless
warning if set.

## The workflow: validate → smoke → full run

### 1. Floating-window validation (~15 min, any card)

Proves `rolling-hf-float` (the "caterpillar": full model on CPU, ≤2 decoder
blocks in VRAM during capture, zero during training) is bit-exact against
`rolling-hf`:

```bash
python3 validate_floating_window.py --model-id Qwen/Qwen3-0.6B \
    --layers 0,1,5,10,15,20,27 --pool-batches 50
```

Pools chain (layer L reads L−1's pool), so the validator walks the chain
contiguously and compares only at the requested layers. Expected result:
`All layers bit-exact: True` with `max_diff=0.000e+00`, timing parity ~1.0×
(the float path trades nothing on capture speed; its win is VRAM).

**2026-07-10 result (Qwen3-0.6B, RTX 6000 Ada):** 7/7 layers bit-exact,
speedups 0.96–1.05×.

### 2. Smoke run (~15 min)

End-to-end trainer — scheduler, SAE loop, checkpointing, pool pipeline:

```bash
nohup python3 run_smollm2_runpod.py --layer-range 0,2 --pool-batches 50 \
    --max-steps 500 --capture rolling-hf-float --no-pretok --no-push \
    > /workspace/smoke.log 2>&1 &
tail -f /workspace/smoke.log
```

**2026-07-10 result:** ~295k tok/s sustained (vs ~130k A10 ceiling), provider
0.1ms, fwd_bwd ~110ms — SAE math is the bottleneck, as designed.

### 3. Full atlas run (hours; this is the keeper)

```bash
nohup python3 run_qwen3_0_6b_runpod.py --layer-range 0,27 \
    --capture rolling-hf-float --pool-batches 500 --max-steps 5000 \
    --target-l0 50 --microbatch-tokens 32768 \
    > /workspace/qwen_atlas.log 2>&1 &
```

Each completed layer uploads to the HF repo immediately (crash-safe) and logs a
W&B run per layer. Rule of thumb: **50 pool batches for validation/smoke, 500
for anything whose weights you intend to keep.**

## Known trainer facts that matter on RunPod

- `SAE_MODEL_ID` should always match the target model; runner scripts set it.
  (The tokenizer-fallback bug where a mismatched default poisoned token shards
  was fixed in `dbae9c4` — the trainer now fails loudly instead.)
- `validate_floating_window.py` uses its own `--scratch`
  (default `/tmp/validate_floating_window`), not `SAE_SCRATCH_DIR`.
- Triton kernel stays off (`SAE_USE_TRITON=0`); torch.compile gives ~nothing on
  Ampere/Ada for this SAE shape. Eager PyTorch is the production path.
- Watch dead-feature % and EV per layer in W&B; the scheduler self-drives
  (SmolLM2 baseline: 30/30 layers, zero dead features, zero intervention).

## Cost reference (2026-07)

| Job | Card | Time | Cost |
|---|---|---|---|
| Validation (7 layers) | RTX 6000 Ada | ~15 min | ~$0.20 |
| Smoke (3 layers × 500 steps) | RTX 6000 Ada | ~15 min | ~$0.20 |
| Full 28-layer Qwen3-0.6B atlas | RTX 6000 Ada | ~6–8 h | ~$5–6 |

Stop the pod when done — the volume keeps everything for the next session.
