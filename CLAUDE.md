# Ultra Review Brief: Event-Aware SAE Trainer

## Review goal

Perform a whole-system, evidence-based architecture review of this PyTorch SAE
trainer. The project must make sparse-autoencoder training practical for
individual researchers and small teams: reliable convergence with no per-layer
retuning, correct stopping, fast throughput, and the minimum feasible VRAM.

Prioritize findings in this order:

1. Correctness or numerical stability bugs that can prevent or fake convergence.
2. Stop criteria that waste expensive training steps or terminate a run before
   it has reached its best valid SAE.
3. VRAM residency, allocation-lifetime, synchronization, and transfer issues.
4. Throughput bottlenecks that materially affect single-GPU users.
5. Complexity that undermines the self-tuning/model-agnostic promise.

Do not propose generic "use FSDP/DeepSpeed/activation checkpointing" advice
unless it is compatible with the explicit residency invariants below and has a
clear advantage for the supported single-GPU workflow. Prefer precise,
testable changes over large rewrites.

## Read these files first

| Area | Files |
| --- | --- |
| SAE architecture, loss, training loop, capture, checkpoints | `sae_trainer_rolling.py` |
| Control system and convergence state machine | `sae_scheduler.py` |
| Model presets and CLI assembly | `run_atlas.py`, `configs/` |
| Scheduler regression coverage | `tests/test_scheduler.py`, `tests/test_pin_tail_guard.py` |
| VRAM-offload invariant coverage | `tests/test_floating_window.py` |
| SAE, revival, sparse-decode, and diagnostics coverage | `tests/test_sae_model.py`, `tests/test_revival.py`, `tests/test_sparse_decode.py`, `tests/` |

Review the entire listed system, not only a local diff. Trace each reported
finding to its call sites and relevant tests.

## Non-negotiable VRAM residency contract

The `rolling-float` and `rolling-hf-float` capture modes deliberately keep the
full LLM on CPU. During activation-pool production:

- Only the active decoder block, plus an immediately adjacent block when a
  production path genuinely needs it, may be resident in GPU VRAM.
- All inactive decoder blocks must remain on CPU/unallocated from CUDA.
- The small shared components that every production pass needs may stay on the
  GPU: embeddings, rotary embeddings, and the Gemma per-layer embedding and
  projection components.
- `FloatingLayerWindow` is the ownership boundary for this policy. It must
  release active blocks on normal completion and exceptions.
- SAE training occurs separately after model eviction; do not retain LLM
  activations, caches, hooks, or parameters on CUDA across that boundary.

Do not recommend moving the full base model, a full decoder stack, or
unbounded activation/KV caches to GPU as a performance shortcut. Evaluate
whether non-blocking transfers, pinned host memory, stream use, double
buffering, tensor lifetime, `empty_cache`, and Python-side synchronization
improve the real peak-memory/throughput tradeoff without violating this
contract.

For `auto` capture, audit forward-hook registration/removal, hook output
ownership, exception cleanup, and repeated per-layer capture. Look specifically
for stale hooks, retained graphs, or GPU tensors escaping their intended
lifetime.

## Mathematical/control invariants

The SAE has JumpReLU features:

```text
pre = (x - b_dec) @ W_enc + b_enc
f = ReLU(pre - threshold)
x_hat = f @ W_dec + b_dec
```

The intended optimization problem is:

```text
minimize reconstruction_loss
subject to mean_L0 <= target_l0 (K)
```

The training-loop sparsity term is intentionally one-sided:

```text
slack = max(0, mean_L0 - K)
L_sparse = lambda * slack + (mu / 2) * slack^2
L_total = L_recon + L_sparse + L_aux
```

The scheduler's dual update is projected, signed ascent:

```text
lambda <- clip(lambda + alpha * gain * (mean_L0 - control_target),
               lambda_min, lambda_max)
```

Above target, the controller may deliberately use a temporary slingshot
control target below `K` to approach the constraint from the sparse side.
Below target, lambda may relax, but the primal hinge remains zero. Threshold
nudges, adaptive STE bandwidth, encoder-gradient dampening, warm starts, and
AECS learning-rate modes are auxiliary actuators. They must not fight the
dual controller near convergence or silently invalidate the constraint.

Audit these specifically:

- Units, reduction domains, gradient paths, dtype transitions, and
  microbatch/gradient-accumulation scaling for every loss term.
- Whether `lambda`, `mu`, thresholds, and the STE have stable bounded behavior
  for deep-layer initial L0 values several times larger than `K`, very small
  `K`, and L0 undershoot.
- Whether controller updates observe representative signals rather than noisy
  or stale samples, and whether the various actuator cadences interact safely.
- Whether live tuning, checkpoint restore, resampling, and rollback preserve
  all state needed for equivalent continued training.

## Convergence and stopping contract

The scheduler has two interacting state machines:

- AECS optimization modes: `BASELINE`, `RECOVERY`, `EXPLORE`, `STABILIZE`.
- Sparsity phases: `DESCENT`, `PIN`, and reserved `FINETUNE`.

`PIN` is not merely a label. On entry it freezes the dual and direct threshold
nudges so reconstruction quality can settle after L0 reaches the narrow PIN
band. PIN must release if L0 escapes materially or the ultra-active feature
tail collapses. Current early stopping requires a stable L0 window on the
sparse side of target, a lambda plateau, and either PIN EV readiness or PIN
timeout. The trainer also has dead-feature rollback, post-peak-quality guards,
and an aggressive-low-K stop path.

Assess whether this combination is the best practical stopping policy. Look
for:

- False convergence from lambda saturation, clamping, frozen dual state,
  low-variance but biased L0, or smoothed EV hiding deterioration.
- Runs that cannot stop due to incompatible cadences, unimplemented transitions,
  unreachable gates, or phase-state inconsistencies.
- Premature stops that select inferior weights, especially after a resample,
  rollback, or noisy EV sample.
- Selection criteria that should use held-out activation batches, confidence
  intervals/sequential tests, Pareto criteria (EV, L0, dead-feature health),
  or an explicit improvement budget rather than a brittle single threshold.
- Whether checkpoints distinguish latest, best-quality, best-feasible, and
  rollback states correctly.

Retain the product intent: end as soon as further training is unlikely to
produce a meaningfully better *feasible and healthy* SAE, not merely when a
single metric is flat.

## Performance and usability review criteria

Measure or reason from concrete code paths; do not assume a GPU configuration.
Focus on consumer/single-GPU feasibility:

- Peak allocated and reserved VRAM for each capture mode and during SAE
  training, including optimizer state, full/microbatch tensors, error buffers,
  checkpoint clones, activation pools, and logging probes.
- Avoidable CUDA synchronizations (`.item()`, host reads, per-step allocation,
  device moves, `empty_cache`) versus deliberately necessary ones.
- Whether H2D transfer and activation production can overlap safely with work.
- Disk pool size, format, lifetime, resume behavior, and I/O stalls.
- Exactness and fallback behavior of `SAE_FAST_PREBIAS`, sparse decode, Triton,
  and optional `torch.compile`; performance claims need a benchmark or
  profiler-backed validation plan.
- Defaults and CLI/config ergonomics for a user with limited VRAM and no
  ability to babysit a run.

## Required review output

Return only actionable findings. For each finding, include:

1. Severity: `critical`, `high`, `medium`, or `low`.
2. File and line range, plus the relevant execution path.
3. Why it affects convergence quality, stopping correctness, VRAM, throughput,
   or unattended usability.
4. A minimal safe remedy and any tradeoff.
5. The regression test, property test, benchmark, or profiler measurement that
   would prove the remedy.

Separate confirmed defects from performance hypotheses. Do not report
speculative micro-optimizations without a measurement plan. If no issue exists
in an area, state the invariant and evidence that made it safe.

## Validation expectations for proposed changes

Run targeted CPU tests first, especially:

```bash
pytest tests/test_scheduler.py tests/test_pin_tail_guard.py tests/test_floating_window.py
```

For changes to loss, SAE architecture, revival, or sparse decoding, include
the corresponding focused tests. Any CUDA optimization must also specify a
small reproducible GPU smoke/profile command that reports peak allocated VRAM,
peak reserved VRAM, tokens/second, and final L0/EV/dead-feature metrics.
