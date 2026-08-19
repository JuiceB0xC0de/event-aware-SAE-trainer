# Contributing

Thanks for your interest in improving event-aware-sae-trainer.

## Ground rules

- **Target hardware is H100/A100.** Optimizations should hold up on data-center GPUs; don't
  add consumer-GPU hedges that complicate the hot path.
- **The model is `google/gemma-4-E2B-it`.** Not gemma-3.
- **Profile before optimizing.** Claims about memory or throughput should be backed by a
  `torch.profiler` / `torch.cuda.max_memory_allocated()` measurement, not estimates. PRs that
  change the training hot path should include before/after numbers.
- **Don't change training dynamics silently.** Anything that alters loss, λ behavior, or the
  early-stop criterion needs a note on why and, ideally, an EV/L0 comparison on at least one
  layer.

## Workflow

1. Fork and branch from `main`.
2. Keep changes scoped — separate analytics/logging changes from training-loop changes.
3. `python -m py_compile sae_trainer_rolling.py sae_scheduler.py` before pushing.
4. Open a PR describing what changed, why, and how you verified it.

## Code style

- Match the surrounding style; inline comments explain *why*, not *what*.
- No new heavyweight dependencies without discussion.
- `sae_trainer_rolling.py` is the single trainer and the source of truth for the SAE arch,
  datasets, and training loop. `sae_scheduler.py` is the only module it imports from this repo.
