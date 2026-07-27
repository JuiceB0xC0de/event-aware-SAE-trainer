# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2026-07-27

> Not numerically identical to 0.5.0: the encoder backward changes accumulation
> order. The result is the same quantity by exact algebra, not the same bits.

### Added
- **Reassociated encoder pre-bias gradient** (`_EncoderPreBias`). A JumpReLU step
  is six GEMMs of `2*B*d*F`, all the same size. One of them exists only because
  `b_dec` needs a gradient: autograd forms the full `[B,F] x [F,d]` grad_input and
  then reduces it over the batch. The batch sum commutes with the right-multiply,

      d(loss)/d(b_dec) = -sum_b (dpre_b @ W_enc) = -(sum_b dpre_b) @ W_enc

  so the same value comes from an `[F]` reduction plus one GEMV. That removes ~1/6
  of the step's FLOPs. Exact, not an approximation.

  Falls back to plain autograd whenever `x` requires grad, so callers that need
  `grad_x` are unaffected. `SAE_FAST_PREBIAS=0` disables it outright.

  **Measured:** gradients match plain autograd to 1e-10 in float64 and the forward
  to 1e-12 (`tests/test_prebias_reassociation.py`). Encoder-only throughput on CPU
  improves 18% at `B=512,d=256,F=8192` and 24% at `B=1024,d=512,F=16384`, the
  trend rising with size as GEMMs come to dominate. **The full-step speedup on GPU
  is not yet measured** — the prediction is ~17%, i.e. `fwd_bwd_cuda` moving from
  123–125ms to roughly 103–108ms at CodVa's shape. Treat that as a hypothesis with
  a test attached, not a result.

## [0.5.0] - 2026-07-27

> Runs are **not numerically comparable to 0.4.0**: controller behaviour and the
> default early-stop thresholds both changed.

### Added
- **`rolling-generic` capture**: single-block rolling for custom architectures
  whose decoder blocks take `(x, cos, sin)` positionally, using
  `get_input_embeddings()` and a `_get_rope`-style helper. `auto` promotes to it
  automatically, but only after verifying the single-block walk reproduces the
  model's own forward on a probe batch — a wrong block signature would produce
  garbage activations silently rather than raise. `auto` previously ran a **full
  model forward per layer** on these models, i.e. O(n_layers²) block evaluations.
  Measured on CodVa-1-Small: 69.5k → 205k tok/s, max relative error 0.00e+00.
- **Capture timing breakdown**: per-layer read / forward / write split and a
  MoE-vs-dense label on the produce line.
- **`OBS` trace**: step 1 then every `SAE_OBS_EVERY` steps (default 100),
  reporting L0, EV, dead% and threshold, colour-coded. `LOG_EVERY` is
  deliberately untouched — it is the AL dual cadence and the dead/fire-rate
  window, not a print interval.
- **`--timing-every`** on both entry points; `[STEP-TIME]` now defaults to every
  25 steps instead of off.
- **`SAE_STOP_L0_REL` / `SAE_STOP_EV_FLOOR`**: the Aggressive-K stop thresholds
  are no longer literals.

### Changed
- **Threshold nudge is two-sided again.** The overshoot gate compared L0 and
  target against an absolute feature count tuned for `K=500`. At `K=50` both
  sides sat permanently under it, so the downward nudge never fired for an
  entire run while λ pushed down continuously — a one-sided pulse actuator
  against an integrator. The gate is now relative to target and behaves
  identically at either K.
- **Undershoot nudge scales with severity** instead of being capped at a flat
  gain, and escalates its frequency when far under target, mirroring the
  overshoot path.
- **Dual update has a recovery gain below target.** λ wound up at up to 24×
  through the slingshot and unwound at exactly 1×, so λ ratcheted upward across
  a run even with L0 centred on target (observed: 2.4e-3 → 6.9e-3 while L0 sat
  at 46–56).
- **Revived features get the live median threshold**, not `INIT_THRESHOLD`.
  0.1 is an absolute activation-scale constant: ~0.75σ on an early layer and
  ~0.03σ on a deep one, so revival injected a depth-dependent discrete L0 step
  that the controller had no visibility into.
- **Default early-stop EV floor 0.95 → 0.90.** EV declines with depth as the
  residual-stream norm grows, so a floor tuned on shallow layers blocked the
  stop on every deep one and burned the full step budget.
- **Config `env:` overrides the inherited shell**, matching the documented
  `CLI > config > defaults` precedence. Previously `setdefault` let a stale
  export beat an explicit config value.
- **Rolling chain resumes at `start_layer`** when its pool is already on disk
  instead of rebuilding from L0, and removes pools below the entry point that
  the retention loop can never reach.

### Fixed
- **`vocab_size` is read from the model config**, not `tokenizer.vocab_size`.
  The latter excludes added tokens, so every model with FIM / ChatML / domain
  tags failed the token-range check and refused to train.
- **HF credentials resolve from `hf auth login`** as well as `HF_TOKEN`; a
  CLI-only login no longer aborts a push-enabled run, and one call site could
  raise `KeyError` outright.

## [0.4.0] - 2026-07-26

### Added
- **Mid-chain bootstrap**: rolling runs can start at any layer without
  regenerating the pool chain from L0. Entry uses one hooked full-forward
  capture pass, and `activation_norm_ref` is restored from the previous
  layer's `meta.json` so the chain norm stays consistent. Warns and falls
  back to self-referencing if no reference is found — pass `--norm-ref`
  to pin it explicitly.
- **`pool_retention` / `--pool-retention`**: how many previous layers'
  pools stay on disk as rollback insulation.

### Changed
- Pool retention now defaults to **3** (was 1). Catching a bad layer late
  no longer means recomputing the whole chain — you can restart from L-3.
  Costs up to 2 additional pools of disk; lower it with
  `--pool-retention 1` if the volume is tight.

## [0.3.0] - 2026-07-26

### Added
- **PIN stop gate**: the scheduler pins convergence once L0 holds inside the target
  band and stops the layer early; the pin is released if L0 escapes the band.
- **Hard 1% dead-feature ceiling** with rollback to the last clean window.
- **Per-dimension EV logging** alongside the scalar EV.
- **`--norm-ref` / activation-norm reference**: reuse a prior run's activation
  normalization when retraining, persisted with the resume pool.
- **`rolling-float` / `rolling-hf-float` capture modes**: hoist/drop blocks so only
  the active sandwich sits in VRAM.
- **`run_atlas.py` config-driven runner**: one YAML per model under `configs/`,
  CLI overrides on top; replaces the per-model runner scripts.
- **CI**: pytest suite runs on every push and pull request.
- **Streaming fallback** when pre-tokenized shards are absent.

### Fixed
- AuxK dead-set cache was being reset every step, neutering dead-feature revival.
- Streaming-tokenizer fallback could use the wrong model's tokenizer.
- `HF_SAE_REPO` env var handling was case-sensitive.
- Global `*.json` gitignore rule was swallowing `configs/`.
- `validate_floating_window` walks the pool chain contiguously for sparse layer
  lists and creates the token dir before shard capture.

### Removed
- Per-model and cloud-runner scripts; `run_atlas.py` + `configs/` is the single
  entry point above the trainer.

## [0.2.1] - 2026-06-20

### Fixed
- **Pinned coherent dependency stack** for Gemma-4, Qwen3, and Qwen3.5 model families:
  `torch>=2.6.0,<2.8.0`, `transformers>=5.0.0`, `torchvision`, `torchaudio`,
  `accelerate>=1.6.0`, `datasets>=3.0.0`, `numpy<3.0.0`.
- **RunPod setup no longer inherits system-site packages**: `setup_runpod.sh` now
  creates an isolated venv and installs an ABI-matched CUDA 12.4 torch stack from the
  PyTorch index, preventing `undefined symbol` import crashes caused by the image's
  pre-installed torchaudio/torchvision wheels.

### Changed
- `setup_runpod.sh` accepts `MODEL_ID` env var and pre-caches that tokenizer instead
  of hardcoding `google/gemma-4-E2B-it`.

## [0.2.0] - 2026-06-13

### Added
- **Rolling resume pool persistence**: the last completed layer's activation pool is
  persisted on disk as a resume checkpoint. A restart resumes the residual chain from
  that layer instead of regenerating pools from layer 0. Exactly one resume pool is
  kept at a time.
- **LLM VRAM eviction during SAE training**: the base model is moved to CPU while the
  SAE trains (it is idle then), freeing GPU memory for larger microbatches / less
  gradient accumulation. Enabled by default; disable with `--no-model-evict`.
- **Double-buffered activation transfer**: H2D copies of the next activation batch
  now run on a dedicated CUDA stream and overlap with the current training step's
  forward/backward/optimizer work.
- **`--target-l0` CLI/runtime override**: override the global L0 target `K` (default
  500) without editing the source. Useful for aggressive sparsity tests (e.g. K=50).
- **`gemma4_sae_k50_test.py` Modal app**: clone the trainer from GitHub at image build
  time and run a small-scope K=50 test on Gemma-4 E2B.

### Changed
- **Auto capture is now truncated per layer**: instead of hooks + `_EarlyExit`, the
  decoder ModuleList is sliced to `[:layer+1]` for each layer's capture. Same
  correctness, less hook overhead, and the forward naturally stops after the target
  layer.

## [0.1.0] - 2026-06-06

### Added
- **Self-tuning SAE trainer** with event-aware scheduler (AECS mode machine + Augmented-Lagrangian L0 integrator)
- **Model-agnostic activation capture** via forward hooks (works with any `AutoModelForCausalLM`)
- **Gemma-optimized rolling capture** for layers 0–14 (single-block walk, VRAM/compute efficient)
- **Comprehensive test suite** covering SAE architecture, scheduler, capture, datasets, and CLI
- **Validation script** (`validate_rolling_cache.py`) proving bit-exactness of rolling capture
- **Example usage script** (`examples/use_trained_sae.py`) for loading and inspecting trained SAEs
- **pip-installable package** via `pyproject.toml`
- MIT license and contribution guidelines

### Key Features
- Set an L0 target and walk away — each layer converges independently
- No per-layer hyperparameter retuning required
- Dead features held near-zero via aux-loss revival + threshold reset + resampling
- One command sweeps all decoder layers in a single unattended run
