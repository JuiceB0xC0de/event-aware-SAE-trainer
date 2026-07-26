# Changelog

All notable changes to this project will be documented in this file.

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
