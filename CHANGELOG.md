# Changelog

All notable changes to this project will be documented in this file.

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
