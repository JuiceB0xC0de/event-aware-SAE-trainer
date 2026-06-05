"""
Example SAE training configs.

These mirror the module-level constants at the top of ``sae_trainer_rolling.py``.
To use one, edit those constants (they are module globals), or adapt the trainer
to read from a dict like the ones below.

All configs target google/gemma-4-E2B-it, layers 0-14, on a single H100/A100.
"""

# Default — matches sae_trainer_rolling.py as shipped. 32x expansion, the
# high-quality config used for the full layer atlas.
DEFAULT = {
    "d_in": 1536,
    "n_features": 32 * 1536,   # 49152
    "k": 500,                  # target L0
    "batch_tokens": 32_768,
    "seq_len": 2_048,
    "aux_k": 128,
    "pool_batches": 4000,
}

# Faster / cheaper sweep — smaller dictionary. NOTE: this is a DIFFERENT
# experiment, not a free speedup. Fewer features means a different (coarser)
# feature basis, not the same atlas at lower cost. Use for quick iteration.
SMALL_8X = {
    **DEFAULT,
    "n_features": 8 * 1536,    # 12288
    "k": 200,
}

# Smoke test — a couple of layers, few steps, live tokenization, no Hub upload.
# Mirrors: python sae_trainer_rolling.py --start-layer 0 --end-layer 2 \
#              --max-steps 500 --no-pretok --no-push
SMOKE = {
    **DEFAULT,
    "start_layer": 0,
    "end_layer": 2,
    "max_steps": 500,
    "use_pretok": False,
    "push": False,
}
