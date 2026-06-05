"""
Example run configurations.

These mirror the CLI flags of sae_trainer_rolling.py. The trainer is model-agnostic:
d_in and layer count are auto-detected from the model, so configs only carry the knobs
you actually choose. Nothing pushes anywhere unless `hub_id` is set.
"""

# Default — model-agnostic hook capture, 32x dictionary, no upload.
DEFAULT = {
    "model_id": "google/gemma-4-E2B-it",   # any AutoModelForCausalLM works
    "capture": "auto",
    "start_layer": 0,
    "end_layer": 9,
    "expansion": 32,
    "pool_batches": 4000,
    "use_pretok": True,
    "hub_id": None,                        # set to "me/my-saes" to publish
    "wandb_project": None,                 # set + WANDB_API_KEY to log
}

# Gemma fast path — single-block capture (VRAM/compute-optimized, Gemma-3n/4 only,
# layers 0-14). Same SAEs, much cheaper to produce activations for.
GEMMA_ROLLING = {
    **DEFAULT,
    "capture": "rolling",
    "end_layer": 15,
}

# A different model entirely — proves there's nothing Gemma-specific in the core.
LLAMA_3_2_1B = {
    **DEFAULT,
    "model_id": "meta-llama/Llama-3.2-1B",
    "capture": "auto",
    "end_layer": 16,
}

# Smaller dictionary for quick iteration. NOTE: a different basis, not a free speedup.
SMALL_8X = {
    **DEFAULT,
    "expansion": 8,
}

# Smoke test — two layers, few steps, live tokenization, no upload.
SMOKE = {
    **DEFAULT,
    "start_layer": 0,
    "end_layer": 2,
    "max_steps": 500,
    "use_pretok": False,
}
