"""Optional Gemma-4 Triton attention bootstrap.

The kernel package carries a Transformers 5.5.4 compatibility patch that must
run before importing Gemma-4's Transformers configuration or model modules.
Keeping this adapter separate lets both run_atlas.py and the direct trainer CLI
perform that setup without making the kernel package a hard dependency.
"""
from __future__ import annotations

import importlib

_prepared: dict[str, bool] = {}


def _is_gemma4(model_id: str) -> bool:
    return "gemma-4" in model_id.lower()


def prepare_gemma4_attention(model_id: str) -> bool:
    """Register the optional kernel before any Transformers model import."""
    if not _is_gemma4(model_id):
        return False
    if model_id in _prepared:
        return _prepared[model_id]

    try:
        kernel = importlib.import_module("gemma_triton_flash_attn")
    except ModuleNotFoundError:
        _prepared[model_id] = False
        print("  [attention] gemma_triton_flash_attn unavailable; using Transformers default")
        return False

    kernel.patch_transformers_5_5_4_flash_attn_key()
    kernel.register_triton_attention()
    _prepared[model_id] = True
    return True


def select_gemma4_attention(model, enabled: bool) -> None:
    """Select the registered backend on both multimodal and text configs."""
    if not enabled:
        return
    model.config._attn_implementation = "triton_gqa"
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        text_config._attn_implementation = "triton_gqa"
    print(
        "  [attention]",
        model.config._attn_implementation,
        getattr(text_config, "_attn_implementation", "(no text config)"),
    )
