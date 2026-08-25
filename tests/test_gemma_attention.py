import sys
import types

import gemma_attention


def test_non_gemma_model_does_not_import_optional_kernel(monkeypatch):
    monkeypatch.setattr(gemma_attention, "_prepared", {})
    monkeypatch.delitem(sys.modules, "gemma_triton_flash_attn", raising=False)

    assert gemma_attention.prepare_gemma4_attention("meta-llama/Llama-3.2-1B") is False


def test_gemma_bootstrap_is_idempotent_and_selects_backend(monkeypatch):
    calls = []
    fake_kernel = types.SimpleNamespace(
        patch_transformers_5_5_4_flash_attn_key=lambda: calls.append("patch"),
        register_triton_attention=lambda: calls.append("register"),
    )
    monkeypatch.setitem(sys.modules, "gemma_triton_flash_attn", fake_kernel)
    monkeypatch.setattr(gemma_attention, "_prepared", {})

    assert gemma_attention.prepare_gemma4_attention("google/gemma-4-E2B-it") is True
    assert gemma_attention.prepare_gemma4_attention("google/gemma-4-E2B-it") is True
    assert calls == ["patch", "register"]

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(text_config=types.SimpleNamespace())
    )
    gemma_attention.select_gemma4_attention(model, enabled=True)
    assert model.config._attn_implementation == "triton_gqa"
    assert model.config.text_config._attn_implementation == "triton_gqa"


def test_missing_optional_kernel_leaves_gemma_unchanged(monkeypatch):
    monkeypatch.setattr(gemma_attention, "_prepared", {})
    def missing(_name):
        raise ModuleNotFoundError

    monkeypatch.setattr(gemma_attention.importlib, "import_module", missing)

    assert gemma_attention.prepare_gemma4_attention("google/gemma-4-E2B-it") is False
