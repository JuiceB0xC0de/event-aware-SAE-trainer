"""Tests for the floating-window / sliding-block loader.

These tests run on CPU (no CUDA required) and verify that the
FloatingLayerWindow manager only touches the decoder blocks it is asked to,
keeping the rest of the model on CPU. This is the core residency invariant
behind `--capture rolling-hf-float`.
"""
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import sae_trainer_rolling as t


class _FakeRotaryEmb(nn.Module):
    """Minimal rotary_emb stand-in that returns position_embeddings."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, position_ids=None, **kwargs):
        # Return cos/sin tuple like real rotary_emb
        return (torch.zeros_like(x), torch.zeros_like(x))


class _FakeTextModel(nn.Module):
    """Minimal text_model exposing embed_tokens, rotary_emb, and a layer list."""

    def __init__(self, n_layers, d, vocab=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.rotary_emb = _FakeRotaryEmb()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])


class _FakeDecoderLayer(nn.Module):
    """Layer that accepts position_embeddings like LlamaDecoderLayer."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x, *args, attention_mask=None, position_ids=None,
                position_embeddings=None, use_cache=False, **kwargs):
        return (self.lin(x),)


class _FakeTextModelWithProperLayers(nn.Module):
    """Like _FakeTextModel but layers accept the HF signature."""

    def __init__(self, n_layers, d, vocab=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.rotary_emb = _FakeRotaryEmb()
        self.layers = nn.ModuleList([_FakeDecoderLayer(d) for _ in range(n_layers)])


def _all_on_cpu(module):
    """True if every parameter/buffer of module is on CPU."""
    return all(p.device.type == "cpu" for p in module.parameters()) and \
           all(b.device.type == "cpu" for b in module.buffers())


def _any_on_device(module, device):
    """True if any parameter/buffer of module is on the requested device."""
    dev_name = device.type
    return any(p.device.type == dev_name for p in module.parameters()) or \
           any(b.device.type == dev_name for b in module.buffers())


def test_floating_window_moves_only_active_blocks():
    d = 8
    n_layers = 4
    text_model = _FakeTextModel(n_layers, d)
    decoder_layers = text_model.layers

    # Use a fake CUDA device if available, otherwise just test CPU->CPU moves
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    win = t.FloatingLayerWindow(text_model, decoder_layers, device)

    # Initially everything on CPU
    assert _all_on_cpu(text_model), "full model should start on CPU"

    # Shared components are pinned to target device
    assert _any_on_device(text_model.embed_tokens, device)
    assert _any_on_device(text_model.rotary_emb, device)

    # Activate layer 1
    win.activate(1)
    assert _any_on_device(decoder_layers[1], device)
    assert _all_on_cpu(decoder_layers[0])
    assert _all_on_cpu(decoder_layers[2])
    assert _all_on_cpu(decoder_layers[3])

    # Activate layer 2 alongside layer 1
    win.activate(2)
    assert _any_on_device(decoder_layers[1], device)
    assert _any_on_device(decoder_layers[2], device)
    assert _all_on_cpu(decoder_layers[0])
    assert _all_on_cpu(decoder_layers[3])

    # set_active should drop layer 1 and add layer 3
    win.set_active({2, 3})
    assert _all_on_cpu(decoder_layers[0])
    assert _all_on_cpu(decoder_layers[1])
    assert _any_on_device(decoder_layers[2], device)
    assert _any_on_device(decoder_layers[3], device)

    # deactivate_all clears everything
    win.deactivate_all()
    assert _all_on_cpu(text_model)


def test_floating_window_context_manager_cleans_up():
    d = 8
    n_layers = 4
    text_model = _FakeTextModel(n_layers, d)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with t.FloatingLayerWindow(text_model, text_model.layers, device) as win:
        win.activate(0)
        win.activate(1)
        assert _any_on_device(text_model.layers[0], device)

    # After context exit everything is back on CPU
    assert _all_on_cpu(text_model)


def test_floating_window_ignore_out_of_range_layers():
    d = 8
    n_layers = 3
    text_model = _FakeTextModel(n_layers, d)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    win = t.FloatingLayerWindow(text_model, text_model.layers, device)

    # These should be no-ops, not raise
    win.activate(-1)
    win.activate(100)
    win.deactivate(-1)
    win.deactivate(100)
    win.set_active([-5, 1, 100])
    assert _any_on_device(text_model.layers[1], device)
    assert _all_on_cpu(text_model.layers[0])
    assert _all_on_cpu(text_model.layers[2])


class _FakeGemmaTextModel(nn.Module):
    """Gemma-3n/4-style text model: PLE table + projection alongside the basics."""

    def __init__(self, n_layers, d, ple=4, vocab=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.rotary_emb = _FakeRotaryEmb()
        self.embed_tokens_per_layer = nn.Embedding(vocab, n_layers * ple)
        self.per_layer_model_projection = nn.Linear(d, n_layers * ple, bias=False)
        self.per_layer_projection_norm = nn.LayerNorm(ple)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])


def test_floating_window_pins_gemma_ple_components():
    """rolling-float needs the PLE table/projection resident for _make_invariants."""
    text_model = _FakeGemmaTextModel(n_layers=3, d=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    win = t.FloatingLayerWindow(text_model, text_model.layers, device)

    assert _any_on_device(text_model.embed_tokens_per_layer, device)
    assert _any_on_device(text_model.per_layer_model_projection, device)
    assert _any_on_device(text_model.per_layer_projection_norm, device)
    # Decoder blocks stay hoisted out until activated
    assert all(_all_on_cpu(l) for l in text_model.layers) or device.type == "cpu"

    win.activate(1)
    assert _any_on_device(text_model.layers[1], device)
    win.deactivate_all()
    assert _all_on_cpu(text_model.layers[1])


def test_rolling_float_capture_family_resume_compat(tmp_path, monkeypatch):
    """A resume pool persisted under 'rolling' must resume under 'rolling-float'."""
    import json
    monkeypatch.setattr(t, "ROLLCACHE", str(tmp_path))
    mpath = t._resume_manifest_path(0)
    rdir = t._resume_pool_dir(0)
    rdir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        torch.save(torch.zeros(1, 2, 4, dtype=torch.bfloat16), rdir / f"shard_{i:05d}.pt")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w") as f:
        json.dump({"seed": 0, "pool_batches": 2, "model_id": "m",
                   "capture": "rolling", "last_completed_layer": 3}, f)

    assert t._find_resume_layer(0, 2, "m", "rolling") == 3
    assert t._find_resume_layer(0, 2, "m", "rolling-float") == 3
    assert t._find_resume_layer(0, 2, "m", "rolling-hf") == -1


def test_rolling_float_uses_rolling_pool_retention_cleanup():
    assert t._uses_rolling_pool_retention("rolling-float")


def test_rolling_hf_float_production_runs_with_fake_model(tmp_path):
    """End-to-end floating-window production path on a tiny fake model."""
    d = 8
    n_layers = 3
    text_model = _FakeTextModelWithProperLayers(n_layers, d)
    decoder_layers = text_model.layers
    device = torch.device("cpu")  # run entirely on CPU for test determinism

    # Write 2 token shards
    tok_dir = Path(tmp_path) / "tok"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        torch.save(torch.randint(0, 64, (2, 4)), tok_dir / f"shard_{i:05d}.pt")

    dst_dir = Path(tmp_path) / "poolL0"
    src_dir = None

    win = t.FloatingLayerWindow(text_model, decoder_layers, device)
    win.activate(0)

    t._produce_pool_hf_rolling(text_model, text_model, decoder_layers, 0,
                               tok_dir, src_dir, dst_dir, device)

    assert len(t._shard_paths(dst_dir)) == 2
    shard = t._read_shard(dst_dir, 0)
    assert shard.dtype == torch.bfloat16
    assert shard.shape == (2, 4, d)
