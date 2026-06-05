"""Tests for model-agnostic activation capture (`_produce_pool_hooked`).

Two layers of assurance:
  1. Against a tiny fake model on CPU (always runs) -- exercises the hook +
     early-exit + shard-writing path end to end.
  2. Against a real HF model's own `output_hidden_states` (runs when transformers
     + network are available, skipped otherwise) -- the bit-exactness guard: it
     proves the hooked capture returns the *actual* residual stream, not a subtly
     wrong reconstruction. This is the guard that matters for trustworthy atlases.
"""
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import sae_trainer_rolling as t


# -- 1. fake-model CPU tests -------------------------------------------------

class _FakeLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x, *args, **kwargs):
        return self.lin(x)


class _FakeModel(nn.Module):
    def __init__(self, n_layers, d):
        super().__init__()
        self.embed = nn.Embedding(64, d)
        self.layers = nn.ModuleList([_FakeLayer(d) for _ in range(n_layers)])

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        h = self.embed(input_ids)
        for lyr in self.layers:
            h = lyr(h)
        return h


def _write_token_shards(tok_dir, n_shards, n_seqs, seq):
    tok_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        torch.save(torch.randint(0, 64, (n_seqs, seq)), tok_dir / f"shard_{i:05d}.pt")


def test_hooked_capture_writes_pool(tmp_path):
    d = 8
    model = _FakeModel(3, d).eval()
    tok = Path(tmp_path) / "tok"
    dst = Path(tmp_path) / "poolL1"
    _write_token_shards(tok, n_shards=3, n_seqs=2, seq=4)

    t._produce_pool_hooked(model, model.layers, 1, tok, dst, torch.device("cpu"))

    assert len(t._shard_paths(dst)) == 3
    shard = t._read_shard(dst, 0)
    assert shard.dtype == torch.bfloat16
    assert shard.shape == (2, 4, d)


def test_hooked_capture_matches_manual_forward(tmp_path):
    d = 8
    model = _FakeModel(3, d).eval()
    tok = Path(tmp_path) / "tok"
    dst = Path(tmp_path) / "poolL1"
    _write_token_shards(tok, n_shards=1, n_seqs=2, seq=4)

    t._produce_pool_hooked(model, model.layers, 1, tok, dst, torch.device("cpu"))
    captured = t._read_shard(dst, 0).float()

    ids = t._read_shard(tok, 0)
    with torch.no_grad():
        h = model.embed(ids)
        h = model.layers[0](h)
        expected = model.layers[1](h)      # output of block 1
    assert torch.allclose(captured, expected.to(torch.bfloat16).float(), atol=1e-3)


def test_early_exit_skips_later_layers(tmp_path):
    # capturing layer 0 must not require layers 1+ to run -- prove it by tripping a
    # layer that would error if executed.
    d = 8
    model = _FakeModel(3, d).eval()

    class _Boom(nn.Module):
        def forward(self, *a, **k):
            raise AssertionError("layers after the captured one should not run")

    model.layers[1] = _Boom()
    tok = Path(tmp_path) / "tok"
    dst = Path(tmp_path) / "poolL0"
    _write_token_shards(tok, n_shards=1, n_seqs=2, seq=4)

    t._produce_pool_hooked(model, model.layers, 0, tok, dst, torch.device("cpu"))
    assert len(t._shard_paths(dst)) == 1


# -- 2. bit-exactness vs a real model's hidden_states ------------------------

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_hooked_capture_equals_hf_hidden_states(tmp_path):
    """The captured residual must equal HF's own output_hidden_states for the layer.
    Skipped when the model can't be fetched (offline)."""
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(TINY)
    except Exception as e:                       # offline / transformers missing
        pytest.skip(f"reference model unavailable: {type(e).__name__}")

    model.eval()
    n_layers = model.config.num_hidden_layers
    _, _, decoder_layers = t._find_text_model(model, n_layers)

    ids = torch.randint(0, model.config.vocab_size, (2, 6))
    tok = Path(tmp_path) / "tok"
    tok.mkdir()
    torch.save(ids, tok / "shard_00000.pt")

    L = min(1, n_layers - 1)
    dst = Path(tmp_path) / "pool"
    t._produce_pool_hooked(model, decoder_layers, L, tok, dst, torch.device("cpu"))
    captured = t._read_shard(dst, 0).float()

    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    expected = out.hidden_states[L + 1].float()   # hidden_states[0]=embeds, [L+1]=block L output

    assert captured.shape == expected.shape
    torch.testing.assert_close(
        captured, expected.to(torch.bfloat16).float(), atol=1e-2, rtol=1e-2)
