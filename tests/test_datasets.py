"""Tests for dataset construction/dispatch.

The live streaming dataset needs HF network + transformers, so we only test that
the dispatcher returns the right *type* without iterating it. The pre-tokenized
dataset is fully exercised against a tiny on-disk fixture (numpy only)."""
import json

import numpy as np
import torch

import sae_trainer_rolling as t


def test_build_dataset_dispatch_streaming():
    ds = t._build_token_dataset(hf_token="x", batch_tokens=16, use_pretok=False)
    assert isinstance(ds, t.StreamingBatchDataset)


def test_build_dataset_dispatch_pretok(tmp_path):
    ds = t._build_token_dataset(
        hf_token="x", batch_tokens=16, use_pretok=True, pretok_dir=str(tmp_path))
    assert isinstance(ds, t.PreTokenizedDataset)


def _make_pretok_fixture(dir_path, batch_tokens, shard_len):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "manifest.json").write_text(json.dumps({"n_shards": 1}))
    arr = np.arange(shard_len, dtype=np.int64)
    np.save(dir_path / "shard_00.npy", arr)


def test_pretokenized_dataset_yields_batches(tmp_path):
    batch_tokens = 16
    _make_pretok_fixture(tmp_path, batch_tokens, shard_len=batch_tokens * 4)
    ds = t.PreTokenizedDataset(pretok_dir=str(tmp_path), batch_tokens=batch_tokens, seed=0)
    it = iter(ds)
    for _ in range(5):
        batch = next(it)
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.long
        assert batch.shape == (batch_tokens,)


def test_pretokenized_dataset_values_in_range(tmp_path):
    batch_tokens = 8
    shard_len = batch_tokens * 3
    _make_pretok_fixture(tmp_path, batch_tokens, shard_len)
    ds = t.PreTokenizedDataset(pretok_dir=str(tmp_path), batch_tokens=batch_tokens, seed=1)
    batch = next(iter(ds))
    # values are a contiguous slice of arange(shard_len)
    assert batch.min() >= 0
    assert batch.max() < shard_len
