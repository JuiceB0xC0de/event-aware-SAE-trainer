"""Tests for the activation-pool I/O helpers (filesystem + bf16 roundtrip)."""
from pathlib import Path

import torch

import sae_trainer_rolling as t


def test_write_read_roundtrip_is_bf16(tmp_path):
    d = Path(tmp_path) / "pool"
    original = torch.randn(2, 3)
    t._write_shard(d, 0, original)
    read = t._read_shard(d, 0)
    assert read.dtype == torch.bfloat16
    # _write_shard casts to bf16, so the readback equals the bf16 cast exactly
    assert torch.equal(read, original.to(torch.bfloat16))


def test_shard_paths_sorted_and_counted(tmp_path):
    d = Path(tmp_path) / "pool"
    for i in (2, 0, 1):                      # write out of order
        t._write_shard(d, i, torch.randn(2, 3))
    paths = t._shard_paths(d)
    assert len(paths) == 3
    names = [p.name for p in paths]
    assert names == sorted(names)            # lexicographic == numeric for zero-padded
    assert names[0] == "shard_00000.pt"


def test_shard_filename_zero_padding(tmp_path):
    d = Path(tmp_path) / "pool"
    t._write_shard(d, 42, torch.randn(1, 1))
    assert (d / "shard_00042.pt").exists()


def test_rm_pool_deletes_directory(tmp_path):
    d = Path(tmp_path) / "pool"
    t._write_shard(d, 0, torch.randn(2, 3))
    assert d.exists()
    t._rm_pool(d)
    assert not d.exists()


def test_rm_pool_is_safe_on_missing(tmp_path):
    # should not raise on a non-existent directory
    t._rm_pool(Path(tmp_path) / "does_not_exist")


def test_pool_dir_composes_under_rollcache():
    pd = t._pool_dir("tokens_s0")
    assert pd.name == "tokens_s0"
    assert str(pd).startswith(str(t.ROLLCACHE))
