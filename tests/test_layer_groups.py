"""Tests for Group-SAE layer grouping (arXiv:2410.21508).

Covers both halves of the integration:
  * the standalone grouping primitives in `layer_groups`, and
  * the trainer-side wiring in `sae_trainer_rolling` (`_reuse_sae_for_layer`
    and the `--group-similarity` CLI flag) that turns a grouping decision into
    a shared SAE without launching any training (no GPU / model / network).
"""
import json

import torch

import layer_groups as g
import sae_trainer_rolling as t


# -- grouping primitives (new module) --------------------------------------

def test_mean_signature_reduces_pool_shard_to_a_vector():
    shard = torch.randn(4, 8, 5)                # [n_seqs, seq_len, d]
    sig = g.mean_activation_signature(shard)
    assert sig.shape == (5,)
    assert torch.allclose(sig, shard.reshape(-1, 5).float().mean(dim=0))


def test_cosine_similarity_bounds_and_degenerate():
    a = torch.tensor([1.0, 0.0, 0.0])
    assert g.cosine_similarity(a, a) == 1.0
    assert abs(g.cosine_similarity(a, torch.tensor([0.0, 1.0, 0.0]))) < 1e-6
    assert g.cosine_similarity(a, torch.zeros(3)) == 0.0     # no NaN on zero norm


def test_contiguous_grouping_splits_on_the_similarity_floor():
    base = torch.tensor([1.0, 0.0])
    near = torch.tensor([0.99, 0.14])          # ~cos 0.99 with base
    far = torch.tensor([0.0, 1.0])             # orthogonal -> new group
    sigs = [base, near, near, far, far]
    groups = g.group_contiguous_layers(sigs, threshold=0.9)
    assert groups == [[0, 1, 2], [3, 4]]
    # only two SAEs get trained instead of five
    assert g.sae_training_savings(groups) == 3


def test_high_threshold_falls_back_to_one_group_per_layer():
    sigs = [torch.randn(6) for _ in range(4)]
    groups = g.group_contiguous_layers(sigs, threshold=1.01)  # unreachable floor
    assert groups == [[0], [1], [2], [3]]
    assert g.sae_training_savings(groups) == 0


# -- trainer wiring (existing module) --------------------------------------

def _make_anchor_dir(tmp_path, monkeypatch, layer, seed):
    monkeypatch.setattr(t, "SAE_DIR", str(tmp_path))
    d = tmp_path / f"layer_{layer:02d}_s{seed}"
    d.mkdir(parents=True)
    torch.save({"w": torch.ones(2, 2)}, d / "sae.pt")
    (d / "meta.json").write_text(json.dumps(
        {"layer": layer, "seed": seed, "final_metrics": {"ev": 0.9, "l0": 500.0}}))
    return d


def test_reuse_sae_copies_anchor_and_rewrites_provenance(tmp_path, monkeypatch):
    _make_anchor_dir(tmp_path, monkeypatch, layer=3, seed=0)
    metrics = t._reuse_sae_for_layer(anchor_layer=3, member_layer=4, seed=0)

    member = tmp_path / "layer_04_s0"
    assert (member / "sae.pt").exists()        # dictionary copied verbatim
    meta = json.loads((member / "meta.json").read_text())
    assert meta["layer"] == 4                  # id rewritten to the member
    assert meta["shared_from"] == 3            # provenance recorded
    # returns a train_sae_on_activations-shaped metrics dict, tagged as shared
    assert metrics["ev"] == 0.9
    assert metrics["shared_from"] == 3


def test_reuse_sae_raises_when_anchor_untrained(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "SAE_DIR", str(tmp_path))
    try:
        t._reuse_sae_for_layer(anchor_layer=1, member_layer=2, seed=0)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError when the anchor has no SAE")


def test_group_similarity_flag_reaches_run_atlas_rolling(monkeypatch):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(t, "run_atlas_rolling", fake_run)
    monkeypatch.setattr("sys.argv",
                        ["sae_trainer_rolling.py", "--group-similarity", "0.95"])
    t.main()
    assert captured["group_similarity"] == 0.95


def test_group_similarity_defaults_off(monkeypatch):
    captured = {}
    monkeypatch.setattr(t, "run_atlas_rolling", lambda **kw: captured.update(kw))
    monkeypatch.setattr("sys.argv", ["sae_trainer_rolling.py"])
    t.main()
    assert captured["group_similarity"] is None
