"""Tests for the argparse CLI wiring in `main()`.

We monkeypatch `run_atlas_rolling` so we can assert the parsed flags reach it
correctly without launching any training (no GPU / model / network)."""
import sae_trainer_rolling as t


def _capture_main_kwargs(monkeypatch, argv):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(t, "run_atlas_rolling", fake_run)
    monkeypatch.setattr("sys.argv", ["sae_trainer_rolling.py"] + argv)
    t.main()
    return captured


def test_defaults(monkeypatch):
    kw = _capture_main_kwargs(monkeypatch, [])
    assert kw["start_layer"] == 0
    assert kw["end_layer"] == 9
    assert kw["seed"] == t.DEFAULT_SEED
    assert kw["pool_batches"] == t.POOL_BATCHES_DEFAULT
    assert kw["use_pretok"] is True
    assert kw["push"] is True
    assert kw["max_steps"] == t.N_STEPS
    # model-agnostic defaults: no model override, no push target, no wandb
    assert kw["capture"] == "auto"
    assert kw["model_id"] is None
    assert kw["hub_id"] is None
    assert kw["hub_repo_type"] is None
    assert kw["wandb_project"] is None
    assert kw["expansion"] is None
    assert kw["evict_model"] is True
    assert kw["target_l0"] is None


def test_flag_overrides(monkeypatch):
    kw = _capture_main_kwargs(monkeypatch, [
        "--start-layer", "1", "--end-layer", "3", "--seed", "7",
        "--pool-batches", "100", "--max-steps", "10",
        "--no-pretok", "--no-push", "--no-model-evict",
        "--capture", "rolling", "--model-id", "meta-llama/Llama-3.2-1B",
        "--hub-id", "me/my-saes", "--wandb-project", "run1", "--expansion", "16",
        "--hub-repo-type", "dataset",
        "--target-l0", "50",
    ])
    assert kw["start_layer"] == 1
    assert kw["end_layer"] == 3
    assert kw["seed"] == 7
    assert kw["pool_batches"] == 100
    assert kw["max_steps"] == 10
    assert kw["use_pretok"] is False
    assert kw["push"] is False
    assert kw["evict_model"] is False
    assert kw["target_l0"] == 50
    assert kw["capture"] == "rolling"
    assert kw["model_id"] == "meta-llama/Llama-3.2-1B"
    assert kw["hub_id"] == "me/my-saes"
    assert kw["hub_repo_type"] == "dataset"
    assert kw["wandb_project"] == "run1"
    assert kw["expansion"] == 16
