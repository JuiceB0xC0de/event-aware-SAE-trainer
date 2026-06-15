"""Tests for RevivalController -- dead-feature revival ownership (Branch 5).

CPU-only. conftest.py puts the repo root on sys.path so sae_trainer_rolling
imports cleanly.
"""
import pytest
import torch

import sae_trainer_rolling as t


def _rc(n_features=81920, policy="legacy"):
    return t.RevivalController(n_features, torch.device("cpu"), aux_k_policy=policy)


# -- AuxK policy (Branch 5B) --------------------------------------------------

def test_legacy_aux_k_at_k500_is_128():
    assert _rc(policy="legacy").effective_k_aux(500) == 128


def test_legacy_aux_k_preserves_low_l0_cap():
    rc = _rc(policy="legacy")
    assert rc.effective_k_aux(50) == 25    # min(128, 50//2)
    assert rc.effective_k_aux(10) == 8     # max(8, min(128, 5)) floor
    assert rc.effective_k_aux(300) == 128  # min(128, 150) cap


def test_feature_fraction_policy():
    # 81920 features -> 81920//64 = 1280 -> capped at 512
    assert _rc(n_features=81920, policy="feature_fraction").effective_k_aux(500) == 512
    # smaller dict stays below the 512 cap: 8192//64 = 128
    assert _rc(n_features=8192, policy="feature_fraction").effective_k_aux(500) == 128


def test_invalid_aux_k_policy_raises():
    with pytest.raises(ValueError):
        t.RevivalController(64, torch.device("cpu"), aux_k_policy="bogus")


def test_default_policy_is_legacy():
    rc = t.RevivalController(81920, torch.device("cpu"))
    assert rc.aux_k_policy == "legacy"
    assert rc.effective_k_aux(500) == 128  # behavior-preserving default


def test_revival_metrics_shape():
    rc = _rc(n_features=81920, policy="feature_fraction")
    m = rc.revival_metrics(500)
    assert m == {
        "aux_k_policy": "feature_fraction",
        "effective_aux_k": 512,
        "target_l0": 500,
        "n_features": 81920,
        "dead_count": 0,
        "reset_count": 0,
        "resampled_count": 0,
        "total_resampled": 0,
    }


# -- Reset / resample scheduling decisions (Branch 5C) ------------------------

def test_should_reset_cadence():
    rc = _rc()
    assert rc.should_reset(0) is False          # step 0 excluded
    assert rc.should_reset(t.RESET_EVERY) is True
    assert rc.should_reset(t.RESET_EVERY + 1) is False
    assert rc.should_reset(2 * t.RESET_EVERY) is True


def test_reset_threshold_matches_helper():
    rc = _rc()
    for k in (50, 100, 250, 500):
        assert rc.reset_threshold_for(k) == t._aggressive_k_reset_threshold(k)


def test_dead_masks_use_thresholds():
    rc = _rc(n_features=4)
    rc.steps_since_fired = torch.tensor([0, 1200, 600, 3000], dtype=torch.long)
    thr = rc.reset_threshold_for(500)  # default 1500
    # very_dead: silent >= reset threshold
    assert rc.very_dead_mask(500).tolist() == [s >= thr for s in (0, 1200, 600, 3000)]
    # resample dead: silent >= max(500, thr - 250)
    rd_thr = max(500, thr - 250)
    assert rc.resample_dead_mask(500).tolist() == [s >= rd_thr for s in (0, 1200, 600, 3000)]


def test_is_resample_step_and_schedule():
    rc = _rc()
    schedule = rc.resample_schedule(500)
    assert rc.is_resample_step(schedule[0], 500) is True
    assert rc.is_resample_step(schedule[-1], 500) is True
    assert rc.is_resample_step(schedule[-1] + 1, 500) is False


def test_record_reset_and_resample_metrics():
    rc = _rc()
    rc.record_reset(7)
    rc.record_resample(n_dead=10, n_resampled=4)
    rc.record_resample(n_dead=6, n_resampled=3)
    assert rc.last_reset_count == 7
    assert rc.last_resample_dead_count == 6
    assert rc.last_resampled_count == 3
    assert rc.total_resampled == 7  # 4 + 3 accumulated
    m = rc.revival_metrics(500)
    assert m["reset_count"] == 7 and m["resampled_count"] == 3 and m["total_resampled"] == 7
