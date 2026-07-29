"""PIN feature-tail guard and EV smoothing.

Both defend the same failure: a control decision made from a signal that is too
noisy or too lagging to support it. The tail guard watches the ultra-active
feature count while the dual is frozen; the EV smoother stops single-batch
measurement spread from driving the stop gates.
"""
import statistics

import pytest

import sae_scheduler as s
from sae_trainer_rolling import _ev_smoothed, EV_SMOOTH_MIN, EV_SMOOTH_N
from collections import deque

from test_scheduler import _make_scheduler


def _pin(**overrides):
    """A scheduler already sitting in PIN with a captured tail baseline."""
    sched = _make_scheduler(target_l0=500.0, pin_l0_band_abs=0.5, **overrides)
    sched.lambda_l0 = 1e-3
    sched._maybe_update_phase(l0=500.0, ev=0.90)
    assert sched.phase == "PIN"
    # First ultra reading inside PIN becomes the baseline, no judgement yet.
    sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.95)
    assert sched.pin_ultra_entry == pytest.approx(0.95)
    assert sched.phase == "PIN"
    return sched


def test_tail_collapse_releases_pin():
    sched = _pin(pin_ultra_release_frac=0.70, pin_ultra_patience=2)
    # 0.60/0.95 = 0.63 -> below the release ratio, but patience is 2.
    sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.60)
    assert sched.phase == "PIN"
    assert sched.pin_ultra_low_count == 1
    sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.60)
    assert sched.phase == "DESCENT"
    assert sched.pin_ultra_releases == 1


def test_healthy_tail_does_not_release_pin():
    sched = _pin(pin_ultra_release_frac=0.70, pin_ultra_patience=2)
    for _ in range(10):
        # Drifting down but never through the ratio (0.72 of entry).
        sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.685)
    assert sched.phase == "PIN"
    assert sched.pin_ultra_releases == 0


def test_low_count_resets_on_recovery():
    """Patience must mean *consecutive* windows, so one bad read cannot bank
    credit toward a release that a recovered tail should have cancelled."""
    sched = _pin(pin_ultra_release_frac=0.70, pin_ultra_patience=2)
    sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.50)
    assert sched.pin_ultra_low_count == 1
    sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.90)
    assert sched.pin_ultra_low_count == 0
    assert sched.phase == "PIN"


def test_repeated_collapse_stops_instead_of_cycling():
    """If the tail keeps shedding after being handed back to the dual, more
    training is buying EV by killing features. Stop rather than loop."""
    sched = _pin(pin_ultra_release_frac=0.70, pin_ultra_patience=1,
                 pin_ultra_max_releases=2)
    for expected_release in (1, 2):
        sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.10)
        assert sched.phase == "DESCENT"
        assert sched.pin_ultra_releases == expected_release
        assert not sched.should_stop
        # Re-enter PIN and re-capture a baseline, as a real run would.
        sched._maybe_update_phase(l0=500.0, ev=0.90)
        sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.95)
    sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.10)
    assert sched.should_stop
    assert "feature-tail collapse" in sched.stop_reason


def test_guard_is_inert_without_an_ultra_signal():
    """Callers that never supply ultra_frac must see the pre-guard behavior."""
    sched = _make_scheduler(target_l0=500.0, pin_l0_band_abs=0.5)
    sched.lambda_l0 = 1e-3
    sched._maybe_update_phase(l0=500.0, ev=0.90)
    for _ in range(20):
        sched._maybe_update_phase(l0=500.0, ev=0.90)
    assert sched.phase == "PIN"
    assert sched.pin_ultra_entry is None
    assert not sched.should_stop


def test_pin_entry_resets_stale_tail_baseline():
    """A baseline carried over from a previous PIN would compare the new tail
    against an old dictionary. Entering PIN must clear it."""
    sched = _pin()
    sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.10)
    sched._maybe_update_phase(l0=500.0, ev=0.90, ultra_frac=0.10)
    assert sched.phase == "DESCENT"
    sched._maybe_update_phase(l0=500.0, ev=0.90)
    assert sched.phase == "PIN"
    assert sched.pin_ultra_entry is None
    assert sched.pin_ultra_low_count == 0


# -- EV smoothing -----------------------------------------------------------

def test_smoother_passes_raw_through_until_the_window_fills():
    hist = deque(maxlen=EV_SMOOTH_N)
    for _ in range(EV_SMOOTH_MIN - 1):
        hist.append(0.10)
        assert _ev_smoothed(hist, 0.93) == 0.93


def test_smoother_rejects_a_single_outlier_batch():
    """The real failure: one lucky/unlucky batch swinging a gate. A mean would
    pass a fraction of the outlier through; the median must not move at all."""
    hist = deque([0.85, 0.86, 0.85, 0.86], maxlen=EV_SMOOTH_N)
    clean = _ev_smoothed(hist, 0.85)
    hist.append(0.20)
    assert _ev_smoothed(hist, 0.20) == pytest.approx(clean, abs=0.01)
    assert statistics.mean(hist) < clean - 0.10  # a mean would have moved


def test_smoother_tracks_a_real_shift():
    """Robustness must not mean deafness: a sustained move has to come through."""
    hist = deque(maxlen=EV_SMOOTH_N)
    for v in (0.85, 0.85, 0.85, 0.85, 0.85):
        hist.append(v)
    assert _ev_smoothed(hist, 0.85) == pytest.approx(0.85)
    for v in (0.40, 0.41, 0.40, 0.41, 0.40):
        hist.append(v)
    assert _ev_smoothed(hist, 0.40) == pytest.approx(0.40, abs=0.01)


def test_smoother_is_none_safe():
    """Non-log steps carry ev=None; the smoother must not invent a value."""
    assert _ev_smoothed(deque([0.9, 0.9, 0.9]), None) is None
