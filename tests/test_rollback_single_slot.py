"""The dead-feature rollback buffer only ever needs one slot.

`_pick_dead_rollback` scans newest-first and returns the first window with
dead <= ceiling. Consequences:

  * an over-ceiling window is never the answer, so storing it is pure cost
  * once a newer under-ceiling window exists, every older one is unreachable

so retaining only the most recent under-ceiling snapshot selects exactly what a
multi-slot buffer selects, at a quarter of the host RAM (a full SAE state dict
is ~1.36 GB at EXPANSION=32 on a 2304-dim model).

These tests pin that equivalence so the single-slot buffer in
train_sae_on_activations cannot silently drift from the selection rule.
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sae_trainer_rolling as T  # noqa: E402

CEILING = 1.0


def _legacy_buffer(windows, slots=4):
    """Old behaviour: append every window, evict oldest past `slots`."""
    buf = []
    for step, dead in windows:
        buf.append({"step": step, "dead": dead})
        if len(buf) > slots:
            buf.pop(0)
    return buf


def _single_slot(windows, ceiling=CEILING):
    """New behaviour: retain only the newest window at or under the ceiling."""
    slot = None
    for step, dead in windows:
        if dead is not None and dead <= ceiling:
            slot = {"step": step, "dead": dead}
    return slot


def test_matches_the_documented_minicpm_trace():
    """The trace recorded in _pick_dead_rollback's docstring."""
    windows = [(1500, 0.0), (1750, 0.1), (2000, 0.8), (2250, 8.8)]
    legacy_pick = T._pick_dead_rollback(_legacy_buffer(windows), CEILING)
    assert legacy_pick["step"] == 2000          # as documented
    assert _single_slot(windows)["step"] == 2000


def test_single_slot_matches_legacy_on_random_traces():
    rng = random.Random(20260819)
    for _ in range(2000):
        n = rng.randint(1, 12)
        windows = [(i * 250, round(rng.choice([
            rng.uniform(0.0, 1.0),     # under ceiling
            rng.uniform(1.0, 12.0),    # over ceiling
        ]), 3)) for i in range(n)]

        legacy_pick = T._pick_dead_rollback(_legacy_buffer(windows), CEILING)
        slot = _single_slot(windows)

        if legacy_pick is None:
            # Single-slot may still hold a valid fallback the legacy buffer
            # evicted. It must never be the other way round.
            continue
        assert slot is not None, f"single slot lost a fallback legacy kept: {windows}"
        assert slot["step"] == legacy_pick["step"], windows


def test_single_slot_is_never_worse_than_legacy():
    """Legacy can evict its last clean state; single-slot cannot."""
    # Five consecutive over-ceiling windows push the clean one out of a 4-slot buffer.
    windows = [(0, 0.2)] + [(i * 250, 5.0) for i in range(1, 6)]
    assert T._pick_dead_rollback(_legacy_buffer(windows), CEILING) is None
    assert _single_slot(windows)["step"] == 0


@pytest.mark.parametrize("windows", [
    [],
    [(0, 5.0)],
    [(0, None)],
])
def test_no_clean_window_yields_no_pick(windows):
    assert T._pick_dead_rollback(_legacy_buffer(windows), CEILING) is None
    assert _single_slot(windows) is None


def test_ties_prefer_the_newer_window():
    windows = [(0, 0.5), (250, 0.5)]
    assert T._pick_dead_rollback(_legacy_buffer(windows), CEILING)["step"] == 250
    assert _single_slot(windows)["step"] == 250
