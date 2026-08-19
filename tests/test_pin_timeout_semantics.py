"""What pin_timeout_steps actually does (A6).

The config comment claimed "max steps in PIN before bailing back to DESCENT".
That bail is staged but unwired: _maybe_update_phase logs

    [PHASE] PIN would timeout ... [observe-only: no return to DESCENT yet]

and takes no action. The knob's only live effect is in _check_early_stop, where
a timed-out PIN satisfies the stop gate that otherwise waits for PIN
EV-readiness.

These tests pin both halves of that, so the day someone wires the DESCENT bail
they will see exactly which behaviour changed instead of discovering it in a run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_scheduler import _make_scheduler  # noqa: E402

TARGET_L0 = 500.0


def _pinned_scheduler(pin_timeout_steps=2000, elapsed=0):
    sched = _make_scheduler(target_l0=TARGET_L0,
                            pin_timeout_steps=pin_timeout_steps)
    sched.phase = "PIN"
    sched.pin_entry_step = 0
    sched.total_steps = elapsed
    for _ in range(4):
        sched.buffer.push_sae(l0=TARGET_L0)
    # A settled dual so the AL gates pass and only the PIN gate is under test.
    sched._lambda_history = [5e-4, 5e-4, 5e-4]
    sched.lambda_l0 = 5e-4
    return sched


def test_pin_timeout_does_not_return_to_descent():
    """The documented bail is not implemented. If this ever fails, someone wired
    it up -- update the config comment and the stop-path expectations too."""
    sched = _pinned_scheduler(pin_timeout_steps=100, elapsed=5000)
    sched._maybe_update_phase(l0=TARGET_L0, ev=0.99)
    assert sched.phase == "PIN"


def test_pin_timeout_enables_the_stop():
    """The live effect: a timed-out PIN satisfies the gate that otherwise waits
    for pin_ev_count to reach pin_ev_patience."""
    sched = _pinned_scheduler(pin_timeout_steps=100, elapsed=5000)
    sched.pin_ev_count = 0                      # not EV-ready
    for _ in range(sched.config.ev_stop_patience):
        sched._check_early_stop(ev=0.99)
    assert sched.should_stop is True
    assert "timeout" in sched.stop_reason.lower()


def test_pin_not_timed_out_and_not_ev_ready_does_not_stop():
    sched = _pinned_scheduler(pin_timeout_steps=100_000, elapsed=10)
    sched.pin_ev_count = 0
    for _ in range(sched.config.ev_stop_patience + 2):
        sched._check_early_stop(ev=0.99)
    assert sched.should_stop is False


def test_ev_ready_stops_without_timeout():
    sched = _pinned_scheduler(pin_timeout_steps=100_000, elapsed=10)
    sched.pin_ev_count = sched.config.pin_ev_patience
    for _ in range(sched.config.ev_stop_patience):
        sched._check_early_stop(ev=0.99)
    assert sched.should_stop is True
    assert "EV-ready" in sched.stop_reason


def test_outside_pin_never_stops_on_al_convergence_alone():
    """A run that converges without ever entering PIN keeps training."""
    sched = _pinned_scheduler(pin_timeout_steps=100, elapsed=5000)
    sched.phase = "DESCENT"
    for _ in range(sched.config.ev_stop_patience + 2):
        sched._check_early_stop(ev=0.99)
    assert sched.should_stop is False
