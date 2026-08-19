"""The AL plateau test must mean "settled", not merely "not rising".

Two defects this pins:

A2 -- the old test was `max(recent) / recent[0] - 1 < tol`. For a monotonically
      decreasing lambda, max(recent) IS recent[0], so the expression is 0.0 and a
      collapsing dual was reported as converged.

A1 -- lambda is projected onto [lambda_l0_min, lambda_l0_max]. A dual pinned to
      the ceiling is perfectly flat because it ran out of authority, which is not
      a KKT point. The floor is deliberately NOT treated the same way: lambda == 0
      with L0 inside the band is complementary slackness, i.e. the sparsity
      constraint is simply inactive, which is a legitimate stop.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_scheduler import _make_scheduler  # noqa: E402

TARGET_L0 = 500.0


def _verdict(lambdas, lambda_max=5e-3, l0=None):
    """True if the given lambda window is accepted as an AL plateau.

    The L0 gate is satisfied independently so the lambda verdict is the only
    thing under test. _check_early_stop advances its AL-convergence counter
    exactly when both gates pass.
    """
    sched = _make_scheduler(target_l0=TARGET_L0, lambda_l0_max=lambda_max)
    l0 = TARGET_L0 if l0 is None else l0
    for _ in range(4):
        sched.buffer.push_sae(l0=l0)
    # _check_early_stop appends the CURRENT lambda to the history before reading
    # the window, so seed the history with all but the last reading and set
    # lambda_l0 to the last one. The window under test is then exactly `lambdas`.
    lambdas = list(lambdas)
    if lambdas:
        sched._lambda_history = lambdas[:-1]
        sched.lambda_l0 = lambdas[-1]
    else:
        sched._lambda_history = []
        sched.lambda_l0 = 0.0
    before = sched._ev_above_floor_count
    sched._check_early_stop(ev=0.99)
    return sched._ev_above_floor_count > before


def test_l0_gate_alone_is_not_enough():
    """Sanity: without a lambda plateau nothing converges, so the other tests
    are measuring the lambda verdict and not an always-true gate."""
    assert _verdict([]) is False


def test_flat_lambda_is_converged():
    assert _verdict([5e-4, 5e-4, 5e-4, 5e-4]) is True


def test_rising_lambda_is_not_converged():
    assert _verdict([1e-4, 2e-4, 4e-4, 8e-4]) is False


def test_falling_lambda_is_not_converged():
    """A2: max/first cannot see a decline, so this used to report converged."""
    assert _verdict([8e-4, 4e-4, 2e-4, 1e-4]) is False


def test_oscillating_lambda_is_not_converged():
    assert _verdict([5e-4, 8e-4, 5e-4, 8e-4]) is False


def test_lambda_pinned_at_ceiling_is_not_converged():
    """A1: saturated high is flat, but the constraint is being held by force."""
    assert _verdict([5e-3] * 4, lambda_max=5e-3) is False


def test_just_below_ceiling_still_converges():
    """Only an actual pin disqualifies; a high-but-free dual is a real plateau."""
    assert _verdict([4e-3] * 4, lambda_max=5e-3) is True


def test_raising_the_ceiling_unpins_the_dual():
    """The warning tells the operator to raise lambda_l0_max; doing so must
    restore convergence for the same lambda window."""
    assert _verdict([5e-3] * 4, lambda_max=5e-3) is False
    assert _verdict([5e-3] * 4, lambda_max=5e-2) is True


def test_tiny_but_free_lambda_is_converged():
    """A dual resting near the floor is complementary slackness, not saturation."""
    assert _verdict([1e-6, 1e-6, 1e-6, 1e-6]) is True


@pytest.mark.parametrize("window", [[5e-4] * 3, [5e-4] * 2, [5e-4], []])
def test_short_windows_never_converge(window):
    assert _verdict(window) is False


def test_l0_outside_band_blocks_convergence():
    """Even a perfect plateau must not stop while L0 is off target."""
    assert _verdict([5e-4] * 4, l0=TARGET_L0 * 2) is False
