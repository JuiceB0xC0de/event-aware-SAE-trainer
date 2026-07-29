"""Gradient norm is a control signal, not a log value.

From 7920d89 (2026-06-15) until this test landed, the trainer passed 0.0 for
grad_norm on non-log steps, on the stated theory that it was "only needed for
logging/W&B". It is not. step() pushes it into a 50-deep window that
_detect_base_event() reads on EVERY step, so a synthetic zero is not a skipped
sample, it is a false observation.

With LOG_EVERY=250 against a 50-deep window the buffer held at most one real
reading and was all-zero for 200 of every 250 steps. The last-10 mean then sat
under plateau_grad_norm_thresh (1e-4), PLATEAU fired, and because
_maybe_transition() returns early when the target mode equals the current mode,
EXPLORE latched: 1.5x LR and 0.5x weight decay for the remainder of any run past
event_warmup_steps + 10.

No shipped atlas layer was affected (accepted runs stopped between 1501 and 4751
steps, below the 5000-step warmup), but N_STEPS defaults to 15_000, so the
exposure was entirely in long runs -- exactly the ones already struggling.

Every other scheduler test sets event_warmup_steps=10_000 to stay in BASELINE
for short runs, which is why none of them could see this.
"""
import inspect

from test_scheduler import _make_scheduler, _sig


def _run(sched, n, grad_norm):
    for _ in range(n):
        sched.step(_sig(50.0, grad_norm=grad_norm))


def test_honest_grad_norm_every_step_stays_baseline():
    """The contract: a real reading every step leaves the mode machine alone."""
    sched = _make_scheduler(target_l0=50.0, event_warmup_steps=20, cooldown_steps=1)
    _run(sched, 120, grad_norm=1.0)
    assert sched.mode == "BASELINE"


def test_synthetic_zeros_latch_the_mode_machine_into_explore():
    """The regression: zeros on non-log steps drive a phantom PLATEAU."""
    sched = _make_scheduler(target_l0=50.0, event_warmup_steps=20, cooldown_steps=1)
    _run(sched, 20, grad_norm=1.0)
    assert sched.mode == "BASELINE"

    # The old trainer behaviour between two log steps.
    _run(sched, 120, grad_norm=0.0)
    assert sched.mode == "EXPLORE"

    # Departure from BASELINE is one-way. Honest readings afterwards do not undo
    # it: the mixed window's variance crosses reentry_grad_norm_tol and trips
    # UNSTABLE -> STABILIZE, which has no exit event of its own. No base event
    # ever maps back to BASELINE, so the run finishes in a modulated LR mode
    # whichever way the window happens to fall.
    _run(sched, 60, grad_norm=1.0)
    assert sched.mode in ("EXPLORE", "STABILIZE")
    assert sched.mode != "BASELINE"


def test_gradient_spike_is_dead_under_the_zero_pattern():
    """The safety path went with it.

    grad_norm_ema over 49 zeros and one real reading g gives mu = 0.03g and
    sigma = 0.168g, so the z-score is 5.77 for ANY g. Against the shipped
    instability_z_thresh of 10.0 that detector could never fire, no matter how
    badly the gradients blew up.
    """
    sched = _make_scheduler(target_l0=50.0, event_warmup_steps=20, cooldown_steps=1)
    _run(sched, 60, grad_norm=0.0)
    for g in (1.0, 10.0, 1000.0):
        sched.buffer.grad_norms[-1] = g
        assert sched.buffer.grad_norm_zscore() < 6.0
        sched.buffer.grad_norms[-1] = 0.0


def test_trainer_does_not_reintroduce_the_synthetic_zero():
    """Source guard: the deferral is a tempting optimisation that already shipped
    once. If grad_norm ever needs deferring again, teach the scheduler that None
    means "no observation" and rescale its window first."""
    from sae_trainer_rolling import train_sae_on_activations

    src = inspect.getsource(train_sae_on_activations)
    assert "grad_norm_val = 0.0" not in src
    assert "grad_norm_val = grad_norm_t.item()" in src
