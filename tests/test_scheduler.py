"""Tests for the AECS scheduler -- focus on the Augmented-Lagrangian lambda
integrator, which is the part the trainer depends on for sparsity convergence."""
import sae_scheduler as s
from conftest import make_dummy_optimizer


def _make_scheduler(**overrides):
    cfg_kwargs = dict(
        use_augmented_lagrangian=True,
        target_l0=500.0,
        lambda_l0_init=0.0,
        lambda_l0_min=0.0,
        lambda_l0_max=5e-3,
        al_dual_step=1e-7,
        warmup_steps=1,
        event_warmup_steps=10_000,   # keep us in BASELINE for these short runs
        verbose=False,
        mode_verbose=False,
    )
    cfg_kwargs.update(overrides)
    cfg = s.SAEAECSConfig(**cfg_kwargs)
    return s.SAEEventControlScheduler(make_dummy_optimizer(), cfg)


def _sig(l0, loss=1.0, grad_norm=1.0):
    return {"loss": loss, "grad_norm": grad_norm, "l0": l0}


def test_initial_state():
    sched = _make_scheduler()
    assert sched.mode == "BASELINE"
    assert sched.lambda_l0 == 0.0
    assert sched.should_stop is False


def test_lambda_increases_when_l0_above_target():
    sched = _make_scheduler()
    prev = sched.lambda_l0
    for _ in range(20):
        sched.step(_sig(l0=600.0))   # above target -> integrator climbs
        assert sched.lambda_l0 >= prev
        prev = sched.lambda_l0
    assert sched.lambda_l0 > 0.0


def test_lambda_clamped_to_max():
    # huge step + above target -> should saturate at lambda_l0_max
    sched = _make_scheduler(al_dual_step=1.0)
    sched.step(_sig(l0=600.0))
    assert sched.lambda_l0 == 5e-3


def test_lambda_decreases_when_l0_below_target_and_floored():
    sched = _make_scheduler(al_dual_step=1.0)
    sched.lambda_l0 = 3e-3                     # pretend the integrator built up
    sched.step(_sig(l0=100.0))                 # below target -> relax
    assert sched.lambda_l0 < 3e-3
    # keep pushing below target; lambda must not go below the floor
    for _ in range(5):
        sched.step(_sig(l0=100.0))
    assert sched.lambda_l0 >= 0.0
    assert sched.lambda_l0 == 0.0


def test_step_returns_mode_and_sets_optimizer_lr():
    sched = _make_scheduler()
    mode = sched.step(_sig(l0=550.0))
    assert mode in s.SAEEventControlScheduler.MODES
    lr = sched.optimizer.param_groups[0]["lr"]
    assert isinstance(lr, float) and lr > 0.0


def test_summary_has_expected_keys():
    sched = _make_scheduler()
    sched.step(_sig(l0=550.0))
    summary = sched.summary()
    assert "mode" in summary
    assert "lambda_l0" in summary


def test_legacy_pcontroller_path_runs():
    # use_augmented_lagrangian=False exercises the other lambda branch
    sched = _make_scheduler(use_augmented_lagrangian=False, lambda_adjust_cooldown=1)
    for _ in range(5):
        sched.step(_sig(l0=600.0))
    assert sched.lambda_l0 >= 0.0


def test_ev_delta_edge_cases():
    import math
    buf = s.SAESignalBuffer(window=10)

    # Empty buffer
    assert buf.ev_delta() == 0.0

    # Single item
    buf.push_sae(ev=0.5)
    assert buf.ev_delta() == 0.0

    # Two items
    buf.push_sae(ev=0.7)
    assert math.isclose(buf.ev_delta(), 0.2)

    # Three items
    buf.push_sae(ev=0.6)
    assert math.isclose(buf.ev_delta(), -0.1)
