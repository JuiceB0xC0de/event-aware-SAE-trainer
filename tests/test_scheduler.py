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


def test_check_dead_emergency_below_threshold():
    sched = _make_scheduler(dead_emergency_thresh=0.5, dead_emergency_resample_trigger=True)
    # 0.4 < 0.5, so it should not trigger
    result = sched._check_dead_emergency(dead_pct=0.4)
    assert result is None


def test_check_dead_emergency_triggers():
    sched = _make_scheduler(
        dead_emergency_thresh=0.5,
        dead_emergency_resample_trigger=True,
        dead_emergency_cooldown=100
    )
    sched.total_steps = 200
    sched._dead_emergency_last_step = 0
    # 0.6 > 0.5, trigger is True, cooldown passed (200 - 0 >= 100)
    result = sched._check_dead_emergency(dead_pct=0.6)
    assert result == "DEAD_EMERGENCY"
    assert sched._dead_emergency_last_step == 200


def test_check_dead_emergency_no_resample_trigger():
    sched = _make_scheduler(
        dead_emergency_thresh=0.5,
        dead_emergency_resample_trigger=False,
        dead_emergency_cooldown=100
    )
    sched.total_steps = 200
    sched._dead_emergency_last_step = 0
    # 0.6 > 0.5, cooldown passed, but resample_trigger is False
    result = sched._check_dead_emergency(dead_pct=0.6)
    assert result is None
    # Last step should still be updated
    assert sched._dead_emergency_last_step == 200


def test_check_dead_emergency_cooldown():
    sched = _make_scheduler(
        dead_emergency_thresh=0.5,
        dead_emergency_resample_trigger=True,
        dead_emergency_cooldown=100
    )

    # First trigger
    sched.total_steps = 100
    sched._dead_emergency_last_step = -100
    result = sched._check_dead_emergency(dead_pct=0.6)
    assert result == "DEAD_EMERGENCY"
    assert sched._dead_emergency_last_step == 100

    # Second trigger immediately after, inside cooldown
    sched.total_steps = 150
    result = sched._check_dead_emergency(dead_pct=0.6)
    assert result is None
    # Last step should not be updated
    assert sched._dead_emergency_last_step == 100

    # Third trigger after cooldown
    sched.total_steps = 250
    result = sched._check_dead_emergency(dead_pct=0.6)
    assert result == "DEAD_EMERGENCY"
    assert sched._dead_emergency_last_step == 250


def test_early_pulse_lifts_warmup_floor_far_from_target():
    sched = _make_scheduler(
        warmup_steps=1000,
        early_pulse_steps=100,
        early_pulse_warmup_floor=0.35,
        early_pulse_multiplier=1.25,
        convergence_lockout_rel=0.25,
    )
    sched.step(_sig(l0=2000.0))
    lr = sched.optimizer.param_groups[0]["lr"]
    assert lr >= sched.config.base_lr * 0.35


def test_convergence_lockout_suppresses_energy_boosts():
    sched = _make_scheduler(
        initial_lr_multiplier=1.6,
        early_pulse_steps=100,
        early_pulse_multiplier=1.4,
        activation_norm_ref=1.0,
        convergence_lockout_rel=0.25,
    )
    sched.step({**_sig(l0=2000.0), "activation_norm": 4.0})
    boosted = sched.optimizer.param_groups[0]["lr"]

    sched_locked = _make_scheduler(
        initial_lr_multiplier=1.6,
        early_pulse_steps=100,
        early_pulse_multiplier=1.4,
        activation_norm_ref=1.0,
        convergence_lockout_rel=0.25,
    )
    sched_locked.step({**_sig(l0=620.0), "activation_norm": 4.0})
    locked = sched_locked.optimizer.param_groups[0]["lr"]

    assert locked < boosted


def test_stall_pulse_triggers_only_far_from_target():
    sched = _make_scheduler(
        stall_warmup_steps=1,
        stall_cooldown_steps=1,
        stall_pulse_steps=10,
        event_warmup_steps=10_000,
    )
    for l0 in [3000.0, 2500.0, 2200.0, 2100.0, 2098.0, 2097.5, 2097.4]:
        sched.step(_sig(l0=l0))
    assert sched._stall_pulse_remaining > 0

    locked = _make_scheduler(
        stall_warmup_steps=1,
        stall_cooldown_steps=1,
        stall_pulse_steps=10,
        event_warmup_steps=10_000,
    )
    for l0 in [800.0, 700.0, 630.0, 620.0, 615.0, 612.0, 610.0]:
        locked.step(_sig(l0=l0))
    assert locked._stall_pulse_remaining == 0


def test_landing_stall_accelerates_lambda_not_lr_pulse():
    slow = _make_scheduler(
        lambda_l0_max=1.0,
        l0_tolerance=0.20,
        al_landing_zone_rel=0.35,
        al_landing_min_progress=0.08,
        al_landing_gain_max=16.0,
        early_pulse_steps=0,
    )
    slow._prev_l0 = 620.02
    slow.step(_sig(l0=620.0))

    fast = _make_scheduler(
        lambda_l0_max=1.0,
        l0_tolerance=0.20,
        al_landing_zone_rel=0.35,
        al_landing_min_progress=0.08,
        al_landing_gain_max=16.0,
        early_pulse_steps=0,
    )
    fast._prev_l0 = 700.0
    fast.step(_sig(l0=620.0))

    assert slow.lambda_l0 > fast.lambda_l0 * 5
    assert slow._stall_pulse_remaining == 0


def test_constraint_lr_floor_keeps_optimizer_alive_above_tolerance():
    sched = _make_scheduler(
        warmup_steps=1,
        total_steps=10_000,
        l0_tolerance=0.20,
        constraint_lr_floor=0.08,
        early_pulse_steps=0,
    )
    sched.total_steps = 9_500
    sched.step(_sig(l0=620.0))

    lr = sched.optimizer.param_groups[0]["lr"]
    assert lr >= sched.config.base_lr * sched.config.constraint_lr_floor
