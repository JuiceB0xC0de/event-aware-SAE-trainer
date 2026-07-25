"""Tests for the AECS scheduler -- focus on the Augmented-Lagrangian lambda
integrator, which is the part the trainer depends on for sparsity convergence."""
import pytest

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
    above_target = _make_scheduler(
        lambda_l0_max=1.0,
        l0_tolerance=0.20,
        al_slingshot_overshoot_rel=0.10,
        al_slingshot_gain_max=24.0,
        early_pulse_steps=0,
    )
    above_target._prev_l0 = 620.02
    above_target.step(_sig(l0=620.0))

    old_landing = _make_scheduler(
        lambda_l0_max=1.0,
        l0_tolerance=0.20,
        al_landing_zone_rel=0.35,
        al_landing_min_progress=0.08,
        al_landing_gain_max=16.0,
        al_slingshot_overshoot_rel=0.10,
        al_slingshot_gain_max=24.0,
        early_pulse_steps=0,
    )
    old_landing._prev_l0 = 500.02
    old_landing.step(_sig(l0=500.0))

    assert above_target.lambda_l0 > old_landing.lambda_l0 * 10
    assert above_target._stall_pulse_remaining == 0


def test_constraint_lr_floor_keeps_optimizer_alive_until_target_crossed():
    sched = _make_scheduler(
        warmup_steps=1,
        total_steps=10_000,
        l0_tolerance=0.20,
        constraint_lr_floor=0.12,
        early_pulse_steps=0,
    )
    sched.total_steps = 9_500
    sched.step(_sig(l0=599.0))

    lr = sched.optimizer.param_groups[0]["lr"]
    assert lr >= sched.config.base_lr * sched.config.constraint_lr_floor

    crossed = _make_scheduler(
        warmup_steps=1,
        total_steps=10_000,
        l0_tolerance=0.20,
        constraint_lr_floor=0.12,
        early_pulse_steps=0,
    )
    crossed.total_steps = 9_500
    crossed.step(_sig(l0=499.0))
    assert crossed.optimizer.param_groups[0]["lr"] < lr


def test_early_stop_requires_sparse_side_crossing():
    sched = _make_scheduler(
        l0_tolerance=0.20,
        al_slingshot_overshoot_rel=0.10,
        ev_stop_patience=1,
        ev_check_every=1,
    )
    sched.lambda_l0 = 1e-3
    sched._lambda_history = [1e-3, 1e-3, 1e-3]
    for l0 in [599.0, 599.0, 599.0]:
        sched.step({**_sig(l0=l0), "ev": 0.99})
    assert sched.should_stop is False

    # Sparse-side crossing alone no longer stops immediately: the PIN gate
    # additionally requires phase==PIN (L0 within pin_l0_band_abs of target)
    # plus EV-ready (pin_ev_patience good windows) or PIN timeout.
    crossed = _make_scheduler(
        l0_tolerance=0.20,
        al_slingshot_overshoot_rel=0.10,
        ev_stop_patience=1,
        ev_check_every=1,
    )
    crossed.lambda_l0 = 1e-3
    crossed._lambda_history = [1e-3, 1e-3, 1e-3]
    for l0 in [499.8, 499.8, 499.8, 499.8]:
        crossed.step({**_sig(l0=l0), "ev": 0.99})
    assert crossed.should_stop is True


def test_pin_gate_holds_al_stop_until_ev_ready_or_timeout():
    """The Qwen L0/L1 regression: AL converges while EV is still climbing.
    The EV-blind AL stop must hold until PIN is EV-ready or times out."""
    sched = _make_scheduler(
        l0_tolerance=0.20,
        al_slingshot_overshoot_rel=0.10,
        ev_stop_patience=1,
        ev_check_every=1,
    )
    sched.lambda_l0 = 1e-3
    sched._lambda_history = [1e-3, 1e-3, 1e-3]
    # L0 locked in the PIN band, but EV well below pin_ev_thresh: no stop.
    for _ in range(6):
        sched.step({**_sig(l0=499.8), "ev": 0.70})
    assert sched.phase == "PIN"
    assert sched.should_stop is False

    # Backdate PIN entry past pin_timeout_steps: the polish budget is spent,
    # now the AL stop may fire even with low EV.
    sched.pin_entry_step = sched.total_steps - sched.config.pin_timeout_steps
    sched.step({**_sig(l0=499.8), "ev": 0.70})
    assert sched.should_stop is True
    assert "PIN timeout" in sched.stop_reason


# -- Phase observability (Branch 1) -------------------------------------------

def test_scheduler_starts_in_descent():
    sched = _make_scheduler()
    assert sched.phase == "DESCENT"
    assert sched.phase_step == 0
    assert sched.pin_entry_step is None
    assert sched.summary()["phase"] == "DESCENT"


def test_descent_enters_pin_when_l0_in_band():
    sched = _make_scheduler(target_l0=500.0, pin_l0_band_abs=0.5)
    assert sched.phase == "DESCENT"
    # Outside the band -> stays in DESCENT.
    sched._maybe_update_phase(l0=600.0, ev=0.90)
    assert sched.phase == "DESCENT"
    # Inside the band -> transitions to PIN and captures entry state.
    sched.lambda_l0 = 1e-3
    sched._maybe_update_phase(l0=500.2, ev=0.90)
    assert sched.phase == "PIN"
    assert sched.phase_step == 0
    assert sched.pin_entry_step == sched.total_steps
    assert sched.pinned_lambda == 1e-3


def test_old_checkpoint_without_phase_fields_restores_safely(tiny_sae, tmp_path):
    """A checkpoint saved before the phase machine existed (no phase keys in
    scheduler_state) must restore with safe defaults, not raise."""
    import torch
    import sae_trainer_rolling as t

    sae = tiny_sae
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    sched = _make_scheduler()
    # Dirty the phase state so we can prove the restore overwrites it.
    sched.phase = "PIN"
    sched.phase_step = 7
    sched.pin_ev_count = 2

    n = sum(1 for _ in sae.parameters())  # any size; roundtrip only needs a tensor
    ffc = torch.zeros(8, dtype=torch.long)
    ssf = torch.zeros(8, dtype=torch.long)
    rng = {"cuda": None, "cpu": torch.get_rng_state()}
    path = t._save_full_checkpoint(tmp_path, 100, sae, opt, sched, rng, ffc, ssf, None)

    # Emulate an OLD checkpoint: strip every phase key from scheduler_state.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("phase", "phase_step", "pin_entry_step", "pinned_lambda",
              "pin_ev_count", "pin_retry_count"):
        ckpt["scheduler_state"].pop(k, None)
    torch.save(ckpt, path)

    # Restore into a fresh (also dirtied) scheduler.
    sched2 = _make_scheduler()
    sched2.phase = "FINETUNE"
    sched2.phase_step = 999
    loaded = t._load_full_checkpoint(path, sae, opt, sched2, device="cpu")
    assert loaded is not None, "old-style checkpoint failed to load"
    assert sched2.phase == "DESCENT"
    assert sched2.phase_step == 0
    assert sched2.pin_entry_step is None
    assert sched2.pinned_lambda is None
    assert sched2.pin_ev_count == 0
    assert sched2.pin_retry_count == 0


# -- PIN lambda gates (Branch 2) ----------------------------------------------

def test_dual_update_frozen_in_pin():
    sched = _make_scheduler(al_dual_step=1.0, lambda_l0_max=1.0)
    sched.total_steps = 10
    sched.lambda_l0 = 2e-3
    # DESCENT: the integrator moves lambda when L0 is above target.
    sched.phase = "DESCENT"
    sched._dual_update(600.0)
    assert sched.lambda_l0 > 2e-3
    # PIN: lambda is frozen regardless of L0 error (no windup).
    pinned = sched.lambda_l0
    sched.phase = "PIN"
    for l0 in (600.0, 1200.0, 300.0):
        sched._dual_update(l0)
        assert sched.lambda_l0 == pinned


def test_live_tune_lambda_override_ignored_in_pin(tmp_path):
    import json
    lt = tmp_path / "live_tune.json"
    lt.write_text(json.dumps({"lambda_l0_override": 0.5}))
    missing_alt = str(tmp_path / "does_not_exist.json")  # avoid stray /tmp/live_tune.json

    # DESCENT: the override is applied (sets lambda directly).
    sched = _make_scheduler(lambda_l0_max=1.0)
    sched.config.live_tune_path = str(lt)
    sched.config.live_tune_path_alt = missing_alt
    sched.phase = "DESCENT"
    sched._live_tune_mtime = 0.0
    sched._apply_live_tune()
    assert sched.lambda_l0 == 0.5

    # PIN: the same override is ignored, lambda unchanged.
    sched2 = _make_scheduler(lambda_l0_max=1.0)
    sched2.config.live_tune_path = str(lt)
    sched2.config.live_tune_path_alt = missing_alt
    sched2.phase = "PIN"
    sched2.lambda_l0 = 3e-3
    sched2._live_tune_mtime = 0.0
    sched2._apply_live_tune()
    assert sched2.lambda_l0 == 3e-3


def test_finetune_entry_repins_lambda_ceiling():
    # Ceiling lowered below pinned lambda -> FINETUNE entry raises it back up.
    sched = _make_scheduler(lambda_l0_max=0.1)
    sched.phase = "PIN"
    sched.pinned_lambda = 0.5
    sched._enter_phase("FINETUNE", "test")
    assert sched.phase == "FINETUNE"
    assert sched.config.lambda_l0_max == 0.5  # raised to pinned_lambda

    # Ceiling already above pinned lambda -> left unchanged.
    sched2 = _make_scheduler(lambda_l0_max=0.8)
    sched2.phase = "PIN"
    sched2.pinned_lambda = 0.5
    sched2._enter_phase("FINETUNE", "test")
    assert sched2.config.lambda_l0_max == 0.8


# -- Threshold nudge gate (Branch 3) ------------------------------------------

def test_threshold_nudge_suppressed_in_pin():
    sched = _make_scheduler()  # target_l0=500, default nudge gain 0.08
    # DESCENT: L0 well above target and outside the deadband -> nudge fires.
    sched.phase = "DESCENT"
    nudge, should_apply = sched.compute_threshold_nudge(current_l0=600.0, step=0)
    assert should_apply is True
    assert nudge > 0.0
    # PIN: identical inputs -> fully suppressed.
    sched.phase = "PIN"
    assert sched.compute_threshold_nudge(current_l0=600.0, step=0) == (0.0, False)


def test_threshold_nudge_active_outside_pin():
    # DESCENT nudges when L0 is outside the deadband.
    sched = _make_scheduler()
    sched.phase = "DESCENT"
    nudge, should_apply = sched.compute_threshold_nudge(current_l0=650.0, step=0)
    assert should_apply is True and nudge > 0.0
    # FINETUNE still allows the nudge (gate is PIN-only, not global).
    sched.phase = "FINETUNE"
    nudge_f, should_apply_f = sched.compute_threshold_nudge(current_l0=650.0, step=0)
    assert should_apply_f is True and nudge_f > 0.0


# -- Deep-layer slingshot gain scaling (Branch 4) -----------------------------

def _slingshot_sched(ref=None, probe=None, layer=None):
    sched = _make_scheduler(al_slingshot_gain_max=24.0, deep_layer_slingshot_gain=8.0,
                            slingshot_norm_alpha=-0.5)
    sched.config.activation_norm_ref = ref
    sched._activation_norm_preflight = probe
    sched.layer = layer
    return sched


def test_slingshot_gain_norm_scaled_from_preflight():
    # ref/probe 1/1 -> full gain 24.
    assert _slingshot_sched(ref=1.0, probe=1.0)._effective_slingshot_gain() == 24.0
    # ratio 4, alpha -0.5 -> 4**-0.5 = 0.5 -> 24 * 0.5 = 12.
    assert _slingshot_sched(ref=1.0, probe=4.0)._effective_slingshot_gain() == 12.0
    # ratio 100 -> 0.1 < floor_scale (8/24) -> floors at deep_layer_slingshot_gain.
    assert _slingshot_sched(ref=1.0, probe=100.0)._effective_slingshot_gain() == pytest.approx(8.0)


def test_slingshot_gain_uses_landing_path():
    # The norm-scaled gain is what the dual update sees while L0 is above target.
    sched = _slingshot_sched(ref=1.0, probe=4.0)
    assert sched._landing_lambda_gain(current_l0=600.0, control_error=150.0) == 12.0


def test_slingshot_gain_fallback_without_preflight_stats():
    # Missing preflight stats -> deterministic layer-number fallback.
    assert _slingshot_sched(ref=None, probe=None, layer=3)._effective_slingshot_gain() == 8.0
    assert _slingshot_sched(ref=None, probe=None, layer=24)._effective_slingshot_gain() == 8.0
    assert _slingshot_sched(ref=None, probe=None, layer=2)._effective_slingshot_gain() == 24.0
    assert _slingshot_sched(ref=None, probe=None, layer=None)._effective_slingshot_gain() == 24.0


# --- PIN release on L0 escape -------------------------------------------------
# Regression guard for the MiniCPM5-1B L13-L15 control failure: a mass dead-feature
# reset blew L0 from 50 to ~1700 while the phase machine was PINned, and because the
# dual is frozen in PIN there was no restoring force. All three layers coasted to
# max_steps at ~34x the target sparsity.


def _pinned_sched(**overrides):
    """A scheduler already sitting in PIN with L0 at target."""
    sched = _make_scheduler(**overrides)
    sched._maybe_update_phase(l0=500.0, ev=0.90)
    assert sched.phase == "PIN"
    return sched


def test_pin_releases_when_l0_escapes_band():
    sched = _pinned_sched()
    # Blowout well outside +/-25% of target 500 -> release back to DESCENT.
    sched._maybe_update_phase(l0=1700.0, ev=0.97)
    assert sched.phase == "DESCENT"
    assert sched.pin_ev_count == 0


def test_pin_release_unfreezes_the_dual():
    # The point of releasing is that the integrator runs again. In PIN the dual
    # update early-returns and lambda cannot move; after release it must.
    sched = _pinned_sched()
    lam_pinned = sched.lambda_l0
    sched._dual_update(1700.0)
    assert sched.lambda_l0 == lam_pinned, "lambda moved while PINned"

    sched._maybe_update_phase(l0=1700.0, ev=0.97)
    assert sched.phase == "DESCENT"
    sched._dual_update(1700.0)
    assert sched.lambda_l0 > lam_pinned, "lambda did not move after PIN release"


def test_pin_holds_through_normal_oscillation():
    # Observed healthy PIN behaviour oscillates a few counts around target. The
    # release band is deliberately wide so this can never trip it.
    sched = _pinned_sched()
    for l0 in (497.0, 503.0, 490.0, 510.0, 520.0, 480.0):
        sched._maybe_update_phase(l0=l0, ev=0.96)
        assert sched.phase == "PIN", f"released on benign L0={l0}"


def test_pin_release_band_is_configurable():
    sched = _pinned_sched(pin_l0_release_frac=0.10)
    # 10% of 500 -> release above 550. 540 holds, 600 releases.
    sched._maybe_update_phase(l0=540.0, ev=0.96)
    assert sched.phase == "PIN"
    sched._maybe_update_phase(l0=600.0, ev=0.96)
    assert sched.phase == "DESCENT"


# --- dead-feature ceiling rollback --------------------------------------------
# Regression guard for the MiniCPM5-1B deep-layer collapse: at PIN entry the feature
# tail starves inside a single 250-step window (L13 observed 0.8% -> 8.8% dead while
# EV moved 0.949 -> 0.948). AuxK cannot respond because AUX_DEAD_THRESHOLD is 250
# steps of silence. So the trainer buffers recent log windows and walks back.

import sae_trainer_rolling as t


def _w(step, dead):
    return {"step": step, "dead": dead, "ev": 0.95, "l0": 49.0, "state": None}


def test_dead_rollback_picks_newest_under_ceiling():
    # The real L13 buffer at the breach.
    buf = [_w(1500, 0.0), _w(1750, 0.1), _w(2000, 0.8), _w(2250, 8.8)]
    pick = t._pick_dead_rollback(buf, 1.0)
    assert pick["step"] == 2000, "should take the most-trained state under the ceiling"


def test_dead_rollback_returns_none_when_all_breached():
    buf = [_w(2250, 8.8), _w(2500, 18.4), _w(2750, 26.0)]
    assert t._pick_dead_rollback(buf, 1.0) is None


def test_dead_rollback_ceiling_is_inclusive():
    buf = [_w(1000, 0.5), _w(1250, 1.0)]
    assert t._pick_dead_rollback(buf, 1.0)["step"] == 1250


def test_dead_rollback_skips_none_dead():
    # Non-log windows carry dead=None and must never be selected.
    buf = [_w(1000, 0.4), _w(1250, None), _w(1500, None)]
    assert t._pick_dead_rollback(buf, 1.0)["step"] == 1000
