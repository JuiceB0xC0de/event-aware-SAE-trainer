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

def test_grad_norm_zscore_few_elements():
    buf = s.SAESignalBuffer()
    # Less than 5 elements
    for i in range(2):
        buf.push_base(loss=1.0, grad_norm=float(i))
    assert buf.grad_norm_zscore() == 0.0

def test_grad_norm_zscore_zero_variance():
    buf = s.SAESignalBuffer()
    # 5 identical elements
    for _ in range(5):
        buf.push_base(loss=1.0, grad_norm=2.0)
    assert buf.grad_norm_zscore() == 0.0

def test_grad_norm_zscore_valid():
    buf = s.SAESignalBuffer()
    # Varying elements
    buf.push_base(loss=1.0, grad_norm=1.0)
    buf.push_base(loss=1.0, grad_norm=2.0)
    buf.push_base(loss=1.0, grad_norm=3.0)
    buf.push_base(loss=1.0, grad_norm=4.0)
    buf.push_base(loss=1.0, grad_norm=5.0)

    # Calculate expected mathematically
    # mu_0 = 1.0
    # var_0 = 0.0

    # mu_1 = 0.97 * 1.0 + 0.03 * 2.0 = 1.03
    # var_1 = 0.97 * 0.0 + 0.03 * (2.0 - 1.03)^2 = 0.028227

    # We can just verify it's a non-zero float since testing the exact EMA implementation logic
    # might be brittle if EMA formula changes.
    zscore = buf.grad_norm_zscore()
    assert isinstance(zscore, float)
    assert zscore != 0.0
    # With increasing values, the last value is higher than the EMA mean, so zscore > 0
    assert zscore > 0.0

    # Let's push a very small value to see if zscore becomes negative
    buf.push_base(loss=1.0, grad_norm=0.0)
    zscore_neg = buf.grad_norm_zscore()
    assert zscore_neg < 0.0
