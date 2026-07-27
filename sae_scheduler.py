"""
SAE-extended AECS -- Dual-Loop Adaptive Control for JumpReLU SAE Training.

AECS base (AECS-scheduler): LR scheduler via 4-mode state machine
    (BASELINE, RECOVERY, EXPLORE, STABILIZE) driven by gradient/loss dynamics.

SAE extension: LAMBDA_L0 as a second actuator, responding to:
    - L0 error: L0 vs target K (PI controller)
    - EV floor protection: EV drops -> pull LAMBDA back
    - Dead feature emergency: dead_pct > threshold -> STABILIZE
    - Early stop: EV stays above floor for N consecutive windows

Two control loops, one event detection engine.
"""
from __future__ import annotations

__version__ = "0.5.0"

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple



@dataclass
class SAEAECSConfig:
    """Config for SAE-extended AECS scheduler.

    Inherit from AECSConfig base + SAE-specific knobs."""
    # -- Base LR scheduling (AECS) -----------------------------------------
    base_lr: float = 2e-4
    warmup_steps: int = 1000               # LR warmup duration
    event_warmup_steps: int = 1500         # separate from LR warmup: suppress event detection while loss is still settling. Set >= warmup_steps + slack.
    total_steps: int = 60000
    loss_window: int = 50
    grad_window: int = 50
    instability_z_thresh: float = 10.0
    loss_spike_ratio: float = 1.5          # was 1.15 -- too tight for SAE training where per-batch recon naturally bounces 20-40% in early epochs
    loss_spike_min_recent: int = 50        # window for recent_min; was hardcoded 10, too noisy at this batch size
    plateau_grad_norm_thresh: float = 1e-4
    recovery_lr_factor: float = 0.3
    recovery_momentum_factor: float = 0.5
    recovery_clip_boost: float = 1.5
    recovery_min_steps: int = 100
    recovery_max_steps: int = 1000
    explore_lr_factor: float = 1.5
    explore_noise_std: float = 1e-5
    event_persistence: int = 5
    cooldown_steps: int = 200
    reentry_grad_norm_tol: float = 0.1
    verbose: bool = True
    mode_verbose: bool = True  # print mode transitions

    # -- L0 PI controller --------------------------------------------------
    target_l0: float = 32.0
    l0_tolerance: float = 0.15       # +/-15% of target before reacting
    lambda_l0_init: float = 0.0          # AL: start lambda at 0; integrator builds it up from the constraint violation
    lambda_l0_min: float = 0.0
    lambda_l0_max: float = 1.0           # sanity cap

    # -- Augmented Lagrangian for L0 <= target constraint --------------------
    # Replaces the legacy P-controller. lambda updates per projected dual ascent:
    #     lambda_{t+1} = clip[0, max] ( lambda_t + alpha * (L0_avg - target) )
    # Sparsity loss in the trainer uses the hinge form for one-sided constraint:
    #     L_sparse = lambda * max(0, L0 - target) + (mu/2) * max(0, L0 - target)^2
    # Theory: standard projected-dual-ascent for inequality-constrained optimization.
    # mu provides curvature far from target (so lambda doesn't need to be large yet);
    # lambda provides the steady-state pressure near target (built up by the integrator).
    use_augmented_lagrangian: bool = True
    al_mu: float = 1e-8                  # quadratic penalty coefficient -- curvature
    al_dual_step: float = 5e-9           # dual ascent step size -- integrator gain
    al_log_every: int = 250              # print lambda update at this cadence (every step is too noisy)
    al_convergence_rel_tol: float = 0.005  # lambda "plateau" tolerance: max(recent)/recent[0]-1 < this -> considered converged
    al_landing_zone_rel: float = 0.35    # final-stretch lambda boost zone above target
    al_landing_min_progress: float = 0.08  # L0/step; below this while above band = stalled
    al_landing_gain_max: float = 16.0    # max multiplier on dual gain in final stretch
    al_recovery_gain_max: float = 8.0    # max multiplier when UNDER target, so lambda can
                                         # unwind at a rate comparable to how it wound up
    al_slingshot_overshoot_rel: float = 0.10  # aim below K before releasing lambda pressure
    al_slingshot_gain_max: float = 24.0   # dual gain while L0 is still above K

    # -- Legacy P-controller knobs (kept for use_augmented_lagrangian=False) -
    lambda_adjust_factor: float = 1.0    # multiplicative step per adjustment (1.0 = frozen)
    lambda_adjust_cooldown: int = 9999   # min steps between L0 adjustments
    lambda_freeze_l0_progress: float = 100.0

    # -- EV floor protection -----------------------------------------------
    ev_floor: float = 0.97
    ev_floor_patience: int = 3        # consecutive log-windows below floor -> RECOVERY
    ev_drop_thresh: float = -0.005    # log-window EV delta that triggers immediate RECOVERY
    ev_stop_thresh: float = 0.97
    ev_stop_patience: int = 5
    ev_check_every: int = 500         # only push EV / check floor at this step interval (= LOG_EVERY); decouples patience from training-step rate

    # -- Dead feature emergency --------------------------------------------
    dead_emergency_thresh: float = 15.0   # % dead -> force STABILIZE
    dead_emergency_cooldown: int = 5000   # min steps between dead emergency triggers
    dead_emergency_resample_trigger: bool = True  # auto-trigger resample on emergency

    # -- Revival AuxK policy -----------------------------------------------
    # "legacy"          -> max(8, min(128, target_l0 // 2))  (current behavior, default)
    # "feature_fraction"-> min(512, n_features // 64)        (new; A/B only)
    aux_k_policy: str = "legacy"

    # -- L0 stabilization detection ----------------------------------------
    l0_stabilize_window: int = 3          # consecutive windows to detect stabilization
    l0_stabilize_std_thresh: float = 0.5  # std of L0 window below this -> stabilize

    # -- Preemptive LR energy ------------------------------------------------
    # These multipliers are coarse steering only. They are locked out once L0 is
    # near/below target so late convergence remains the lambda integrator's job.
    initial_lr_multiplier: float = 1.0
    initial_lr_decay_steps: int = 3000
    activation_norm_ref: Optional[float] = None
    activation_norm_lr_alpha: float = 0.5
    activation_norm_lr_min: float = 0.85
    activation_norm_lr_max: float = 1.45
    lr_energy_max: float = 2.0
    convergence_lockout_rel: float = 0.25
    constraint_lr_floor: float = 0.12    # min LR fraction while L0 is still above target

    # -- Early pulse ---------------------------------------------------------
    early_pulse_steps: int = 400
    early_pulse_multiplier: float = 1.0
    early_pulse_warmup_floor: float = 0.50
    early_pulse_dampen: float = 0.5

    # -- L0 momentum stall pulses ------------------------------------------
    stall_pulse_enabled: bool = True
    stall_warmup_steps: int = 750
    stall_cooldown_steps: int = 500
    stall_pulse_steps: int = 150
    stall_pulse_max_extra: float = 0.45
    stall_pulse_min_multiplier: float = 1.08
    stall_fast_alpha: float = 0.35
    stall_slow_alpha: float = 0.05
    stall_crossover_ratio: float = 0.35
    stall_min_slow_progress: float = 0.10
    stall_abs_progress_floor: float = 0.05
    stall_dead_suppress_pct: float = 12.0

    # -- Direct threshold nudge (bypasses STE gradient bottleneck) ------------
    # When L0 is far above target, the JumpReLU STE only gives gradient to
    # features whose pre-activations are near the threshold.  Features far
    # above threshold are "stuck on" — the gradient can't reach them.
    # The threshold nudge directly steps log_threshold based on L0 error,
    # bypassing the STE entirely.  This is a proportional controller:
    #   nudge = nudge_gain * (L0 - target) / L0
    # applied to log_threshold of ALL features (not just near-threshold ones).
    # Positive nudge pushes threshold up → fewer features fire → lower L0.
    #
    # The nudge is L0-proportional:
    #   - Gain scales with overshoot: nudge_gain * max(1, overshoot / approach_scale)
    #   - Frequency scales with overshoot: more frequent when L0 >> target
    #   - Symmetric undershoot dampener: when L0 < target, applies a gentle
    #     downward nudge to prevent overshoot from pulling threshold too high.
    #   - Dead band near target: suppress nudge within tolerance to avoid
    #     oscillation with the AL integrator.
    #
    # Interaction guard (Fable): bandwidth floor (0.15) ensures STE gradient
    # stays wide enough to provide a restoring force alongside the nudge.
    # The undershoot dampener provides symmetry — if L0 drops below target,
    # the nudge reverses to lower thresholds, preventing the nudge from
    # overshooting alone.
    threshold_nudge_gain: float = 0.08      # proportional gain (0 = disabled; 0.08 for E4B)
    threshold_nudge_every: int = 50          # apply every N steps (base frequency)
    threshold_nudge_l0_min: float = 550.0   # only nudge when L0 > this (buffer below target)
    # L0-proportional nudge extensions:
    threshold_nudge_overshoot_scale: float = 2.0   # overshoot ratio where gain reaches max
    threshold_nudge_gain_max: float = 0.25          # max effective gain (cap proportional scaling)
    threshold_nudge_freq_overshoot: float = 3.0    # overshoot ratio where frequency doubles
    threshold_nudge_undershoot_gain: float = 0.04   # gain for downward nudge when L0 < target
    threshold_nudge_deadband_rel: float = 0.10      # suppress nudge within +/-10% of target

    # -- W_enc gradient dampening -----------------------------------------------
    # When L0 >> target, W_enc weights adapt to keep features on despite rising
    # sparsity pressure — a classic SAE pathology where the encoder "inflates"
    # to overpower the threshold.  This dampens the W_enc gradient proportional
    # to L0 overshoot, reducing the encoder's ability to fight back while L0
    # converges.  Blunt but effective — revisit if recon degrades.
    wenc_dampen_enabled: bool = True
    wenc_dampen_overshoot_start: float = 2.0   # start dampening at 2x target L0
    wenc_dampen_max_factor: float = 0.5         # max dampening: scale gradient by (1 - this)

    # -- STE bandwidth (gradient channel width) ---------------------------------
    # Default 0.1 matches the JumpReLU paper. Widen (e.g. 0.2-0.3) to give
    # gradient signal to more features near threshold when L0 is stuck.
    # Live-tuneable: write {"ste_bandwidth": 0.3} to live_tune.json.
    ste_bandwidth: float = 0.2

    # -- L0-adaptive STE bandwidth -----------------------------------------------
    # When L0 >> target, widen the STE bandwidth so gradient can reach features
    # that are "stuck on" (far above threshold).  As L0 approaches target,
    # narrow back toward the base.  Floor at ste_bandwidth_floor to prevent
    # the gradient channel from collapsing entirely.
    # Interaction guard (Fable): the bandwidth floor (0.15) prevents the
    # adaptive bandwidth from going so narrow that the threshold nudge (#2)
    # overshoots alone — the STE gradient channel stays wide enough to
    # provide a restoring force.
    ste_adaptive_bandwidth: bool = True          # enable L0-adaptive bandwidth
    ste_bandwidth_floor: float = 0.15           # never go below this
    ste_bandwidth_max: float = 0.5              # cap when L0 is very far above target
    ste_bandwidth_approach_scale: float = 2.0   # overshoot ratio where bandwidth reaches max

    # -- L0-aware lambda warmup --------------------------------------------------
    # In deep layers, L0 can start 4-8x above target. Starting lambda at 0 means
    # the AL integrator has to build up from zero, which wastes thousands of steps
    # while L0 runs away. Seed lambda proportional to the initial overshoot so
    # the integrator has a head start.
    # lambda_warmup = warmup_base + warmup_per_overshoot * max(0, initial_l0/target - 1)
    # Capped at lambda_l0_max. Set warmup_per_overshoot=0.0 to disable.
    lambda_warmup_base: float = 0.0            # base lambda seed (0 = start from zero normally)
    lambda_warmup_per_overshoot: float = 5e-3   # lambda added per unit of overshoot ratio

    # -- Live-tune dials (runtime overrides via JSON file) --------------------
    # Write a JSON file to this path with any of these keys to override at runtime:
    #   lambda_l0_max, al_mu, al_dual_step, target_l0, lambda_l0_override,
    #   ste_bandwidth, threshold_nudge_gain
    # The scheduler picks up changes every live_tune_every steps.
    # Set to None or "" to disable. Set to a path like "live_tune.json" to enable.
    live_tune_path: Optional[str] = None
    live_tune_path_alt: str = "/tmp/live_tune.json"  # fallback: always container-local
    live_tune_every: int = 50        # check for overrides every N steps

    # -- Phase control (DESCENT / PIN / FINETUNE) -----------------------------
    # Strategic phase machine layered on top of AECS disturbance handling:
    #   DESCENT  — drive L0 into the band around target (existing behavior).
    #   PIN      — L0 in band; freeze sparsity pressure (lambda + threshold
    #              nudge) and let EV catch up.
    #   FINETUNE — release lambda gently to settle on the final equilibrium.
    # These knobs are config-only until the corresponding gates are wired in
    # later branches; the legacy ev_stop_thresh/ev_stop_patience above stay
    # authoritative for stopping until FINETUNE is implemented.
    pin_l0_band_abs: float = 0.5        # |L0 - target| <= this -> DESCENT enters PIN
    pin_l0_release_frac: float = 0.25   # |L0 - target| > this*target while PINned -> back to DESCENT
    pin_timeout_steps: int = 2000       # max steps in PIN before bailing back to DESCENT
    pin_ev_thresh: float = 0.95         # EV window counts toward PIN success above this
    pin_ev_patience: int = 3            # consecutive good EV windows -> ready for FINETUNE
    finetune_dual_step: float = 1e-9    # dual ascent step used during FINETUNE
    finetune_lambda_release_frac: float = 0.10  # lambda set to this fraction of pinned on release
    deep_layer_slingshot_gain: float = 8.0      # slingshot gain floor (deep-layer fallback)
    slingshot_norm_alpha: float = -0.5          # exponent for activation-norm slingshot scaling


class SAESignalBuffer:
    """Ring buffer for AECS base signals + SAE-specific signals."""

    def __init__(self, window: int = 50):
        self.losses: deque = deque(maxlen=window)
        self.grad_norms: deque = deque(maxlen=window)
        self.layer_grad_norms: deque = deque(maxlen=window)
        self.l0_values: deque = deque(maxlen=window)
        self.ev_values: deque = deque(maxlen=window)
        self.dead_pcts: deque = deque(maxlen=window)
        self.steps: int = 0

    def push_base(self, loss: float, grad_norm: float, layer_grad_norms=None):
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)
        self.layer_grad_norms.append(layer_grad_norms or [])
        self.steps += 1

    def push_sae(self, l0: float = None, ev: float = None, dead_pct: float = None):
        if l0 is not None:
            self.l0_values.append(l0)
        if ev is not None:
            self.ev_values.append(ev)
        if dead_pct is not None:
            self.dead_pcts.append(dead_pct)

    def loss_min_recent(self, n: int = 10) -> float:
        if len(self.losses) == 0:
            return float('inf')
        return min(list(self.losses)[-n:])

    def grad_norm_ema(self, alpha: float = 0.97) -> Tuple[float, float]:
        if len(self.grad_norms) == 0:
            return 0.0, 1.0
        mu = self.grad_norms[0]
        var = 0.0
        for g in list(self.grad_norms)[1:]:
            mu = alpha * mu + (1 - alpha) * g
            var = alpha * var + (1 - alpha) * (g - mu) ** 2
        return mu, max(math.sqrt(max(var, 1e-8)), 1e-8)

    def grad_norm_zscore(self) -> float:
        if len(self.grad_norms) < 5:
            return 0.0
        mu, sigma = self.grad_norm_ema()
        if sigma < 1e-8:
            return 0.0
        return (self.grad_norms[-1] - mu) / sigma

    def grad_norm_variance(self) -> float:
        if len(self.grad_norms) < 5:
            return 0.0
        vals = list(self.grad_norms)
        mean = sum(vals) / len(vals)
        return sum((x - mean) ** 2 for x in vals) / len(vals)

    def redundancy_score(self) -> float:
        if len(self.grad_cosines) < max(1, self.grad_cosines.maxlen // 2):
            return 0.0
        vals = list(self.grad_cosines)
        return sum(vals) / len(vals)

    def l0_mean(self, window: int = 3) -> float:
        if len(self.l0_values) < window:
            return 0.0
        return sum(list(self.l0_values)[-window:]) / window

    def ev_delta(self) -> float:
        if len(self.ev_values) < 2:
            return 0.0
        vals = list(self.ev_values)
        return vals[-1] - vals[-2]

    def ev_below_floor_count(self, floor: float) -> int:
        """Count consecutive log-windows EV has been below floor."""
        count = 0
        for v in reversed(list(self.ev_values)):
            if v < floor:
                count += 1
            else:
                break
        return count


class SAEEventControlScheduler:
    """
    Dual-loop adaptive scheduler: AECS (LR) + SAE LAMBDA controller.

    The base EventControlScheduler handles LR via 4-mode state machine.
    This class extends it with a second control loop on LAMBDA_L0.

    Usage::

        sae_cfg = SAEAECSConfig(
            target_l0=32.0, ev_floor=0.97, lambda_l0_init=5e-4,
        )
        scheduler = SAEEventControlScheduler(optimizer, sae_cfg)

        for step in range(N_STEPS):
            pre = sae.encode_pre(acts)
            feat_acts = sae.apply_jumprelu(pre)
            x_hat = sae.decode(feat_acts)

            recon_loss = (acts - x_hat).pow(2).mean()
            gate = sae.l0_indicator(pre)
            l0 = gate.sum(dim=-1).mean().item()
            ev = compute_explained_variance(acts, x_hat)
            dead_pct = compute_dead_pct(feat_acts)

            mode = scheduler.step({
                "loss": recon_loss.item(),
                "grad_norm": grad_norm,
                "l0": l0,
                "ev": ev,
                "dead_pct": dead_pct,
            })

            effective_lambda = scheduler.lambda_l0  # read current LAMBDA
            loss = recon_loss + scheduler.lambda_l0 * l0
            ...
    """

    MODES = ["BASELINE", "RECOVERY", "EXPLORE", "STABILIZE"]

    def __init__(self, optimizer, config: SAEAECSConfig = None, mode_label: str = "",
                 layer: Optional[int] = None):
        self.config = config or SAEAECSConfig()
        self.optimizer = optimizer
        self.mode_label = mode_label  # "L0", "L1", etc. for logging
        self.layer = layer            # decoder layer index; drives deep-layer gain scaling

        # Base AECS state
        self.buffer = SAESignalBuffer(
            window=max(self.config.loss_window, self.config.grad_window)
        )
        self.mode: str = "BASELINE"
        self.mode_steps: int = 0
        self.total_steps: int = 0
        self.event_counter: Dict[str, int] = {m: 0 for m in self.MODES}
        self.transition_log: List[Dict] = []
        self.base_lrs: List[float] = [g["lr"] for g in optimizer.param_groups]
        self.base_betas = [g.get("betas", (0.9, 0.999)) for g in optimizer.param_groups]
        self.base_weight_decays = [g.get("weight_decay", 0.0) for g in optimizer.param_groups]
        self._loss_ema: float = 0.0
        self._loss_ema_alpha: float = 0.95

        # SAE LAMBDA controller state
        self.lambda_l0: float = self.config.lambda_l0_init
        self.lambda_adjust_last_step: int = 0

        # EV tracking
        self._ev_below_floor_count: int = 0
        self._ev_above_floor_count: int = 0

        # Dead feature tracking
        self._dead_emergency_last_step: int = 0

        # Early stop -- AL convergence detection
        self.should_stop: bool = False
        self.stop_reason: str = ""
        self._lambda_history: list = []     # rolling lambda readings for plateau detection
        self._activation_norm_ema: Optional[float] = None
        # Frozen preflight activation norm — drives the deterministic slingshot
        # gain scaling. Distinct from the live EMA (which is for LR adaptation).
        self._activation_norm_preflight: Optional[float] = None
        self._prev_l0: Optional[float] = None
        self._l0_progress_fast: Optional[float] = None
        self._l0_progress_slow: Optional[float] = None
        self._stall_pulse_remaining: int = 0
        self._stall_pulse_multiplier: float = 1.0
        self._last_stall_pulse_step: int = -10**9
        self._energy_dampen: float = 1.0

        # Live-tune state
        self._live_tune_mtime: float = 0.0
        self._live_tune_applied: Dict[str, object] = {}

        # Phase control state (DESCENT / PIN / FINETUNE).
        # Observable only at this branch: tracked, summarized, and checkpointed
        # but does not yet gate any actuator.
        self.phase: str = "DESCENT"
        self.phase_step: int = 0           # steps since the current phase was entered
        self.pin_entry_step: Optional[int] = None  # total_steps at PIN entry
        self.pinned_lambda: Optional[float] = None  # lambda captured on PIN entry
        self.pin_ev_count: int = 0         # consecutive good-EV windows seen in PIN
        self.pin_retry_count: int = 0      # times PIN timed out back to DESCENT

    def seed_lambda(self, initial_l0: float):
        """Seed lambda proportional to initial L0 overshoot.

        In deep layers, L0 can start 4-8x above target. Starting lambda at
        0 means the AL integrator wastes thousands of steps building up.
        This seeds lambda based on how far above target L0 is at init, so
        the integrator has a head start.

        Call this once from the preflight probe (before training begins).
        """
        cfg = self.config
        if cfg.lambda_warmup_per_overshoot <= 0:
            return  # disabled
        target = max(cfg.target_l0, 1.0)
        overshoot_ratio = max(0.0, initial_l0 / target - 1.0)  # 0 if at/below target
        seed = cfg.lambda_warmup_base + cfg.lambda_warmup_per_overshoot * overshoot_ratio
        seed = min(seed, cfg.lambda_l0_max)  # cap at max
        if seed > self.lambda_l0:
            old = self.lambda_l0
            self.lambda_l0 = seed
            if cfg.mode_verbose:
                prefix = f"[{self.mode_label}] " if self.mode_label else ""
                print(f"{prefix}[LAMBDA WARMUP] lambda {old:.3e} -> {seed:.3e} "
                      f"(initial_l0={initial_l0:.1f}, target={target:.0f}, "
                      f"overshoot_ratio={overshoot_ratio:.2f}x)")

    # -- Phase machine (DESCENT / PIN / FINETUNE) -----------------------------
    # Observation-only at this branch: transitions and counters are tracked and
    # logged, but no actuator (lambda dual update, threshold nudge, stop) reads
    # self.phase yet. Gating lands in later branches.

    def _l0_in_pin_band(self, current_l0: float) -> bool:
        """True when L0 is within the PIN band (+/- pin_l0_band_abs) of target."""
        return abs(current_l0 - self.config.target_l0) <= self.config.pin_l0_band_abs

    def _l0_escaped_pin_band(self, current_l0: float) -> bool:
        """True when a PINned L0 has drifted far enough that PIN's premise is void.

        PIN freezes the dual on the assumption that L0 is locked at target. A mass
        dead-feature reset can revive thousands of features in one step and blow L0
        far past the band; with the dual frozen there is no restoring force, so the
        run coasts to max_steps at the wrong sparsity (observed: MiniCPM5-1B L13-L15
        finished at L0 ~1670-1717 against target 50). Releasing on escape hands
        lambda back to the integrator.

        The release band is deliberately much wider than the entry band, so normal
        PIN oscillation (observed +/-3 around target) can never trip it. Only a
        genuine blowout does.
        """
        cfg = self.config
        return abs(current_l0 - cfg.target_l0) > cfg.pin_l0_release_frac * cfg.target_l0

    def _enter_phase(self, new_phase: str, reason: str):
        """Transition the phase machine, reset phase_step, capture PIN entry state.

        NOTE: this docstring previously claimed no actuator reads self.phase, so a
        transition was purely cosmetic. That is no longer true and was the cause of
        a real control failure. `_update_dual` early-returns on phase == "PIN" (the
        lambda freeze), so entering and leaving PIN directly gates whether the
        sparsity integrator runs at all. Treat phase transitions as actuator changes.
        """
        if new_phase == self.phase:
            return
        old = self.phase
        self.phase = new_phase
        self.phase_step = 0
        if new_phase == "PIN":
            self.pin_entry_step = self.total_steps
            self.pinned_lambda = self.lambda_l0
            self.pin_ev_count = 0
        prefix = f"[{self.mode_label}] " if self.mode_label else ""
        if new_phase == "FINETUNE" and self.pinned_lambda is not None:
            # Re-pin the ceiling so a stale/lowered lambda_l0_max (e.g. from a live
            # tune during PIN) cannot clip the pinned lambda downward on the first
            # re-enabled dual update after release.
            old_max = self.config.lambda_l0_max
            new_max = max(old_max, self.pinned_lambda)
            if new_max != old_max:
                self.config.lambda_l0_max = new_max
                print(f"{prefix}[PHASE] re-pin lambda_l0_max {old_max:.3e} -> {new_max:.3e} "
                      f"(>= pinned_lambda {self.pinned_lambda:.3e})")
        print(f"{prefix}[PHASE] {old} -> {new_phase} @ step {self.total_steps} "
              f"(lambda={self.lambda_l0:.3e}; {reason})")

    def _maybe_update_phase(self, l0, ev):
        """Phase detection. NOT observation-only: the phase gates the dual.

        - DESCENT -> PIN when L0 enters the band around target.
        - PIN -> DESCENT when L0 escapes the release band. This one IS an actuator
          change: _update_dual early-returns on phase == "PIN", so leaving PIN is
          what hands lambda back to the integrator. See _l0_escaped_pin_band.
        - In PIN, count consecutive EV windows at/above pin_ev_thresh, and LOG
          (but do not act on) FINETUNE readiness and PIN timeout. Actual FINETUNE
          release (Task 6A) and timeout return-to-DESCENT (Task 6C) land later.
        """
        if l0 is None:
            return
        cfg = self.config
        prefix = f"[{self.mode_label}] " if self.mode_label else ""
        if self.phase == "DESCENT":
            if self._l0_in_pin_band(float(l0)):
                self._enter_phase(
                    "PIN",
                    f"L0={l0:.2f} within +/-{cfg.pin_l0_band_abs} of target {cfg.target_l0:.1f}",
                )
        elif self.phase == "PIN":
            # Premise check first: a PINned L0 that has escaped the band means the
            # frozen dual is holding the wrong lambda with no way to correct. Release
            # before anything else in this branch reads pin state.
            if l0 is not None and self._l0_escaped_pin_band(float(l0)):
                self._enter_phase(
                    "DESCENT",
                    f"L0={l0:.1f} escaped PIN release band "
                    f"(+/-{cfg.pin_l0_release_frac * cfg.target_l0:.1f} of "
                    f"{cfg.target_l0:.1f}); releasing frozen dual",
                )
                self.pin_ev_count = 0
                return
            if ev is not None:
                if ev >= cfg.pin_ev_thresh:
                    self.pin_ev_count += 1
                else:
                    self.pin_ev_count = 0
            if self.pin_ev_count >= cfg.pin_ev_patience:
                print(f"{prefix}[PHASE] PIN ready for FINETUNE @ step {self.total_steps} "
                      f"(pin_ev_count={self.pin_ev_count} >= {cfg.pin_ev_patience}) "
                      f"[observe-only: no lambda release yet]")
            if (self.pin_entry_step is not None
                    and self.total_steps - self.pin_entry_step >= cfg.pin_timeout_steps):
                print(f"{prefix}[PHASE] PIN would timeout @ step {self.total_steps} "
                      f"({self.total_steps - self.pin_entry_step} >= {cfg.pin_timeout_steps} steps) "
                      f"[observe-only: no return to DESCENT yet]")

    def step(self, signals: Dict) -> str:
        """Advance scheduler one step.

        Args:
            signals: dict with "loss", "grad_norm" + optional "l0", "ev", "dead_pct".

        Returns:
            Current mode string.
        """
        loss = signals.get("loss", 0.0)
        grad_norm = signals.get("grad_norm", 0.0)
        l0 = signals.get("l0")
        ev = signals.get("ev")
        dead_pct = signals.get("dead_pct")
        activation_norm = signals.get("activation_norm")

        # EMA for loss spike detection
        if self.total_steps == 0:
            self._loss_ema = loss
        else:
            self._loss_ema = (
                self._loss_ema_alpha * self._loss_ema
                + (1 - self._loss_ema_alpha) * loss
            )

        # Push to signal buffers. L0 and dead_pct are high-frequency per-step
        # signals. EV is only pushed at ev_check_every boundaries so the deque
        # holds log-window-rate values -- patience knobs then mean "log windows"
        # not "training steps".
        self.buffer.push_base(loss, grad_norm)
        self.total_steps += 1
        self.mode_steps += 1
        self.phase_step += 1   # steps since current phase entered (reset on _enter_phase)

        # -- Live-tune: pick up runtime parameter overrides from JSON file --
        if self.config.live_tune_path and self.total_steps % max(1, self.config.live_tune_every) == 0:
            self._apply_live_tune()

        ev_check_due = (
            ev is not None
            and self.total_steps % max(1, self.config.ev_check_every) == 0
        )
        ev_for_buffer = ev if ev_check_due else None
        if l0 is not None or ev_for_buffer is not None or dead_pct is not None:
            self.buffer.push_sae(l0=l0, ev=ev_for_buffer, dead_pct=dead_pct)

        if activation_norm is not None:
            self._update_activation_norm(float(activation_norm))
        if l0 is not None:
            self._update_l0_progress(float(l0))
            self._maybe_trigger_stall_pulse(float(l0), dead_pct)

        # -- SAE-specific event detection (uses log-window-rate EV signal) --
        sae_event = self._detect_sae_events(signals) if ev_check_due else None

        # -- Phase machine (observation only; no actuator reads self.phase) --
        if ev_check_due:
            self._maybe_update_phase(l0, ev)

        # -- LAMBDA update: Augmented Lagrangian dual ascent (default) OR legacy P-ctrl --
        if l0 is not None:
            if self.config.use_augmented_lagrangian:
                self._dual_update(l0)
            elif self.total_steps - self.lambda_adjust_last_step >= self.config.lambda_adjust_cooldown:
                self._adjust_lambda(l0)

        # -- EV floor / early stop checks -- only at log-window boundaries ---
        if ev_check_due:
            self._check_ev_floor(ev)
            self._check_early_stop(ev)

        # -- Dead feature emergency -------------------------------------
        if dead_pct is not None and sae_event is None:
            sae_event = self._check_dead_emergency(dead_pct)

        # -- Maybe trigger mode transition ------------------------------
        base_event = self._detect_base_event(signals)
        final_event = sae_event or base_event
        if final_event:
            self._maybe_transition(final_event)
            if final_event in ("GRADIENT_SPIKE", "LOSS_SPIKE", "UNSTABLE", "DEAD_EMERGENCY"):
                self._dampen_energy(final_event)

        # -- Compute LR modulation --------------------------------------
        lrs = self._compute_lrs()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

        # -- Apply mode tweaks (betas, weight_decay) --------------------
        self._apply_mode_tweaks()

        return self.mode

    # -- L0 constraint via Augmented Lagrangian --------------------------------

    def _dual_update(self, current_l0: float):
        """Projected dual ascent for the inequality constraint L0_avg <= target_L0.

            lambda_{t+1} = clip[lambda_min, lambda_max] ( lambda_t + alpha * (L0_avg - target_L0) )

        This is the textbook Augmented Lagrangian dual update for inequality
        constraints. lambda is an INTEGRAL of the constraint violation: it grows
        monotonically while L0 > target, plateaus when L0 ~= target, and can
        relax (down to lambda_min) if L0 dips below target.

        The corresponding primal sparsity loss (computed in the trainer) is the
        hinge form so we don't penalize being too sparse:

            L_sparse = lambda * max(0, L0 - target) + (mu/2) * max(0, L0 - target)^2

        Together these form the AL method for the constrained problem
            min recon  s.t.  L0_avg <= target_L0
        which has proper convergence theory (vs the ad-hoc P-controller).
        """
        cfg = self.config
        # PIN freezes sparsity pressure: lambda is pinned at PIN entry and the
        # phase machine owns it, so the integrator must not move it. This early
        # return is the single guarantee against lambda windup during PIN.
        if self.phase == "PIN":
            if cfg.verbose and self.total_steps % max(1, cfg.al_log_every) == 0:
                print(f"  [AL @ step {self.total_steps}] PIN: dual update frozen "
                      f"(lambda={self.lambda_l0:.3e})")
            return
        error = current_l0 - cfg.target_l0   # signed (negative means we're below target)
        control_target = self._dual_control_target(current_l0)
        control_error = current_l0 - control_target
        gain = self._landing_lambda_gain(current_l0, control_error)
        new_lambda = self.lambda_l0 + cfg.al_dual_step * gain * control_error
        # Project onto [lambda_min, lambda_max]
        new_lambda = max(cfg.lambda_l0_min, min(cfg.lambda_l0_max, new_lambda))
        self.lambda_l0 = new_lambda
        self.lambda_adjust_last_step = self.total_steps

        # Throttle logging -- every step is too noisy
        if cfg.verbose and self.total_steps % cfg.al_log_every == 0:
            print(
                f"  [AL @ step {self.total_steps}] lambda={self.lambda_l0:.3e} "
                f"(L0={current_l0:.1f}, target={cfg.target_l0}, error={error:+.1f}, "
                f"control_target={control_target:.1f}, gain={gain:.1f}x)"
            )

    def _dual_control_target(self, current_l0: float) -> float:
        cfg = self.config
        if current_l0 > cfg.target_l0:
            return cfg.target_l0 * (1.0 - cfg.al_slingshot_overshoot_rel)
        return cfg.target_l0

    def _effective_slingshot_gain(self) -> float:
        """Slingshot dual-gain, scaled by frozen preflight activation norm.

        Plant-aware: deeper layers have larger activation norms and overshoot
        lambda under the early-layer slingshot. Scale the gain down by the
        preflight norm ratio (probe/ref) ** slingshot_norm_alpha, with the floor
        derived from deep_layer_slingshot_gain so the curve asymptotes to the
        validated deep-layer value. Uses the FROZEN preflight norm (deterministic),
        never the live EMA. Falls back to the layer-number rule when preflight
        stats are missing.
        """
        cfg = self.config
        max_gain = cfg.al_slingshot_gain_max
        floor_gain = min(max_gain, cfg.deep_layer_slingshot_gain)

        ref = cfg.activation_norm_ref
        probe = self._activation_norm_preflight

        if ref is not None and ref > 0 and probe is not None and probe > 0:
            floor_scale = floor_gain / max(max_gain, 1e-12)
            ratio = probe / ref
            scale = max(floor_scale, min(1.0, ratio ** cfg.slingshot_norm_alpha))
            return max_gain * scale

        if self.layer is not None and self.layer >= 3:
            return floor_gain
        return max_gain

    def _landing_lambda_gain(self, current_l0: float, control_error: float) -> float:
        cfg = self.config
        if control_error <= 0:
            # Recovery gain. lambda winds UP through the slingshot branch below at up
            # to _effective_slingshot_gain() (24x on early layers) but historically
            # unwound at a flat 1.0x, so a warmup overshoot took thousands of steps to
            # give back. Scale the downward gain with how far under target we are,
            # capped well below the slingshot so this can never outrun it. Near target
            # this is ~1.0x, leaving endgame convergence behaviour untouched.
            target = max(cfg.target_l0, 1e-8)
            under_rel = min(1.0, -control_error / target)     # 0 at target, 1 at L0=0
            return 1.0 + (cfg.al_recovery_gain_max - 1.0) * under_rel
        if current_l0 > cfg.target_l0:
            return self._effective_slingshot_gain()
        target = max(cfg.target_l0, 1e-8)
        error_rel = control_error / target
        if error_rel <= cfg.l0_tolerance:
            return 1.0
        if error_rel > cfg.al_landing_zone_rel:
            return 1.0

        progress = self._l0_progress_fast
        if progress is not None and progress >= cfg.al_landing_min_progress:
            return 1.0

        span = max(cfg.al_landing_zone_rel - cfg.l0_tolerance, 1e-8)
        zone_frac = min(1.0, max(0.0, (error_rel - cfg.l0_tolerance) / span))
        stall_frac = 1.0
        if progress is not None:
            stall_frac = min(1.0, max(0.0, (cfg.al_landing_min_progress - progress) / cfg.al_landing_min_progress))
        return 1.0 + (cfg.al_landing_gain_max - 1.0) * max(zone_frac, stall_frac)

    # -- L0-adaptive STE bandwidth -----------------------------------------------

    def adaptive_ste_bandwidth(self, current_l0: float) -> float:
        """Compute adaptive STE bandwidth based on L0 overshoot.

        When L0 >> target, widen bandwidth so gradient can reach features
        that are far above threshold (the "stuck on" problem).  As L0
        approaches target, narrow back toward the base.  Floor at
        ste_bandwidth_floor to keep the gradient channel open.

        Returns the adaptive bandwidth; caller writes it to sae_cfg.ste_bandwidth.
        """
        cfg = self.config
        if not cfg.ste_adaptive_bandwidth:
            return cfg.ste_bandwidth

        target = max(cfg.target_l0, 1.0)
        overshoot = current_l0 / target  # e.g. 4.0 means L0 is 4x target

        # Map overshoot to a bandwidth scale in [0, 1].
        # At overshoot = 1.0 (L0 = target), scale = 0 → bandwidth = floor.
        # At overshoot = ste_bandwidth_approach_scale, scale = 1 → bandwidth = max.
        # Between, linearly interpolate.
        scale = max(0.0, min(1.0, (overshoot - 1.0) / max(cfg.ste_bandwidth_approach_scale - 1.0, 0.01)))
        bw = cfg.ste_bandwidth_floor + (cfg.ste_bandwidth_max - cfg.ste_bandwidth_floor) * scale
        return bw

    # -- L0-proportional threshold nudge ------------------------------------------

    def compute_threshold_nudge(self, current_l0: float, step: int) -> Tuple[float, bool]:
        """Compute threshold nudge value and whether to apply it this step.

        Returns (nudge_value, should_apply).
        Positive nudge → push threshold up (reduce L0 when above target).
        Negative nudge → push threshold down (increase L0 when below target).
        Zero → dead band near target, don't nudge.

        The nudge is proportional to overshoot:
          - Gain scales with L0 overshoot ratio, capped at gain_max.
          - Frequency increases with overshoot (every n steps, faster when far).
          - Symmetric undershoot dampener: gentle downward nudge when L0 < target.
          - Dead band: suppress nudge within tolerance of target.
        """
        cfg = self.config
        # PIN freezes direct threshold manipulation while EV catches up — the
        # phase machine owns the actuators here. This gate is PIN-only: the nudge
        # stays fully available in DESCENT and FINETUNE and is NOT disabled globally.
        if self.phase == "PIN":
            return 0.0, False
        if cfg.threshold_nudge_gain <= 0:
            return 0.0, False

        target = max(cfg.target_l0, 1.0)
        overshoot = current_l0 / target  # >1 when above target, <1 when below

        # Dead band: suppress nudge within tolerance of target.
        # This prevents the nudge from fighting the AL integrator near convergence.
        within_deadband = abs(current_l0 - target) / target <= cfg.threshold_nudge_deadband_rel
        if within_deadband:
            return 0.0, False

        # Overshoot nudge (L0 > target): scale gain with overshoot ratio
        if current_l0 > target:
            # Scale gain: gain * max(1, overshoot / approach_scale), capped at gain_max
            scale = max(1.0, overshoot / max(cfg.threshold_nudge_overshoot_scale, 0.01))
            effective_gain = min(cfg.threshold_nudge_gain * scale, cfg.threshold_nudge_gain_max)
            rel_error = (current_l0 - target) / max(current_l0, 1.0)
            nudge = effective_gain * rel_error

            # Scale frequency: apply more often when overshoot is large.
            # Base: every threshold_nudge_every steps. At overshoot > freq_overshoot,
            # halve the interval (apply twice as often).
            freq = cfg.threshold_nudge_every
            if overshoot > cfg.threshold_nudge_freq_overshoot:
                freq = max(1, freq // 2)
            should_apply = (step % freq == 0)

            # Safety net, expressed RELATIVE to target. The old form compared L0 and
            # target against threshold_nudge_l0_min=550, an absolute count tuned for
            # K=500. At K=50 both sides are permanently under 550, so this branch was
            # switched off for the whole run -- the nudge could only ever push L0 UP,
            # never down, while lambda pushed down continuously. A one-sided pulse
            # actuator against an integrator is a limit cycle, which is the shakiness
            # around target. Relative form behaves identically at K=500 and K=50.
            if (current_l0 - target) / target < cfg.threshold_nudge_deadband_rel:
                should_apply = False

            return nudge, should_apply

        # Undershoot dampener (L0 < target): gentle downward nudge to reverse
        # any threshold overshoot and let L0 recover.  This is the symmetric
        # counterpart — without it, the nudge can overshoot alone because
        # there's no restoring force pulling thresholds back down.
        # Mirror the overshoot branch so severity actually reaches the actuator.
        # The old law was `-gain * (1 - L0/target)`. Since `1 - L0/target` -> 1.0 as
        # L0 -> 0, the pull-up was hard-capped at `undershoot_gain` (0.04) no matter
        # how far under target the layer sat, while the push-down side scales to
        # gain_max (0.25). A layer that starts at L0=2.6 against target 50 therefore
        # crawled back at the same rate as one sitting at 45.
        undershoot = 1.0 - overshoot                       # 0..1, ->1 as L0 -> 0
        severity = 1.0 / max(overshoot, 1e-3)              # target/L0, unbounded
        scale = max(1.0, severity / max(cfg.threshold_nudge_overshoot_scale, 0.01))
        effective_gain = min(cfg.threshold_nudge_undershoot_gain * scale,
                             cfg.threshold_nudge_gain_max)
        nudge = -effective_gain * undershoot
        # Escalate frequency when far under, matching the overshoot path.
        freq = cfg.threshold_nudge_every
        if severity > cfg.threshold_nudge_freq_overshoot:
            freq = max(1, freq // 2)
        should_apply = (step % freq == 0)
        return nudge, should_apply

    # -- W_enc gradient dampening ------------------------------------------------

    def wenc_dampen_factor(self, current_l0: float) -> float:
        """Compute W_enc gradient dampening factor based on L0 overshoot.

        Returns a multiplier in [1.0 - wenc_dampen_max_factor, 1.0].
        1.0 = no dampening (L0 at or below target).
        0.5 = halve the W_enc gradient (L0 very far above target).

        The dampening scales linearly from 1.0 at wenc_dampen_overshoot_start
        to (1 - wenc_dampen_max_factor) at 2x overshhoot_start.
        """
        cfg = self.config
        if not cfg.wenc_dampen_enabled:
            return 1.0

        target = max(cfg.target_l0, 1.0)
        overshoot = current_l0 / target

        if overshoot <= cfg.wenc_dampen_overshoot_start:
            return 1.0

        # Scale dampening linearly from 0 to max between overshoot_start and 2*overshoot_start
        scale = min(1.0, (overshoot - cfg.wenc_dampen_overshoot_start) / max(cfg.wenc_dampen_overshoot_start, 0.01))
        dampen = 1.0 - cfg.wenc_dampen_max_factor * scale
        return max(1.0 - cfg.wenc_dampen_max_factor, dampen)

    # -- L0 P-Controller (legacy -- kept for use_augmented_lagrangian=False) ----

    def _adjust_lambda(self, current_l0: float):
        """P-controller adjustment of LAMBDA_L0 with ANTI-WINDUP.

        Sign convention:
          error = l0_mean - target
          error > tol  -> L0 ABOVE target -> INCREASE lambda to add sparsity pressure
          error < -tol -> L0 BELOW target -> DECREASE lambda to relieve sparsity pressure

        Anti-windup rule:
          If L0 is already moving toward target faster than `min_progress_per_window`,
          freeze lambda. Pushing the actuator while the system is responding causes overshoot
          and adversarial W_enc growth (encoder weights inflate to keep features above
          a rising threshold -- a classic SAE pathology).
        """
        target = self.config.target_l0
        tol = target * self.config.l0_tolerance
        l0_mean = self.buffer.l0_mean(window=3)
        error = l0_mean - target

        # Anti-windup: compute L0 trajectory over last 3 windows
        l0_trend = 0.0
        if len(self.buffer.l0_values) >= 3:
            recent = list(self.buffer.l0_values)[-3:]
            l0_trend = recent[-1] - recent[0]  # negative = decreasing
        min_progress = self.config.lambda_freeze_l0_progress  # default 5% of target

        # If above target AND already decreasing fast enough -> freeze lambda (system is responding)
        if error > tol and l0_trend < -max(min_progress, target * 0.5):
            adjust = 1.0
            reason = "frozen (L0 decreasing)"
        # If above target AND L0 is rising despite high lambda -> adversarial W_enc growth -- STOP pushing
        elif error > tol and l0_trend > target * 0.5 and self.lambda_l0 > self.config.lambda_l0_init * 5:
            adjust = 1.0
            reason = "frozen (L0 rising at high lambda -- adversarial)"
        # Below target -> pull lambda back
        elif error < -tol:
            adjust = 1.0 / self.config.lambda_adjust_factor
            reason = "decrease"
        # Above target and not moving toward it -> push lambda
        elif error > tol:
            adjust = self.config.lambda_adjust_factor
            reason = "increase"
        else:
            adjust = 1.0
            reason = "in target"

        self.lambda_l0 *= adjust
        self.lambda_l0 = max(self.config.lambda_l0_min, min(self.config.lambda_l0_max, self.lambda_l0))
        self.lambda_adjust_last_step = self.total_steps

        if self.config.verbose:
            print(
                f"  [LAMBDA @ step {self.total_steps}] {self.lambda_l0:.2e} "
                f"(l0={l0_mean:.1f}, target={target}, error={error:.1f}, "
                f"trend={l0_trend:+.0f}, {reason})"
            )

    # -- EV Floor Protection ----------------------------------------------------

    def _check_ev_floor(self, ev: float):
        """Track consecutive windows EV is below floor."""
        if ev < self.config.ev_floor:
            self._ev_below_floor_count += 1
        else:
            self._ev_below_floor_count = 0

    def _check_early_stop(self, ev: float):
        """Early stop when the Augmented Lagrangian has CONVERGED.

        Convergence here means BOTH:
          - L0 within tolerance of target (the constraint is satisfied)
          - lambda has plateaued (the integrator has found its KKT-point value)

        We measure lambda plateau as: relative growth of lambda over the last 4 log-window
        readings is below `al_convergence_rel_tol`. EV is no longer the stop signal --
        EV may peak before L0 hits target and then decline as AL squeezes harder;
        waiting for high EV after convergence just burns compute.
        """
        l0_mean = self.buffer.l0_mean(window=3)
        l0_in_target = (
            l0_mean > 0
            and self.config.target_l0 * (1.0 - self.config.al_slingshot_overshoot_rel)
                <= l0_mean
                <= self.config.target_l0
        )

        # Track lambda history (separate from the buffer, since buffer fields are signals not state)
        self._lambda_history.append(self.lambda_l0)
        if len(self._lambda_history) > 8:
            self._lambda_history.pop(0)

        lambda_converged = False
        if len(self._lambda_history) >= 4:
            recent = self._lambda_history[-4:]
            if recent[0] > 1e-12:
                rel_growth = max(recent) / recent[0] - 1.0
                lambda_converged = rel_growth < self.config.al_convergence_rel_tol

        if l0_in_target and lambda_converged:
            self._ev_above_floor_count += 1
        else:
            self._ev_above_floor_count = 0

        if self._ev_above_floor_count >= self.config.ev_stop_patience:
            # PIN stop gate (Branch 6 / Task 6B): AL convergence alone must not
            # end the run the moment sparsity locks -- PIN exists to let EV catch
            # up with lambda frozen. Hold the stop until PIN reports EV-ready
            # (pin_ev_count windows at/above pin_ev_thresh) or PIN times out.
            # A run that converges without ever entering PIN keeps training
            # (bounded by max_steps) rather than stopping mid-DESCENT.
            pin_ev_ready = self.pin_ev_count >= self.config.pin_ev_patience
            pin_elapsed = (self.total_steps - self.pin_entry_step
                           if self.pin_entry_step is not None else 0)
            pin_timed_out = (self.pin_entry_step is not None
                             and pin_elapsed >= self.config.pin_timeout_steps)
            if self.phase != "PIN" or not (pin_ev_ready or pin_timed_out):
                return
            pin_note = ("PIN EV-ready" if pin_ev_ready
                        else f"PIN timeout after {pin_elapsed} steps")
            self.should_stop = True
            self.stop_reason = (
                f"AL converged: L0 {l0_mean:.1f} in target +/-{self.config.l0_tolerance*100:.0f}% of "
                f"{self.config.target_l0}, lambda plateaued at {self.lambda_l0:.3e} "
                f"({self.config.ev_stop_patience} windows of stability); {pin_note}. "
                f"Final EV={ev:.3f}."
            )

    # -- Dead Feature Emergency --------------------------------------------------

    def _check_dead_emergency(self, dead_pct: float) -> Optional[str]:
        """If dead_pct exceeds emergency threshold, trigger STABILIZE."""
        if dead_pct > self.config.dead_emergency_thresh:
            if self.total_steps - self._dead_emergency_last_step >= self.config.dead_emergency_cooldown:
                self._dead_emergency_last_step = self.total_steps
                if self.config.mode_verbose:
                    print(f"[SAE-ACS] Dead feature emergency: {dead_pct:.1f}% dead "
                          f"-> STABILIZE + resample")
                if self.config.dead_emergency_resample_trigger:
                    return "DEAD_EMERGENCY"
        return None

    # -- Base AECS event detection ----------------------------------------------

    def _detect_base_event(self, signals) -> Optional[str]:
        """Detect events from the base AECS signal buffer."""
        buf = self.buffer
        cfg = self.config

        # Suppress event detection during a settling window AFTER LR warmup.
        # event_warmup_steps is decoupled from LR warmup_steps -- at LR-warmup boundary
        # the loss is still volatile (the optimizer is finding its footing) and a
        # 15-50% per-batch bounce is normal SAE training behavior, not a real spike.
        if self.total_steps < cfg.event_warmup_steps:
            return None

        if buf.steps < cfg.event_persistence + 5:
            return None

        if buf.steps < 5:
            return None

        if buf.grad_norm_zscore() > cfg.instability_z_thresh:
            return "GRADIENT_SPIKE"

        # Use a longer window for recent_min to reduce noise sensitivity.
        recent_min = buf.loss_min_recent(n=cfg.loss_spike_min_recent)
        if recent_min > 0 and buf.losses[-1] > recent_min * cfg.loss_spike_ratio:
            return "LOSS_SPIKE"

        if buf.grad_norm_variance() > cfg.reentry_grad_norm_tol:
            return "UNSTABLE"

        if len(buf.grad_norms) >= 10:
            recent_avg = sum(list(buf.grad_norms)[-10:]) / 10
            if recent_avg < cfg.plateau_grad_norm_thresh:
                return "PLATEAU"

        return None

    # -- SAE-specific event detection -------------------------------------------

    def _detect_sae_events(self, signals: Dict) -> Optional[str]:
        """Detect events from SAE-specific signals. Overrides base AECS decisions."""
        signals.get("l0")
        ev = signals.get("ev")

        # Same warmup gate as _detect_base_event: during sparsity calibration the
        # EV naturally drops 3-10% per log window as JumpReLU thresholds tighten
        # and features drop out of the active set. That's not pathology, it's the
        # main training signal. Suppress until the SAE has settled.
        if self.total_steps < self.config.event_warmup_steps:
            return None

        # EV drop threshold -> immediate RECOVERY
        if ev is not None and self.buffer.ev_delta() < self.config.ev_drop_thresh:
            if self.config.mode_verbose:
                print(f"[SAE-ACS] EV drop detected: {self.buffer.ev_delta():.4f} -> RECOVERY")
            return "LOSS_SPIKE"  # maps to RECOVERY

        # EV floor patience exceeded -> RECOVERY
        ev_below = self.buffer.ev_below_floor_count(self.config.ev_floor)
        if ev_below >= self.config.ev_floor_patience:
            if self.config.mode_verbose:
                print(f"[SAE-ACS] EV {ev_below} consecutive windows below {self.config.ev_floor} -> RECOVERY")
            return "LOSS_SPIKE"

        # EV stabilization -> suggest BASELINE exit (handled by _maybe_transition)
        return None

    # -- Mode transition logic --------------------------------------------------

    def _maybe_transition(self, event: str):
        """Apply hysteresis + cooldown before transitioning modes."""
        cfg = self.config

        if self.mode_steps < cfg.cooldown_steps and self.mode != "BASELINE":
            return

        if self.mode == "RECOVERY":
            if self.mode_steps < cfg.recovery_min_steps:
                return
            if self.mode_steps >= cfg.recovery_max_steps:
                self._enter_mode("BASELINE", "recovery_max_duration")
                return

        # Map event to target mode
        event_mode_map = {
            "GRADIENT_SPIKE": "RECOVERY",
            "LOSS_SPIKE": "RECOVERY",
            "UNSTABLE": "STABILIZE",
            "PLATEAU": "EXPLORE",
            "DEAD_EMERGENCY": "STABILIZE",
        }
        target = event_mode_map.get(event, "BASELINE")

        if target == self.mode:
            return

        # REENTRY: require stable grad norm before returning to BASELINE
        if target == "BASELINE" and self.mode in ("RECOVERY", "STABILIZE"):
            if self.buffer.grad_norm_variance() > cfg.reentry_grad_norm_tol:
                return

        self._enter_mode(target, event)

    def _enter_mode(self, new_mode: str, cause: str):
        old_mode = self.mode
        self.mode = new_mode
        self.mode_steps = 0
        self.event_counter[new_mode] += 1
        self.transition_log.append({
            "step": self.total_steps,
            "from": old_mode,
            "to": new_mode,
            "cause": cause,
            "lr": self.optimizer.param_groups[0]["lr"],
            "lambda_l0": self.lambda_l0,
        })
        if self.config.mode_verbose:
            prefix = f"[{self.mode_label}] " if self.mode_label else ""
            print(f"{prefix}[AECS] Step {self.total_steps}: {old_mode} -> {new_mode} ({cause})")

    # -- LR computation ---------------------------------------------------------

    def _update_activation_norm(self, activation_norm: float):
        if activation_norm <= 0:
            return
        if self._activation_norm_ema is None:
            self._activation_norm_ema = activation_norm
        else:
            self._activation_norm_ema = 0.95 * self._activation_norm_ema + 0.05 * activation_norm

    def _update_l0_progress(self, current_l0: float):
        if self._prev_l0 is None:
            self._prev_l0 = current_l0
            return
        # Positive progress means L0 is moving down toward the target from above.
        progress = self._prev_l0 - current_l0
        self._prev_l0 = current_l0
        if self._l0_progress_fast is None:
            self._l0_progress_fast = progress
            self._l0_progress_slow = progress
            return
        cfg = self.config
        self._l0_progress_fast = (
            cfg.stall_fast_alpha * progress
            + (1.0 - cfg.stall_fast_alpha) * self._l0_progress_fast
        )
        self._l0_progress_slow = (
            cfg.stall_slow_alpha * progress
            + (1.0 - cfg.stall_slow_alpha) * self._l0_progress_slow
        )

    def _convergence_locked(self, l0: Optional[float] = None) -> bool:
        if l0 is None:
            if self.buffer.l0_values:
                l0 = self.buffer.l0_values[-1]
            else:
                l0 = self.buffer.l0_mean(window=3)
        if l0 <= 0:
            return False
        target = self.config.target_l0
        return l0 <= target * (1.0 + self.config.convergence_lockout_rel)

    def _maybe_trigger_stall_pulse(self, current_l0: float, dead_pct: Optional[float]):
        cfg = self.config
        if not cfg.stall_pulse_enabled:
            return
        if self.total_steps < cfg.stall_warmup_steps:
            return
        if self._convergence_locked(current_l0):
            self._stall_pulse_remaining = 0
            self._stall_pulse_multiplier = 1.0
            return
        if self._stall_pulse_remaining > 0:
            return
        if self.total_steps - self._last_stall_pulse_step < cfg.stall_cooldown_steps:
            return
        if dead_pct is not None and dead_pct >= cfg.stall_dead_suppress_pct:
            return
        if self._l0_progress_fast is None or self._l0_progress_slow is None:
            return

        fast = self._l0_progress_fast
        slow = self._l0_progress_slow
        structural_stall = (
            slow > cfg.stall_min_slow_progress
            and fast < slow * cfg.stall_crossover_ratio
        )
        flat_stall = abs(fast) < cfg.stall_abs_progress_floor and abs(slow) < cfg.stall_abs_progress_floor
        if not (structural_stall or flat_stall):
            return

        distance = max(0.0, current_l0 - cfg.target_l0)
        distance_frac = min(1.0, distance / max(cfg.target_l0 * 3.0, 1.0))
        extra = cfg.stall_pulse_max_extra * distance_frac
        pulse = max(cfg.stall_pulse_min_multiplier, 1.0 + extra)
        self._stall_pulse_multiplier = min(pulse, cfg.lr_energy_max)
        self._stall_pulse_remaining = cfg.stall_pulse_steps
        self._last_stall_pulse_step = self.total_steps

        if cfg.mode_verbose:
            prefix = f"[{self.mode_label}] " if self.mode_label else ""
            print(
                f"{prefix}[SAE-ACS] L0 stall pulse x{self._stall_pulse_multiplier:.2f} "
                f"for {cfg.stall_pulse_steps} steps "
                f"(L0={current_l0:.1f}, fast={fast:.3f}, slow={slow:.3f})"
            )

    def _dampen_energy(self, cause: str):
        cfg = self.config
        old = self._energy_dampen
        self._energy_dampen = max(0.25, self._energy_dampen * cfg.early_pulse_dampen)
        self._stall_pulse_remaining = 0
        self._stall_pulse_multiplier = 1.0
        if cfg.mode_verbose and old != self._energy_dampen:
            prefix = f"[{self.mode_label}] " if self.mode_label else ""
            print(f"{prefix}[SAE-ACS] energy dampened x{self._energy_dampen:.2f} ({cause})")

    # -- Live-tune: runtime parameter overrides from JSON file -----------------

    # Tunable knobs and their types. Write any subset to live_tune.json:
    #   {"lambda_l0_max": 0.1, "al_mu": 5e-5, "target_l0": 300, ...}
    # Special key "lambda_l0_override" directly sets lambda (nuclear option).
    # Delete the file or set keys to null to revert to config defaults.
    _LIVE_TUNE_KEYS = {
        "lambda_l0_max": float,
        "lambda_l0_min": float,
        "al_mu": float,
        "al_dual_step": float,
        "al_slingshot_gain_max": float,
        "al_landing_zone_rel": float,
        "al_landing_min_progress": float,
        "al_landing_gain_max": float,
        "al_slingshot_overshoot_rel": float,
        "target_l0": float,
        "l0_tolerance": float,
        "constraint_lr_floor": float,
        "lambda_l0_override": float,  # directly set lambda, bypasses integrator
        "ste_bandwidth": float,        # widen STE gradient channel (default 0.1)
        "ste_adaptive_bandwidth": bool, # enable/disable L0-adaptive bandwidth
        "ste_bandwidth_floor": float,  # min adaptive bandwidth
        "ste_bandwidth_max": float,    # max adaptive bandwidth
        "threshold_nudge_gain": float, # direct threshold stepping gain (0 = disabled)
        "pin_ev_thresh": float,        # EV level PIN needs before early stop may fire
        "pin_ev_patience": float,      # consecutive good EV windows for PIN EV-ready
        "pin_timeout_steps": float,    # max PIN polish steps before stop is allowed
        "pin_l0_band_abs": float,      # |L0-target| band for DESCENT -> PIN entry
    }

    def _apply_live_tune(self):
        """Check live_tune.json for runtime parameter overrides.

        Checks TWO paths: live_tune_path (Volume, may lag) and
        live_tune_path_alt (/tmp, always container-local). The alt path
        takes priority if both exist. Picks up changes by comparing mtime;
        only re-reads when the file changes. Prints a summary of overrides.
        """
        cfg = self.config
        paths = [p for p in [cfg.live_tune_path, getattr(cfg, 'live_tune_path_alt', None)]
                 if p]
        if not paths:
            return
        # Find the most recently modified file that exists
        best_p = None
        best_mtime = 0
        for path in paths:
            p = Path(path)
            try:
                if p.exists():
                    mt = p.stat().st_mtime
                    if mt > best_mtime:
                        best_mtime = mt
                        best_p = p
            except OSError:
                continue
        if best_p is None:
            # Neither file exists — revert all overrides to config defaults
            if self._live_tune_applied:
                for key in list(self._live_tune_applied):
                    setattr(cfg, key, getattr(cfg, key))
                self._live_tune_applied = {}
            return
        if best_mtime == self._live_tune_mtime:
            return  # no change
        self._live_tune_mtime = best_mtime
        try:
            text = best_p.read_text()
            if not text.strip():
                return
            overrides = json.loads(text)
        except (json.JSONDecodeError, OSError) as e:
            if cfg.verbose and self.total_steps % 500 == 0:
                print(f"  [live-tune] WARNING: failed to read {best_p}: {e}")
            return
        applied = {}
        for key, expected_type in self._LIVE_TUNE_KEYS.items():
            if key not in overrides:
                continue
            val = overrides[key]
            if val is None:
                # Revert to default — remove from applied, don't set
                if key in self._live_tune_applied:
                    self._live_tune_applied.pop(key, None)
                continue
            try:
                val = expected_type(val)
            except (ValueError, TypeError):
                continue
            if key == "lambda_l0_override":
                if self.phase == "PIN":
                    # PIN owns lambda; an external hot-tune file must not silently
                    # steal lambda ownership while sparsity pressure is frozen.
                    if cfg.verbose:
                        prefix = f"[{self.mode_label}] " if self.mode_label else ""
                        print(f"  {prefix}[live-tune @ step {self.total_steps}] IGNORED "
                              f"lambda_l0_override={val:.3e} during PIN "
                              f"(lambda pinned at {self.lambda_l0:.3e})")
                    continue
                # Directly set lambda, bypass integrator
                self.lambda_l0 = val
                applied[key] = val
                continue
            # Override the config attribute
            getattr(cfg, key, None)
            setattr(cfg, key, val)
            applied[key] = val
            self._live_tune_applied[key] = val

        if applied and cfg.verbose:
            prefix = f"[{self.mode_label}] " if self.mode_label else ""
            parts = [f"{k}={v:.3e}" if isinstance(v, float) and abs(v) < 0.01 else f"{k}={v}"
                      for k, v in applied.items()]
            print(f"  {prefix}[live-tune @ step {self.total_steps}] {', '.join(parts)}")

    def _activation_lr_multiplier(self) -> float:
        cfg = self.config
        if cfg.activation_norm_ref is None or cfg.activation_norm_ref <= 0:
            return 1.0
        if self._activation_norm_ema is None or self._activation_norm_ema <= 0:
            return 1.0
        raw = (self._activation_norm_ema / cfg.activation_norm_ref) ** cfg.activation_norm_lr_alpha
        return max(cfg.activation_norm_lr_min, min(cfg.activation_norm_lr_max, raw))

    def _initial_lr_multiplier(self) -> float:
        cfg = self.config
        if cfg.initial_lr_multiplier <= 1.0:
            return 1.0
        if cfg.initial_lr_decay_steps <= 0:
            return cfg.initial_lr_multiplier
        decay = max(0.0, 1.0 - self.total_steps / cfg.initial_lr_decay_steps)
        return 1.0 + (cfg.initial_lr_multiplier - 1.0) * decay

    def _early_pulse_multiplier(self) -> float:
        cfg = self.config
        if cfg.early_pulse_steps <= 0 or cfg.early_pulse_multiplier <= 1.0:
            return 1.0
        if self.total_steps > cfg.early_pulse_steps:
            return 1.0
        decay = max(0.0, 1.0 - self.total_steps / cfg.early_pulse_steps)
        return 1.0 + (cfg.early_pulse_multiplier - 1.0) * decay

    def _stall_lr_multiplier(self) -> float:
        if self._stall_pulse_remaining <= 0:
            return 1.0
        cfg = self.config
        decay = self._stall_pulse_remaining / max(cfg.stall_pulse_steps, 1)
        self._stall_pulse_remaining -= 1
        return 1.0 + (self._stall_pulse_multiplier - 1.0) * decay

    def _energy_lr_multiplier(self) -> float:
        if self._convergence_locked():
            return 1.0
        mult = (
            self._initial_lr_multiplier()
            * self._activation_lr_multiplier()
            * self._early_pulse_multiplier()
            * self._stall_lr_multiplier()
            * self._energy_dampen
        )
        return max(0.25, min(self.config.lr_energy_max, mult))

    def _constraint_lr_floor(self) -> float:
        cfg = self.config
        if not self.buffer.l0_values:
            return 0.0
        l0 = self.buffer.l0_values[-1]
        if l0 > cfg.target_l0:
            return cfg.constraint_lr_floor
        return 0.0

    def _compute_lrs(self) -> List[float]:
        cfg = self.config
        step = self.total_steps

        # Cosine backbone
        if step < cfg.warmup_steps:
            backbone = step / max(cfg.warmup_steps, 1)
        else:
            progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
            backbone = 0.5 * (1.0 + math.cos(math.pi * progress))

        if not self._convergence_locked() and step <= cfg.early_pulse_steps:
            backbone = max(backbone, cfg.early_pulse_warmup_floor)
        backbone = max(backbone, self._constraint_lr_floor())

        # Mode modulation
        mult = 1.0
        if self.mode == "RECOVERY":
            mult = cfg.recovery_lr_factor
        elif self.mode == "EXPLORE":
            mult = cfg.explore_lr_factor
        elif self.mode == "STABILIZE":
            mult = cfg.recovery_lr_factor * 0.8

        energy_mult = self._energy_lr_multiplier()
        return [base * backbone * mult * energy_mult for base in self.base_lrs]

    def _apply_mode_tweaks(self):
        cfg = self.config
        for group, base_b, base_wd in zip(
            self.optimizer.param_groups, self.base_betas, self.base_weight_decays
        ):
            if "betas" in group:
                beta1, beta2 = base_b
                if self.mode == "RECOVERY":
                    group["betas"] = (beta1 * cfg.recovery_momentum_factor, beta2)
                else:
                    group["betas"] = (beta1, beta2)
            if "weight_decay" in group:
                if self.mode == "EXPLORE":
                    group["weight_decay"] = base_wd * 0.5
                elif self.mode == "RECOVERY":
                    group["weight_decay"] = base_wd * 1.2
                else:
                    group["weight_decay"] = base_wd

    # -- Summary & diagnostics --------------------------------------------------

    def summary(self) -> Dict:
        return {
            "mode": self.mode,
            "phase": self.phase,
            "phase_step": self.phase_step,
            "pin_entry_step": self.pin_entry_step,
            "pinned_lambda": self.pinned_lambda,
            "pin_ev_count": self.pin_ev_count,
            "pin_retry_count": self.pin_retry_count,
            "total_steps": self.total_steps,
            "lambda_l0": self.lambda_l0,
            "transitions": len(self.transition_log),
            "event_counter": self.event_counter,
            "ev_below_floor": self._ev_below_floor_count,
            "ev_above_floor": self._ev_above_floor_count,
            "activation_norm_ema": self._activation_norm_ema,
            "activation_norm_preflight": self._activation_norm_preflight,
            "effective_slingshot_gain": self._effective_slingshot_gain(),
            "l0_progress_fast": self._l0_progress_fast,
            "l0_progress_slow": self._l0_progress_slow,
            "stall_pulse_remaining": self._stall_pulse_remaining,
            "energy_dampen": self._energy_dampen,
            "recent_transitions": self.transition_log[-5:],
        }
