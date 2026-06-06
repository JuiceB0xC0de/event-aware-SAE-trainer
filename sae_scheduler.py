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

__version__ = "0.1.0"

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


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

    # -- L0 stabilization detection ----------------------------------------
    l0_stabilize_window: int = 3          # consecutive windows to detect stabilization
    l0_stabilize_std_thresh: float = 0.5  # std of L0 window below this -> stabilize


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

    def instability_score(self) -> float:
        return self.grad_norm_zscore()

    def l0_mean(self, window: int = 3) -> float:
        if len(self.l0_values) < window:
            return 0.0
        return sum(list(self.l0_values)[-window:]) / window

    def ev_last(self) -> float:
        return self.ev_values[-1] if self.ev_values else 0.0

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

    def ev_above_floor_count(self, floor: float) -> int:
        """Count consecutive log-windows EV has been above floor."""
        count = 0
        for v in reversed(list(self.ev_values)):
            if v >= floor:
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

    def __init__(self, optimizer, config: SAEAECSConfig = None, mode_label: str = ""):
        self.config = config or SAEAECSConfig()
        self.optimizer = optimizer
        self.mode_label = mode_label  # "L0", "L1", etc. for logging

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

        ev_check_due = (
            ev is not None
            and self.total_steps % max(1, self.config.ev_check_every) == 0
        )
        ev_for_buffer = ev if ev_check_due else None
        if l0 is not None or ev_for_buffer is not None or dead_pct is not None:
            self.buffer.push_sae(l0=l0, ev=ev_for_buffer, dead_pct=dead_pct)

        # -- SAE-specific event detection (uses log-window-rate EV signal) --
        sae_event = self._detect_sae_events(signals) if ev_check_due else None

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
        error = current_l0 - cfg.target_l0   # signed (negative means we're below target)
        new_lambda = self.lambda_l0 + cfg.al_dual_step * error
        # Project onto [lambda_min, lambda_max]
        new_lambda = max(cfg.lambda_l0_min, min(cfg.lambda_l0_max, new_lambda))
        self.lambda_l0 = new_lambda
        self.lambda_adjust_last_step = self.total_steps

        # Throttle logging -- every step is too noisy
        if cfg.verbose and self.total_steps % cfg.al_log_every == 0:
            print(
                f"  [AL @ step {self.total_steps}] lambda={self.lambda_l0:.3e} "
                f"(L0={current_l0:.1f}, target={cfg.target_l0}, error={error:+.1f})"
            )

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
            and abs(l0_mean - self.config.target_l0)
                < self.config.target_l0 * self.config.l0_tolerance
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
            self.should_stop = True
            self.stop_reason = (
                f"AL converged: L0 {l0_mean:.1f} in target +/-{self.config.l0_tolerance*100:.0f}% of "
                f"{self.config.target_l0}, lambda plateaued at {self.lambda_l0:.3e} "
                f"({self.config.ev_stop_patience} windows of stability). Final EV={ev:.3f}."
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
        l0 = signals.get("l0")
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

    def _compute_lrs(self) -> List[float]:
        cfg = self.config
        step = self.total_steps

        # Cosine backbone
        if step < cfg.warmup_steps:
            backbone = step / max(cfg.warmup_steps, 1)
        else:
            progress = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
            backbone = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Mode modulation
        mult = 1.0
        if self.mode == "RECOVERY":
            mult = cfg.recovery_lr_factor
        elif self.mode == "EXPLORE":
            mult = cfg.explore_lr_factor
        elif self.mode == "STABILIZE":
            mult = cfg.recovery_lr_factor * 0.8

        return [base * backbone * mult for base in self.base_lrs]

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
            "total_steps": self.total_steps,
            "lambda_l0": self.lambda_l0,
            "transitions": len(self.transition_log),
            "event_counter": self.event_counter,
            "ev_below_floor": self._ev_below_floor_count,
            "ev_above_floor": self._ev_above_floor_count,
            "recent_transitions": self.transition_log[-5:],
        }