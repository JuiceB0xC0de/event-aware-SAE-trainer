# Event-Aware SAE Trainer: Review Contract

This document defines **binding invariants** for this codebase. It is not a
request for opinions. Every numbered rule below is a property the code either
satisfies or violates. Report violations.

## What counts as a finding

A finding is a specific, cited violation of a numbered rule below, or a bug in
the code paths those rules govern. For each finding, state:

1. the rule number violated,
2. the file and line range where the violation occurs,
3. the concrete input or run condition under which it manifests,
4. what the code does versus what the rule requires.

Do **not** report: style, formatting, missing type hints, test coverage
generally, or "consider refactoring" suggestions. Those are out of scope for
this review and will be discarded.

A rule is violated even if the surrounding code is old, and even if the
violation is spread across several files. Age is not a defense. "Pre-existing"
is not a defense — this review is explicitly an audit of existing behavior.

## Read these files first

| Area | Files |
| --- | --- |
| SAE architecture, loss, training loop, capture, checkpoints | `sae_trainer_rolling.py` |
| Control system, convergence state machine, stopping logic | `sae_scheduler.py` |
| Model presets and CLI assembly | `run_atlas.py`, `configs/` |
| Scheduler regression coverage | `tests/test_scheduler.py`, `tests/test_pin_tail_guard.py` |
| SAE, revival, sparse-decode coverage | `tests/test_sae_model.py`, `tests/test_revival.py`, `tests/test_sparse_decode.py` |

Trace each finding to its call sites. A rule about the stopping policy is not
satisfied by reading `_check_early_stop` alone; check who calls it, how often,
and with what state.

---

## Section A: Stopping and convergence policy

Governing code: `sae_scheduler.py::_check_early_stop` (~line 1046),
`_check_ev_floor` (~1039), `_maybe_update_phase` (~539), `_maybe_transition`
(~1191), `_enter_mode` (~1225), and the `step()` driver (~626).

**A1. A saturated dual MUST NOT be read as a converged dual.**
`lambda_l0` is projected onto `[lambda_l0_min, lambda_l0_max]`
(`_dual_update` ~line 761, `_adjust_lambda` ~line 1027). A lambda resting at
either bound is flat for reasons that have nothing to do with reaching a
KKT point. The plateau test MUST distinguish "stopped moving because it found
its value" from "stopped moving because it hit the clip." Report if it cannot.

**A2. The plateau test MUST detect decline as well as growth.**
`_check_early_stop` computes `rel_growth = max(recent) / recent[0] - 1.0` and
declares convergence when that is below `al_convergence_rel_tol`. Verify the
behavior when lambda is *decreasing* across the window, and when lambda
oscillates with equal magnitude up and down. If a monotonically collapsing
dual satisfies the convergence test, that is a violation.

**A3. Reported stop criteria MUST match enforced stop criteria.**
The `l0_in_target` gate uses `al_slingshot_overshoot_rel` for its lower bound
and `target_l0` for its upper bound. The `stop_reason` string reports
`l0_tolerance`. Any divergence between the tolerance the code enforces and the
tolerance the operator is told about is a violation — an operator tuning
`l0_tolerance` to change stopping behavior must actually change it.

**A4. The stop gate MUST be evaluated against held-out activations.**
Early-stop quality signals MUST NOT be computed on the same activation batch
the optimizer just consumed. If `ev` as passed into `_check_early_stop`
originates from the training batch, that is a violation. Trace the provenance
of `ev` from its call site in `step()`.

**A5. Every AECS mode MUST have a bounded, non-pathological exit to BASELINE.**
Modes are `BASELINE`, `RECOVERY`, `EXPLORE`, `STABILIZE`. `RECOVERY` has an
explicit duration bound (`recovery_max_steps`, ~line 1201). Establish, by
tracing actual reachability rather than reading the transition table, whether
each of `EXPLORE` and `STABILIZE` can return to `BASELINE` on a healthy step.
Pay attention to the guard controlling whether `_maybe_transition` is called
at all, and to which events can reach the default branch of `event_mode_map`.
A mode that can only be escaped by a fault condition violates this rule.

**A6. A single knob MUST NOT drive two contradictory behaviors.**
`pin_timeout_steps` is documented as "max steps in PIN before bailing back to
DESCENT" (~line 242) and is used that way in `_maybe_update_phase` (~line 621).
It is *also* used in `_check_early_stop` (~line 1093) to terminate the run.
Determine which fires first under realistic ordering, whether a PIN timeout
can both bail to DESCENT and stop the run in the same step, and whether the
outcome depends on call order within `step()`. Ambiguous ordering is a
violation.

**A7. PIN MUST release when its preconditions break.**
PIN freezes dual and threshold nudges. It MUST release if L0 escapes the band
materially (`_l0_escaped_pin_band`, ~line 485) or the ultra-active feature tail
collapses. Verify the release path is reachable from inside PIN, and that
freezing the dual cannot itself prevent the escape condition from ever being
observed.

**A8. Checkpoint selection MUST NOT return weights from an unstable window.**
The trainer distinguishes latest, best-quality, best-feasible, and rollback
states. A stop MUST NOT select a checkpoint recorded during, or within the
cooldown following, a resample, dead-feature revival, or rollback event.
Report if the selection logic can return a checkpoint captured mid-perturbation.

**A9. Stopping MUST be reachable.**
Identify any configuration of reachable settings under which no stop condition
can ever fire, so the run terminates only at `max_steps`. Incompatible
cadences between the log-window boundary (`ev_check_due`) and the patience
counters (`ev_stop_patience`, `pin_ev_patience`) are the likely mechanism.
Confirm the counters can actually accumulate at the rate the checks run.

**A10. Counters MUST reset on the conditions their names imply.**
`_ev_above_floor_count` is incremented on AL convergence, not on EV being above
a floor. Verify every patience counter resets when its underlying condition
breaks, and that no counter can survive a mode or phase transition that should
invalidate it.

---

## Section B: Convergence mathematics

Governing code: `_dual_update` (~727), `_dual_control_target` (~773),
`_effective_slingshot_gain` (~779), `_landing_lambda_gain` (~807),
`adaptive_ste_bandwidth` (~841), `compute_threshold_nudge` (~868),
`wenc_dampen_factor` (~953), `_adjust_lambda` (~980), plus the JumpReLU and
loss code in `sae_trainer_rolling.py`.

**B1. The dual update MUST be sign-correct.**
`lambda` MUST increase when measured L0 exceeds `target_l0` and relax when it
falls below. Verify the sign through every gain path — `_effective_slingshot_gain`,
`_landing_lambda_gain`, and any dampening — not just the base update. A gain
that can go negative silently inverts the controller.

**B2. Two mechanisms MUST NOT push the same quantity without coordination.**
Both the dual (`lambda`) and the direct threshold nudge
(`compute_threshold_nudge`) drive L0. Establish that they cannot fight each
other: one pushing sparsity up while the other pushes it down, or both acting
in the same step so the effective gain is the sum rather than the intended
value. PIN freezing only one of the two is a violation of this rule.

**B3. JumpReLU threshold gradients MUST use the intended straight-through
estimator.** Verify the STE bandwidth from `adaptive_ste_bandwidth` is actually
applied in the backward path, that it is positive everywhere it is used, and
that no code path silently bypasses the STE and takes a true (zero) gradient
through the step function.

**B4. Dead-feature revival MUST leave the optimizer in a consistent state.**
When a feature is resampled or revived, its encoder/decoder rows change
discontinuously. The corresponding optimizer moments MUST be reset alongside.
Stale Adam state on a revived feature is a violation. Check `test_revival.py`
covers this and report if it does not.

**B5. Normalization MUST be consistent between fit and use.**
Verify `_update_activation_norm` (~1244) and any activation scaling are applied
identically during pool production, training, and evaluation. A norm fitted on
one distribution and applied to another silently biases both L0 and EV, which
in turn corrupts every gate in Section A.

**B6. Sparse decode MUST be numerically equivalent to dense decode.**
`SAE_FAST_PREBIAS`, sparse decode, and any Triton or `torch.compile` path MUST
produce results equivalent to the reference path within documented tolerance.
Report any path where the fast route can diverge, especially in edge cases:
zero active features, all features active, or a feature revived this step.

**B7. Metrics feeding control MUST be computed on the same basis they are
compared against.** L0 measured with one threshold convention and compared to a
`target_l0` defined under another is a violation. Verify the L0 fed to
`_dual_update` and the L0 compared against `target_l0` in `_check_early_stop`
are the same quantity, computed the same way, over the same axis.

---

## Explicitly out of scope for this review

Do not spend effort on these. They are being handled separately and findings
about them will be discarded:

- VRAM residency and the offload contract. This has been validated empirically.
- Throughput, kernel efficiency, transfer overlap, and profiling.
- Documentation, README, packaging, CI, and test coverage as a general concern.
- Proposals to restructure or rewrite working subsystems.

## Required output

Group findings by section (A or B), most severe first. For each: rule number,
file and line range, the triggering condition, and observed versus required
behavior. If a rule is satisfied, say so in one line — a clean bill on a
specific rule is a useful result. Do not pad with generalities.
