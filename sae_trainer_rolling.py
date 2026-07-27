"""
Event-aware SAE trainer -- train one sparse autoencoder per decoder layer, unattended.
======================================================================================
THE POINT of this trainer is the event-aware scheduler (sae_scheduler.py): an AECS
mode controller plus an Augmented-Lagrangian L0 integrator that makes training
*self-tuning*. You set an L0 target; the scheduler overshoots it, then the dual
integrator pulls lambda back, L0 oscillates and converges as close to target as the
model allows, while dead features are held ~0 (aux-loss revival + threshold reset +
resampling + a dead-feature emergency mode). It sweeps every layer in a single run
with no per-layer hyperparameter retuning and no babysitting -- the failure mode that
forces restart-and-retune loops in fixed-schedule SAE trainers.

The scheduler and the training loop are MODEL-AGNOSTIC: they consume a stream of
activation batches from a provider and never touch model internals. How activations
are produced is pluggable (`--capture`):

  auto         (default) -- forward-hook the residual stream of any AutoModelForCausalLM.
                            Correct by construction (observes the real forward). Re-runs a
                            (truncated) forward per layer: 1 pool of disk, N x compute.
  rolling      (opt-in)  -- Gemma-3n/4 single-block walk. Gemma-specific invariants
                            (per-layer embeddings, sliding/full masks, KV sharing).
                            Guarded by validate_rolling_cache.py.
  rolling-float (opt-in) -- same walk as rolling, but the full model stays on CPU
                            and only the active block (+ pinned shared components)
                            is hoisted to GPU. Same math as rolling: identical
                            _produce_pool path, weights merely change device.
  rolling-hf   (opt-in)  -- generic Llama/SmolLM2/Qwen-style single-block walk using
                            the model's own rotary_emb and position_embeddings. Same
                            disk/compute win as rolling, but works for any HF decoder
                            whose layers accept position_embeddings.

Run (plain Python, expects a CUDA GPU; H100/A100 target):
    python sae_trainer_rolling.py --model-id meta-llama/Llama-3.2-1B --end-layer 16
    python sae_trainer_rolling.py --capture rolling --end-layer 15      # Gemma fast path
    python sae_trainer_rolling.py --model-id HuggingFaceTB/SmolLM2-135M-Instruct --capture rolling-hf --end-layer 30  # Llama-style fast path
    python sae_trainer_rolling.py --hub-id me/my-saes --wandb-project my-run

Config: $SAE_DATA_DIR (default ./data), $SAE_MODEL_ID, $SAE_HUB_ID, $WANDB_PROJECT,
$SAE_SCRATCH_DIR; HF_TOKEN read from the environment. d_in is auto-detected.
"""
from __future__ import annotations

__version__ = "0.1.0"

import math
import os
from pathlib import Path

class IterableDataset:
    """Tiny import-time base for the local streaming datasets.

    The trainer iterates these datasets directly, so importing the module should
    not initialize torch/OpenMP on CPU-only control hosts just to read constants
    or parse CLI wiring.
    """
    pass

def _slug(model_id: str) -> str:
    """Filesystem-safe tag from a HF model id ('google/gemma-4-E2B-it' ->
    'google_gemma-4-e2b-it'). Keeps outputs from different models separate."""
    return model_id.strip("/").replace("/", "_").replace(" ", "_").lower()


def _l0_crossed_target(l0: float, target_l0: float) -> bool:
    """True once the sparsity controller has pushed L0 through the target."""
    return l0 <= target_l0


def _post_peak_decline_is_bad(ev: float, best_ev: float, margin: float, floor: float) -> bool:
    """Catastrophe guard: ignore normal EV sag unless quality is also absolutely low."""
    return (best_ev - ev) >= margin and ev < floor


# Default model -- override per-run with --model-id / $SAE_MODEL_ID. Nothing below is
# hardcoded to a model family except the opt-in 'rolling' capture path.
MODEL_ID   = os.environ.get("SAE_MODEL_ID", "google/gemma-4-E2B-it")
# HF upload target. NO default org: push happens only when set explicitly (--hub-id /
# $SAE_HUB_ID), so a clone-and-run can never push somewhere unintended or fail on perms.
SAE_HUB_ID = os.environ.get("SAE_HUB_ID")
# wandb project. Off unless set (--wandb-project / $WANDB_PROJECT), independent of any key.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

# -- Local data layout (replaces the Modal /data volume) --------------------
# Everything reads/writes under $SAE_DATA_DIR (default ./data). Activation pools
# can be redirected to fast scratch via $SAE_SCRATCH_DIR (defaults under DATA_DIR).
DATA_DIR   = Path(os.environ.get("SAE_DATA_DIR", "./data")).expanduser()
SAE_DIR    = str(DATA_DIR / "saes" / _slug(MODEL_ID))        # persistent SAE outputs (per-model)
PRETOK_DIR = str(DATA_DIR / "pretok" / "fineweb-edu")        # optional pre-tokenized shards
# Activation pools are large and ephemeral -- point this at the fastest local disk
# you have. Pools are deleted incrementally during a run (we hold ~one pool at a time).
ROLLCACHE  = os.environ.get("SAE_SCRATCH_DIR", str(DATA_DIR / "rollcache"))
HARD_STOP_LAYER = 15                       # exclusive upper bound; never touch 15+

# -- Corpus + model loading (set via CLI/config only, no env) ---------------
CORPUS_ID         = "HuggingFaceFW/fineweb-edu"
CORPUS_TEXT_FIELD = "text"
CORPUS_PREFIX     = ""      # prepended to every corpus text (e.g. a domain tag)
TRUST_REMOTE_CODE = False

# -- Hyperparameters --------------------------------------------------------
D_IN          = 1536        # fallback residual width; auto-detected from the model config at run time
EXPANSION     = 32          # SAE dictionary size = EXPANSION * d_in (Llama-Scope-class config)
N_FEATURES    = EXPANSION * D_IN   # 49152 at d_in=1536; recomputed for the detected d_in
K             = 500         # FINAL target L0. Natural L0 settles ~550 at 32x; target<natural keeps lambda slightly positive.
K_INIT        = 500         # Curriculum disabled (== K)
K_CURRICULUM_STEPS = 1      # effectively disabled (target_l0 = K from step 1)
BATCH_TOKENS  = int(os.environ.get("SAE_BATCH_TOKENS", "32768"))  # total tokens per SAE step (must be divisible by SEQ_LEN)
MICROBATCH_TOKENS = int(os.environ.get("SAE_MICROBATCH_TOKENS", str(BATCH_TOKENS)))  # tokens per microbatch for gradient accumulation
SEQ_LEN       = 2_048       # length passed to the model (attention is O(seq^2))
N_STEPS       = 15_000      # peak-EV early-stop usually fires well before this
LR            = 2e-4
LOG_EVERY     = 250
# Observation-only L0 trace. LOG_EVERY is load-bearing -- it is the AL dual-update
# cadence, the dead-feature window and the fire-rate window -- so it must not be
# lowered just to see more. This is a pure print cadence and gates nothing. It
# exists because the sparsity collapse happens inside the first LOG_EVERY steps,
# where nothing was being recorded at all.
OBS_EVERY     = int(os.environ.get("SAE_OBS_EVERY", "100"))
# Aggressive-K early-stop gate. EV is not comparable across depth -- it falls as the
# residual stream's norm grows, so a floor tuned on shallow layers silently blocks the
# stop on every deep one and burns the full step budget for nothing.
K_STOP_L0_REL   = float(os.environ.get("SAE_STOP_L0_REL", "0.15"))
K_STOP_EV_FLOOR = float(os.environ.get("SAE_STOP_EV_FLOOR", "0.95"))
LR_WARMUP_STEPS = 300
TIMING_EVERY_DEFAULT = 25   # [STEP-TIME] cadence; override via SAE_TIMING or --timing-every


def _resolve_hf_token():
    """Explicit env vars first, then the token stored by `hf auth login`.

    Being logged in through the CLI is the normal case; requiring HF_TOKEN in the
    environment on top of that is a papercut, not a security boundary.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

# JumpReLU / aux-loss / resampling knobs
INIT_THRESHOLD = 0.1        # initial per-feature threshold
STE_BANDWIDTH  = 0.1        # epsilon for the straight-through estimator
BDEC_INIT_BATCHES = 50      # batches used to estimate b_dec mean
AUX_DEAD_THRESHOLD = 250    # steps without firing before a feature gets aux-loss revival
AUX_K          = 128        # top-k dead features per token to revive
AUX_COEFF      = 1 / 32     # Gao 2024 standard
RESET_EVERY    = 1_000      # sweep + theta-reset cadence for chronically-dead features
RESET_THRESHOLD = 1_500     # silent-steps required to qualify for theta reset
CHECKPOINT_EVERY = 1_000
RESAMPLE_STEPS = (K_CURRICULUM_STEPS, 12_000)   # steps where full W_enc/W_dec resample fires
ERR_BUFFER_SZ  = 16_384     # high-error activation buffer (tokens)
RESAMPLE_SCALE = 1.0        # scale of resampled encoder rows vs mean alive norm
DEFAULT_SEED   = 0

# Numeric encoding of scheduler phase for W&B plotting (string is logged too).
_PHASE_IDX = {"DESCENT": 0, "PIN": 1, "FINETUNE": 2}

# Triton fused kernel flag - DISABLED.
# The current fused kernel is ~100x slower than PyTorch forward-only and its
# backward kernel exceeds A10 shared-memory limits. Keep the env-var read for
# future kernels, but force the path off until a new implementation exists.
_USE_TRITON_ENV = os.environ.get("SAE_USE_TRITON", "0").lower() in ("1", "true", "yes", "on")
USE_TRITON = False
if _USE_TRITON_ENV:
    print("[TRITON] SAE_USE_TRITON=1 ignored: current kernel is disabled (see sae_trainer_rolling.py)")


def _aggressive_k_aux_k(target_l0: int) -> int:
    """At very low L0, reviving 128 dead features per token is nonsensical; cap AUX_K
    to a fraction of the target so the aux loss doesn't fight the sparsity budget."""
    return max(8, min(AUX_K, target_l0 // 2))


# --- console colour -----------------------------------------------------------
# The per-step panel scrolls fast and the three numbers that decide whether a layer
# is any good (L0, dead, EV) were the same shade as the twenty that don't matter.
# Colour is written into the log FILE too, on purpose: runs live under nohup, so
# isatty() is False and a tty check would make the output monotone exactly where
# it's read from. Set SAE_NO_COLOR=1 to turn it off.
_SAE_COLOR = os.environ.get("SAE_NO_COLOR", "").strip().lower() not in ("1", "true", "yes", "on")


def _c(text, code: str) -> str:
    """Wrap text in an ANSI SGR code, or return it untouched when colour is off."""
    return f"\033[{code}m{text}\033[0m" if _SAE_COLOR else str(text)


def _c_dead(v: float, ceiling: float = 1.0) -> str:
    """dead%: red once it breaches the ceiling, yellow at half of it, green below."""
    s = f"{v:.1f}%"
    if v > ceiling:
        return _c(s, "1;91")
    if v > ceiling / 2:
        return _c(s, "93")
    return _c(s, "92")


def _c_ev(v: float, floor: float = 0.95) -> str:
    """EV: green at or above the quality gate, yellow within 5 points, red below."""
    s = f"{v:.3f}"
    if v >= floor:
        return _c(s, "1;92")
    if v >= floor - 0.05:
        return _c(s, "93")
    return _c(s, "91")


def _c_l0(v: float, target: float) -> str:
    """L0: green inside 5% of target, yellow inside 20%, red outside."""
    s = f"{v:.1f}"
    if target <= 0:
        return s
    rel = abs(v - target) / target
    if rel <= 0.05:
        return _c(s, "1;92")
    if rel <= 0.20:
        return _c(s, "93")
    return _c(s, "1;91")


def _pick_dead_rollback(buf, ceiling: float):
    """Newest buffered log window whose dead_pct is at or under `ceiling`, else None.

    Newest-first because we want the most-trained state that still has a live feature
    tail, not the healthiest one. On MiniCPM5-1B L13 the buffer at the breach held
    steps 1500/1750/2000/2250 at 0.0/0.1/0.8/8.8% dead, and step 2000 is the pick:
    L0 49.3, EV 0.949, dead 0.8%.
    """
    for s in reversed(buf):
        if s["dead"] is not None and s["dead"] <= ceiling:
            return s
    return None


def _aggressive_k_reset_threshold(target_l0: int) -> int:
    """Dead features accumulate faster at low L0; reset/resample them sooner."""
    if target_l0 <= 100:
        return 750
    if target_l0 <= 250:
        return 1000
    return RESET_THRESHOLD


def _aggressive_k_resample_steps(target_l0: int):
    """Add an early resample window for aggressive low-L0 runs."""
    if target_l0 <= 100:
        return (K_CURRICULUM_STEPS, 2000, 6000, 12_000)
    return (K_CURRICULUM_STEPS, 12_000)

# Pool sizing: T batches of fresh activations per layer. Early-stop converges
# ~3500 steps, so T=4000 gives fresh-every-step (no epoching) with headroom.
POOL_BATCHES_DEFAULT = 4000


class RevivalController:
    """Owns dead-feature revival state and bookkeeping.

    Branch 5A scope: ownership of the fire/dead trackers, dead-percent
    calculation, and checkpoint pack/unpack only. The AuxK dead-mask policy
    (5B) and threshold-reset / resample policy (5C) move in here in later
    passes. Behavior is identical to the previous inline trackers.

    Revival is always-on and independent of scheduler mode/phase.
    """

    AUX_K_POLICIES = ("legacy", "feature_fraction")

    def __init__(self, n_features: int, device, aux_k_policy: str = "legacy"):
        import torch  # lazy import: this module imports torch inside scopes only
        if aux_k_policy not in self.AUX_K_POLICIES:
            raise ValueError(
                f"unknown aux_k_policy {aux_k_policy!r}; "
                f"expected one of {self.AUX_K_POLICIES}")
        self.n_features = n_features
        self.aux_k_policy = aux_k_policy
        self.feature_fire_counts = torch.zeros(n_features, device=device, dtype=torch.long)
        self.steps_since_fired = torch.zeros(n_features, device=device, dtype=torch.long)
        # Revival metrics (per-layer running totals; not checkpointed -- matches
        # the previous inline behavior where these reset on resume).
        self.total_resampled = 0
        self.last_dead_count = 0
        self.last_reset_count = 0
        self.last_resampled_count = 0
        self.last_resample_dead_count = 0

    # -- AuxK dead-feature revival policy (Branch 5B) -------------------------
    def aux_dead_mask(self):
        """Boolean mask of features eligible for aux-loss revival this step."""
        return self.steps_since_fired >= AUX_DEAD_THRESHOLD

    def effective_k_aux(self, target_l0) -> int:
        """Top-k dead features per token to revive, per the configured policy.

        legacy           -> max(8, min(AUX_K, target_l0 // 2))  (low-L0 safety cap)
        feature_fraction  -> min(512, n_features // 64)
        """
        if self.aux_k_policy == "legacy":
            return _aggressive_k_aux_k(int(target_l0))
        if self.aux_k_policy == "feature_fraction":
            return min(512, self.n_features // 64)
        raise ValueError(f"unknown aux_k_policy {self.aux_k_policy!r}")

    def revival_metrics(self, target_l0) -> dict:
        """Lightweight metrics for logging."""
        return {
            "aux_k_policy": self.aux_k_policy,
            "effective_aux_k": self.effective_k_aux(target_l0),
            "target_l0": int(target_l0),
            "n_features": self.n_features,
            "dead_count": self.last_dead_count,
            "reset_count": self.last_reset_count,
            "resampled_count": self.last_resampled_count,
            "total_resampled": self.total_resampled,
        }

    # -- threshold-reset and resample scheduling decisions (Branch 5C) --------
    # These own the *decisions*; the model-weight mutations stay inline in the
    # training loop, readable and close to the existing implementation.
    def reset_threshold_for(self, target_l0) -> int:
        return _aggressive_k_reset_threshold(int(target_l0))

    def resample_schedule(self, target_l0):
        return _aggressive_k_resample_steps(int(target_l0))

    def should_reset(self, step: int) -> bool:
        return step % RESET_EVERY == 0 and step > 0

    def very_dead_mask(self, target_l0):
        """Features silent long enough to qualify for a theta reset."""
        return self.steps_since_fired >= self.reset_threshold_for(target_l0)

    def resample_dead_mask(self, target_l0):
        """Features dead enough to qualify for full W_enc/W_dec resample."""
        return self.steps_since_fired >= max(500, self.reset_threshold_for(target_l0) - 250)

    def should_buffer_err(self, step: int, target_l0) -> bool:
        """True when a resample step is near enough to start buffering high-error acts."""
        schedule = self.resample_schedule(target_l0)
        nxt = min((s for s in schedule if s >= step), default=None)
        return nxt is not None and (nxt - step) <= ERR_BUFFER_SZ // 64

    def is_resample_step(self, step: int, target_l0) -> bool:
        return step in self.resample_schedule(target_l0)

    def clear_silence(self, mask):
        """Reset the silence counter for the masked features (post reset/resample).

        Keeps the controller the sole owner of steps_since_fired mutations even
        though the model-weight mutation it accompanies stays inline in the loop.
        """
        self.steps_since_fired[mask] = 0

    def record_reset(self, n_reset: int):
        self.last_reset_count = n_reset

    def record_resample(self, n_dead: int, n_resampled: int):
        self.last_resample_dead_count = n_dead
        self.last_resampled_count = n_resampled
        self.total_resampled += n_resampled

    def update_fire_state(self, fired_accum):
        """Advance per-feature silence counters using this step's fired mask."""
        self.steps_since_fired += 1
        self.steps_since_fired[fired_accum] = 0
        self.feature_fire_counts += fired_accum.long()

    def dead_pct(self, window: int) -> float:
        """Percent of features silent for at least `window` steps."""
        return (self.steps_since_fired >= window).float().mean().item() * 100

    def fire_rate(self, window: int):
        """Per-feature fire rate over the accumulation window."""
        return self.feature_fire_counts.float() / max(window, 1)

    def reset_fire_counts(self):
        self.feature_fire_counts.zero_()

    # -- checkpoint pack/unpack (keeps the legacy top-level key names) --------
    def state_tensors(self):
        """Return (feature_fire_counts, steps_since_fired) for checkpointing."""
        return self.feature_fire_counts, self.steps_since_fired

    def load_state(self, feature_fire_counts, steps_since_fired):
        self.feature_fire_counts = feature_fire_counts
        self.steps_since_fired = steps_since_fired


# ===========================================================================
#  SAE model  (JumpReLU, inlined -- this file is the single source of truth)
# ===========================================================================

JumpReLUSAE = None


def _ensure_sae_classes():
    """Define torch-backed SAE classes lazily.

    Importing this module should only read constants and CLI wiring. Pulling in
    torch/OpenMP at import time can fail on lightweight control hosts and in
    sandboxed subprocess tests.
    """
    global JumpReLUSAE
    if JumpReLUSAE is not None:
        return JumpReLUSAE

    import torch
    import torch.nn as nn

    class _JumpReLU(torch.autograd.Function):
        """JumpReLU activation with straight-through estimator for threshold gradient."""
        @staticmethod
        def forward(ctx, pre, log_threshold, bandwidth):
            threshold = log_threshold.exp()
            gate = (pre > threshold).to(pre.dtype)
            ctx.save_for_backward(pre, threshold, gate)
            ctx.bandwidth = bandwidth
            return pre * gate

        @staticmethod
        def backward(ctx, grad_output):
            pre, threshold, gate = ctx.saved_tensors
            eps = ctx.bandwidth
            grad_pre = grad_output * gate

            # Optimization: avoid boolean mask to float casting and elementwise multiplication.
            # Using torch.where reduces O(N) ops to O(M), skipping irrelevant zeros.
            in_band_mask = (pre - threshold).abs() < eps
            sum_dims = tuple(range(grad_output.ndim - 1))
            masked_vals = torch.where(in_band_mask, pre * grad_output, 0.0)
            grad_threshold = -(masked_vals.sum(dim=sum_dims) / (2 * eps))
            grad_log_threshold = grad_threshold * threshold
            return grad_pre, grad_log_threshold, None

    class _JumpReLUWithGate(torch.autograd.Function):
        """Fused JumpReLU activation + L0 indicator.

        Returns (feat_acts, gate) from a single pass: threshold.exp() and the
        (pre > threshold) gate are computed once instead of once each in
        _JumpReLU and _L0Indicator, and backward shares one in-band STE mask
        across both threshold-gradient contributions. Mathematically identical
        to applying _JumpReLU and _L0Indicator separately to the same
        (pre, log_threshold) -- see test_fused_jumprelu_matches_separate.
        """
        @staticmethod
        def forward(ctx, pre, log_threshold, bandwidth):
            threshold = log_threshold.exp()
            gate = (pre > threshold).to(pre.dtype)
            ctx.save_for_backward(pre, threshold, gate)
            ctx.bandwidth = bandwidth
            return pre * gate, gate

        @staticmethod
        def backward(ctx, grad_feat, grad_gate):
            pre, threshold, gate = ctx.saved_tensors
            eps = ctx.bandwidth
            # grad w.r.t. pre: only the feat path (the L0 gate is a step fn -> 0).
            grad_pre = grad_feat * gate if grad_feat is not None else None
            # grad w.r.t. log_threshold: both STE contributions share one mask.
            #   feat path: where(in_band, pre * grad_feat)
            #   gate path: where(in_band, grad_gate)
            combined = None
            if grad_feat is not None:
                combined = pre * grad_feat
            if grad_gate is not None:
                combined = grad_gate if combined is None else combined + grad_gate
            if combined is None:
                return grad_pre, None, None
            in_band = (pre - threshold).abs() < eps
            sum_dims = tuple(range(pre.ndim - 1))
            masked = torch.where(in_band, combined, 0.0)
            grad_threshold = -(masked.sum(dim=sum_dims) / (2 * eps))
            grad_log_threshold = grad_threshold * threshold
            return grad_pre, grad_log_threshold, None

    class _L0Indicator(torch.autograd.Function):
        """Step function H(pre - theta) with STE for L0 sparsity loss."""
        @staticmethod
        def forward(ctx, pre, log_threshold, bandwidth):
            threshold = log_threshold.exp()
            ctx.save_for_backward(pre, threshold)
            ctx.bandwidth = bandwidth
            return (pre > threshold).to(pre.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            pre, threshold = ctx.saved_tensors
            eps = ctx.bandwidth

            # Optimization: avoid boolean mask to float casting and elementwise multiplication.
            # Using torch.where is faster and more memory efficient.
            in_band_mask = (pre - threshold).abs() < eps
            sum_dims = tuple(range(grad_output.ndim - 1))
            masked_vals = torch.where(in_band_mask, grad_output, 0.0)
            grad_threshold = -(masked_vals.sum(dim=sum_dims) / (2 * eps))
            grad_log_threshold = grad_threshold * threshold
            return None, grad_log_threshold, None

    class _JumpReLUSAE(nn.Module):
        """JumpReLU SAE using plain PyTorch ops + a backward hook for threshold STE.

        The custom autograd Functions in earlier versions forced a torch.compile
        graph break and ran Python backward code every step, creating ~150ms/step
        of CPU stall on H100.  Here the forward is just `pre > threshold.exp()` and
        `pre * gate`, which compile can optimize end-to-end.  The threshold
        gradient is injected via a parameter hook that combines the feat-acts and
        L0-indicator STE contributions using the same in-band straight-through
        estimator as before.
        """

        def __init__(self, d_in: int, n_features: int):
            super().__init__()
            self.d_in = d_in
            self.n_features = n_features
            self.W_enc = nn.Linear(d_in, n_features, bias=True)
            self.W_dec = nn.Linear(n_features, d_in, bias=False)
            self.b_dec = nn.Parameter(torch.zeros(d_in))
            self.log_threshold = nn.Parameter(
                torch.full((n_features,), math.log(INIT_THRESHOLD))
            )
            # STE bandwidth as a mutable instance attribute so it can be
            # overridden at runtime (live-tune knob for widening the gradient
            # channel).  Default matches the module constant.
            self.ste_bandwidth = STE_BANDWIDTH
            # W_dec orthonormal, W_enc tied to W_dec.T at init (Anthropic recipe)
            nn.init.orthogonal_(self.W_dec.weight)
            self._normalize_decoder()
            with torch.no_grad():
                self.W_enc.weight.copy_(self.W_dec.weight.t())
                self.W_enc.bias.zero_()

            # Storage for backward-hook state (cleared each forward).
            self._saved_pre: "Optional[torch.Tensor]" = None
            self._saved_grad_feat: "Optional[torch.Tensor]" = None
            self._saved_grad_gate: "Optional[torch.Tensor]" = None
            self._log_threshold_hook_handle = self.log_threshold.register_hook(
                self._log_threshold_hook
            )

        def _normalize_decoder(self):
            with torch.no_grad():
                # W_dec.weight is [d_in, n_features]; each FEATURE direction is a COLUMN.
                norms = self.W_dec.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
                self.W_dec.weight.div_(norms)

        def _feat_hook_save_grad(self, grad):
            self._saved_grad_feat = grad
            return grad

        def _gate_hook_save_grad(self, grad):
            self._saved_grad_gate = grad
            return grad

        def _log_threshold_hook(self, grad):
            """Backward hook: replace the autograd gradient for log_threshold
            with the in-band straight-through estimator gradient.

            When `gate` is used in both feat_acts = pre * gate and gate.sum(),
            the gate tensor's hook receives the combined gradient
                grad_gate_total = pre * grad_feat + grad_gate
            which is exactly the combined term we need. If only the feat path is
            active we fall back to pre * grad_feat; if only the L0 indicator path
            is active we use grad_gate directly.
            """
            if self._saved_pre is None:
                return grad
            pre = self._saved_pre
            threshold = self.log_threshold.exp()
            eps = self.ste_bandwidth
            in_band = (pre - threshold).abs() < eps
            sum_dims = tuple(range(pre.ndim - 1))

            if self._saved_grad_gate is not None:
                # gate is used directly in loss -> its hook already received
                # pre * grad_feat + grad_gate.
                combined = self._saved_grad_gate
            elif self._saved_grad_feat is not None:
                # Only feat_acts used in loss.
                combined = pre * self._saved_grad_feat
            else:
                return grad

            masked = torch.where(in_band, combined, 0.0)
            grad_threshold = -(masked.sum(dim=sum_dims) / (2 * eps))
            return grad_threshold * threshold

        def _jumprelu_forward(self, pre: "torch.Tensor", need_gate: bool):
            """Plain-PyTorch JumpReLU forward; registers hooks for backward STE.

            The hard gate `pre > threshold.exp()` is non-differentiable. We use a
            detached hard gate for the JumpReLU output (so autograd gives the
            correct grad_pre = grad_feat * gate) and a separate zero-value
            "gradient bridge" gate for the L0 indicator path. The bridge depends on
            log_threshold, letting us attach a hook that captures grad_gate and a
            parameter hook that injects the STE threshold gradient.
            """
            # Ensure any stale grad storage from a previous forward is cleared.
            self._saved_pre = None
            self._saved_grad_feat = None
            self._saved_grad_gate = None

            threshold = self.log_threshold.exp()
            gate_hard = (pre > threshold).to(pre.dtype)
            # Zero-value bridge: value is unchanged, but gives the gate tensor a
            # grad_fn linked to log_threshold so its hook captures grad_gate.
            bridge = (threshold - threshold.detach()).view(1, -1) * 0.0
            gate = gate_hard + bridge
            # feat_acts uses the bridge gate so log_threshold participates in the
            # graph and the parameter hook fires. The gate hook will receive the
            # combined gradient (pre * grad_feat + grad_gate) when both paths are
            # used, which is exactly what the STE needs.
            feat_acts = pre * gate
            # Hooks are only needed during training forward (with gradients). In
            # torch.no_grad() contexts like preflight probes, skip them.
            if torch.is_grad_enabled():
                self._saved_pre = pre.detach()
                feat_acts.register_hook(self._feat_hook_save_grad)
                if need_gate:
                    gate.register_hook(self._gate_hook_save_grad)
            return (feat_acts, gate) if need_gate else feat_acts

        def encode(self, x: "torch.Tensor", k: int = K) -> "torch.Tensor":
            pre = self.W_enc(x - self.b_dec)
            return self._jumprelu_forward(pre, need_gate=False)

        def encode_pre(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.W_enc(x - self.b_dec)

        def apply_jumprelu(self, pre: "torch.Tensor") -> "torch.Tensor":
            return self._jumprelu_forward(pre, need_gate=False)

        def l0_indicator(self, pre: "torch.Tensor") -> "torch.Tensor":
            _, gate = self._jumprelu_forward(pre, need_gate=True)
            return gate

        def jumprelu_with_gate(self, pre: "torch.Tensor"):
            """Fused activation + L0 gate: returns (feat_acts, gate) in one pass."""
            return self._jumprelu_forward(pre, need_gate=True)

        def decode(self, acts: "torch.Tensor") -> "torch.Tensor":
            return self.W_dec(acts) + self.b_dec

        def forward(self, x: "torch.Tensor", k: int = K):
            acts = self.encode(x, k=k)
            x_hat = self.decode(acts)
            return x_hat, acts

    JumpReLUSAE = _JumpReLUSAE
    return JumpReLUSAE


def _make_sae(d_in: int, n_features: int, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    sae_cls = _ensure_sae_classes()
    return sae_cls(d_in, n_features)




# ===========================================================================
#  Data streaming  (token datasets, inlined)
# ===========================================================================

class StreamingBatchDataset(IterableDataset):
    """IterableDataset over the full FineWeb-Edu corpus, yielding ready-to-train
    1-D LongTensors of length `batch_tokens`. Tokenizes on the fly; multiple
    DataLoader workers each get a disjoint shard subset via HF's `.shard(...)`.
    Module-level class so it stays picklable for worker processes."""

    def __init__(self, hf_token, model_id, batch_tokens,
                 max_seq_len: int = 4096, shuffle_buffer: int = 10_000, seed: int = 0,
                 corpus_id: str = None, corpus_text_field: str = None,
                 corpus_prefix: str = None, trust_remote_code: bool = False):
        super().__init__()
        self.hf_token = hf_token
        self.model_id = model_id
        self.batch_tokens = batch_tokens
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.corpus_id = corpus_id or CORPUS_ID
        self.corpus_text_field = corpus_text_field or CORPUS_TEXT_FIELD
        self.corpus_prefix = corpus_prefix if corpus_prefix is not None else CORPUS_PREFIX
        self.trust_remote_code = trust_remote_code

    def __iter__(self):
        import random as _random
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        _random.seed(self.seed * 7919 + worker_id)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=self.hf_token,
            trust_remote_code=self.trust_remote_code)

        def _open_corpus():
            ds = load_dataset(
                self.corpus_id,
                split="train", streaming=True, token=self.hf_token,
            )
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
            if num_workers > 1:
                ds = ds.shard(num_shards=num_workers, index=worker_id)
            return ds

        fw_iter = iter(_open_corpus())

        def _next_text():
            nonlocal fw_iter
            while True:
                try:
                    row = next(fw_iter)
                    text = row.get(self.corpus_text_field, "")
                    return (self.corpus_prefix + text) if text else text
                except StopIteration:
                    fw_iter = iter(_open_corpus())

        def _tokenize(text: str):
            return tokenizer(
                text, truncation=True, max_length=self.max_seq_len,
                return_tensors="pt", add_special_tokens=True,
            ).input_ids[0]

        token_buffer: list = []
        while True:
            while sum(len(t) for t in token_buffer) < self.batch_tokens:
                text = _next_text()
                if not text.strip():
                    continue
                ids = _tokenize(text)
                if len(ids) < 8:
                    continue
                token_buffer.append(ids)
            flat = torch.cat(token_buffer)
            yield flat[: self.batch_tokens].contiguous()
            leftover = flat[self.batch_tokens :]
            token_buffer = [leftover] if leftover.numel() > 0 else []


class PreTokenizedDataset(IterableDataset):
    """IterableDataset over pre-tokenized FineWeb-Edu shards on disk (memmap reads,
    no tokenization on the hot path). Build the shards with a pretokenize pass."""

    def __init__(self, pretok_dir: str, batch_tokens: int, seed: int = 0):
        super().__init__()
        self.pretok_dir = pretok_dir
        self.batch_tokens = batch_tokens
        self.seed = seed

    def __iter__(self):
        import json
        import random as _random
        import numpy as np
        import torch

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        _random.seed(self.seed * 7919 + worker_id)

        with open(f"{self.pretok_dir}/manifest.json") as f:
            manifest = json.load(f)
        all_idx = list(range(manifest["n_shards"]))
        my_idx = [i for i in all_idx if i % num_workers == worker_id]
        if not my_idx:
            my_idx = [worker_id % manifest["n_shards"]]

        shards = []
        for i in my_idx:
            path = f"{self.pretok_dir}/shard_{i:02d}.npy"
            arr = np.load(path, mmap_mode="r")
            if arr.shape[0] >= self.batch_tokens + 1:
                shards.append(arr)

        if not shards:
            raise RuntimeError(
                f"Worker {worker_id} got no usable shards. my_idx={my_idx}"
            )

        while True:
            shard = _random.choice(shards)
            max_start = shard.shape[0] - self.batch_tokens
            start = _random.randint(0, max_start)
            tokens = shard[start : start + self.batch_tokens].copy()
            yield torch.from_numpy(tokens).long()


def _build_token_dataset(hf_token, batch_tokens, seed: int = 0,
                         use_pretok: bool = False, pretok_dir=None, model_id=None):
    """Construct the streaming dataset. If `use_pretok=True`, reads pre-tokenized
    memmap shards (faster); otherwise streams + tokenizes FineWeb-Edu on the fly.
    `model_id` picks the tokenizer for streaming; defaults to the module-level
    MODEL_ID (set via $SAE_MODEL_ID) for backward compatibility."""
    if use_pretok:
        if pretok_dir is None:
            pretok_dir = PRETOK_DIR
        return PreTokenizedDataset(pretok_dir=pretok_dir, batch_tokens=batch_tokens, seed=seed)
    return StreamingBatchDataset(
        hf_token=hf_token, model_id=model_id or MODEL_ID,
        batch_tokens=batch_tokens, max_seq_len=4096, seed=seed,
        corpus_id=CORPUS_ID, corpus_text_field=CORPUS_TEXT_FIELD,
        corpus_prefix=CORPUS_PREFIX, trust_remote_code=TRUST_REMOTE_CODE,
    )


# ===========================================================================
#  b2 single-block invocation  (proven bit-exact in validate_rolling_cache.py)
# ===========================================================================

def _find_text_model(model, n_layers):
    """Return (text_model, layers_attr_name, decoder_layers)."""
    import torch.nn as nn
    queue = [(model, None, None)]
    while queue:
        node, parent, attr = queue.pop(0)
        for name, child in node.named_children():
            if isinstance(child, nn.ModuleList) and len(child) == n_layers:
                return node, name, child
            queue.append((child, node, name))
    raise RuntimeError(f"Cannot find decoder ModuleList of length {n_layers}")


def _make_invariants(text_model, tcfg, ids):
    """Per-batch quantities that are CONSTANT across layers for these tokens:
    per_layer_inputs [B,S,n_layers,ple], per-type causal masks, per-type rotary,
    position_ids. Mirrors the head of Gemma4TextModel.forward."""
    import torch
    try:
        from transformers.masking_utils import (
            create_causal_mask, create_sliding_window_causal_mask)
    except Exception:
        from transformers.models.gemma4.modeling_gemma4 import (
            create_causal_mask, create_sliding_window_causal_mask)

    unique_layer_types = getattr(text_model, "unique_layer_types", None) \
        or sorted(set(tcfg.layer_types))
    with torch.no_grad():
        inputs_embeds0 = text_model.embed_tokens(ids)
        ple_raw = text_model.get_per_layer_inputs(ids, None)
        per_layer_inputs = text_model.project_per_layer_inputs(inputs_embeds0, ple_raw)
        S = inputs_embeds0.shape[1]
        position_ids = torch.arange(S, device=inputs_embeds0.device).unsqueeze(0)
        mask_kwargs = {
            "config": tcfg, "inputs_embeds": inputs_embeds0,
            "attention_mask": None, "past_key_values": None,
            "position_ids": position_ids,
        }
        masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }
        pos_emb = {lt: text_model.rotary_emb(inputs_embeds0, position_ids, lt)
                   for lt in unique_layer_types}
    return {
        "inputs_embeds0": inputs_embeds0, "per_layer_inputs": per_layer_inputs,
        "masks": masks, "pos_emb": pos_emb, "position_ids": position_ids,
    }


def _run_block(decoder_layers, tcfg, layer, hidden, inv):
    """Run ONLY block `layer` on `hidden`, exactly as Gemma4TextModel.forward's
    loop does. layers 0..14 are kv-own, so a fresh empty shared_kv dict is fine."""
    import torch
    lt = tcfg.layer_types[layer]
    with torch.no_grad():
        out = decoder_layers[layer](
            hidden,
            inv["per_layer_inputs"][:, :, layer, :],
            shared_kv_states={},  # nobody reads it in 0..14; storers 13/14 write harmlessly
            position_embeddings=inv["pos_emb"][lt],
            attention_mask=inv["masks"][lt],
            position_ids=inv["position_ids"],
            past_key_values=None,
        )
    return out[0] if isinstance(out, tuple) else out


def _is_hf_rolling_supported(text_model, decoder_layers):
    """True for Llama/SmolLM2/Qwen-style decoders that expose shared rotary embeddings."""
    import inspect
    if not hasattr(text_model, "embed_tokens"):
        return False
    if not hasattr(text_model, "rotary_emb"):
        return False
    if len(decoder_layers) == 0:
        return False
    try:
        sig = inspect.signature(decoder_layers[0].forward)
        return "position_embeddings" in sig.parameters
    except Exception:
        return False


def _find_rope_fn(model, text_model):
    """Locate a callable returning (cos, sin) for a sequence length.

    Custom-code decoders usually precompute RoPE on the top-level module rather
    than exposing an HF `rotary_emb` submodule. Returns None if nothing matches.
    """
    for obj in (model, text_model, getattr(model, "model", None)):
        if obj is None:
            continue
        fn = getattr(obj, "_get_rope", None)
        if callable(fn):
            return fn
    return None


def _is_generic_rolling_supported(model, text_model, decoder_layers):
    """True for custom decoders whose blocks take (x, cos, sin) positionally.

    This is the escape hatch for `trust_remote_code` architectures that cannot use
    the Llama-shaped path: they still walk one block at a time, which is the entire
    point of rolling capture. Without it `auto` re-runs the FULL model once per
    layer -- O(n_layers^2) block evaluations instead of O(n_layers).
    """
    import inspect
    if len(decoder_layers) == 0:
        return False
    if model.get_input_embeddings() is None:
        return False
    if _find_rope_fn(model, text_model) is None:
        return False
    try:
        params = list(inspect.signature(decoder_layers[0].forward).parameters)
    except Exception:
        return False
    return len(params) >= 3 and params[1] == "cos" and params[2] == "sin"


def _make_generic_invariants(model, text_model, ids):
    """Per-batch invariants for custom (x, cos, sin) decoder blocks."""
    import torch
    embed = model.get_input_embeddings()
    rope_fn = _find_rope_fn(model, text_model)
    with torch.no_grad():
        inputs_embeds = embed(ids)
        cos, sin = rope_fn(inputs_embeds.shape[1], inputs_embeds.device, torch.float32)
    return {"inputs_embeds": inputs_embeds, "cos": cos, "sin": sin}


def _run_generic_block(layer, hidden, inv):
    """Run one custom decoder block in rolling mode."""
    import torch
    with torch.no_grad():
        out = layer(hidden, inv["cos"], inv["sin"])
    return out[0] if isinstance(out, tuple) else out


def _verify_generic_rolling(model, text_model, decoder_layers, ids, device, tol=2e-2):
    """Check the single-block walk reproduces the model's own forward at layer 0.

    A wrong block signature or a missing per-layer input (structural bias, an
    alternating layer type) would silently produce garbage activations rather than
    raise, and every SAE downstream would train on them. So prove it once against
    a forward hook before trusting the fast path.
    """
    import torch
    grabbed = {}

    def _grab(_m, _i, out):
        grabbed["hidden"] = out[0] if isinstance(out, tuple) else out

    handle = decoder_layers[0].register_forward_hook(_grab)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in grabbed:
        return False, "hook never fired"

    inv = _make_generic_invariants(model, text_model, ids)
    rolled = _run_generic_block(decoder_layers[0], inv["inputs_embeds"], inv)

    ref = grabbed["hidden"].float()
    got = rolled.float()
    if ref.shape != got.shape:
        return False, f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
    denom = ref.abs().max().clamp_min(1e-6)
    err = (ref - got).abs().max() / denom
    return bool(err <= tol), f"max rel err {err:.2e} (tol {tol:g})"


def _produce_pool_generic_rolling(model, text_model, decoder_layers, layer, tok_dir,
                                  src_dir, dst_dir, device):
    """Custom-arch single-block rolling capture.

    layer 0: embed + block 0 over the token pool.
    layer>=1: block L over src_dir (= pool[L-1]).
    """
    import time
    import torch

    tok_paths = _shard_paths(tok_dir)
    n = len(tok_paths)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if len(_shard_paths(dst_dir)) >= n:
        print(f"  [produce L{layer}] pool already present ({n} shards) -- skip")
        return

    print(f"  [produce L{layer}] generic rolling block {layer} over {n} batches ...")
    t0 = time.time()
    report_every = max(1, n // 10)
    for i in range(n):
        ids = _read_shard(tok_dir, i).to(device)
        inv = _make_generic_invariants(model, text_model, ids)
        if layer == 0:
            hidden = inv["inputs_embeds"]
        else:
            hidden = _read_shard(src_dir, i).to(device, dtype=inv["inputs_embeds"].dtype)
        out = _run_generic_block(decoder_layers[layer], hidden, inv)
        _write_shard(dst_dir, i, out)
        if (i + 1) % report_every == 0:
            tok_s = (i + 1) * BATCH_TOKENS / (time.time() - t0)
            print(f"    produced {i+1}/{n}  ({tok_s/1e3:.1f}k tok/s)")
    print(f"  [produce L{layer}] done in {(time.time()-t0)/60:.1f}min -> {dst_dir}")


def _make_llama_invariants(text_model, ids):
    """Per-batch invariants for generic HF decoder rolling single-block capture.

    Mirrors the head of LlamaModel.forward: shared position_ids and rotary cos/sin
    passed to every decoder layer. The layer builds its own causal mask when
    attention_mask=None.
    """
    import torch
    with torch.no_grad():
        inputs_embeds = text_model.embed_tokens(ids)
        S = inputs_embeds.shape[1]
        cache_position = torch.arange(S, device=ids.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids=position_ids)
    return {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "attention_mask": None,
        "position_embeddings": position_embeddings,
    }


def _run_hf_block(layer, hidden, inv):
    """Run one Llama-like decoder layer in rolling mode."""
    import torch
    with torch.no_grad():
        out = layer(
            hidden,
            attention_mask=inv["attention_mask"],
            position_ids=inv["position_ids"],
            position_embeddings=inv["position_embeddings"],
            use_cache=False,
        )
    return out[0] if isinstance(out, tuple) else out


def _produce_pool_hf_rolling(model, text_model, decoder_layers, layer, tok_dir, src_dir,
                             dst_dir, device):
    """Generic HF single-block rolling capture (Llama/SmolLM2/Qwen).

    layer 0: embed_tokens + block 0 over the token pool.
    layer>=1: block L over src_dir (= pool[L-1]).
    """
    import time
    import torch

    tok_paths = _shard_paths(tok_dir)
    n = len(tok_paths)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if len(_shard_paths(dst_dir)) >= n:
        print(f"  [produce L{layer}] pool already present ({n} shards) -- skip")
        return

    print(f"  [produce L{layer}] HF rolling block {layer} over {n} batches ...")
    t0 = time.time()
    for i in range(n):
        ids = _read_shard(tok_dir, i).to(device)
        inv = _make_llama_invariants(text_model, ids)
        if layer == 0:
            hidden = inv["inputs_embeds"]
        else:
            hidden = _read_shard(src_dir, i).to(device, dtype=inv["inputs_embeds"].dtype)
        out = _run_hf_block(decoder_layers[layer], hidden, inv)
        _write_shard(dst_dir, i, out)
        if (i + 1) % 500 == 0:
            tok_s = (i + 1) * BATCH_TOKENS / (time.time() - t0)
            print(f"    produced {i+1}/{n}  ({tok_s/1e3:.1f}k tok/s)")
    print(f"  [produce L{layer}] done in {(time.time()-t0)/60:.1f}min -> {dst_dir}")


# ===========================================================================
#  Floating layer window (only active decoder blocks on GPU)
# ===========================================================================

class FloatingLayerWindow:
    """Keep the full LLM on CPU and move only the decoder blocks currently
    computing onto GPU. This is the "caterpillar" / sliding-window loader:
    at any production step only ~2 layers touch VRAM, everything else stays
    on host memory.

    Shared small components (embed_tokens, rotary_emb, and for Gemma-3n/4 the
    per-layer-embedding table + projection) are moved to GPU once and left there
    because every layer's production needs them. Decoder blocks are activated
    and deactivated around each production pass.
    """

    # Every module _make_invariants / _make_llama_invariants touches. Gemma-only
    # attrs are skipped via hasattr on Llama-style models and vice versa.
    SHARED_COMPONENTS = (
        "embed_tokens",              # all models
        "rotary_emb",                # all models
        "embed_tokens_per_layer",    # Gemma-3n/4 PLE token-identity table
        "per_layer_model_projection",  # Gemma-3n/4 PLE context projection
        "per_layer_projection_norm",   # Gemma-3n/4 PLE norm
    )

    def __init__(self, text_model, decoder_layers, device):
        import torch
        self.text_model = text_model
        self.decoder_layers = decoder_layers
        self.device = device
        self._active: set = set()
        # Pin shared components to GPU once. They are tiny compared to the
        # decoder stack and are needed by every layer's production.
        for name in self.SHARED_COMPONENTS:
            mod = getattr(text_model, name, None)
            if mod is not None:
                mod.to(device)

    def activate(self, layer: int):
        """Ensure decoder block `layer` is on GPU."""
        import torch
        if layer < 0 or layer >= len(self.decoder_layers):
            return
        if layer not in self._active:
            self.decoder_layers[layer].to(self.device)
            self._active.add(layer)

    def deactivate(self, layer: int):
        """Move decoder block `layer` back to CPU."""
        import torch
        if layer in self._active:
            self.decoder_layers[layer].to("cpu")
            self._active.discard(layer)

    def deactivate_all(self):
        """Move all active blocks back to CPU and clear the window."""
        import torch
        for layer in list(self._active):
            self.decoder_layers[layer].to("cpu")
        self._active.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_active(self, layers):
        """Activate the given set of layers, deactivating any others."""
        layers = set(int(l) for l in layers if 0 <= l < len(self.decoder_layers))
        to_deactivate = self._active - layers
        to_activate = layers - self._active
        for layer in to_deactivate:
            self.deactivate(layer)
        for layer in to_activate:
            self.activate(layer)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deactivate_all()
        return False


# ===========================================================================
#  Pool I/O  (activation shards on container-local NVMe at ROLLCACHE)
# ===========================================================================

def _pool_dir(tag: str) -> Path:
    return Path(ROLLCACHE) / tag


def _write_shard(dir_path: Path, i: int, tensor):
    import torch
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.to(torch.bfloat16).cpu(), dir_path / f"shard_{i:05d}.pt")


def _read_shard(dir_path: Path, i: int):
    import torch
    return torch.load(dir_path / f"shard_{i:05d}.pt", map_location="cpu", weights_only=True)


def _shard_paths(dir_path: Path):
    return sorted(dir_path.glob("shard_*.pt"))


def _rm_pool(dir_path: Path):
    """Delete a layer's activation pool once it is no longer needed."""
    import shutil
    if dir_path.exists():
        shutil.rmtree(dir_path, ignore_errors=True)


def _resume_pool_dir(seed: int) -> Path:
    """Persistent rolling-resume directory. Holds the pool of the last completed
    layer so a restart can resume the residual chain without regenerating from L0."""
    return Path(ROLLCACHE) / f"resume_pool_s{seed}"


def _resume_manifest_path(seed: int) -> Path:
    return _resume_pool_dir(seed) / "manifest.json"


def _save_resume_pool(pool_dir: Path, layer: int, seed: int, pool_batches: int,
                      model_id: str, capture: str, activation_norm_ref: float = None):
    """Persist pool_dir as the rolling resume checkpoint for `layer`.

    Keeps exactly one resume pool on disk (overwrites the previous one). Copying is
    O(one pool); on NVMe this is a few seconds per layer vs. hours of training, and
    it removes any dependency on the transient active pool_dir which gets consumed
    during the next layer's production.
    """
    import json
    import shutil
    rdir = _resume_pool_dir(seed)
    _rm_pool(rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    for p in _shard_paths(pool_dir):
        shutil.copyfile(p, rdir / p.name)
    manifest = {
        "last_completed_layer": layer,
        "seed": seed,
        "pool_batches": pool_batches,
        "model_id": model_id,
        "capture": capture,
        "activation_norm_ref": activation_norm_ref,
    }
    with open(_resume_manifest_path(seed), "w") as f:
        json.dump(manifest, f)


def _find_resume_layer(seed: int, pool_batches: int, model_id: str, capture: str) -> int:
    """Return the last completed layer from a rolling resume checkpoint, or -1 if
    none exists or it was produced with incompatible settings."""
    import json
    mpath = _resume_manifest_path(seed)
    if not mpath.exists():
        return -1
    try:
        with open(mpath) as f:
            m = json.load(f)
        # Float variants produce bit-identical pools to their base mode (same walk,
        # weights merely hoisted between devices), so resume across the family.
        def _capture_family(c):
            return {"rolling-float": "rolling", "rolling-hf-float": "rolling-hf"}.get(c, c)
        if (m.get("seed") != seed or
            m.get("pool_batches") != pool_batches or
            m.get("model_id") != model_id or
            _capture_family(m.get("capture")) != _capture_family(capture)):
            return -1
        rdir = _resume_pool_dir(seed)
        n_saved = len(_shard_paths(rdir)) if rdir.exists() else 0
        # A complete resume pool must have the expected number of shards. A partial
        # copy (e.g., interrupted save) is treated as absent so we regenerate cleanly.
        if n_saved != pool_batches:
            return -1
        return int(m.get("last_completed_layer", -1))
    except Exception:
        return -1


# ===========================================================================
#  Token pool capture  (run once -- the same tokens flow through every layer)
# ===========================================================================

def _capture_token_pool(hf_token, seed, pool_batches, use_pretok, tok_dir: Path, bos_token_id,
                        model_id=None, vocab_size=None):
    """Materialize T batches of BOS-prepended [n_seqs, SEQ_LEN] token shards so
    every layer trains on identical tokens (residual pool[L]=block L output == input
    to block L+1 requires this).

    Iterates the IterableDataset DIRECTLY in the main process -- no DataLoader worker
    procs. A persistent-worker spawn DataLoader crashes the interpreter on teardown
    (PyGILState_Release during finalization). With use_pretok the dataset is memmap
    reads (instant); live streaming is single-threaded but this is a one-time capture.

    Token shards are validated against `vocab_size`; stale caches (e.g. from a
    previous model) are discarded and regenerated.  Pre-tokenized shards are only
    used when their manifest vocab size matches the target model; otherwise we fall
    back to streaming tokenization.
    """
    pretok_dir = None
    import torch

    paths = _shard_paths(tok_dir)
    if paths and len(paths) >= pool_batches:
        if vocab_size is None:
            print(f"  [tokens] reusing {len(paths)} cached token shards")
            return
        sample = _read_shard(tok_dir, 0)
        if sample.min() >= 0 and sample.max() < vocab_size:
            print(f"  [tokens] reusing {len(paths)} cached token shards")
            return
        print(f"  [tokens] cached shards invalid for {model_id} "
              f"(ids not in [0,{vocab_size})); regenerating")
        _rm_pool(tok_dir)

    # Resolve model-specific pretok dir when one is provided.
    # PRETOK_DIR is the legacy default; callers can pass a model-specific path.
    if use_pretok and pretok_dir is None:
        from pathlib import Path as _Path
        pretok_dir = str(_Path(PRETOK_DIR).parent / _slug(model_id or MODEL_ID))

    # Pre-tokenized shards are model-specific.  If they are absent or their vocab
    # doesn't match, fall back to on-the-fly tokenization with the correct tokenizer.
    if use_pretok:
        import json
        from pathlib import Path as _Path
        manifest_path = _Path(pretok_dir) / "manifest.json"
        if not manifest_path.exists():
            print(f"  [tokens] no pretok shards at {pretok_dir}; "
                  f"falling back to streaming tokenization")
            use_pretok = False
        elif vocab_size is not None:
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                pretok_vocab = manifest.get("vocab_size")
                if pretok_vocab is not None and pretok_vocab != vocab_size:
                    print(f"  [tokens] pretok vocab_size {pretok_vocab} != model {vocab_size}; "
                          f"falling back to streaming")
                    use_pretok = False
            except Exception:
                pass

    dataset = _build_token_dataset(
        hf_token=hf_token, batch_tokens=BATCH_TOKENS, seed=seed, use_pretok=use_pretok,
        pretok_dir=pretok_dir, model_id=model_id)
    it = iter(dataset)                                     # main-process iteration

    # Pre-tokenized shards are tokenizer-specific.  Validate the first batch against
    # the model's vocab; if it doesn't fit, stream + tokenize with the right tokenizer.
    if vocab_size is not None:
        probe = next(it)
        if probe.min() < 0 or probe.max() >= vocab_size:
            print(f"  [tokens] pretok ids out of range (max={probe.max()}, "
                  f"vocab={vocab_size}); falling back to streaming tokenization")
            del it, dataset
            use_pretok = False
            dataset = _build_token_dataset(
                hf_token=hf_token, batch_tokens=BATCH_TOKENS, seed=seed, use_pretok=False,
                model_id=model_id)
            it = iter(dataset)
            # Re-validate: if the rebuilt stream still emits out-of-range ids the
            # tokenizer itself is wrong -- fail loudly instead of writing poison.
            probe = next(it)
            if probe.min() < 0 or probe.max() >= vocab_size:
                raise RuntimeError(
                    f"streaming tokenizer for {model_id} produced ids outside "
                    f"[0,{vocab_size}) (max={int(probe.max())}); check model_id/tokenizer")
            import itertools
            it = itertools.chain([probe], it)
        else:
            import itertools
            it = itertools.chain([probe], it)

    n_seqs = BATCH_TOKENS // SEQ_LEN
    tok_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [tokens] capturing {pool_batches} token shards ([{n_seqs},{SEQ_LEN}]) ...")
    for i in range(pool_batches):
        batch = next(it)                                   # [BATCH_TOKENS]
        real = batch[: n_seqs * (SEQ_LEN - 1)].view(n_seqs, SEQ_LEN - 1)
        bos = torch.full((n_seqs, 1), bos_token_id, dtype=real.dtype)
        ids = torch.cat([bos, real], dim=1)                # [n_seqs, SEQ_LEN]
        torch.save(ids.cpu(), tok_dir / f"shard_{i:05d}.pt")
        if (i + 1) % 500 == 0:
            print(f"    token shard {i+1}/{pool_batches}")
    del it, dataset
    print(f"  [tokens] done -> {tok_dir}")


# ===========================================================================
#  Pool production  (capture activations -> pool[L] shards)
#  Two backends, both writing the same shard format the provider reads:
#    _produce_pool_hooked  -- model-agnostic forward-hook capture (default)
#    _produce_pool         -- Gemma-3n/4 single-block walk (opt-in, --capture rolling)
# ===========================================================================

def _produce_pool_hooked(model, decoder_layers, layer, tok_dir, dst_dir, device):
    """Model-agnostic capture: run the model forward over the token pool and record
    the residual-stream output of decoder block `layer`.

    Works for any AutoModelForCausalLM (and text-only forwards of multimodal models).
    Correct by construction -- it observes the real forward, so there is no bit-exact
    risk. We use output_hidden_states=True to extract the layer output; this is slower
    than a truncated forward but avoids model-family-specific assumptions about layer
    truncation (Llama, Qwen, etc. break when decoder_layers is shortened).

    Each layer is captured independently (one forward per layer), so disk stays at one
    pool while compute is N_layers x a full forward."""
    import torch
    import time

    tok_paths = _shard_paths(tok_dir)
    n = len(tok_paths)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if len(_shard_paths(dst_dir)) >= n:
        print(f"  [produce L{layer}] pool already present ({n} shards) -- skip")
        return

    print(f"  [produce L{layer}] hidden-states capture over {n} batches ...")
    t0 = time.time()

    # Custom-code models may accept output_hidden_states but never populate it
    # (out.hidden_states stays None). Fall back to a forward hook on the target
    # block; hooks observe the real forward, so correctness is unaffected.
    hook_state = {}

    def _grab(_module, _inp, out):
        hook_state["hidden"] = out[0] if isinstance(out, tuple) else out

    hook_handle = None
    for i in range(n):
        ids = _read_shard(tok_dir, i).to(device)           # [n_seqs, SEQ_LEN] int
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        # hidden_states[0] = embeddings input, hidden_states[k] = output of block k-1
        if out.hidden_states is not None:
            hidden = out.hidden_states[layer + 1]
        else:
            if hook_handle is None:
                print(f"  [produce L{layer}] model ignores output_hidden_states; "
                      f"switching to forward-hook capture")
                hook_handle = decoder_layers[layer].register_forward_hook(_grab)
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
            hidden = hook_state.pop("hidden")
        _write_shard(dst_dir, i, hidden)
        if (i + 1) % 500 == 0:
            tok_s = (i + 1) * BATCH_TOKENS / (time.time() - t0)
            print(f"    produced {i+1}/{n}  ({tok_s/1e3:.1f}k tok/s)")
    if hook_handle is not None:
        hook_handle.remove()
    print(f"  [produce L{layer}] done in {(time.time()-t0)/60:.1f}min -> {dst_dir}")


def _produce_pool(model, text_model, decoder_layers, tcfg, layer, tok_dir, src_dir,
                  dst_dir, device):
    """Gemma-3n/4 single-block walk. Produce pool[layer] (block-L output for every batch).
    layer 0: embed_tokens + block 0 over the token pool.
    layer>=1: block L over src_dir (= pool[L-1]).
    Tokens are always needed for per_layer_inputs (PLE).

    Source pools are NOT deleted during production. This keeps the previous layer's
    pool intact so a crash during the next layer's training can resume from it. Pool
    retention is managed by the orchestrator (keeps last 3 pools, deletes older ones).
    """
    import time

    tok_paths = _shard_paths(tok_dir)
    n = len(tok_paths)
    dst_dir.mkdir(parents=True, exist_ok=True)
    # idempotent skip if already produced
    if len(_shard_paths(dst_dir)) >= n:
        print(f"  [produce L{layer}] pool already present ({n} shards) -- skip")
        return
    print(f"  [produce L{layer}] running block {layer} over {n} batches ...")
    t0 = time.time()
    for i in range(n):
        ids = _read_shard(tok_dir, i).to(device)           # [n_seqs, SEQ_LEN] int
        inv = _make_invariants(text_model, tcfg, ids)
        if layer == 0:
            hidden = inv["inputs_embeds0"]                  # scaled word embeds
        else:
            hidden = _read_shard(src_dir, i).to(device, dtype=inv["inputs_embeds0"].dtype)
        out = _run_block(decoder_layers, tcfg, layer, hidden, inv)
        _write_shard(dst_dir, i, out)
        if (i + 1) % 500 == 0:
            tok_s = (i + 1) * BATCH_TOKENS / (time.time() - t0)
            print(f"    produced {i+1}/{n}  ({tok_s/1e3:.1f}k tok/s)")
    print(f"  [produce L{layer}] done in {(time.time()-t0)/60:.1f}min -> {dst_dir}")


# ===========================================================================
#  Activation provider  (reads pool[L] shards, prefetched, yields SAE batches)
# ===========================================================================

class RollingActivationProvider:
    """Yields [BATCH_TOKENS, D_IN] bf16 activations on device from pool[L] shards.
    Background thread prefetches the next shard so the SAE hot loop never waits.

    The reader pool is intentionally simple: resume restores model/optimizer/
    scheduler state, while the activation stream restarts from the cached pool.
    That keeps checkpointing useful for preemption without serializing a large
    asynchronous prefetch queue.
    """

    def __init__(self, pool_dir: Path, device, seed=0, queue_size=12, n_workers=4):
        import threading
        import queue as _queue
        self.paths = sorted(_shard_paths(pool_dir))
        if not self.paths:
            raise RuntimeError(f"No pool shards in {pool_dir}")
        self.device = device
        self.seed = seed
        self._q = _queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        # Multiple reader threads: torch.load releases the GIL during the file read,
        # so N threads overlap N shard reads. A single reader could not saturate even
        # local NVMe (each shard is ~168MB at d_in=2560), leaving the H100 waiting.
        n_workers = max(1, min(n_workers, len(self.paths)))
        self._threads = [
            threading.Thread(target=self._worker, args=(w, n_workers), daemon=True)
            for w in range(n_workers)
        ]
        for t in self._threads:
            t.start()

    def _worker(self, wid, n_workers):
        import random as _random
        import queue as _queue
        import torch
        rng = _random.Random(self.seed * 7919 + wid)
        my_idx = [i for i in range(len(self.paths)) if i % n_workers == wid]
        if not my_idx:
            return
        while not self._stop.is_set():
            rng.shuffle(my_idx)
            for idx in my_idx:
                if self._stop.is_set():
                    return
                t = torch.load(self.paths[idx], map_location="cpu", weights_only=True)  # [n_seqs,SEQ_LEN,d] bf16
                t = t.reshape(-1, t.shape[-1])
                if self.device.type == "cuda" and torch.cuda.is_available() and not t.is_pinned():
                    t = t.pin_memory()
                while not self._stop.is_set():
                    try:
                        self._q.put(t, timeout=0.5)
                        break
                    except _queue.Full:
                        continue

    def next_batch(self):
        """Return the next activation batch as a CPU pinned bf16 tensor.

        The caller (typically a double-buffer wrapper) controls when and on which
        CUDA stream the H2D transfer happens, so the transfer can overlap with GPU
        computation instead of serializing on the default stream.
        """
        t = self._q.get()
        return t                                            # [BATCH_TOKENS, D_IN] bf16, CPU pinned

    def next_batch_to_device(self, non_blocking=True):
        """Convenience path for callers that do not need transfer overlap."""
        return self.next_batch().to(self.device, non_blocking=non_blocking)

    # Stubs retained so any external caller that touches resume_state on a
    # fresh run (e.g. the trainer's checkpoint-restore path) gets a no-op
    # rather than AttributeError. The provider itself does not support
    # resume in this build.
    def get_state(self):
        return None

    def restore_state(self, state):
        pass

    def close(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass
        for t in getattr(self, "_threads", []):
            t.join(timeout=1.0)


class _ActivationDoubleBuffer:
    """Overlap CPU->GPU transfer of batch N+1 with GPU computation on batch N.

    The provider's reader threads keep the next batch in CPU pinned memory.
    This wrapper moves that batch to GPU on a dedicated transfer stream while the
    training loop is still processing the current batch on the default stream.
    """

    def __init__(self, provider: RollingActivationProvider, device):
        import torch
        self.provider = provider
        self.device = device
        self._transfer_stream = torch.cuda.Stream(device=device)
        self._next_gpu = None
        self._ready_event = None
        self._prefetch()

    def _prefetch(self):
        import torch
        cpu = self.provider.next_batch()                   # CPU pinned bf16
        gpu = None
        with torch.cuda.stream(self._transfer_stream):
            gpu = cpu.to(self.device, non_blocking=True)  # H2D on transfer stream
        self._next_gpu = gpu
        self._ready_event = torch.cuda.Event()
        self._ready_event.record(self._transfer_stream)

    def next_batch(self):
        """Return a GPU-ready activation batch, then start transferring the next one."""
        self._ready_event.wait()                           # ensure H2D is done
        gpu = self._next_gpu
        # Kick off the next transfer immediately so it overlaps with the upcoming
        # optimizer/compute work on the default stream.
        self._prefetch()
        return gpu

    def next_batch_to_device(self, non_blocking=True):
        """Passthrough for callers that need the legacy interface."""
        return self.next_batch()

    def get_state(self):
        return self.provider.get_state()

    def restore_state(self, state):
        return self.provider.restore_state(state)

    def close(self):
        self.provider.close()


# ===========================================================================
#  SAE training body  (trains one JumpReLU SAE on cached activations)
# ===========================================================================

def _save_full_checkpoint(out_dir, step, sae, optimizer, scheduler, rng_states,
                          feature_fire_counts, steps_since_fired, provider_state):
    """Save a complete checkpoint: model, optimizer, scheduler, RNG, dead-feature stats."""
    import torch
    energy_config_keys = [
        "initial_lr_multiplier", "initial_lr_decay_steps",
        "activation_norm_ref", "activation_norm_lr_alpha",
        "activation_norm_lr_min", "activation_norm_lr_max",
        "lr_energy_max", "convergence_lockout_rel",
        "constraint_lr_floor",
        "early_pulse_steps", "early_pulse_multiplier",
        "early_pulse_warmup_floor", "early_pulse_dampen",
        "al_landing_zone_rel", "al_landing_min_progress",
        "al_landing_gain_max", "al_slingshot_overshoot_rel",
        "al_slingshot_gain_max",
        "lambda_l0_max", "lambda_l0_min", "al_mu", "al_dual_step",
        "target_l0", "l0_tolerance",
        "stall_pulse_enabled", "stall_warmup_steps",
        "stall_cooldown_steps", "stall_pulse_steps",
        "stall_pulse_max_extra", "stall_pulse_min_multiplier",
        "stall_fast_alpha", "stall_slow_alpha",
        "stall_crossover_ratio", "stall_min_slow_progress",
        "stall_abs_progress_floor", "stall_dead_suppress_pct",
        "threshold_nudge_gain", "threshold_nudge_every", "threshold_nudge_l0_min",
        "ste_bandwidth",
    ]
    ckpt = {
        "step": step,
        "sae_state": {k: v.cpu().clone() for k, v in sae.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": {
            "lambda_l0": scheduler.lambda_l0,
            "mode": scheduler.mode,
            "mode_steps": scheduler.mode_steps,
            "total_steps": scheduler.total_steps,
            "event_counter": scheduler.event_counter,
            "transition_log": scheduler.transition_log[-100:],  # last 100 transitions
            "_lambda_history": scheduler._lambda_history,
            "_activation_norm_ema": getattr(scheduler, "_activation_norm_ema", None),
            "_activation_norm_preflight": getattr(scheduler, "_activation_norm_preflight", None),
            "_prev_l0": getattr(scheduler, "_prev_l0", None),
            "_l0_progress_fast": getattr(scheduler, "_l0_progress_fast", None),
            "_l0_progress_slow": getattr(scheduler, "_l0_progress_slow", None),
            "_stall_pulse_remaining": getattr(scheduler, "_stall_pulse_remaining", 0),
            "_stall_pulse_multiplier": getattr(scheduler, "_stall_pulse_multiplier", 1.0),
            "_last_stall_pulse_step": getattr(scheduler, "_last_stall_pulse_step", -10**9),
            "_energy_dampen": getattr(scheduler, "_energy_dampen", 1.0),
            # Phase control state (observable; gated in later branches).
            "phase": getattr(scheduler, "phase", "DESCENT"),
            "phase_step": getattr(scheduler, "phase_step", 0),
            "pin_entry_step": getattr(scheduler, "pin_entry_step", None),
            "pinned_lambda": getattr(scheduler, "pinned_lambda", None),
            "pin_ev_count": getattr(scheduler, "pin_ev_count", 0),
            "pin_retry_count": getattr(scheduler, "pin_retry_count", 0),
            "energy_config": {
                k: getattr(scheduler.config, k)
                for k in energy_config_keys
                if hasattr(scheduler.config, k)
            },
        },
        "rng_state": rng_states,
        "feature_fire_counts": feature_fire_counts.cpu().clone(),
        "steps_since_fired": steps_since_fired.cpu().clone(),
        "provider_state": provider_state,
    }
    ckpt_path = out_dir / "checkpoint_full.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def _load_full_checkpoint(ckpt_path, sae, optimizer, scheduler, device):
    """Load a full checkpoint, returning (step, rng_states, feature_fire_counts,
    steps_since_fired, provider_state) or None if loading fails."""
    import torch
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sae.load_state_dict({k: v.to(device) for k, v in ckpt["sae_state"].items()})
        optimizer.load_state_dict(ckpt["optimizer_state"])
        sched_state = ckpt["scheduler_state"]
        scheduler.lambda_l0 = sched_state["lambda_l0"]
        scheduler.mode = sched_state["mode"]
        scheduler.mode_steps = sched_state["mode_steps"]
        scheduler.total_steps = sched_state["total_steps"]
        scheduler.event_counter = sched_state["event_counter"]
        scheduler.transition_log = sched_state["transition_log"]
        scheduler._lambda_history = sched_state.get("_lambda_history", [])
        scheduler._activation_norm_ema = sched_state.get("_activation_norm_ema")
        scheduler._activation_norm_preflight = sched_state.get("_activation_norm_preflight")
        scheduler._prev_l0 = sched_state.get("_prev_l0")
        scheduler._l0_progress_fast = sched_state.get("_l0_progress_fast")
        scheduler._l0_progress_slow = sched_state.get("_l0_progress_slow")
        scheduler._stall_pulse_remaining = sched_state.get("_stall_pulse_remaining", 0)
        scheduler._stall_pulse_multiplier = sched_state.get("_stall_pulse_multiplier", 1.0)
        scheduler._last_stall_pulse_step = sched_state.get("_last_stall_pulse_step", -10**9)
        scheduler._energy_dampen = sched_state.get("_energy_dampen", 1.0)
        # Phase control state — backward-compatible defaults for old checkpoints
        # saved before the phase machine existed.
        scheduler.phase = sched_state.get("phase", "DESCENT")
        scheduler.phase_step = sched_state.get("phase_step", 0)
        scheduler.pin_entry_step = sched_state.get("pin_entry_step", None)
        scheduler.pinned_lambda = sched_state.get("pinned_lambda", None)
        scheduler.pin_ev_count = sched_state.get("pin_ev_count", 0)
        scheduler.pin_retry_count = sched_state.get("pin_retry_count", 0)
        default_constraint_lr_floor = scheduler.config.constraint_lr_floor
        energy_config = sched_state.get("energy_config", {})
        for k, v in energy_config.items():
            if hasattr(scheduler.config, k):
                setattr(scheduler.config, k, v)
        # Older checkpoints were saved before the slingshot controller existed.
        # Keep their dynamic state, but upgrade the final-stretch policy.
        if "al_slingshot_gain_max" not in energy_config:
            scheduler.config.constraint_lr_floor = max(
                scheduler.config.constraint_lr_floor,
                default_constraint_lr_floor,
            )
        return {
            "step": ckpt["step"],
            "rng_state": ckpt["rng_state"],
            "feature_fire_counts": ckpt["feature_fire_counts"].to(device),
            "steps_since_fired": ckpt["steps_since_fired"].to(device),
            "provider_state": ckpt.get("provider_state"),
        }
    except Exception as e:
        print(f"  WARNING: checkpoint load failed ({e})")
        return None


def train_sae_on_activations(layer, d_in, seed, provider, *, frozen_decoder=False,
                             max_steps=N_STEPS, bdec_batches=BDEC_INIT_BATCHES,
                             microbatch_tokens=MICROBATCH_TOKENS,
                             resume_from=None, push=True, activation_norm_ref=None,
                             cpu=False):
    """Train one JumpReLU SAE on activations supplied by `provider`.

    Gradient accumulation: if `microbatch_tokens < BATCH_TOKENS`, accumulate gradients
    over `BATCH_TOKENS // microbatch_tokens` microbatches before stepping.

    Resume: if `resume_from` points to a `checkpoint_full.pt`, restore model,
    optimizer, scheduler, RNG, and dead-feature stats.

    bf16 path: activations are kept in bf16 through the forward; loss/reductions in fp32.
    """
    import contextlib
    import json
    import math
    import time
    import threading
    import torch
    import torch.nn as nn
    from sae_scheduler import SAEAECSConfig, SAEEventControlScheduler

    def _autocast():
        if device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    n_steps = int(min(max_steps, N_STEPS))
    accum_steps = BATCH_TOKENS // microbatch_tokens
    assert BATCH_TOKENS % microbatch_tokens == 0, \
        f"BATCH_TOKENS ({BATCH_TOKENS}) must be divisible by microbatch_tokens ({microbatch_tokens})"

    device = torch.device("cuda" if not cpu else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Critical for H100
    torch.set_float32_matmul_precision("high")

    # -- LAMBDA tier (layers 0-14 only -> hot/trough/mid) ---------------------
    if layer < 10:
        tier = "hot"
    elif layer < 12:
        tier = "trough"
    else:
        tier = "mid"
    {"hot": 1e-3, "trough": 3e-4, "mid": 5e-4}[tier]

    # Aggressive low-L0 runs (K <= 100) converge fast and then collapse into
    # feature death if training continues. Tighten the AL convergence window and
    # the dead-feature guards for this regime while keeping K=500 defaults intact.
    aggressive_k = K <= 100
    ev_check_every = 250 if aggressive_k else 500
    ev_stop_patience = 2 if aggressive_k else 5
    dead_emergency_thresh = 10.0 if aggressive_k else 20.0
    dead_emergency_cooldown = 1500 if aggressive_k else 5000

    sae_cfg = SAEAECSConfig(
        base_lr=LR, warmup_steps=min(LR_WARMUP_STEPS, max(1, n_steps // 5)), total_steps=n_steps,
        event_warmup_steps=5_000, instability_z_thresh=10.0,
        loss_spike_ratio=3.0, loss_spike_min_recent=50,
        plateau_grad_norm_thresh=1e-4,
        recovery_lr_factor=0.7, recovery_momentum_factor=0.5, explore_lr_factor=1.5,
        cooldown_steps=250, event_persistence=3, mode_verbose=True,
        target_l0=float(K_INIT), l0_tolerance=0.20,
        use_augmented_lagrangian=True, lambda_l0_init=0.0, lambda_l0_min=0.0,
        # E4B-tuned: threshold nudge + wider STE bandwidth for L0 descent.
        # Lambda ceiling at 1.0 — the nudge does the heavy lifting on threshold,
        # lambda provides supporting gradient pressure through the STE channel.
        lambda_l0_max=1.0, al_mu=5e-5, al_dual_step=2e-8, al_log_every=LOG_EVERY,
        ev_floor=0.0, ev_floor_patience=10**9, ev_drop_thresh=-1.0,
        ev_stop_thresh=0.88, ev_stop_patience=ev_stop_patience,
        ev_check_every=ev_check_every,
        dead_emergency_thresh=dead_emergency_thresh,
        dead_emergency_cooldown=dead_emergency_cooldown,
        # live_tune_path is set after out_dir is created below
    )
    out_suffix = "_frozen" if frozen_decoder else ""

    # -- wandb ----------------------------------------------------------------
    use_wandb = False
    wandb = None
    if WANDB_PROJECT and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb as _wandb
            try:
                if _wandb.run is not None:
                    _wandb.finish(exit_code=1)
            except Exception:
                pass
            _wandb.init(project=WANDB_PROJECT, name=f"L{layer:02d}_s{seed}",
                        reinit="finish_previous",
                        config={"layer": layer, "seed": seed, "model_id": MODEL_ID,
                                "n_features": N_FEATURES, "k": K, "batch_tokens": BATCH_TOKENS,
                                "microbatch_tokens": microbatch_tokens, "accum_steps": accum_steps,
                                "lr": LR, "n_steps": N_STEPS, "tier": tier, "resume_from": resume_from})
            wandb = _wandb
            use_wandb = True
            print(f"  WandB: {wandb.run.url}")
        except Exception as e:
            print(f"  WARNING: wandb init failed ({e})")

    print(f"\n{'='*60}\nROLLING SAE  layer={layer}  seed={seed}  tier={tier}  d_in={d_in}")
    print(f"  accum_steps={accum_steps} (microbatch={microbatch_tokens//1024}k tokens)")
    if resume_from:
        print(f"  resume_from={resume_from}")
    print(f"{'='*60}")

    out_dir = Path(SAE_DIR) / f"layer_{layer:02d}_s{seed}{out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Live-tune dials -------------------------------------------------------
    # Write a JSON file to override AL/scheduler params at runtime.
    # Keys: lambda_l0_max, al_mu, al_dual_step, target_l0, lambda_l0_override, ...
    # The trainer checks TWO paths: the layer output dir (on Volume, may lag)
    # and /tmp/live_tune.json (always container-local, always fresh).
    # To crank dials on a running Modal container, use:
    #   modal exec <app> -- bash -c 'echo "{\"lambda_l0_max\": 0.3}" > /tmp/live_tune.json'
    # Or from the training container's shell directly.
    sae_cfg.live_tune_path = str(out_dir / "live_tune.json")
    sae_cfg.live_tune_path_alt = "/tmp/live_tune.json"

    sae = _make_sae(d_in, N_FEATURES, seed).to(device)
    # Sync STE bandwidth from config to SAE (live-tune can override it later)
    sae.ste_bandwidth = sae_cfg.ste_bandwidth
    if frozen_decoder:
        sae.W_dec.weight.requires_grad_(False)

    # Try torch.compile on the SAE forward. The timing showed ~200ms/step of
    # fwd+bwd CPU stall from Python kernel-launch overhead on H100, which
    # compile is designed to eliminate. We compile only the SAE module, not
    # the whole training loop, so resample/reset/scheduler branches don't
    # poison the graph. Custom autograd Functions cause graph breaks around
    # the JumpReLU, but the matmuls still get fused/optimized. If compile
    # fails or slows things, disable with SAE_COMPILE=0.
    use_compile = os.environ.get("SAE_COMPILE", "0").lower() in ("1", "true", "yes", "on")
    if use_compile and device.type == "cuda":
        try:
            import torch
            # default mode: balanced compile time vs runtime speed
            # fullgraph=False allows graph breaks at the custom autograd Function.
            sae = torch.compile(sae, mode="default", fullgraph=False, dynamic=False)
            print(f"  torch.compile enabled on SAE (default, fullgraph=False)")
        except Exception as e:
            print(f"  WARNING: torch.compile failed ({e}), falling back to eager SAE")
    else:
        print(f"  torch.compile disabled (SAE_COMPILE={use_compile}, device={device.type})")

    # torch.compile is applied above to the SAE module only, not the whole loop.

    optimizer = torch.optim.Adam(
        [{"params": sae.W_enc.parameters(), "weight_decay": 0},
         {"params": sae.W_dec.parameters(), "weight_decay": 1e-4},
         {"params": [sae.b_dec], "weight_decay": 0},
         {"params": [sae.log_threshold], "weight_decay": 0}],
        lr=LR, betas=(0.9, 0.999), fused=(device.type == "cuda"))
    scheduler = SAEEventControlScheduler(optimizer, sae_cfg, mode_label=f"L{layer:02d}",
                                         layer=layer)

    preflight_stats = {
        "activation_norm_probe": None,
        "activation_norm_ref": activation_norm_ref,
        "initial_l0_probe": None,
        "initial_lr_multiplier": 1.0,
        "early_pulse_multiplier": 1.0,
        "layer_lr_multiplier": 1.0,
    }

    # -- Resume from checkpoint -----------------------------------------------
    start_step = 1
    revival = RevivalController(N_FEATURES, device,
                                aux_k_policy=getattr(sae_cfg, "aux_k_policy", "legacy"))
    err_buffer = []

    if resume_from and Path(resume_from).exists():
        loaded = _load_full_checkpoint(resume_from, sae, optimizer, scheduler, device)
        if loaded:
            start_step = loaded["step"] + 1
            revival.load_state(loaded["feature_fire_counts"], loaded["steps_since_fired"])
            # Restore RNG states
            if "cuda" in loaded["rng_state"]:
                torch.cuda.set_rng_state(loaded["rng_state"]["cuda"])
            if "cpu" in loaded["rng_state"]:
                torch.set_rng_state(loaded["rng_state"]["cpu"])
            print(f"  Resumed from step {start_step}, lambda={scheduler.lambda_l0:.3e}, mode={scheduler.mode}")
            # Provider resume is optional. The current async provider restarts
            # its cached activation stream instead of serializing queued shards.
            if loaded.get("provider_state"):
                provider.restore_state(loaded["provider_state"])

    # -- b_dec init from provider activations (skip if resumed) ---------------
    if start_step == 1:
        print(f"Estimating b_dec from {bdec_batches} batches ...")
        acc = torch.zeros(d_in, device=device, dtype=torch.float32)
        n_acc = 0
        probe_batch = None
        for _ in range(bdec_batches):
            a = provider.next_batch_to_device()
            if probe_batch is None:
                probe_batch = a
            acc += a.sum(dim=0)
            n_acc += a.shape[0]
        if n_acc > 0:
            with torch.no_grad():
                sae.b_dec.copy_(acc / n_acc)
            print(f"  b_dec set from {n_acc} tokens, ||b_dec||={sae.b_dec.norm().item():.4f}")
        if probe_batch is not None:
            probe_tokens = min(probe_batch.shape[0], max(1024, min(microbatch_tokens, 8192)))
            probe = probe_batch[:probe_tokens]
            with torch.no_grad(), _autocast():
                probe_pre = sae.encode_pre(probe)
                initial_l0 = sae.l0_indicator(probe_pre).sum(dim=-1).float().mean().item()
            activation_norm = probe.float().pow(2).mean().sqrt().item()
            ref_norm = activation_norm_ref if activation_norm_ref and activation_norm_ref > 0 else activation_norm
            norm_ratio = activation_norm / max(ref_norm, 1e-8)

            # -- Threshold warmup from empirical percentile -----------------------
            # When L0 >> target at init (deep layers can be 4-8x), shift thresholds
            # up from the pre-activation distribution so features start closer to
            # the right sparsity level.  Target a WARMUP L0 that's well ABOVE the
            # final target (3x target, or 30% of initial, whichever is smaller) —
            # aiming for the exact target L0 overcorrects because setting all
            # thresholds uniformly kills natural feature variation and drops L0
            # below target, leaving the AL integrator with nothing to integrate.
            if initial_l0 > K * 2:  # only warm up when L0 is significantly above target
                warmup_l0_target = min(float(K) * 3, initial_l0 * 0.3)
                warmup_l0_target = max(warmup_l0_target, float(K) * 1.5)  # floor at 1.5x target
                warmup_frac = warmup_l0_target / float(N_FEATURES)  # fraction of features we want active
                # percentile of |pre-activations| to use as threshold
                pctile = max(0.5, (1.0 - warmup_frac) * 100.0)
                # Subsample to ~1M elements for quantile — full tensor is too large
                # (probe_tokens × N_FEATURES can be >600M elements, quantile OOMs)
                abs_pre_flat = probe_pre.abs().flatten().float()
                max_quantile_elems = 1_000_000
                if abs_pre_flat.numel() > max_quantile_elems:
                    perm = torch.randperm(abs_pre_flat.numel(), device=abs_pre_flat.device)[:max_quantile_elems]
                    abs_pre_sample = abs_pre_flat[perm]
                else:
                    abs_pre_sample = abs_pre_flat
                warmup_threshold = torch.quantile(abs_pre_sample, pctile / 100.0).item()
                warmup_threshold = max(warmup_threshold, 0.1)  # floor at 0.1 to avoid degenerate threshold
                current_thr_mean = sae.log_threshold.exp().mean().item()
                with torch.no_grad():
                    # Set all thresholds to the warmup value.  This gives every feature
                    # a reasonable starting point instead of the fixed INIT_THRESHOLD which
                    # can be wildly wrong for deep layers with large activations.
                    sae.log_threshold.data.fill_(math.log(warmup_threshold))
                    # Clear Adam state for log_threshold so the optimizer doesn't
                    # carry momentum from the old threshold values.
                    state = optimizer.state.get(sae.log_threshold, {})
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                        state["exp_avg_sq"].zero_()
                print(f"  [THRESH WARMUP] L0_probe={initial_l0:.0f} >> target={K} → "
                      f"warmup_target_L0={warmup_l0_target:.0f} (pctile={pctile:.1f}%) "
                      f"warmup_thr={warmup_threshold:.4f} old_thr_mean={current_thr_mean:.4f}")
                # Re-measure L0 after warmup to update the scheduler's initial state
                with torch.no_grad(), _autocast():
                    warmup_l0 = sae.l0_indicator(sae.encode_pre(probe)).sum(dim=-1).float().mean().item()
                print(f"  [THRESH WARMUP] L0 after warmup: {warmup_l0:.1f} (was {initial_l0:.1f}, "
                      f"target_warmup_L0={warmup_l0_target:.0f})")
                initial_l0 = warmup_l0  # use updated L0 for the rest of preflight
            norm_mult = max(0.90, min(1.35, norm_ratio ** 0.5))
            layer_mult = 1.0
            if layer >= 16:
                layer_mult = min(1.35, 1.15 + 0.01 * (layer - 16))
            violation = max(0.0, initial_l0 - float(K)) / max(float(K), 1.0)
            coupled_mult = 1.0 + min(0.30, 0.06 * violation)
            early_pulse_mult = 1.0 + min(0.35, 0.08 * violation)
            if layer >= 16:
                early_pulse_mult += 0.08
            initial_lr_mult = min(1.65, norm_mult * layer_mult * coupled_mult)

            sae_cfg.activation_norm_ref = ref_norm
            sae_cfg.initial_lr_multiplier = initial_lr_mult
            sae_cfg.early_pulse_multiplier = min(1.45, early_pulse_mult)
            scheduler._activation_norm_ema = activation_norm
            # Freeze the preflight norm for deterministic slingshot gain scaling
            # (distinct from the live EMA, which stays for LR adaptation only).
            scheduler._activation_norm_preflight = activation_norm
            # Seed lambda proportional to initial L0 overshoot so the AL
            # integrator doesn't start from zero when L0 is 4-8x above target.
            scheduler.seed_lambda(initial_l0)
            preflight_stats.update({
                "activation_norm_probe": activation_norm,
                "activation_norm_ref": ref_norm,
                "initial_l0_probe": initial_l0,
                "initial_lr_multiplier": sae_cfg.initial_lr_multiplier,
                "early_pulse_multiplier": sae_cfg.early_pulse_multiplier,
                "layer_lr_multiplier": layer_mult,
            })
            for group, lr in zip(optimizer.param_groups, scheduler._compute_lrs()):
                group["lr"] = lr
            print(
                f"  [PREFLIGHT] act_norm={activation_norm:.4f} ref={ref_norm:.4f} "
                f"initial_L0={initial_l0:.1f} lr_mult={sae_cfg.initial_lr_multiplier:.2f} "
                f"pulse={sae_cfg.early_pulse_multiplier:.2f}"
            )
            print(
                f"  [SLINGSHOT] base={sae_cfg.al_slingshot_gain_max:.1f} "
                f"floor={sae_cfg.deep_layer_slingshot_gain:.1f} "
                f"ref={ref_norm:.4f} probe={activation_norm:.4f} "
                f"ratio={norm_ratio:.3f} alpha={sae_cfg.slingshot_norm_alpha:.2f} "
                f"gain={scheduler._effective_slingshot_gain():.1f}"
            )

    # Wrap the provider in a double-buffer for CUDA so the next batch's H2D transfer
    # overlaps with the current step's forward/backward/optimizer work. On CPU this is
    # a no-op (just passes through).
    if device.type == "cuda":
        provider = _ActivationDoubleBuffer(provider, device)

    metrics = {"recon_loss": [], "mean_l0": [], "dead_pct": [], "resampled": [],
               "ev": [], "nonlinear_err": [], "linear_err": []}
    log_window_start = time.time()
    log_window_tokens = 0

    # -- step-level timing --------------------------------------------------------
    # SAE_TIMING=N -> print every N steps; unset/true/on -> every TIMING_EVERY_DEFAULT
    # steps; 0/off/false -> disabled. SAE_TIMING=1 prints every step, which is a wall
    # of text on a fast layer -- use it for debugging, not for watching a run.
    # Accumulators for the [TIMING @ LOG_EVERY] aggregate run on every step regardless
    # of the print interval.
    _timing_env = os.environ.get("SAE_TIMING", "").strip().lower()
    if _timing_env == "":
        timing_every = TIMING_EVERY_DEFAULT
    elif _timing_env in ("0", "off", "false", "no"):
        timing_every = 0
    elif _timing_env in ("true", "yes", "on"):
        timing_every = TIMING_EVERY_DEFAULT
    elif _timing_env.isdigit() and int(_timing_env) > 0:
        timing_every = int(_timing_env)
    else:
        timing_every = TIMING_EVERY_DEFAULT
    do_timing = timing_every > 0
    if do_timing:
        timing = {
            "t_provider": 0.0,      # wall-time accumulator (last LOG_EVERY steps)
            "t_errbuf": 0.0,
            "t_overhead": 0.0,
            "t_fwd_bwd_wall": 0.0,  # CPU wall of the accum loop (fwd+bwd+.item waits)
            "t_cpu_post": 0.0,      # CPU wall of scheduler.step + nudge + bandwidth
            "n_steps": 0,
            "_last_provider": 0.0,
            "_last_errbuf": 0.0,
            "_last_fwd_bwd": 0.0,
            "_last_opt_norm": 0.0,
            "evt_fwd_bwd_start": None,
            "evt_fwd_bwd_end": None,
            "evt_opt_start": None,
            "evt_opt_end": None,
            "evt_norm_start": None,
            "evt_norm_end": None,
        }
        def _new_cuda_event():
            import torch
            return torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    else:
        timing = None

    best_ev = -float("inf"); best_ev_step = 0; best_ev_l0 = 0.0
    best_state_in_memory = None; best_state_pending = False
    best_ev_persist_margin = 0.005
    # Dead-feature ceiling with rollback. Deep layers starve their feature tail the
    # moment PIN freezes the dual: observed MiniCPM5-1B L13 going 0.8% -> 8.8% dead in
    # a single 250-step window at PIN entry, while EV moved 0.949 -> 0.948. AuxK cannot
    # catch it (AUX_DEAD_THRESHOLD is 250 steps of silence, longer than the collapse).
    # So instead of trying to revive forward, keep the last few logged states and walk
    # BACK to the newest one still under the ceiling. On L13 that lands step 2000:
    # L0 49.3, EV 0.949, dead 0.8% -- four thousandths of EV for 9x fewer dead features.
    # Ceiling of 1.0% never fires on layers that behave (0-12 here peaked at 0.58%),
    # so this is inert on healthy runs.
    dead_stop_pct = 1.0
    dead_rollback_slots = 4
    dead_state_buf = []  # list of dicts: step / dead / ev / l0 / state (CPU tensors)
    ev_decline_margin = 0.035
    ev_decline_floor = 0.95
    ev_decline_patience = 8
    ev_decline_warmup_steps = 1500
    ev_decline_stop_enabled = False
    ev_below_peak_streak = 0

    def resample_dead_neurons(dead_mask, err_buf):
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0 or len(err_buf) == 0:
            return 0
        candidates = torch.cat(err_buf, dim=0)
        with torch.no_grad():
            idx = torch.randint(0, candidates.shape[0], (n_dead,), device=candidates.device)
            samples = candidates[idx]
            norms = samples.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            alive_mask = ~dead_mask
            alive_norm = sae.W_enc.weight[alive_mask].norm(dim=1).mean() if alive_mask.any() \
                else torch.tensor(1.0, device=candidates.device)
            resampled_rows = ((samples / norms) * RESAMPLE_SCALE * alive_norm).to(sae.W_enc.weight.dtype)
            sae.W_enc.weight.data[dead_mask] = resampled_rows
            sae.W_enc.bias.data[dead_mask] = 0.0
            rand_dec = torch.randn(sae.d_in, n_dead, device=candidates.device, dtype=sae.W_dec.weight.dtype)
            rand_dec = rand_dec / rand_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
            sae.W_dec.weight.data[:, dead_mask] = rand_dec
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p, {})
                    if "exp_avg" not in state:
                        continue
                    if p is sae.W_enc.weight or p is sae.W_enc.bias:
                        state["exp_avg"][dead_mask] = 0.0; state["exp_avg_sq"][dead_mask] = 0.0
                    elif p is sae.W_dec.weight:
                        state["exp_avg"][:, dead_mask] = 0.0; state["exp_avg_sq"][:, dead_mask] = 0.0
        return n_dead

    # -- training loop --------------------------------------------------------
    last_step = start_step - 1
    t_step_start_prev = None
    # AuxK dead-set cache. These MUST live outside the step loop: the refresh only
    # runs every AUX_DEAD_THRESHOLD steps and every other step reuses the cached set.
    # They used to be initialised inside the loop, which silently reset them to
    # empty on 249 of every 250 steps -- so the aux revival loss only ever applied
    # on refresh steps (0.4% of training), and `last_dead_count` logged 0 forever
    # because logging lands on step % 250 == 0 while the refresh is step % 250 == 1.
    # That left the emergency mass reset as the only path that ever revived a
    # feature, which is how a layer gets to 34% dead before anything intervenes.
    aux_n_dead = 0
    aux_dead_indices = None
    aux_eff_k = 0
    for step in range(start_step, n_steps + 1):
        last_step = step
        if do_timing:
            t_step_start = time.perf_counter()
        if scheduler.should_stop:
            print(f"  EARLY STOP @ step {step}: {scheduler.stop_reason}")
            if use_wandb:
                wandb.log({"train/early_stop": True}, step=step)
            break

        sae_cfg.target_l0 = float(K)  # curriculum disabled (K_INIT==K)

        # Gradient accumulation: accumulate over microbatches before stepping
        optimizer.zero_grad()
        accum_recon_loss = 0.0
        accum_l0 = 0.0
        accum_sparsity_loss = 0.0
        accum_aux_loss = 0.0
        fired_accum = torch.zeros(N_FEATURES, device=device, dtype=torch.bool)

        # Get full batch and split into microbatches for accumulation. The double-buffer
        # wrapper returns the batch already on GPU; its transfer overlapped with the
        # previous step's compute on a separate CUDA stream.
        if do_timing:
            t_provider_start = time.perf_counter()
        full_batch = provider.next_batch()  # [BATCH_TOKENS, d_in] bf16, on device
        if do_timing:
            t_provider_elapsed = time.perf_counter() - t_provider_start
            timing["t_provider"] += t_provider_elapsed
            timing["_last_provider"] = t_provider_elapsed
            timing["evt_fwd_bwd_start"] = _new_cuda_event()
            if timing["evt_fwd_bwd_start"] is not None:
                timing["evt_fwd_bwd_start"].record()
        # Activation norm feeds the scheduler's LR-adaptation EMA (alpha=0.95, slow)
        # and is otherwise nearly stationary within a layer. Computing it every step
        # is a full-tensor cast+reduce+sync for a number that barely moves -- refresh
        # every 50 steps and pass None otherwise (the scheduler guards on None). Log
        # steps (LOG_EVERY=250, divisible by 50) always land on a refresh.
        if step % 50 == 0:
            activation_norm_step = full_batch.float().pow(2).mean().sqrt().item()
        else:
            activation_norm_step = None
        microbatch_size = microbatch_tokens  # tokens per microbatch

        # Pre-compute dead feature parameters for the aux loss. L0/sparsity stay
        # inside each microbatch forward so the JumpReLU threshold gradient is live.
        # At aggressive low L0, cap AUX_K to a fraction of the target so the aux loss
        # doesn't try to revive more features than the sparsity budget allows.
        #
        # Dead-set cadence: recomputing the dead feature set every step costs a GPU
        # reduce+sync (.sum().item()) plus torch.where. The set evolves slowly, so
        # refresh it only every AUX_DEAD_THRESHOLD steps (or on the first step).
        # Between refreshes we reuse the cached dead_indices/eff_k/n_dead.
        aux_k_eff = revival.effective_k_aux(K)
        if step % AUX_DEAD_THRESHOLD == 1 or step == start_step:
            dead_mask_aux = revival.aux_dead_mask()
            aux_n_dead = int(dead_mask_aux.sum().item())
            if aux_n_dead > 0:
                aux_dead_indices = torch.where(dead_mask_aux)[0]
                aux_eff_k = min(aux_k_eff, aux_n_dead)
            else:
                aux_dead_indices = None
                aux_eff_k = 0
        # else: reuse aux_dead_indices / aux_eff_k / aux_n_dead from the last refresh
        # (they persist across steps now -- see the note above the loop).
        n_dead = aux_n_dead
        dead_indices = aux_dead_indices
        eff_k = aux_eff_k
        revival.last_dead_count = n_dead

        buffer_err_this_step = revival.should_buffer_err(step, K)
        err_candidates = [] if buffer_err_this_step else None

        if do_timing:
            t_fwd_bwd_wall_start = time.perf_counter()
        # Triton fused forward - optional drop-in replacement
        use_triton_fwd = USE_TRITON and device.type == "cuda"
        if use_triton_fwd:
            try:
                from triton_sae_kernel import fused_sae_forward
                print(f"  [TRITON] Using fused SAE kernel")
            except ImportError:
                use_triton_fwd = False
                print(f"  [TRITON] Kernel not available, falling back to PyTorch")

        for accum_idx in range(accum_steps):
            start_idx = accum_idx * microbatch_size
            end_idx = start_idx + microbatch_size
            acts_mb = full_batch[start_idx:end_idx]  # bf16, [microbatch_tokens, d_in]

            # bf16 forward: keep activations in bf16, cast only for loss computation.
            # Everything the step needs (recon, L0, sparsity, aux, fired-mask) comes
            # from THIS forward -- the two extra full-batch encode passes that used to
            # bracket this loop (one for L0, one for the fired mask) are gone.
            with _autocast():
                triton_ok_this_step = use_triton_fwd
                if triton_ok_this_step:
                    # The fused kernel's backward does not yet implement the dead-feature
                    # aux loss.  Use the standard PyTorch path whenever dead features need
                    # revival so training behaviour is preserved.
                    if n_dead > 0:
                        triton_ok_this_step = False

                if triton_ok_this_step:
                    # Triton fused forward with autograd.  If the kernel fails to compile
                    # for this shape/device, fall back cleanly to the PyTorch path.
                    try:
                        x_hat, l0_per_token = fused_sae_forward(acts_mb, sae)
                    except Exception as e:
                        print(f"  [TRITON] Fused forward failed ({e}); falling back to PyTorch for this step")
                        triton_ok_this_step = False

                if triton_ok_this_step:
                    l0_mb = l0_per_token.mean()
                    residual_float = acts_mb.float() - x_hat.float()
                    recon_loss = residual_float.pow(2).mean() / accum_steps
                    slack = (l0_mb - sae_cfg.target_l0).clamp(min=0.0)
                    sparsity_loss = (
                        scheduler.lambda_l0 * slack
                        + 0.5 * sae_cfg.al_mu * slack * slack
                    ) / accum_steps
                    aux_loss = torch.zeros((), device=acts_mb.device, dtype=torch.float32)
                    loss = recon_loss + sparsity_loss + aux_loss
                else:
                    # Standard PyTorch forward (also used as Triton fallback).
                    pre = sae.encode_pre(acts_mb)
                    feat_acts, gate = sae.jumprelu_with_gate(pre)
                    x_hat = sae.decode(feat_acts)

                    # Cast to fp32 only for loss computation (numerical stability)
                    residual_float = acts_mb.float() - x_hat.float()
                    recon_loss = residual_float.pow(2).mean() / accum_steps

                    # Sparsity penalty -- IN-GRAPH so the L0 straight-through estimator
                    # actually trains the JumpReLU thresholds.
                    l0_mb = gate.sum(dim=-1, dtype=torch.float32).mean()
                    slack = (l0_mb - sae_cfg.target_l0).clamp(min=0.0)
                    sparsity_loss = (
                        scheduler.lambda_l0 * slack
                        + 0.5 * sae_cfg.al_mu * slack * slack
                    ) / accum_steps

                    # Aux loss for dead features (scaled by accum_steps)
                    if n_dead > 0:
                        pre_dead = pre[:, dead_indices].relu()
                        topk_vals, topk_idx = pre_dead.topk(eff_k, dim=-1)
                        aux_acts = torch.zeros_like(pre_dead)
                        aux_acts.scatter_(-1, topk_idx, topk_vals)
                        W_dec_dead = sae.W_dec.weight.t()[dead_indices]
                        x_aux = aux_acts @ W_dec_dead
                        residual_target = residual_float.detach()
                        aux_loss = ((residual_target - x_aux.float()).pow(2).mean() * AUX_COEFF) / accum_steps
                    else:
                        aux_loss = torch.zeros((), device=acts_mb.device, dtype=torch.float32)

                    loss = recon_loss + sparsity_loss + aux_loss

            loss.backward()

            accum_recon_loss += recon_loss.detach().item() * accum_steps
            accum_l0 += l0_mb.detach().item()
            accum_sparsity_loss += sparsity_loss.detach().item() * accum_steps
            accum_aux_loss += aux_loss.detach().item() * accum_steps
            with torch.no_grad():
                if triton_ok_this_step:
                    # The fused kernel does not return gate, so recompute it cheaply
                    # for the fire-state tracker (no backward needed).
                    pre = sae.encode_pre(acts_mb)
                    fired_accum |= (sae.l0_indicator(pre) > 0).any(dim=0)
                else:
                    fired_accum |= (feat_acts.detach() > 0).any(dim=0)

            # Collect high-error activation candidates for the upcoming resample.
            # Reuses the main forward's residual (computed above) instead of running a
            # second full encode/decode pass, which was a major tok/s sink near
            # resample steps.
            if buffer_err_this_step:
                with torch.no_grad():
                    per_tok_err = residual_float.detach().pow(2).sum(dim=-1)
                    k = min(64, per_tok_err.numel())
                    if k > 0:
                        topk_vals, topk_idx = per_tok_err.topk(k)
                        err_candidates.append((
                            acts_mb[topk_idx].detach().cpu(),
                            topk_vals.detach().cpu(),
                        ))

        if do_timing:
            _fwd_bwd_elapsed = time.perf_counter() - t_fwd_bwd_wall_start
            timing["t_fwd_bwd_wall"] += _fwd_bwd_elapsed
            timing["_last_fwd_bwd"] = _fwd_bwd_elapsed
            timing["evt_fwd_bwd_end"] = _new_cuda_event()
            if timing["evt_fwd_bwd_end"] is not None:
                timing["evt_fwd_bwd_end"].record()
            timing["evt_opt_start"] = _new_cuda_event()
            if timing["evt_opt_start"] is not None:
                timing["evt_opt_start"].record()
            t_opt_start = time.perf_counter()

        # Single optimizer step after accumulation.
        # grad_norm is only needed for logging/W&B, so defer the .item() sync
        # to log steps. clip_grad_norm_ returns a tensor; we keep it as such.
        grad_norm_t = nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

        # -- W_enc gradient dampening when L0 >> target -------------------------
        # Scale down W_enc gradient proportional to L0 overshoot. This prevents
        # the encoder from "inflating" to keep features on while sparsity pressure
        # is trying to push them off.  Use the current step's accumulated L0 for
        # immediate responsiveness.
        current_step_l0 = accum_l0 / max(accum_steps, 1)
        wenc_factor = scheduler.wenc_dampen_factor(current_step_l0)
        if wenc_factor < 1.0:
            if sae.W_enc.weight.grad is not None:
                sae.W_enc.weight.grad.mul_(wenc_factor)
            if sae.W_enc.bias.grad is not None:
                sae.W_enc.bias.grad.mul_(wenc_factor)

        optimizer.step()

        if do_timing:
            timing["evt_opt_end"] = _new_cuda_event()
            if timing["evt_opt_end"] is not None:
                timing["evt_opt_end"].record()
            timing["evt_norm_start"] = _new_cuda_event()
            if timing["evt_norm_start"] is not None:
                timing["evt_norm_start"].record()

        if not frozen_decoder:
            sae._normalize_decoder()
        with torch.no_grad():
            enc_norms = sae.W_enc.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
            sae.W_enc.weight.div_(enc_norms)

        if do_timing:
            timing["evt_norm_end"] = _new_cuda_event()
            if timing["evt_norm_end"] is not None:
                timing["evt_norm_end"].record()
            timing["_last_opt_norm"] = time.perf_counter() - t_opt_start

        # accum_recon_loss / accum_l0 are already Python floats (summed from the
        # microbatch .item() calls); use them directly instead of bouncing through
        # a GPU tensor and back, which forced two extra device syncs per step.

        # Dead/fired bookkeeping reuses the fired mask gathered during the forward
        # passes above -- no extra encode of the full batch.
        with torch.no_grad():
            revival.update_fire_state(fired_accum)
            log_window_tokens += BATCH_TOKENS

        # K-aware reset/resample cadence (decisions owned by RevivalController;
        # the model-weight mutations stay inline here for readability).
        if revival.should_reset(step):
            with torch.no_grad():
                very_dead = revival.very_dead_mask(K)
                n_reset = int(very_dead.sum().item())
                if n_reset > 0:
                    sae.log_threshold.data[very_dead] = math.log(INIT_THRESHOLD)
                    state = optimizer.state.get(sae.log_threshold, {})
                    if "exp_avg" in state:
                        state["exp_avg"][very_dead] = 0.0; state["exp_avg_sq"][very_dead] = 0.0
                    revival.clear_silence(very_dead)
                    print(f"  [RESET @ {step}] theta->{INIT_THRESHOLD} for {n_reset} dead")
                revival.record_reset(n_reset)

        if buffer_err_this_step:
            if do_timing:
                t_errbuf_start = time.perf_counter()
            with torch.no_grad():
                if err_candidates:
                    cand_acts = torch.cat([acts for acts, _ in err_candidates], dim=0)
                    cand_errs = torch.cat([errs for _, errs in err_candidates], dim=0)
                    k = min(64, cand_errs.numel())
                    if k > 0:
                        top_idx = cand_errs.topk(k).indices
                        err_buffer.append(cand_acts[top_idx].cpu())
                    if sum(t.shape[0] for t in err_buffer) > ERR_BUFFER_SZ:
                        err_buffer = err_buffer[-ERR_BUFFER_SZ // 64:]
            if do_timing:
                t_errbuf_elapsed = time.perf_counter() - t_errbuf_start
                timing["t_errbuf"] += t_errbuf_elapsed
                timing["_last_errbuf"] = t_errbuf_elapsed
        else:
            if do_timing:
                timing["_last_errbuf"] = 0.0

        if revival.is_resample_step(step, K):
            dead_mask = revival.resample_dead_mask(K)
            n_dead = int(dead_mask.sum().item())
            n_res = resample_dead_neurons(dead_mask, [t.to(device) for t in err_buffer])
            if n_res > 0:
                with torch.no_grad():
                    sae.log_threshold.data[dead_mask] = math.log(INIT_THRESHOLD)
                    state = optimizer.state.get(sae.log_threshold, {})
                    if "exp_avg" in state:
                        state["exp_avg"][dead_mask] = 0.0; state["exp_avg_sq"][dead_mask] = 0.0
                revival.clear_silence(dead_mask)
            revival.record_resample(n_dead, n_res)
            print(f"  [RESAMPLE @ {step}] reinit {n_res}/{n_dead} dead")
            err_buffer = []

        recon_val = accum_recon_loss / accum_steps
        is_log_step = (step % LOG_EVERY == 0)
        # grad_norm is only needed for logging/W&B; sync it only on log steps.
        if is_log_step:
            grad_norm_val = grad_norm_t.item()
        else:
            grad_norm_val = 0.0
        l0_val = accum_l0 / accum_steps

        # Observation-only trace through the warmup collapse window. L0 and recon are
        # already accumulated per step, so this costs one threshold reduction.
        if OBS_EVERY > 0 and (step == 1 or (step % OBS_EVERY == 0 and not is_log_step)):
            # EV here is the same global-variance form the log line uses: batch
            # variance plus the recon already accumulated this step, so it costs one
            # reduction and no extra SAE forward. dead_pct only reads silence
            # counters. Both are the numbers that say whether the layer is any good.
            with torch.no_grad():
                thr_obs = sae.log_threshold.exp().mean().item()
                total_var_obs = full_batch.float().var().item()
            ev_obs = 1.0 - (recon_val / total_var_obs) if total_var_obs > 0 else 0.0
            dead_obs = revival.dead_pct(LOG_EVERY)
            print(f"{_c('  >> OBS', '1;96')} {_c(f'{step:>5}', '1;96')}  "
                  f"L0={_c_l0(l0_val, sae_cfg.target_l0)}  "
                  f"EV={_c_ev(ev_obs)}  dead={_c_dead(dead_obs)}  thr={thr_obs:.4f}")

        if do_timing:
            t_step_total = time.perf_counter() - t_step_start
            timing["t_overhead"] += max(
                0.0,
                t_step_total - timing["_last_provider"] - timing["_last_errbuf"]
            )
            timing["n_steps"] += 1

            # Live per-step timing: no CUDA sync, so usable every step on CPU/GPU.
            _prev = t_step_start_prev if t_step_start_prev is not None else t_step_start
            dt_step = t_step_start - _prev
            t_step_start_prev = t_step_start
            step_tok_s = BATCH_TOKENS / max(t_step_total, 1e-9)
            wall_tok_s = BATCH_TOKENS / max(dt_step, 1e-9)
            t_other = max(
                0.0,
                t_step_total
                - timing["_last_provider"]
                - timing["_last_errbuf"]
                - timing["_last_fwd_bwd"]
                - timing["_last_opt_norm"],
            )
            if step % timing_every == 0:
                print(
                    f"  [STEP-TIME {step:>4}] "
                    f"total={t_step_total*1000:>6.1f}ms "
                    f"provider={timing['_last_provider']*1000:>6.1f}ms "
                    f"fwd_bwd={timing['_last_fwd_bwd']*1000:>6.1f}ms "
                    f"opt_norm={timing['_last_opt_norm']*1000:>6.1f}ms "
                    f"errbuf={timing['_last_errbuf']*1000:>6.1f}ms "
                    f"other={t_other*1000:>6.1f}ms "
                    f"step_tok/s={step_tok_s:>7.1f} "
                    f"wall_tok/s={wall_tok_s:>7.1f}"
                )

        if is_log_step:
            dead = revival.dead_pct(LOG_EVERY)
            with torch.no_grad():
                total_var = full_batch.float().var().item()  # .item(): keep ev a python float
                ev = 1.0 - (recon_val / total_var) if total_var > 0 else 0.0
                # Per-dim normalized EV: equal weight per channel, so a few
                # high-magnitude residual dims (Qwen-style outlier channels)
                # can't dominate the score like they do in global-variance EV.
                # Probed on the last microbatch; logging-only signal.
                xh_p, _ = sae(acts_mb)
                res_var_d = (acts_mb.float() - xh_p.float()).pow(2).mean(dim=0)
                var_d = acts_mb.float().var(dim=0)
                ev_perdim = 1.0 - (res_var_d / var_d.clamp_min(1e-6)).mean().item()

            # -- dead-feature ceiling with rollback -----------------------------
            # Buffer this window, then if dead has breached the ceiling walk back to
            # the newest buffered state still under it and stop there. Runs before
            # anything downstream reads dead/ev/l0_val, so the printed line, the W&B
            # row, meta.json and sae.pt all describe the SAME weights.
            dead_state_buf.append({
                "step": step, "dead": dead, "ev": ev, "l0": l0_val,
                "state": {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()},
            })
            if len(dead_state_buf) > dead_rollback_slots:
                dead_state_buf.pop(0)
            if dead > dead_stop_pct:
                _pick = _pick_dead_rollback(dead_state_buf, dead_stop_pct)
                if _pick is not None:
                    sae.load_state_dict(_pick["state"])
                    _tag = _c(f"  [DEAD ROLLBACK @ {step}]", "1;95")
                    _bad = _c(f"{dead:.2f}%", "1;91")
                    _pd = _c(f"{_pick['dead']:.2f}%", "1;92")
                    _pe = _c_ev(_pick["ev"])
                    _pl = _c_l0(_pick["l0"], sae_cfg.target_l0)
                    _ps = _c(str(_pick["step"]), "1;97")
                    print(f"{_tag} dead={_bad} breached {dead_stop_pct:.2f}% ceiling"
                          f" -> restored step {_ps} (dead={_pd} ev={_pe} L0={_pl})")
                    # Re-point the reported metrics at the restored state.
                    dead = _pick["dead"]; ev = _pick["ev"]; l0_val = _pick["l0"]
                    scheduler.stop_reason = (
                        f"dead-feature ceiling: breached {dead_stop_pct:.2f}%, rolled back to "
                        f"step {_pick['step']} (dead={_pick['dead']:.2f}%, EV={_pick['ev']:.4f}, "
                        f"L0={_pick['l0']:.2f})")
                else:
                    print(f"  [DEAD CEILING @ {step}] dead={dead:.2f}% breached "
                          f"{dead_stop_pct:.2f}% ceiling and no buffered window is under it; "
                          f"stopping without rollback")
                    scheduler.stop_reason = (
                        f"dead-feature ceiling: breached {dead_stop_pct:.2f}% with no clean "
                        f"state in the last {len(dead_state_buf)} windows")
                scheduler.should_stop = True
        else:
            dead = None; ev = None; ev_perdim = None

        if do_timing:
            t_cpu_post_start = time.perf_counter()
        scheduler.step({"loss": recon_val, "grad_norm": grad_norm_val,
                        "l0": l0_val, "ev": ev, "dead_pct": dead,
                        "activation_norm": activation_norm_step})

        # -- L0-proportional threshold nudge: bypasses STE gradient bottleneck -------
        # Scales gain and frequency with overshoot, has symmetric undershoot
        # dampener, and respects a dead band near target.  Computed by the
        # scheduler so it has access to the full state.
        nudge_val, nudge_apply = scheduler.compute_threshold_nudge(l0_val, step)
        if nudge_apply and abs(nudge_val) > 0:
            with torch.no_grad():
                sae.log_threshold.data += nudge_val
            if is_log_step:
                direction = "UP" if nudge_val > 0 else "DOWN"
                print(f"  [THRESH NUDGE @ {step}] direction={direction}"
                      f"nudge={nudge_val:+.5f} L0={l0_val:.1f} "
                      f"target={sae_cfg.target_l0:.0f} "
                      f"new_thr_mean={sae.log_threshold.exp().mean().item():.4f}")

        # -- L0-adaptive STE bandwidth -----------------------------------------------
        # When L0 >> target, widen the STE bandwidth so gradient can reach
        # "stuck on" features far above threshold.  As L0 approaches target,
        # narrow back.  Floor at ste_bandwidth_floor (0.15) to keep the gradient
        # channel open (interaction guard: prevents threshold nudge overshoot).
        adaptive_bw = scheduler.adaptive_ste_bandwidth(l0_val)
        # Let live-tune overrides take priority when present
        cfg_bw = getattr(sae_cfg, 'ste_bandwidth', STE_BANDWIDTH)
        # If adaptive bandwidth is enabled, use the adaptive value unless
        # live-tune has overridden ste_bandwidth directly.
        use_adaptive = getattr(sae_cfg, 'ste_adaptive_bandwidth', True)
        if use_adaptive:
            # Only override if live-tune hasn't manually set a different bandwidth
            # (live-tune writes to sae_cfg.ste_bandwidth; we check if it differs
            # from what the adaptive computation would produce)
            new_bw = adaptive_bw
        else:
            new_bw = cfg_bw
        # Write adaptive value back to config so next step reads it consistently
        sae_cfg.ste_bandwidth = new_bw
        if abs(new_bw - sae.ste_bandwidth) > 1e-6:
            old_bw = sae.ste_bandwidth
            sae.ste_bandwidth = new_bw
            if is_log_step:
                print(f"  [STE-BW @ {step}] bandwidth {old_bw:.4f} -> {new_bw:.4f} "
                      f"(L0={l0_val:.1f}, target={sae_cfg.target_l0:.0f}, "
                      f"overshoot={l0_val/max(sae_cfg.target_l0,1):.2f}x)")

        if do_timing:
            timing["t_cpu_post"] += time.perf_counter() - t_cpu_post_start

        if is_log_step:
            with torch.no_grad():
                thr = sae.log_threshold.exp()
                fire_rate = revival.fire_rate(LOG_EVERY)
                ultra_active = (fire_rate > 0.10).float().sum().item()
            now = time.time()
            tokens_per_sec = log_window_tokens / max(now - log_window_start, 1e-6)
            log_window_start = now; log_window_tokens = 0
            revival.reset_fire_counts()

            if do_timing:
                import torch
                # Synchronize CUDA events only on log steps
                cuda_ms = {}
                for key, (start, end) in (
                    ("fwd_bwd", (timing["evt_fwd_bwd_start"], timing["evt_fwd_bwd_end"])),
                    ("opt", (timing["evt_opt_start"], timing["evt_opt_end"])),
                    ("norm", (timing["evt_norm_start"], timing["evt_norm_end"])),
                ):
                    if start is not None and end is not None:
                        start.synchronize()
                        end.synchronize()
                        cuda_ms[key] = start.elapsed_time(end)
                    else:
                        cuda_ms[key] = 0.0
                # Average wall-time buckets over the last LOG_EVERY steps
                n = max(1, timing["n_steps"])
                t_provider_avg = timing["t_provider"] / n
                t_errbuf_avg = timing["t_errbuf"] / n
                t_overhead_avg = timing["t_overhead"] / n
                t_fwd_bwd_wall_avg = timing["t_fwd_bwd_wall"] / n
                t_cpu_post_avg = timing["t_cpu_post"] / n
                # fwd_bwd_wall - fwd_bwd_cuda = CPU stall/launch overhead in the loop
                # (CPU blocked on .item() / feeding kernels while GPU could be idle).
                fwd_bwd_stall = max(0.0, t_fwd_bwd_wall_avg * 1000 - cuda_ms["fwd_bwd"])
                print(f"  [TIMING @ {step}] per_step "
                      f"provider={t_provider_avg*1000:.1f}ms "
                      f"fwd_bwd_cuda={cuda_ms['fwd_bwd']:.1f}ms "
                      f"fwd_bwd_wall={t_fwd_bwd_wall_avg*1000:.1f}ms "
                      f"fwd_bwd_stall={fwd_bwd_stall:.1f}ms "
                      f"opt_cuda={cuda_ms['opt']:.1f}ms "
                      f"norm_cuda={cuda_ms['norm']:.1f}ms "
                      f"cpu_post={t_cpu_post_avg*1000:.1f}ms "
                      f"errbuf={t_errbuf_avg*1000:.1f}ms "
                      f"overhead={t_overhead_avg*1000:.1f}ms "
                      f"(n={n})")
                # Reset accumulators for the next window
                for k in ("t_provider", "t_errbuf", "t_overhead",
                          "t_fwd_bwd_wall", "t_cpu_post", "n_steps"):
                    timing[k] = 0.0

            if ev > best_ev:
                best_ev = ev; best_ev_step = step; best_ev_l0 = l0_val
                best_state_in_memory = {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()}
                best_state_pending = True
            elif best_state_pending and ev < best_ev - best_ev_persist_margin:
                best_ckpt = Path(SAE_DIR) / f"layer_{layer:02d}_s{seed}{out_suffix}_latest" / "checkpoint_best.pt"
                best_ckpt.parent.mkdir(parents=True, exist_ok=True)
                bp = {"step": best_ev_step, "sae_state": best_state_in_memory,
                      "best_ev": best_ev, "best_ev_l0": best_ev_l0, "layer": layer,
                      "seed": seed, "tier": tier, "d_in": d_in, "n_features": N_FEATURES,
                      "target_l0_final": K, "path": "rolling"}
                def _savebest(p, path):
                    torch.save(p, path)
                threading.Thread(target=_savebest, args=(bp, best_ckpt), daemon=True).start()
                best_state_pending = False
                print(f"  [BEST PERSIST @ {step}] peak EV={best_ev:.4f}@{best_ev_step}")

            l0_crossed_target = _l0_crossed_target(l0_val, sae_cfg.target_l0)
            if (
                ev_decline_stop_enabled
                and step >= ev_decline_warmup_steps
                and best_ev > -float("inf")
                and l0_crossed_target
            ):
                if _post_peak_decline_is_bad(ev, best_ev, ev_decline_margin, ev_decline_floor):
                    ev_below_peak_streak += 1
                else:
                    ev_below_peak_streak = 0
                if ev_below_peak_streak >= ev_decline_patience:
                    scheduler.should_stop = True
                    scheduler.stop_reason = (f"Post-peak quality collapse after target cross: EV {ev:.4f} below "
                                             f"floor {ev_decline_floor:.2f} and peak "
                                             f"{best_ev:.4f}@{best_ev_step} by >= {ev_decline_margin:.3f} "
                                             f"for {ev_decline_patience} windows "
                                             f"(L0={l0_val:.1f}, target={sae_cfg.target_l0:.1f})")
                    print(f"  [POST-PEAK STOP @ {step}] {scheduler.stop_reason}")
            else:
                ev_below_peak_streak = 0

            # -- Aggressive-K early stop: when K is small, high EV + L0 near target
            #    is the real convergence signal. Stop before feature death sets in.
            if aggressive_k and step >= 750:
                l0_rel_err = abs(l0_val - K) / max(K, 1.0)
                if l0_rel_err <= K_STOP_L0_REL and ev >= K_STOP_EV_FLOOR:
                    if not hasattr(sae, "_k_converge_counter"):
                        sae._k_converge_counter = 0
                    sae._k_converge_counter += LOG_EVERY
                    if sae._k_converge_counter >= 500:
                        scheduler.should_stop = True
                        scheduler.stop_reason = (
                            f"Aggressive-K convergence: L0={l0_val:.1f} within "
                            f"{K_STOP_L0_REL*100:.0f}% of K={K}, EV={ev:.3f} >= "
                            f"{K_STOP_EV_FLOOR:.2f} for {sae._k_converge_counter} steps"
                        )
                        print(f"  [AGGRESSIVE-K STOP @ {step}] {scheduler.stop_reason}")
                else:
                    if hasattr(sae, "_k_converge_counter"):
                        sae._k_converge_counter = 0

            metrics["recon_loss"].append(round(recon_val, 6))
            metrics["mean_l0"].append(round(l0_val, 2))
            metrics["dead_pct"].append(round(dead, 2))
            metrics["resampled"].append(revival.total_resampled)
            metrics["ev"].append(ev)
            print(f"  step={_c(f'{step:>5}', '1;97')} mode={scheduler.mode:<9s} "
                  f"phase={_c(f'{scheduler.phase:<8s}', '96')} "
                  f"recon={recon_val:.5f} "
                  f"L0={_c_l0(l0_val, sae_cfg.target_l0)} "
                  f"dead={_c_dead(dead, dead_stop_pct)} "
                  f"ev={_c_ev(ev)} "
                  f"evd={ev_perdim:.3f} thr={thr.mean().item():.3f} "
                  f"ultra={int(ultra_active):>4d} lr={scheduler.optimizer.param_groups[0]['lr']:.2e} "
                  f"lam={scheduler.lambda_l0:.2e} tok/s={tokens_per_sec/1e3:.1f}k "
                  f"tokens={step*BATCH_TOKENS/1e6:.1f}M")
            if use_wandb:
                wandb.log({"train/recon_loss": recon_val, "train/mean_l0": l0_val,
                           "train/dead_pct": dead, "train/explained_variance": ev,
                           "train/ev_perdim": ev_perdim,
                           "train/lr": scheduler.optimizer.param_groups[0]["lr"],
                           "train/lambda_l0": scheduler.lambda_l0,
                           "train/activation_norm": activation_norm_step,
                           "scheduler/l0_progress_fast": scheduler._l0_progress_fast,
                           "scheduler/l0_progress_slow": scheduler._l0_progress_slow,
                           "scheduler/energy_dampen": scheduler._energy_dampen,
                           "train/best_ev_so_far": best_ev,
                           "features/ultra_active_count": ultra_active,
                           "timing/tokens_per_sec": tokens_per_sec,
                           "scheduler/mode": scheduler.mode,
                           "scheduler/phase": scheduler.phase,
                           "scheduler/phase_idx": _PHASE_IDX.get(scheduler.phase, -1),
                           "scheduler/effective_slingshot_gain": scheduler._effective_slingshot_gain(),
                           "scheduler/activation_norm_preflight": scheduler._activation_norm_preflight,
                           "timing/tokens_per_sec": tokens_per_sec,
                           **({f"timing/{k}": v for k, v in {
                               "provider_ms": t_provider_avg * 1000,
                               "fwd_bwd_cuda_ms": cuda_ms["fwd_bwd"],
                               "fwd_bwd_wall_ms": t_fwd_bwd_wall_avg * 1000,
                               "fwd_bwd_stall_ms": fwd_bwd_stall,
                               "opt_cuda_ms": cuda_ms["opt"],
                               "norm_cuda_ms": cuda_ms["norm"],
                               "cpu_post_ms": t_cpu_post_avg * 1000,
                               "errbuf_ms": t_errbuf_avg * 1000,
                               "overhead_ms": t_overhead_avg * 1000,
                           }.items()} if do_timing else {}),
                           **{f"revival/{k}": v for k, v in revival.revival_metrics(K).items()}},
                          step=step)

        if step % CHECKPOINT_EVERY == 0:
            rng_states = {"cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                          "cpu": torch.get_rng_state()}
            provider_state = provider.get_state() if hasattr(provider, "get_state") else None
            _save_full_checkpoint(out_dir, step, sae, optimizer, scheduler, rng_states,
                                  revival.feature_fire_counts.clone(),
                                  revival.steps_since_fired.clone(),
                                  provider_state)
            print(f"  [CHECKPOINT @ {step}] {out_dir / 'checkpoint_full.pt'}")

    step = last_step

    # -- final best flush -----------------------------------------------------
    if best_state_pending and best_state_in_memory is not None:
        best_ckpt = Path(SAE_DIR) / f"layer_{layer:02d}_s{seed}{out_suffix}_latest" / "checkpoint_best.pt"
        best_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": best_ev_step, "sae_state": best_state_in_memory,
                    "best_ev": best_ev, "best_ev_l0": best_ev_l0, "layer": layer,
                    "seed": seed, "tier": tier, "d_in": d_in, "n_features": N_FEATURES,
                    "target_l0_final": K, "note": "final flush", "path": "rolling"}, best_ckpt)
        print(f"  [BEST FINAL FLUSH] peak EV={best_ev:.4f}@{best_ev_step}")

    # -- save final + meta + HF push ------------------------------------------
    torch.save(sae.state_dict(), out_dir / "sae.pt")
    sched_summary = scheduler.summary()
    final_metrics = {k: v[-1] if v else None for k, v in metrics.items()}
    final_metrics.update(preflight_stats)
    meta = {"layer": layer, "seed": seed, "model_id": MODEL_ID, "d_in": d_in,
            "n_features": N_FEATURES, "k": K, "batch_tokens": BATCH_TOKENS,
            "n_steps": step, "lr": LR, "lambda_l0": scheduler.lambda_l0, "tier": tier,
            "preflight": preflight_stats,
            "scheduler_mode": scheduler.mode,
            "scheduler_phase": sched_summary["phase"],
            "scheduler_phase_summary": {
                k: sched_summary[k] for k in
                ("phase", "phase_step", "pin_entry_step", "pinned_lambda",
                 "pin_ev_count", "pin_retry_count")
            },
            "scheduler_transitions": sched_summary["transitions"],
            "total_tokens": step * BATCH_TOKENS, "early_stopped": scheduler.should_stop,
            "path": "rolling", "best_ev": best_ev, "best_ev_step": best_ev_step,
            "final_metrics": final_metrics,
            "training_curve": metrics}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save full checkpoint at layer completion (for resume if interrupted between layers)
    rng_states = {"cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                  "cpu": torch.get_rng_state()}
    provider_state = provider.get_state() if hasattr(provider, "get_state") else None
    _save_full_checkpoint(out_dir, step, sae, optimizer, scheduler, rng_states,
                          revival.feature_fire_counts.clone(),
                          revival.steps_since_fired.clone(), provider_state)

    print(f"Saved: {out_dir}/sae.pt + meta.json + checkpoint_full.pt")

    if push and SAE_HUB_ID:
        from huggingface_hub import HfApi
        hf_token = _resolve_hf_token()
        api = HfApi(token=hf_token)
        try:
            api.create_repo(SAE_HUB_ID, repo_type="model", exist_ok=True, private=False)
        except Exception:
            pass
        api.upload_folder(folder_path=str(out_dir), repo_id=SAE_HUB_ID,
                          path_in_repo=f"layer_{layer:02d}_s{seed}", repo_type="model")
        print(f"Pushed layer_{layer:02d}_s{seed} -> {SAE_HUB_ID}")
    elif push and not SAE_HUB_ID:
        print("  [push skipped] no --hub-id / $SAE_HUB_ID set -- SAE saved locally only")
    else:
        print(f"  [push disabled] skipped HF upload for layer {layer}")

    if use_wandb:
        try:
            wandb.summary.update({f"final/{k}": v for k, v in meta["final_metrics"].items()})
        finally:
            try:
                wandb.finish()
            except Exception:
                pass
    return meta["final_metrics"]


# ===========================================================================
#  Orchestrator  (load model once, sweep layers, train one SAE each)
# ===========================================================================

def run_atlas_rolling(start_layer: int = 0, end_layer: int = 9, seed: int = DEFAULT_SEED,
                      pool_batches: int = POOL_BATCHES_DEFAULT, microbatch_tokens: int = None,
                      use_pretok: bool = True, max_steps: int = N_STEPS,
                      bdec_batches: int = BDEC_INIT_BATCHES, resume_from: str = None,
                      push: bool = True, capture: str = "auto", model_id: str = None,
                      hub_id: str = None, wandb_project: str = None, expansion: int = None,
                      evict_model: bool = True, target_l0: int = None, cpu: bool = False,
                      norm_ref: float = None, corpus: str = None,
                      corpus_text_field: str = None, corpus_prefix: str = None,
                      trust_remote_code: bool = False, pool_retention: int = 3):
    """Train one SAE per decoder layer in [start_layer, end_layer] (inclusive).

    capture: "auto" = model-agnostic forward-hook capture (any AutoModelForCausalLM);
             "rolling" = Gemma-3n/4 single-block walk (layers 0..14 only, VRAM-optimized);
             "rolling-hf" = generic Llama/SmolLM2/Qwen single-block walk (no HARD_STOP);
             "rolling-hf-float" = same as rolling-hf but only 1-2 blocks in GPU memory at a time.
    pool_batches: activation batches cached per layer (default 4000; use 500-1000 for limited disk)
    microbatch_tokens: tokens per microbatch for gradient accumulation (default = no accum)
    resume_from: path to checkpoint_full.pt to resume from
    evict_model: move the LLM to CPU during SAE training to free VRAM (default True).
    target_l0: override the global L0 target K (default 500). Use for aggressive sparsity tests.
    cpu: force CPU training (no CUDA). Will be SLOW - for debugging only.
    model_id/hub_id/wandb_project/expansion override module defaults; d_in is auto-detected.
    max_steps/bdec_batches/push: cap work + skip upload for smoke tests."""
    import time
    import torch

    # -- config bus: thread runtime overrides into the module globals that the rest of
    #    the code (train_sae_on_activations, dataset builder, paths) already reads ------
    global MODEL_ID, SAE_DIR, SAE_HUB_ID, WANDB_PROJECT, EXPANSION, N_FEATURES, K, K_INIT
    global CORPUS_ID, CORPUS_TEXT_FIELD, CORPUS_PREFIX, TRUST_REMOTE_CODE
    if model_id:
        MODEL_ID = model_id
        SAE_DIR = str(DATA_DIR / "saes" / _slug(MODEL_ID))
    if corpus:
        CORPUS_ID = corpus
        print(f"  [config] corpus: {CORPUS_ID}")
    if corpus_text_field:
        CORPUS_TEXT_FIELD = corpus_text_field
    if corpus_prefix:
        CORPUS_PREFIX = corpus_prefix
        print(f"  [config] corpus prefix: {CORPUS_PREFIX!r}")
    if trust_remote_code:
        TRUST_REMOTE_CODE = True
        print("  [config] trust_remote_code enabled")
    pool_retention = max(1, int(pool_retention))
    print(f"  [config] pool retention: {pool_retention} previous pool(s) kept on disk")
    if expansion:
        EXPANSION = int(expansion)
    if target_l0 is not None and target_l0 > 0:
        K = int(target_l0)
        K_INIT = K
        print(f"  [config] target L0 overridden: K={K}")
    if hub_id is not None:
        SAE_HUB_ID = hub_id
    if wandb_project is not None:
        WANDB_PROJECT = wandb_project

    assert capture in ("auto", "rolling", "rolling-float", "rolling-hf", "rolling-hf-float",
                       "rolling-generic"), \
        f"unknown --capture {capture!r}"
    if capture in ("rolling", "rolling-float") and end_layer > HARD_STOP_LAYER:
        print(f"  [scope] rolling clamps end_layer {end_layer} -> {HARD_STOP_LAYER} "
              f"(Gemma KV-share boundary)")
        end_layer = HARD_STOP_LAYER

    hf_token = _resolve_hf_token()
    device = torch.device("cpu" if cpu else "cuda")
    torch.manual_seed(seed)

    # -- load model ONCE (generic loader: CausalLM, multimodal fallback) -------
    print(f"Loading {MODEL_ID} ...")
    from transformers import AutoTokenizer
    # Detect if MODEL_ID is a local path
    is_local = os.path.isdir(MODEL_ID)
    tokenizer_kwargs = {"token": hf_token} if not is_local else {}
    if is_local:
        tokenizer_kwargs["local_files_only"] = True
    if TRUST_REMOTE_CODE:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_kwargs)
    bos_token_id = tokenizer.bos_token_id or 2
    try:
        from transformers import AutoModelForCausalLM
        model_kwargs = {"token": hf_token, "dtype": torch.bfloat16, "device_map": "cpu"}
        if is_local:
            model_kwargs["local_files_only"] = True
        if TRUST_REMOTE_CODE:
            model_kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    except (ValueError, KeyError, OSError) as e:
        # Multimodal checkpoints (e.g. Gemma-4) aren't plain CausalLM -- fall back.
        print(f"  CausalLM load failed ({type(e).__name__}); using image-text-to-text loader")
        from transformers import AutoModelForImageTextToText
        model_kwargs = {"token": hf_token, "dtype": torch.bfloat16, "device_map": "cpu"}
        if is_local:
            model_kwargs["local_files_only"] = True
        if TRUST_REMOTE_CODE:
            model_kwargs["trust_remote_code"] = True
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    n_layers = int(tcfg.num_hidden_layers)
    d_in = int(getattr(tcfg, "hidden_size", D_IN))
    N_FEATURES = EXPANSION * d_in              # SAE dict scales with the detected width
    text_model, attr_name, decoder_layers = _find_text_model(model, n_layers)
    if capture in ("rolling-hf", "rolling-hf-float") and not _is_hf_rolling_supported(text_model, decoder_layers):
        print(f"  [warn] --capture {capture} not supported for {type(text_model).__name__}; "
              f"falling back to auto")
        capture = "auto"

    # `auto` re-runs the FULL model once per layer to read one block's output, which
    # is O(n_layers^2) block evaluations across a walk. If the blocks take (x, cos, sin)
    # we can walk them one at a time off the previous pool instead -- O(n_layers) --
    # which is the entire reason the rolling pools exist. Verify against the model's
    # own forward before trusting it; a silent mismatch would poison every layer.
    # Signature introspection is free; the correctness proof needs the token pool, so
    # it runs once that exists (below). The candidate flag is needed here because it
    # changes the disk estimate: rolling keeps a source and a destination pool.
    generic_rolling_candidate = (
        capture == "auto"
        and _is_generic_rolling_supported(model, text_model, decoder_layers)
    )

    # Floating window: keep the full model on CPU, pin tiny shared components to GPU,
    # and let the production loop move only the active 1-2 decoder blocks to GPU.
    use_floating_window = (capture in ("rolling-float", "rolling-hf-float") and device.type == "cuda")
    if use_floating_window:
        print(f"  [floating-window] keeping full model on CPU; only active blocks + "
              f"embed_tokens/rotary_emb will touch GPU")
        floating_window = FloatingLayerWindow(text_model, decoder_layers, device)
    else:
        floating_window = None
        model.to(device)

    end_layer = min(end_layer, n_layers - 1)
    assert 0 <= start_layer <= end_layer < n_layers, \
        f"bad layer range [{start_layer},{end_layer}] for a {n_layers}-layer model"
    layers = list(range(start_layer, end_layer + 1))
    print(f"  model={type(model).__name__}  blocks={len(decoder_layers)}  d_in={d_in}  "
          f"n_features={N_FEATURES} ({EXPANSION}x)  capture={capture}")
    if capture in ("rolling", "rolling-float") and d_in != D_IN:
        print(f"  [warn] rolling capture was tuned at d_in={D_IN}; model reports {d_in}")

    print(f"\n{'#'*60}\n  SAE ATLAS  model={_slug(MODEL_ID)}  layers={layers}  seed={seed}  "
          f"capture={capture}\n{'#'*60}\n")

    # -- disk sizing (we hold ~ONE activation pool at a time) -----------------
    import shutil
    os.makedirs(ROLLCACHE, exist_ok=True)
    shard_gb = (BATCH_TOKENS // SEQ_LEN) * SEQ_LEN * d_in * 2 / 1e9   # bf16 [n_seqs,SEQ_LEN,d]
    # Rolling pipelines two pools (current + pre-produced next) plus a resume copy.
    # Auto/hook capture is independent per layer -- only one pool is on disk at a
    # time (plus resume-copy headroom), so the 2x pipeline term over-estimates peak
    # and would falsely block valid auto runs.
    two_pool = capture in ("rolling", "rolling-float", "rolling-hf") or generic_rolling_candidate
    pool_multiplier = 2.2 if two_pool else 1.2
    n_pools_desc = "2 pipelined pools" if two_pool else "1 pool"
    peak_gb = pool_batches * shard_gb * pool_multiplier + 4
    free_gb = shutil.disk_usage(ROLLCACHE).free / 1e9
    sentinel = free_gb > 1e6                                          # overlay fs -> unreliable
    print(f"  [disk] {ROLLCACHE} peak~{peak_gb:.0f}GB ({n_pools_desc} of {pool_batches} x "
          f"{shard_gb*1e3:.0f}MB); free={'(overlay: unknown)' if sentinel else f'{free_gb:.0f}GB'}")
    if not sentinel:
        assert free_gb > peak_gb, (
            f"insufficient disk at {ROLLCACHE}: {free_gb:.0f}GB free < {peak_gb:.0f}GB peak. "
            f"Lower --pool-batches or set $SAE_SCRATCH_DIR to a bigger disk.")

    # -- token pool (once -- same tokens flow through every layer) ------------
    # Token ids are model-specific; cache them under the model slug so switching
    # models doesn't reuse a stale pool.
    # The model config is the authority: its vocab_size sizes the embedding matrix,
    # which is the real bound on a legal token id. `tokenizer.vocab_size` excludes
    # added special tokens (FIM, ChatML, domain tags), so it under-reports and
    # rejects valid ids. `len(tokenizer)` counts them, hence the fallback order.
    vocab_size = getattr(tcfg, "vocab_size", None) or len(tokenizer)
    tok_dir = _pool_dir(f"tokens_{_slug(MODEL_ID)}_s{seed}")
    _capture_token_pool(hf_token, seed, pool_batches, use_pretok, tok_dir, bos_token_id,
                        model_id=MODEL_ID, vocab_size=vocab_size)

    # Now that the token pool exists, prove the single-block walk reproduces the
    # model's own forward before switching onto it. Costs one 2-sequence forward.
    if generic_rolling_candidate:
        probe_ids = _read_shard(tok_dir, 0)[:2].to(device)
        ok, detail = _verify_generic_rolling(model, text_model, decoder_layers, probe_ids, device)
        if ok:
            print(f"  [capture] blocks take (x, cos, sin) and match the model's own forward "
                  f"({detail}); single-block rolling instead of a full forward per layer")
            capture = "rolling-generic"
        else:
            print(f"  [capture] single-block walk did NOT match the model forward ({detail}); "
                  f"staying on full-forward hooks")

    marker_path = Path(SAE_DIR) / f"atlas_marker_s{seed}.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    def pool_dir_for(L):
        return _pool_dir(f"pool_L{L:02d}_s{seed}")

    results = {}
    # The slingshot gain and LR multipliers scale by probe/ref norm ratio. The ref
    # must be the norm of the chain's FIRST layer (L0), not whichever layer this
    # process happens to train first -- a resume that starts mid-chain with ref=None
    # gives the first retrained layer ratio=1.0 and therefore maximum slingshot gain.
    activation_norm_ref = norm_ref
    t0 = time.time()
    # rolling walks the residual chain; hook capture is independent per layer, so it
    # starts at start_layer. A mid-chain start under rolling/rolling-hf is bootstrapped
    # below via one hooked capture pass instead of walking the chain from 0.
    walk_start = 0 if capture in ("rolling", "rolling-float", "rolling-hf", "rolling-hf-float",
                                  "rolling-generic") else start_layer

    # -- rolling resume: if a previous run persisted the last completed layer's pool,
    #    resume the chain from that layer instead of regenerating everything from 0.
    #    Also handle explicit --resume-from: read the checkpoint's layer and start there.
    resume_layer = -1
    explicit_resume_layer = -1
    if resume_from:
        try:
            import torch as _torch
            _tmp = _torch.load(resume_from, map_location="cpu", weights_only=False)
            explicit_resume_layer = int(_tmp.get("layer", _tmp.get("sae_state", {}).get("layer", -1)))
            if explicit_resume_layer < 0:
                # best-effort: infer from path like .../layer_02_s0/checkpoint_full.pt
                import re
                _m = re.search(r"layer_(\d+)", str(resume_from))
                if _m:
                    explicit_resume_layer = int(_m.group(1))
        except Exception:
            pass

    if capture in ("rolling", "rolling-float", "rolling-hf", "rolling-hf-float",
                   "rolling-generic"):
        if resume_from and explicit_resume_layer >= 0:
            # Explicit --resume-from under rolling. The persistent resume pool is only
            # saved after a layer *completes*, so the chain has no source here; the
            # bootstrap below produces pool[start_layer] directly (or, for float
            # modes, the chain is walked from L0 with training skipped below target).
            walk_start = 0
            start_layer = min(start_layer, explicit_resume_layer)
            print(f"  [resume] explicit checkpoint for layer {explicit_resume_layer}; "
                  f"training from L{start_layer}")
        elif not resume_from:
            # Retained pools are the cheapest resume there is: pool[L] is all that
            # training layer L needs, and pool[L] is also the only input needed to
            # produce pool[L+1]. So if the chain already reaches start_layer, walk
            # from there -- rebuilding earlier pools regenerates data nothing reads.
            # This is what pool_retention keeps them around FOR.
            if (start_layer > walk_start
                    and len(_shard_paths(pool_dir_for(start_layer))) >= pool_batches):
                print(f"  [resume] pool L{start_layer} already on disk; chain resumes "
                      f"at L{start_layer} instead of rebuilding from L{walk_start}")
                walk_start = start_layer
                # Anything below the new entry point is unreachable: the walk starts
                # here and the retention loop only ever walks down to walk_start, so
                # these would sit on disk for the whole run.
                for _stale in range(0, walk_start):
                    _sd = pool_dir_for(_stale)
                    if _sd.exists():
                        _rm_pool(_sd)
                        print(f"  [cleanup] deleted unreachable pool L{_stale}")

            resume_layer = _find_resume_layer(seed, pool_batches, MODEL_ID, capture)
            if resume_layer >= 0:
                walk_start = max(walk_start, resume_layer + 1)
                print(f"  [resume] rolling checkpoint found for layer {resume_layer}; "
                      f"chain resumes at L{walk_start}")
                if activation_norm_ref is None:
                    try:
                        import json as _json
                        with open(_resume_manifest_path(seed)) as _f:
                            _ref = _json.load(_f).get("activation_norm_ref")
                        if _ref and _ref > 0:
                            activation_norm_ref = float(_ref)
                            print(f"  [resume] restored activation_norm_ref={activation_norm_ref:.5f}")
                        else:
                            print("  [resume] WARNING: manifest has no activation_norm_ref; "
                                  "first trained layer will self-reference (max slingshot gain). "
                                  "Pass --norm-ref to pin it.")
                    except Exception:
                        pass
                # Pools below the resume point are pre-crash leftovers the retention
                # loop can never reach (it only walks down to walk_start).
                for _stale in range(0, walk_start):
                    _sd = pool_dir_for(_stale)
                    if _sd.exists():
                        _rm_pool(_sd)
                        print(f"  [cleanup] deleted stale pre-resume pool L{_stale}")
        if start_layer > walk_start and capture in ("rolling", "rolling-hf"):
            # Chain entry ramp: produce pool[start_layer] directly from hooked
            # full forwards instead of walking the chain up from L{walk_start}.
            # One capture pass replaces every intermediate block walk and pool.
            # Float modes keep the chain walk -- a full forward needs all prefix
            # blocks resident in VRAM, which is what those modes exist to avoid.
            print(f"  [bootstrap] chain entry at L{start_layer}: hooked full-forward "
                  f"capture replaces walking L{walk_start}..L{start_layer - 1}")
            _produce_pool_hooked(model, decoder_layers, start_layer, tok_dir,
                                 pool_dir_for(start_layer), device)
            walk_start = start_layer
            if activation_norm_ref is None:
                import json as _json
                _prev_meta = Path(SAE_DIR) / f"layer_{start_layer - 1}_s{seed}" / "meta.json"
                try:
                    with open(_prev_meta) as _f:
                        _ref = _json.load(_f).get("activation_norm_ref")
                except OSError:
                    _ref = None
                if _ref and _ref > 0:
                    activation_norm_ref = float(_ref)
                    print(f"  [bootstrap] restored activation_norm_ref="
                          f"{activation_norm_ref:.5f} from {_prev_meta}")
                else:
                    print("  [bootstrap] WARNING: no activation_norm_ref found in "
                          f"{_prev_meta}; pass --norm-ref to pin the chain norm "
                          "(first trained layer self-references otherwise)")

    for L in range(walk_start, end_layer + 1):
        dst_dir = pool_dir_for(L)
        if capture in ("rolling", "rolling-float", "rolling-hf", "rolling-hf-float",
                       "rolling-generic"):
            src_dir = pool_dir_for(L - 1) if L >= 1 else None
            consume_src = True
            if L == resume_layer + 1 and resume_layer >= 0:
                # The persistent resume pool is the source for the first produced layer.
                # Do not consume it -- it stays on disk until the next layer finishes.
                src_dir = _resume_pool_dir(seed)
                consume_src = False
            if use_floating_window:
                floating_window.activate(L)
            if capture in ("rolling", "rolling-float"):
                _produce_pool(model, text_model, decoder_layers, tcfg, L, tok_dir, src_dir,
                              dst_dir, device)
            elif capture == "rolling-generic":
                _produce_pool_generic_rolling(model, text_model, decoder_layers, L, tok_dir,
                                              src_dir, dst_dir, device)
            else:
                _produce_pool_hf_rolling(model, text_model, decoder_layers, L, tok_dir, src_dir,
                                         dst_dir, device)
            # Pipeline: while the LLM is still hot on GPU, pre-produce the next layer's pool.
            # When training L finishes, pool L+1 is already on disk and we start immediately.
            pre_produced_next = False
            if L + 1 <= end_layer:
                next_dir = pool_dir_for(L + 1)
                if len(_shard_paths(next_dir)) < pool_batches:
                    print(f"  [pipeline] pre-producing pool L{L+1} while model hot ...")
                    if use_floating_window:
                        floating_window.activate(L + 1)
                    if capture in ("rolling", "rolling-float"):
                        _produce_pool(model, text_model, decoder_layers, tcfg, L + 1, tok_dir,
                                      dst_dir, next_dir, device)
                    elif capture == "rolling-generic":
                        _produce_pool_generic_rolling(model, text_model, decoder_layers, L + 1,
                                                      tok_dir, dst_dir, next_dir, device)
                    else:
                        _produce_pool_hf_rolling(model, text_model, decoder_layers, L + 1,
                                                 tok_dir, dst_dir, next_dir, device)
                    pre_produced_next = True
                else:
                    print(f"  [pipeline] pool L{L+1} already present")
            if use_floating_window:
                # Drop all blocks before SAE training. The SAE phase should not pay
                # for any decoder layers to sit in VRAM.
                floating_window.deactivate_all()
            if L < start_layer:
                # Skip-train layers below the explicit resume target: we produced their
                # pool but won't train them. Keep their source pool for the next layer.
                print(f"  [skip-train] L{L} pool produced; below start_layer={start_layer}")
                continue
        else:
            _produce_pool_hooked(model, decoder_layers, L, tok_dir, dst_dir, device)

        # The LLM is idle during SAE training. Evict it from GPU memory so the SAE
        # can use the freed VRAM for a larger microbatch / less gradient accumulation.
        # Reload before the next layer's pool production.
        # In floating-window mode this is already done (blocks were dropped after production).
        if evict_model and not use_floating_window:
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        provider = RollingActivationProvider(dst_dir, device, seed=seed)
        try:
            # Determine resume checkpoint for this layer
            layer_resume = None
            if resume_from:
                # Only apply the explicit resume to the layer the checkpoint belongs to.
                # The orchestrator skips earlier layers via resume_layer / walk_start above.
                layer_resume = resume_from
            res = train_sae_on_activations(L, d_in, seed, provider,
                                           max_steps=max_steps, bdec_batches=bdec_batches,
                                           microbatch_tokens=microbatch_tokens or MICROBATCH_TOKENS,
                                           resume_from=layer_resume, push=push,
                                           activation_norm_ref=activation_norm_ref,
                                           cpu=cpu)
        finally:
            provider.close()
        results[f"layer_{L:02d}"] = res
        if activation_norm_ref is None:
            probe_norm = res.get("activation_norm_probe") if isinstance(res, dict) else None
            if probe_norm:
                activation_norm_ref = probe_norm

        # Bring the LLM back to GPU for the next layer's pool production.
        # Floating-window mode never moved the full model; individual blocks are
        # re-activated inside the production loop as needed.
        if evict_model and not use_floating_window:
            model.to(device)

        if capture in ("rolling", "rolling-hf", "rolling-hf-float", "rolling-generic"):
            # Persist the just-trained layer's pool as the resume checkpoint. This overwrites
            # the previous resume pool, so we keep exactly one layer on disk for restarts.
            if L >= start_layer:
                _save_resume_pool(dst_dir, L, seed, pool_batches, MODEL_ID, capture,
                                  activation_norm_ref=activation_norm_ref)
            # Pool retention policy with pipeline:
            #   - We just trained L. Pool L+1 was pre-produced from L.
            #   - Keep L as source for L+2 production, plus `pool_retention - 1`
            #     older pools as rollback insulation (retraining L-k needs pool[L-k]
            #     on disk; anything deeper re-enters via the hooked bootstrap).
            #   - Delete everything below that to cap disk.
            cutoff = L - pool_retention if pre_produced_next else L - pool_retention - 1
            for old_L in range(cutoff, walk_start - 1, -1):
                if old_L < 0:
                    continue
                old_dir = pool_dir_for(old_L)
                if old_dir.exists():
                    _rm_pool(old_dir)
                    print(f"  [cleanup] deleted pool L{old_L}")

        if capture == "auto":                             # independent capture -> free now
            _rm_pool(dst_dir)

        # Force GPU memory cleanup between layers to prevent OOM
        import gc
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        import json
        with open(marker_path, "w") as f:
            json.dump({"last_completed_layer": L, "seed": seed,
                       "elapsed_min": (time.time() - t0) / 60}, f)
        elapsed = (time.time() - t0) / 60
        print(f"  [ok] L{L} done: {res}  | elapsed {elapsed:.1f}min")

    print(f"\n{'#'*60}\n  SAE ATLAS COMPLETE  layers={layers}  "
          f"wall={ (time.time()-t0)/60:.1f}min\n{'#'*60}")
    return results


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Event-aware SAE trainer: one self-tuning SAE per decoder layer, "
                    "model-agnostic. The scheduler converges L0 to target with no per-layer "
                    "retuning. Default capture works on any AutoModelForCausalLM.")
    p.add_argument("--model-id", default=None,
                   help=f"HF model id to train SAEs on (default {MODEL_ID})")
    p.add_argument("--capture", choices=["auto", "rolling", "rolling-float", "rolling-hf", "rolling-hf-float"], default="auto",
                   help="auto=model-agnostic forward-hook (default); rolling=Gemma single-block fast path; "
                        "rolling-hf=generic Llama/SmolLM2/Qwen single-block fast path; "
                        "rolling-hf-float=rolling-hf with only active blocks in GPU memory")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--end-layer", type=int, default=9,
                   help="inclusive; clamped to model depth (and to 15 under --capture rolling)")
    p.add_argument("--expansion", type=int, default=None,
                   help=f"SAE dict = expansion * d_in (default {EXPANSION})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--pool-batches", type=int, default=POOL_BATCHES_DEFAULT,
                   help="Activation batches cached per layer. Default=4000 (~400GB). Use 500-1000 for limited disk (~50-100GB).")
    p.add_argument("--microbatch-tokens", type=int, default=MICROBATCH_TOKENS,
                   help=f"Tokens per microbatch for gradient accumulation. Default={MICROBATCH_TOKENS//1024}k (no accum). Use 8192 for 4x VRAM savings.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to checkpoint_full.pt to resume training from.")
    p.add_argument("--no-pretok", dest="use_pretok", action="store_false",
                   help="stream + tokenize FineWeb-Edu live instead of using pre-tokenized shards")
    p.add_argument("--timing-every", type=int, default=None,
                   help=f"[STEP-TIME] print cadence in steps (default {TIMING_EVERY_DEFAULT}); 0 disables. Overrides $SAE_TIMING.")
    p.add_argument("--max-steps", type=int, default=N_STEPS, help="cap steps (smoke tests)")
    p.add_argument("--bdec-batches", type=int, default=BDEC_INIT_BATCHES)
    p.add_argument("--hub-id", default=None,
                   help="HF repo to upload SAEs to. No default -- upload is skipped unless set.")
    p.add_argument("--wandb-project", default=None,
                   help="wandb project name. Off unless set (also needs WANDB_API_KEY).")
    p.add_argument("--no-push", dest="push", action="store_false",
                   help="skip the HuggingFace upload even if --hub-id is set")
    p.add_argument("--no-model-evict", dest="evict_model", action="store_false",
                   help="keep the LLM on GPU during SAE training (disables VRAM freeing)")
    p.add_argument("--target-l0", type=int, default=None,
                   help=f"override the L0 target K (default {K})")
    p.add_argument("--cpu", action="store_true",
                   help="force CPU training (no CUDA). Will be SLOW - for debugging only.")
    p.add_argument("--norm-ref", type=float, default=None,
                   help="pin activation_norm_ref (the chain's L0 probe norm). Required when "
                        "retraining a mid-chain layer in a fresh process, otherwise the first "
                        "trained layer self-references and gets maximum slingshot gain.")
    p.add_argument("--corpus", type=str, default=None,
                   help=f"HF dataset id streamed for activations (default {CORPUS_ID})")
    p.add_argument("--corpus-text-field", type=str, default=None,
                   help="column holding the text (default 'text')")
    p.add_argument("--corpus-prefix", type=str, default=None,
                   help="string prepended to every corpus text, e.g. a model's domain tag")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="pass trust_remote_code=True to tokenizer/model loads (custom_code repos)")
    p.add_argument("--pool-retention", type=int, default=3,
                   help="previous layers' pools kept on disk as rollback insulation "
                        "(default 3; use 1 on disk-tight pods -- each pool costs its full disk size)")
    p.set_defaults(use_pretok=True, push=True, evict_model=True, cpu=False)
    args = p.parse_args()

    # Timing cadence is read from the environment deep in the training loop, so the
    # flag lands there rather than threading a parameter through every call site.
    if args.timing_every is not None:
        os.environ["SAE_TIMING"] = str(max(0, args.timing_every))

    res = run_atlas_rolling(
        start_layer=args.start_layer, end_layer=args.end_layer, seed=args.seed,
        pool_batches=args.pool_batches, microbatch_tokens=args.microbatch_tokens,
        use_pretok=args.use_pretok, max_steps=args.max_steps, bdec_batches=args.bdec_batches,
        resume_from=args.resume_from, push=args.push, capture=args.capture,
        model_id=args.model_id, hub_id=args.hub_id, wandb_project=args.wandb_project,
        expansion=args.expansion, evict_model=args.evict_model,
        target_l0=args.target_l0, cpu=args.cpu, norm_ref=args.norm_ref,
        corpus=args.corpus, corpus_text_field=args.corpus_text_field,
        corpus_prefix=args.corpus_prefix, trust_remote_code=args.trust_remote_code,
        pool_retention=args.pool_retention)
    print(f"\nDone. {res}")


if __name__ == "__main__":
    main()
