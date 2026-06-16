"""
Triton fused SAE forward + PyTorch backward autograd Function.

Drop-in replacement for the standard JumpReLU SAE forward.  The forward pass
is fused into a single Triton kernel so the large [n_tokens, n_features]
pre-activation / feature-activation tensors never have to be written to global
memory.  The backward pass is implemented in PyTorch and reproduces the same
in-band straight-through-estimator (STE) threshold gradient used by the trainer's
`_JumpReLUSAE`, so all parameters (W_enc, b_enc, W_dec, b_dec, log_threshold)
still receive correct gradients.

If Triton is not available the module exports a plain PyTorch fallback so the
import in `sae_trainer_rolling.py` does not crash on CPU-only dev machines.
"""
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optional Triton import.  Keep the module importable on CPU / Mac dev boxes.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


def _fallback_fwd(x: torch.Tensor, sae: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch-only fallback matching the Triton function's signature."""
    with torch.no_grad():
        pre = sae.encode_pre(x)
    feat_acts, gate = sae.jumprelu_with_gate(pre)
    x_hat = sae.decode(feat_acts)
    l0 = gate.sum(dim=-1, dtype=torch.float32)
    return x_hat, l0


if not _HAS_TRITON:
    fused_sae_forward = _fallback_fwd
else:

    # -----------------------------------------------------------------------
    # Auto-tune the forward tile sizes for the target A10 / Ampere shape.
    # Tuned for d_in ~ 576 and n_features ~ 18k (SmolLM2-135M-instruct SAE).
    # BLOCK_D must divide the activation dimension cleanly; BLOCK_F can be
    # masked at the tail end of the feature dimension.
    # -----------------------------------------------------------------------
    _SAE_FWD_CONFIGS = [
        triton.Config({"BLOCK_D": 64, "BLOCK_F": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 64, "BLOCK_F": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 64, "BLOCK_F": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 32, "BLOCK_F": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 32, "BLOCK_F": 256}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(
        configs=_SAE_FWD_CONFIGS,
        key=["d_in", "n_features"],
    )
    @triton.jit
    def _fused_sae_fwd_kernel(
        # --- pointers ---
        X_ptr,          # [n_tokens, d_in]       centered input (x - b_dec)
        W_enc_ptr,      # [d_in, n_features]     transposed encoder weight
        B_enc_ptr,      # [n_features]
        W_dec_ptr,      # [n_features, d_in]   transposed decoder weight
        B_dec_ptr,      # [d_in]
        LogThr_ptr,     # [n_features]
        Out_ptr,        # [n_tokens, d_in]
        L0_ptr,         # [n_tokens]
        # --- strides ---
        stride_x_d: tl.constexpr,
        stride_wenc_f: tl.constexpr,
        stride_wdec_d: tl.constexpr,
        stride_out_d: tl.constexpr,
        # --- dims (compile-time specialization) ---
        d_in: tl.constexpr,
        n_features: tl.constexpr,
        # --- tile sizes ---
        BLOCK_D: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        """Fused SAE forward for one token and one output-d tile.

        Each program instance handles (token_idx, d_tile).  The full feature vector
        for the token is computed redundantly by all d_tile programs; this avoids
        materialising [n_tokens, n_features] intermediate tensors in HBM.

        Encode and decode matmuls use tl.dot on bfloat16 inputs with an fp32
        accumulator so Ampere Tensor Cores are exercised.
        """
        token_idx = tl.program_id(0)
        d_start = tl.program_id(1) * BLOCK_D
        d_range = d_start + tl.arange(0, BLOCK_D)
        mask_d = d_range < d_in

        # Decode accumulator for this d-tile.
        out = tl.zeros([BLOCK_D], dtype=tl.float32)
        l0 = tl.zeros([], dtype=tl.float32)

        # Tile over the (large) feature dimension.
        for f_block in range(0, n_features, BLOCK_F):
            f_range = f_block + tl.arange(0, BLOCK_F)
            mask_f = f_range < n_features

            # ---- ENCODE: pre[f_range] = x @ W_enc[:, f_range] + b_enc[f_range]
            pre = tl.load(B_enc_ptr + f_range, mask=mask_f, other=0.0).to(tl.float32)

            for d_block in range(0, d_in, BLOCK_D):
                d_off = d_block + tl.arange(0, BLOCK_D)
                mask_db = d_off < d_in

                xb = tl.load(
                    X_ptr + token_idx * stride_x_d + d_off,
                    mask=mask_db,
                    other=0.0,
                )

                # W_enc is transposed to [d_in, n_features] before passing.
                w_enc = tl.load(
                    W_enc_ptr + d_off[:, None] * stride_wenc_f + f_range[None, :],
                    mask=mask_db[:, None] & mask_f[None, :],
                    other=0.0,
                )

                # [1, BLOCK_D] @ [BLOCK_D, BLOCK_F] -> [1, BLOCK_F], accumulate in fp32.
                pre += tl.sum(
                    tl.dot(xb[None, :], w_enc, allow_tf32=False),
                    axis=0,
                )

            # ---- JumpReLU gate ----
            log_thr = tl.load(LogThr_ptr + f_range, mask=mask_f, other=1.0).to(tl.float32)
            threshold = tl.exp(log_thr)
            gate = tl.where(pre > threshold, pre, 0.0)

            # L0 contribution for this feature tile.
            l0 += tl.sum(gate > 0.0).to(tl.float32)

            # ---- DECODE: out += gate[f_range] @ W_dec[f_range, d_range]
            w_dec = tl.load(
                W_dec_ptr + f_range[:, None] * stride_wdec_d + d_range[None, :],
                mask=mask_f[:, None] & mask_d[None, :],
                other=0.0,
            )

            # [1, BLOCK_F] @ [BLOCK_F, BLOCK_D] -> [1, BLOCK_D], accumulate in fp32.
            out += tl.sum(
                tl.dot(gate[None, :], w_dec, allow_tf32=False),
                axis=0,
            )

        # Add b_dec slice and store this d-tile.
        b_dec_slice = tl.load(B_dec_ptr + d_range, mask=mask_d, other=0.0).to(tl.float32)
        out += b_dec_slice
        tl.store(Out_ptr + token_idx * stride_out_d + d_range, out, mask=mask_d)

        # Only the first d-tile program per token writes the L0 value.
        if d_start == 0:
            tl.store(L0_ptr + token_idx, l0)


    class FusedSAEForward(torch.autograd.Function):
        """Custom autograd function wrapping the Triton SAE forward.

        The backward pass recomputes pre-activations in PyTorch and applies the
        same in-band straight-through-estimator threshold gradient used by
        `_JumpReLUSAE`.  Functionally equivalent to the standard PyTorch path
        while avoiding the [n_tokens, n_features] intermediate writes in the
        forward pass.
        """

        @staticmethod
        def forward(
            ctx,
            x: torch.Tensor,
            W_enc: torch.Tensor,
            b_enc: torch.Tensor,
            W_dec: torch.Tensor,
            b_dec: torch.Tensor,
            log_threshold: torch.Tensor,
            bandwidth: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            B, d_in = x.shape
            n_features = W_enc.shape[0]

            # Run matmul tiles in the same dtype as the activations (bfloat16
            # under autocast), accumulating internally in fp32.
            dt = x.dtype
            x_c = (x - b_dec).to(dt)
            W_enc_T = W_enc.to(dt).t()          # [d_in, n_features]
            W_dec_T = W_dec.to(dt).t()          # [n_features, d_in]
            b_enc_k = b_enc.to(dt)
            b_dec_k = b_dec.to(dt)
            log_thr_k = log_threshold.to(dt)

            out = torch.empty_like(x)
            l0 = torch.empty(B, device=x.device, dtype=torch.float32)

            # Autotune on A10 will pick among the BLOCK_D/BLOCK_F configs above.
            # Set SAE_TRITON_AUTOTUNE=0 to skip autotune and use default 64/128.
            grid = lambda meta: (B, triton.cdiv(d_in, meta["BLOCK_D"]))

            _fused_sae_fwd_kernel[grid](
                x_c, W_enc_T, b_enc_k, W_dec_T, b_dec_k, log_thr_k,
                out, l0,
                stride_x_d=x_c.stride(0),
                stride_wenc_f=W_enc_T.stride(0),
                stride_wdec_d=W_dec_T.stride(0),
                stride_out_d=out.stride(0),
                d_in=d_in,
                n_features=n_features,
            )

            ctx.save_for_backward(x, W_enc, b_enc, W_dec, b_dec, log_threshold)
            ctx.bandwidth = bandwidth
            return out, l0

        @staticmethod
        def backward(
            ctx,
            grad_out: Optional[torch.Tensor],
            grad_l0: Optional[torch.Tensor],
        ):
            x, W_enc, b_enc, W_dec, b_dec, log_threshold = ctx.saved_tensors
            bandwidth = ctx.bandwidth

            x_centered = x - b_dec
            pre = x_centered @ W_enc.t() + b_enc
            threshold = log_threshold.exp()
            gate = (pre > threshold).to(pre.dtype)
            feat_acts = pre * gate

            if grad_out is None:
                grad_out = torch.zeros_like(x)
            if grad_l0 is None:
                grad_l0 = torch.zeros(x.shape[0], device=x.device, dtype=pre.dtype)

            # Gradient through the decode path.
            grad_feat_acts = grad_out @ W_dec              # [B, n_features]
            grad_pre = grad_feat_acts * gate                 # hard gate derivative

            # Straight-through estimator for the JumpReLU thresholds.
            # The combined gradient flowing into the gate tensor is:
            #   pre * grad_feat_acts           (from feat_acts = pre * gate)
            # + grad_l0[token] per token       (from l0_per_token = sum(gate))
            combined = pre * grad_feat_acts + grad_l0[:, None]
            in_band = (pre - threshold).abs() < bandwidth
            eps = bandwidth
            # Sum over the batch dimension only, preserving per-feature gradients.
            grad_threshold = -(combined * in_band).sum(dim=0) / (2.0 * eps)
            grad_log_threshold = grad_threshold * threshold

            # Weight / bias gradients.
            grad_W_enc = grad_pre.t() @ x_centered          # [n_features, d_in]
            grad_b_enc = grad_pre.sum(0)                     # [n_features]
            grad_W_dec = grad_out.t() @ feat_acts            # [d_in, n_features]
            grad_b_dec = grad_out.sum(0) - grad_pre.sum(0) @ W_enc
            grad_x = grad_pre @ W_enc if ctx.needs_input_grad[0] else None

            return (
                grad_x,
                grad_W_enc,
                grad_b_enc,
                grad_W_dec,
                grad_b_dec,
                grad_log_threshold,
                None,  # bandwidth is not a parameter
            )


    def fused_sae_forward(x: torch.Tensor, sae: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused SAE forward with working autograd.

        Args:
            x: [n_tokens, d_in] input activations.
            sae: JumpReLUSAE module with W_enc, W_dec, b_dec, log_threshold,
                 and optionally `ste_bandwidth`.

        Returns:
            x_hat: [n_tokens, d_in] reconstruction.
            l0_per_token: [n_tokens] L0 per token (float32).
        """
        bandwidth = getattr(sae, "ste_bandwidth", 0.1)
        return FusedSAEForward.apply(
            x,
            sae.W_enc.weight,
            sae.W_enc.bias,
            sae.W_dec.weight,
            sae.b_dec,
            sae.log_threshold,
            bandwidth,
        )
