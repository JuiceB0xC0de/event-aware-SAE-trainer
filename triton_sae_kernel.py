"""
Triton fused SAE forward + backward autograd Functions.

Drop-in replacement for the standard JumpReLU SAE forward/backward.  The forward
pass is fused into a single Triton kernel so the large [n_tokens, n_features]
pre-activation / feature-activation tensors never have to be written to global
memory.  The backward pass is also fused and tiled over the feature dimension,
so it never materialises [n_tokens, n_features] either.

If Triton is not available the module exports a plain PyTorch fallback so the
import in `sae_trainer_rolling.py` does not crash on CPU-only dev machines.
"""
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
            ).to(tl.float32)

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


    # -----------------------------------------------------------------------
    # Backward kernel.  Tiled over (feature_tile, d_tile).  Each block loops
    # over token batches and accumulates gradients without ever materialising
    # the full [n_tokens, n_features] pre-activation / activation tensors.
    # -----------------------------------------------------------------------
    _SAE_BWD_CONFIGS = [
        triton.Config({"BLOCK_B": 16, "BLOCK_F": 64, "BLOCK_D": 64},
                      num_warps=4, num_stages=2),
        triton.Config({"BLOCK_B": 32, "BLOCK_F": 64, "BLOCK_D": 64},
                      num_warps=8, num_stages=2),
        triton.Config({"BLOCK_B": 16, "BLOCK_F": 128, "BLOCK_D": 64},
                      num_warps=8, num_stages=2),
    ]

    @triton.autotune(
        configs=_SAE_BWD_CONFIGS,
        key=["d_in", "n_features"],
    )
    @triton.jit
    def _fused_sae_bwd_kernel(
        # --- pointers ---
        X_c_ptr,        # [n_tokens, d_in]       centered input (x - b_dec)
        GradOut_ptr,    # [n_tokens, d_in]       dL/dx_hat
        GradL0_ptr,     # [n_tokens]             dL/dL0_per_token
        W_enc_ptr,      # [d_in, n_features]     transposed encoder weight
        W_dec_ptr,      # [d_in, n_features]     transposed decoder weight
        B_enc_ptr,      # [n_features]
        LogThr_ptr,     # [n_features]
        # --- outputs ---
        Grad_W_enc_ptr,     # [n_features, d_in]
        Grad_b_enc_ptr,     # [n_features]
        Grad_W_dec_ptr,     # [d_in, n_features]
        Grad_log_thr_ptr,   # [n_features]
        # --- strides ---
        stride_x_d: tl.constexpr,
        stride_gradout_d: tl.constexpr,
        stride_wenc_f: tl.constexpr,
        stride_wdec_f: tl.constexpr,
        stride_gwenc_d: tl.constexpr,
        stride_gwdec_f: tl.constexpr,
        # --- dims ---
        n_tokens: tl.constexpr,
        d_in: tl.constexpr,
        n_features: tl.constexpr,
        eps: tl.constexpr,
        # --- tile sizes ---
        BLOCK_B: tl.constexpr,
        BLOCK_F: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused SAE backward for one (feature_tile, d_tile) output block.

        Each program owns (f_tile, d_tile) and loops over batches of tokens,
        accumulating:
            grad_W_enc[f_tile, d_tile]
            grad_W_dec[d_tile, f_tile]
            grad_b_enc[f_tile]
            grad_log_threshold[f_tile]
        """
        f_start = tl.program_id(0) * BLOCK_F
        d_start = tl.program_id(1) * BLOCK_D

        f_range = f_start + tl.arange(0, BLOCK_F)
        d_range = d_start + tl.arange(0, BLOCK_D)
        mask_f = f_range < n_features
        mask_d = d_range < d_in

        # Per-block accumulators for this (f_tile, d_tile) output region.
        acc_gwenc = tl.zeros([BLOCK_F, BLOCK_D], dtype=tl.float32)
        acc_gwdec = tl.zeros([BLOCK_D, BLOCK_F], dtype=tl.float32)
        acc_gbenc = tl.zeros([BLOCK_F], dtype=tl.float32)
        acc_gthr  = tl.zeros([BLOCK_F], dtype=tl.float32)

        # Load weights and thresholds for this feature tile once.
        b_enc_f = tl.load(B_enc_ptr + f_range, mask=mask_f, other=0.0).to(tl.float32)
        log_thr_f = tl.load(LogThr_ptr + f_range, mask=mask_f, other=1.0).to(tl.float32)
        threshold_f = tl.exp(log_thr_f)

        # Loop over token batches.
        for b_block in range(0, n_tokens, BLOCK_B):
            b_range = b_block + tl.arange(0, BLOCK_B)
            mask_b = b_range < n_tokens

            # Load the full-d token tile for pre-activation recomputation.
            # x_c and grad_out are [BLOCK_B, d_in].
            x_c = tl.load(
                X_c_ptr + b_range[:, None] * stride_x_d + tl.arange(0, d_in)[None, :],
                mask=mask_b[:, None],
                other=0.0,
            ).to(tl.float32)
            grad_out = tl.load(
                GradOut_ptr + b_range[:, None] * stride_gradout_d + tl.arange(0, d_in)[None, :],
                mask=mask_b[:, None],
                other=0.0,
            ).to(tl.float32)
            grad_l0_b = tl.load(GradL0_ptr + b_range, mask=mask_b, other=0.0).to(tl.float32)

            # ---- ENCODE: pre[f_range] = x_c @ W_enc[:, f_range] + b_enc[f_range]
            # W_enc is passed transposed: [d_in, n_features].
            w_enc_f = tl.load(
                W_enc_ptr + tl.arange(0, d_in)[:, None] * stride_wenc_f + f_range[None, :],
                mask=mask_f[None, :],
                other=0.0,
            ).to(tl.float32)
            pre_f = tl.dot(x_c, w_enc_f, allow_tf32=False) + b_enc_f  # [BLOCK_B, BLOCK_F]

            # ---- JumpReLU gate ----
            gate_f = (pre_f > threshold_f).to(tl.float32)
            feat_f = pre_f * gate_f

            # ---- grad through decoder: grad_feat = grad_out @ W_dec[:, f_range]
            w_dec_f = tl.load(
                W_dec_ptr + tl.arange(0, d_in)[:, None] * stride_wdec_f + f_range[None, :],
                mask=mask_f[None, :],
                other=0.0,
            ).to(tl.float32)
            grad_feat_f = tl.dot(grad_out, w_dec_f, allow_tf32=False)  # [BLOCK_B, BLOCK_F]
            grad_pre_f = grad_feat_f * gate_f

            # ---- grad_W_enc[f_range, d_range] += grad_pre_f.T @ x_c[:, d_range]
            x_c_d = tl.load(
                X_c_ptr + b_range[:, None] * stride_x_d + d_range[None, :],
                mask=mask_b[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_gwenc += tl.dot(grad_pre_f.trans(1, 0), x_c_d, allow_tf32=False)

            # ---- grad_W_dec[d_range, f_range] += grad_out[:, d_range].T @ feat_f
            grad_out_d = tl.load(
                GradOut_ptr + b_range[:, None] * stride_gradout_d + d_range[None, :],
                mask=mask_b[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_gwdec += tl.dot(grad_out_d.trans(1, 0), feat_f, allow_tf32=False)

            # ---- grad_b_enc[f_range] += sum_b grad_pre_f
            acc_gbenc += tl.sum(grad_pre_f, axis=0)

            # ---- grad_log_threshold via in-band STE ----
            # combined = pre * grad_feat + grad_l0 per token
            combined_f = pre_f * grad_feat_f + grad_l0_b[:, None]
            in_band = tl.abs(pre_f - threshold_f) < eps
            gthr_f = -tl.sum(combined_f * in_band.to(tl.float32), axis=0) / (2.0 * eps)
            acc_gthr += gthr_f * threshold_f

        # Store accumulated gradients for this (f_tile, d_tile) region.
        tl.store(
            Grad_W_enc_ptr + f_range[:, None] * stride_gwenc_d + d_range[None, :],
            acc_gwenc,
            mask=mask_f[:, None] & mask_d[None, :],
        )
        tl.store(
            Grad_W_dec_ptr + d_range[:, None] * stride_gwdec_f + f_range[None, :],
            acc_gwdec,
            mask=mask_d[:, None] & mask_f[None, :],
        )
        tl.store(
            Grad_b_enc_ptr + f_range,
            acc_gbenc,
            mask=mask_f,
        )
        tl.store(
            Grad_log_thr_ptr + f_range,
            acc_gthr,
            mask=mask_f,
        )


    class FusedSAEForward(torch.autograd.Function):
        """Custom autograd function wrapping the Triton SAE forward.

        Backward is implemented by a separate fused Triton kernel that tiles
        over the feature dimension and avoids materialising [n_tokens, n_features].
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
            B, d_in = x.shape
            n_features = W_enc.shape[0]

            if grad_out is None:
                grad_out = torch.zeros_like(x)
            if grad_l0 is None:
                grad_l0 = torch.zeros(B, device=x.device, dtype=torch.float32)

            x_c = (x - b_dec).to(W_enc.dtype)
            grad_out_t = grad_out.to(W_enc.dtype)

            # Output gradient buffers.
            grad_W_enc = torch.empty_like(W_enc)
            grad_W_dec = torch.empty_like(W_dec)
            grad_b_enc = torch.empty_like(b_enc)
            grad_log_threshold = torch.empty_like(log_threshold)

            grid = lambda meta: (
                triton.cdiv(n_features, meta["BLOCK_F"]),
                triton.cdiv(d_in, meta["BLOCK_D"]),
            )

            _fused_sae_bwd_kernel[grid](
                x_c,
                grad_out_t,
                grad_l0,
                W_enc.t(),
                W_dec.t(),
                b_enc,
                log_threshold,
                grad_W_enc,
                grad_b_enc,
                grad_W_dec,
                grad_log_threshold,
                stride_x_d=x_c.stride(0),
                stride_gradout_d=grad_out_t.stride(0),
                stride_wenc_f=W_enc.t().stride(0),
                stride_wdec_f=W_dec.t().stride(0),
                stride_gwenc_d=grad_W_enc.stride(1),
                stride_gwdec_f=grad_W_dec.stride(1),
                n_tokens=B,
                d_in=d_in,
                n_features=n_features,
                eps=float(bandwidth),
            )

            # grad_b_dec closed in host to avoid extra reads in the kernel.
            grad_b_dec = grad_out.sum(0) - grad_b_enc @ W_enc

            return (
                None,              # x is not a leaf we backprop into
                grad_W_enc,
                grad_b_enc,
                grad_W_dec,
                grad_b_dec,
                grad_log_threshold,
                None,              # bandwidth is not a parameter
            )


    def fused_sae_forward(x: torch.Tensor, sae: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused SAE forward with fused Triton backward.

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
