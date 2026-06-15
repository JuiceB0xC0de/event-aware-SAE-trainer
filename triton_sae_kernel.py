"""
Triton fused SAE kernel - fuses encode + JumpReLU + decode + loss.
Reduces memory bandwidth by avoiding intermediate VRAM writes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_sae_encode_decode_kernel(
    # Inputs
    X_ptr,           # [n_tokens, d_in]
    W_enc_ptr,       # [d_in, n_features]
    W_dec_ptr,       # [n_features, d_in]
    B_enc_ptr,       # [n_features]
    B_dec_ptr,       # [d_in]
    Threshold_ptr,   # [n_features]
    # Outputs
    Out_ptr,         # [n_tokens, d_in] - reconstructed
    Gate_sum_ptr,    # [n_tokens] - L0 per token
    # Strides
    stride_x_d: tl.constexpr,
    stride_wenc_d: tl.constexpr,
    stride_wdec_f: tl.constexpr,
    stride_out_d: tl.constexpr,
    # Dimensions
    n_tokens: tl.constexpr,
    d_in: tl.constexpr,
    n_features: tl.constexpr,
    # Block sizes
    BLOCK_D: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """Fused SAE forward: encode → JumpReLU → decode.

    Each program handles BLOCK_D elements of one token's output.
    """
    # Token index and output dimension index
    token_idx = tl.program_id(0)
    d_start = tl.program_id(1) * BLOCK_D

    d_range = d_start + tl.arange(0, BLOCK_D)
    mask_d = d_range < d_in

    # Load input token
    x = tl.load(X_ptr + token_idx * stride_x_d + d_range, mask=mask_d, other=0.0)

    # === ENCODE: pre = X @ W_enc + B_enc ===
    # Accumulate over d_in dimension, output is [n_features]
    pre = tl.zeros([BLOCK_F], dtype=tl.float32)
    for d_block in range(0, d_in, BLOCK_D):
        d_off = d_block + tl.arange(0, BLOCK_D)
        mask_d_block = (d_off < d_in)
        x_block = tl.load(X_ptr + token_idx * stride_x_d + d_off,
                          mask=mask_d_block, other=0.0)
        # Load W_enc slice [BLOCK_D, BLOCK_F]
        w_enc = tl.load(W_enc_ptr + d_off * stride_wenc_d + tl.arange(0, BLOCK_F),
                        mask=mask_d_block[:, None] & (tl.arange(0, BLOCK_F) < n_features),
                        other=0.0)
        pre += tl.dot(x_block[None, :], w_enc)

    b_enc = tl.load(B_enc_ptr + tl.arange(0, BLOCK_F),
                    mask=tl.arange(0, BLOCK_F) < n_features, other=0.0)
    pre += b_enc

    # === JumpReLU ===
    threshold = tl.load(Threshold_ptr + tl.arange(0, BLOCK_F),
                        mask=tl.arange(0, BLOCK_F) < n_features, other=1.0).exp()
    gate = tl.where(pre > threshold, pre, 0.0)  # ReLU-style, zero if below threshold

    # Count active features (L0)
    l0 = tl.sum(gate > 0)

    # === DECODE: out = gate @ W_dec + B_dec ===
    out = tl.zeros([BLOCK_D], dtype=tl.float32)
    for f_block in range(0, n_features, BLOCK_F):
        f_off = f_block + tl.arange(0, BLOCK_F)
        mask_f = f_off < n_features
        gate_block = tl.load(gate + f_off, mask=mask_f, other=0.0)
        w_dec = tl.load(W_dec_ptr + f_off * stride_wdec_f + d_range,
                        mask=mask_f[:, None] & mask_d[None, :], other=0.0)
        out += tl.dot(gate_block[None, :], w_dec)

    b_dec = tl.load(B_dec_ptr + d_range, mask=mask_d, other=0.0)
    out += b_dec

    # Write output
    tl.store(Out_ptr + token_idx * stride_out_d + d_range, out, mask=mask_d)

    # Write L0 (only first program per token does this)
    if d_start == 0:
        tl.store(Gate_sum_ptr + token_idx, l0)


def fused_sae_forward(x, sae):
    """Fused SAE forward pass.

    Args:
        x: [n_tokens, d_in] input activations
        sae: JumpReLUSAE module

    Returns:
        x_hat: [n_tokens, d_in] reconstructed
        l0: [n_tokens] L0 per token
    """
    n_tokens, d_in = x.shape
    n_features = sae.n_features

    # Block sizes - tune these
    BLOCK_D = 64
    BLOCK_F = 128

    # Output buffers
    x_hat = torch.empty_like(x)
    l0_buf = torch.empty(n_tokens, dtype=torch.float32, device=x.device)

    # Launch grid: (n_tokens, ceil(d_in / BLOCK_D))
    grid = (n_tokens, triton.cdiv(d_in, BLOCK_D))

    fused_sae_encode_decode_kernel[grid](
        x, sae.W_enc.weight, sae.W_dec.weight,
        sae.W_enc.bias, sae.b_dec, sae.log_threshold,
        x_hat, l0_buf,
        stride_x_d=x.stride(0),
        stride_wenc_d=sae.W_enc.weight.stride(0),
        stride_wdec_f=sae.W_dec.weight.stride(0),
        stride_out_d=x_hat.stride(0),
        n_tokens=n_tokens, d_in=d_in, n_features=n_features,
        BLOCK_D=BLOCK_D, BLOCK_F=BLOCK_F,
    )

    return x_hat, l0_buf
