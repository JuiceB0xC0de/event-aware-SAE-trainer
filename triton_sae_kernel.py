"""
Triton fused SAE kernel - fuses encode + JumpReLU + decode + loss into one kernel.
Avoids intermediate VRAM writes, reduces memory bandwidth bottlenecks.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_sae_step_kernel(
    # Pointers to inputs
    X_ptr,           # Input activations [n_tokens, d_in]
    W_enc_ptr,       # Encoder weights [d_in, n_features]
    W_dec_ptr,       # Decoder weights [n_features, d_in]
    B_enc_ptr,       # Encoder bias [n_features]
    B_dec_ptr,       # Decoder bias [d_in]
    Threshold_ptr,   # JumpReLU thresholds [n_features]
    # Output pointers
    Loss_ptr,        # Per-token loss [n_tokens]
    L0_ptr,          # Per-token L0 [n_tokens]
    # Dimensions
    n_tokens: tl.constexpr,
    d_in: tl.constexpr,
    n_features: tl.constexpr,
    # Block sizes
    BLOCK_D_IN: tl.constexpr,
    BLOCK_FEATURES: tl.constexpr,
    # Hyperparameters
    target_l0,
    lambda_l0,
    al_mu,
):
    """Fused SAE forward + loss computation.

    Each program handles one token, computing:
    1. pre = X @ W_enc + B_enc
    2. gate = pre > threshold.exp()
    3. acts = pre * gate
    4. x_hat = acts @ W_dec + B_dec
    5. loss = ||X - x_hat||^2 + lambda * max(0, sum(gate) - target)^2
    """
    # Program ID = token index
    token_idx = tl.program_id(0)

    # Load input token [1, d_in]
    d_in_range = tl.arange(0, BLOCK_D_IN)
    mask_d_in = d_in_range < d_in
    x = tl.load(X_ptr + token_idx * d_in + d_in_range, mask=mask_d_in, other=0.0)

    # === ENCODE: pre = X @ W_enc + B_enc ===
    # For each feature block, compute dot product
    pre_acc = tl.zeros([BLOCK_FEATURES], dtype=tl.float32)
    for d_block in range(0, d_in, BLOCK_D_IN):
        d_offset = d_block + d_in_range
        mask = (d_offset < d_in) & mask_d_in
        x_block = tl.load(X_ptr + token_idx * d_in + d_offset, mask=mask, other=0.0)
        w_enc_block = tl.load(W_enc_ptr + d_offset * n_features + tl.arange(0, BLOCK_FEATURES),
                               mask=mask[:, None] & (tl.arange(0, BLOCK_FEATURES) < n_features),
                               other=0.0)
        pre_acc += tl.dot(x_block[None, :], w_enc_block[:, None])

    b_enc = tl.load(B_enc_ptr + tl.arange(0, BLOCK_FEATURES),
                    mask=tl.arange(0, BLOCK_FEATURES) < n_features, other=0.0)
    pre = pre_acc + b_enc

    # === JumpReLU: gate = pre > threshold ===
    threshold = tl.load(Threshold_ptr + tl.arange(0, BLOCK_FEATURES),
                        mask=tl.arange(0, BLOCK_FEATURES) < n_features, other=0.0).exp()
    gate = tl.where(pre > threshold, 1.0, 0.0)
    acts = pre * gate

    # L0 count for this token
    l0 = tl.sum(gate)

    # === DECODE: x_hat = acts @ W_dec + B_dec ===
    x_hat = tl.zeros([BLOCK_D_IN], dtype=tl.float32)
    for f_block in range(0, n_features, BLOCK_FEATURES):
        f_offset = f_block + tl.arange(0, BLOCK_FEATURES)
        mask_f = f_offset < n_features
        acts_block = tl.load(acts + f_offset, mask=mask_f, other=0.0)
        w_dec_block = tl.load(W_dec_ptr + f_offset * d_in + d_in_range,
                               mask=mask_f[:, None] & mask_d_in[None, :], other=0.0)
        x_hat += tl.dot(acts_block[None, :], w_dec_block[:, None])

    b_dec = tl.load(B_dec_ptr + d_in_range, mask=mask_d_in, other=0.0)
    x_hat += b_dec

    # === LOSS: recon + sparsity ===
    residual = x - x_hat
    recon_loss = tl.sum(residual * residual)

    # Sparsity penalty (hinge)
    slack = tl.maximum(0.0, l0 - target_l0)
    sparsity_loss = lambda_l0 * slack + 0.5 * al_mu * slack * slack

    total_loss = recon_loss + sparsity_loss

    # Write outputs
    tl.store(Loss_ptr + token_idx, total_loss)
    tl.store(L0_ptr + token_idx, l0)


def fused_sae_step(x, sae, target_l0, lambda_l0, al_mu):
    """Launch fused SAE kernel.

    Args:
        x: Input activations [n_tokens, d_in]
        sae: JumpReLUSAE module
        target_l0: Target sparsity
        lambda_l0: Current lambda
        al_mu: AL mu parameter

    Returns:
        loss: Total loss (scalar)
        l0: Mean L0 (scalar)
    """
    n_tokens, d_in = x.shape
    n_features = sae.n_features

    # Allocate output buffers
    loss_buf = torch.empty(n_tokens, dtype=torch.float32, device=x.device)
    l0_buf = torch.empty(n_tokens, dtype=torch.float32, device=x.device)

    # Block size tuning - fit in shared memory
    BLOCK_D_IN = triton.next_power_of_2(d_in)
    BLOCK_FEATURES = triton.next_power_of_2(n_features)

    # Cap block sizes to avoid register pressure
    BLOCK_D_IN = min(BLOCK_D_IN, 256)
    BLOCK_FEATURES = min(BLOCK_FEATURES, 256)

    # Launch kernel - one program per token
    grid = (n_tokens,)
    fused_sae_step_kernel[grid](
        x, sae.W_enc.weight, sae.W_dec.weight,
        sae.W_enc.bias, sae.b_dec, sae.log_threshold,
        loss_buf, l0_buf,
        n_tokens=n_tokens, d_in=d_in, n_features=n_features,
        BLOCK_D_IN=BLOCK_D_IN, BLOCK_FEATURES=BLOCK_FEATURES,
        target_l0=target_l0, lambda_l0=lambda_l0, al_mu=al_mu,
    )

    return loss_buf.mean(), l0_buf.mean()
