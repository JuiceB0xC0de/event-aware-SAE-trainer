"""Forward-only diagnostic for the Triton fused SAE kernel on CUDA.

The current fused backward kernel requires more shared memory than an A10
has (672 KB requested vs 101 KB limit) because it materialises the full
padded d_in dimension per block. It is disabled here. This benchmark answers
the narrower question: is the current fused forward worth keeping while we
rewrite the backward?

Run inside the Modal container (or any CUDA box) with:

    python benchmark_triton_kernel.py
"""
import time
import torch

from sae_trainer_rolling import _make_sae
from triton_sae_kernel import fused_sae_forward


def _stage(msg: str):
    print(f"\n[STAGE] {msg}")


def _ok(msg: str):
    print(f"[OK] {msg}")


def _warn(msg: str):
    print(f"[WARN] {msg}")


def _mem():
    return torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2


def _sae_forward_pytorch(sae, x, dtype):
    """Reference forward in a chosen dtype (fp32 or bf16)."""
    with torch.amp.autocast("cuda", dtype=dtype) if dtype == torch.bfloat16 else torch.no_grad():
        pre = sae.encode_pre(x.to(dtype))
        feat, gate = sae.jumprelu_with_gate(pre)
        xhat = sae.decode(feat)
        l0 = gate.sum(dim=-1, dtype=torch.float32)
    return xhat, l0, pre, gate


def _kernel_math_emulation(sae, x, dtype=torch.bfloat16):
    """Emulate the kernel's math: cast operands to bf16, reduce in fp32."""
    xb = x.to(dtype)
    W_enc = sae.W_enc.weight.to(dtype)
    b_enc = sae.W_enc.bias.to(dtype)
    W_dec = sae.W_dec.weight.to(dtype)
    b_dec = sae.b_dec.to(dtype)
    log_thr = sae.log_threshold.to(dtype)

    with torch.no_grad():
        pre = (xb - b_dec) @ W_enc.t() + b_enc
        threshold = torch.exp(log_thr)
        gate = (pre > threshold).to(dtype)
        feat = pre * gate
        xhat = feat @ W_dec.t() + b_dec
        l0 = gate.sum(dim=-1, dtype=torch.float32)
    return xhat, l0, pre, gate


def _err_stats(label, xhat_ref, l0_ref, xhat_tri, l0_tri):
    diff = (xhat_tri - xhat_ref).abs().float()
    l0_diff = (l0_tri - l0_ref).abs().float()
    print(f"\n[REPORT] {label}")
    print(f"  recon mean={diff.mean().item():.2e} "
          f"rms={diff.pow(2).mean().sqrt().item():.2e} "
          f"max={diff.max().item():.2e}")
    print(f"  L0    mean={l0_diff.mean().item():.2e} "
          f"rms={l0_diff.pow(2).mean().sqrt().item():.2e} "
          f"max={l0_diff.max().item():.2e}")
    return diff, l0_diff


def _threshold_margin_stats(pre, log_thr, margin=0.1):
    """Count features whose |pre - threshold| / threshold is within margin."""
    threshold = torch.exp(log_thr)
    rel = (pre.abs() - threshold[None, :]).abs() / (threshold[None, :] + 1e-8)
    near = rel < margin
    total = near.numel()
    count = near.sum().item()
    print(f"[THRESH-MARGIN] features within {margin*100:.0f}% of threshold: "
          f"{count} / {total} ({100*count/total:.3f}%)")
    return count, total


def _time_forward_pytorch(sae, x, n_warm=3, n_rep=10):
    torch.cuda.synchronize()
    for _ in range(n_warm):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pre = sae.encode_pre(x)
            feat, gate = sae.jumprelu_with_gate(pre)
            _ = sae.decode(feat)
            _ = gate.sum(dim=-1, dtype=torch.float32)
    times = []
    for _ in range(n_rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pre = sae.encode_pre(x)
            feat, gate = sae.jumprelu_with_gate(pre)
            xhat = sae.decode(feat)
            l0 = gate.sum(dim=-1, dtype=torch.float32)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times, xhat, l0


def _time_forward_triton(sae, x, n_warm=3, n_rep=10):
    torch.cuda.synchronize()
    for _ in range(n_warm):
        with torch.no_grad():
            _ = fused_sae_forward(x, sae)
    times = []
    for _ in range(n_rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            xhat, l0 = fused_sae_forward(x, sae)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times, xhat, l0


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(0)

    d_in = 576
    n_features = 18431
    batch_tokens = 32768
    b_diag = 1024

    _stage("build SAE and random input")
    t0 = time.perf_counter()
    sae = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    x = torch.randn(batch_tokens, d_in, device=device, dtype=torch.bfloat16)
    x_diag = x[:b_diag]
    torch.cuda.synchronize()
    print(f"[TIME] setup: {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")
    _ok(f"SAE on {torch.cuda.get_device_name(device)} | d_in={d_in} n_features={n_features} B={batch_tokens}")

    # ---------------- diagnostics on x_diag only ----------------
    _stage(f"diagnostics on B={b_diag}")

    # all-off
    with torch.no_grad():
        log_thr = sae.log_threshold
        threshold = torch.exp(log_thr)
        max_thr = threshold.max().item()
        b_enc_off = sae.W_enc.bias - (max_thr + 10.0)
    sae_off = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    with torch.no_grad():
        sae_off.W_enc.bias.copy_(b_enc_off)
    xhat_off, l0_off = fused_sae_forward(x_diag, sae_off)
    off_err = (xhat_off - sae_off.b_dec).abs().max().item()
    print(f"[ALL-OFF] max |xhat - b_dec| = {off_err:.2e}  L0 max = {l0_off.max().item():.2e}")
    assert off_err < 1e-3, f"all-off test failed: {off_err}"
    assert l0_off.max().item() == 0.0, "all-off L0 must be zero"
    _ok("all-off forced test passed")

    # all-on
    b_enc_on = sae.W_enc.bias + (max_thr + 10.0)
    sae_on = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    with torch.no_grad():
        sae_on.W_enc.bias.copy_(b_enc_on)
    xhat_tri_on, l0_tri_on = fused_sae_forward(x_diag, sae_on)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pre_on = sae_on.encode_pre(x_diag)
        feat_on, gate_on = sae_on.jumprelu_with_gate(pre_on)
        xhat_ref_on = sae_on.decode(feat_on)
        l0_ref_on = gate_on.sum(dim=-1, dtype=torch.float32)
    diff_on = (xhat_tri_on - xhat_ref_on).abs().float()
    on_l0_err = (l0_tri_on - l0_ref_on).abs().max().item()
    print(f"[ALL-ON ] recon mean={diff_on.mean().item():.2e} "
          f"rms={diff_on.pow(2).mean().sqrt().item():.2e} "
          f"max={diff_on.max().item():.2e}")
    print(f"[ALL-ON ] max L0 err = {on_l0_err:.2e}")
    assert torch.isfinite(xhat_tri_on).all(), "all-on output contains inf/nan"
    assert on_l0_err < 1e-3, f"all-on L0 test failed: {on_l0_err}"
    _ok("all-on forced test passed (L0 exact, output finite)")

    # default threshold
    with torch.no_grad():
        xhat_bf16, l0_bf16, pre_bf16, _ = _sae_forward_pytorch(sae, x_diag, torch.bfloat16)
        xhat_fp32, l0_fp32, pre_fp32, _ = _sae_forward_pytorch(sae, x_diag, torch.float32)
        xhat_kmath, l0_kmath, pre_kmath, _ = _kernel_math_emulation(sae, x_diag)

    xhat_tri_diag, l0_tri_diag = fused_sae_forward(x_diag, sae)
    _err_stats("Triton vs PyTorch bf16 autocast", xhat_bf16, l0_bf16, xhat_tri_diag, l0_tri_diag)
    _err_stats("Triton vs PyTorch fp32", xhat_fp32, l0_fp32, xhat_tri_diag, l0_tri_diag)
    _err_stats("Triton vs kernel-math emulation", xhat_kmath, l0_kmath, xhat_tri_diag, l0_tri_diag)

    _threshold_margin_stats(pre_fp32, sae.log_threshold, margin=0.05)
    _threshold_margin_stats(pre_fp32, sae.log_threshold, margin=0.10)
    _threshold_margin_stats(pre_fp32, sae.log_threshold, margin=0.20)

    _warn("fused Triton backward is disabled: A10 shared-memory footprint exceeds hardware limit")

    # clear diagnostic tensors before full-B timing
    del xhat_off, l0_off, sae_off
    del xhat_tri_on, l0_tri_on, sae_on, xhat_ref_on, l0_ref_on
    del xhat_bf16, l0_bf16, pre_bf16, xhat_fp32, l0_fp32, pre_fp32
    del xhat_kmath, l0_kmath, pre_kmath, xhat_tri_diag, l0_tri_diag
    torch.cuda.empty_cache()
    print(f"\n[MEM after cache clear] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")

    # ---------------- full-B forward-only timing ----------------
    _stage("full-B forward-only timing")

    pt_times, _, _ = _time_forward_pytorch(sae, x)
    print(f"[PYTORCH FWD] min={min(pt_times):.1f} ms  max={max(pt_times):.1f} ms  "
          f"avg={sum(pt_times)/len(pt_times):.1f} ms  "
          f"{batch_tokens/(sum(pt_times)/len(pt_times))/1000:.1f}k tok/s")

    tri_times, xhat_tri_full, l0_tri_full = _time_forward_triton(sae, x)
    print(f"[TRITON FWD]  min={min(tri_times):.1f} ms  max={max(tri_times):.1f} ms  "
          f"avg={sum(tri_times)/len(tri_times):.1f} ms  "
          f"{batch_tokens/(sum(tri_times)/len(tri_times))/1000:.1f}k tok/s")

    speedup = (sum(pt_times)/len(pt_times)) / (sum(tri_times)/len(tri_times))
    print(f"[SPEEDUP FWD] {speedup:.2f}x")

    # quick sanity on full-B output
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        xhat_pt_full, l0_pt_full, _, _ = _sae_forward_pytorch(sae, x, torch.bfloat16)
    diff_full = (xhat_tri_full - xhat_pt_full).abs().float()
    l0_diff_full = (l0_tri_full - l0_pt_full).abs().float()
    print(f"\n[FULL-B FWD CHECK]")
    print(f"  recon mean={diff_full.mean().item():.2e} rms={diff_full.pow(2).mean().sqrt().item():.2e} max={diff_full.max().item():.2e}")
    print(f"  L0    mean={l0_diff_full.mean().item():.2e} rms={l0_diff_full.pow(2).mean().sqrt().item():.2e} max={l0_diff_full.max().item():.2e}")

    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")
    _ok("forward-only diagnostic benchmark complete")


if __name__ == "__main__":
    main()
