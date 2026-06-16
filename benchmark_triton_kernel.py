"""Micro-benchmark / diagnostic for the Triton fused SAE kernel on CUDA.

Run inside the Modal container (or any CUDA box) with:

    python benchmark_triton_kernel.py

This is *iteration/diagnostic* mode, not final perf tuning. Autotune is
intentionally collapsed to a single known-good config so compile times stay
short while we nail down correctness.
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
    """Emulate the kernel's math exactly: cast operands to bf16, reduce in fp32.

    Uses smaller effective batch (same as x here; keep it cheap) and the same
    tile-invariant reduction order as a plain matmul would. This is a bound, not
    the ground truth.
    """
    xb = x.to(dtype)
    W_enc = sae.W_enc.weight.to(dtype)
    b_enc = sae.W_enc.bias.to(dtype)
    W_dec = sae.W_dec.weight.to(dtype)
    b_dec = sae.b_dec.to(dtype)
    log_thr = sae.log_threshold.to(dtype)

    with torch.no_grad():
        pre = (xb - b_dec) @ W_enc.t() + b_enc  # still fp32 accumulation inside cuBLAS
        threshold = torch.exp(log_thr)
        gate = (pre > threshold).to(dtype)
        feat = pre * gate
        xhat = feat @ W_dec.t() + b_dec
        l0 = gate.sum(dim=-1, dtype=torch.float32)
    return xhat, l0, pre, gate


def _report(ref_label, xhat_ref, l0_ref, xhat_tri, l0_tri):
    diff = (xhat_tri - xhat_ref).abs()
    l0_diff = (l0_tri - l0_ref).abs()
    print(f"\n[REPORT] {ref_label}")
    print(f"  recon p50={diff.float().quantile(0.50).item():.2e}  "
          f"p95={diff.float().quantile(0.95).item():.2e}  "
          f"p99={diff.float().quantile(0.99).item():.2e}  "
          f"max={diff.max().item():.2e}  mean={diff.mean().item():.2e}")
    print(f"  L0    p50={l0_diff.float().quantile(0.50).item():.2e}  "
          f"p95={l0_diff.float().quantile(0.95).item():.2e}  "
          f"p99={l0_diff.float().quantile(0.99).item():.2e}  "
          f"max={l0_diff.max().item():.2e}  mean={l0_diff.mean().item():.2e}")
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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(0)

    d_in = 576
    n_features = 18431
    batch_tokens = 32768

    _stage("build SAE and random input")
    t0 = time.perf_counter()
    sae = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    x = torch.randn(batch_tokens, d_in, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"[TIME] setup: {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")
    _ok(f"SAE on {torch.cuda.get_device_name(device)} | d_in={d_in} n_features={n_features} B={batch_tokens}")

    # ---------------- forced-gate tests ----------------
    _stage("forced-gate correctness tests")

    # all-off: drive every preactivation far below threshold by subtracting a huge bias offset
    with torch.no_grad():
        log_thr = sae.log_threshold
        threshold = torch.exp(log_thr)
        max_thr = threshold.max().item()
        b_enc_off = sae.W_enc.bias - (max_thr + 10.0)
    sae_off = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    with torch.no_grad():
        sae_off.W_enc.bias.copy_(b_enc_off)
    xhat_off, l0_off = fused_sae_forward(x, sae_off)
    b_dec_off = sae_off.b_dec
    off_err = (xhat_off - b_dec_off).abs().max().item()
    print(f"[ALL-OFF] max |xhat - b_dec| = {off_err:.2e}  L0 max = {l0_off.max().item():.2e}")
    assert off_err < 1e-3, f"all-off test failed: {off_err}"
    assert l0_off.max().item() == 0.0, "all-off L0 must be zero"
    _ok("all-off forced test passed")

    # all-on-ish: drive every preactivation far above threshold
    b_enc_on = sae.W_enc.bias + (max_thr + 10.0)
    sae_on = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    with torch.no_grad():
        sae_on.W_enc.bias.copy_(b_enc_on)
    xhat_tri_on, l0_tri_on = fused_sae_forward(x, sae_on)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pre_on = sae_on.encode_pre(x)
        feat_on, gate_on = sae_on.jumprelu_with_gate(pre_on)
        xhat_ref_on = sae_on.decode(feat_on)
        l0_ref_on = gate_on.sum(dim=-1, dtype=torch.float32)
    diff_on = (xhat_tri_on - xhat_ref_on).abs()
    on_recon_err = diff_on.max().item()
    on_l0_err = (l0_tri_on - l0_ref_on).abs().max().item()
    print(f"[ALL-ON ] recon p50={diff_on.float().quantile(0.50).item():.2e}  "
          f"p95={diff_on.float().quantile(0.95).item():.2e}  "
          f"p99={diff_on.float().quantile(0.99).item():.2e}  "
          f"max={on_recon_err:.2e}  mean={diff_on.mean().item():.2e}")
    print(f"[ALL-ON ] max L0 err = {on_l0_err:.2e}")
    assert torch.isfinite(xhat_tri_on).all(), "all-on output contains inf/nan"
    assert on_l0_err < 1e-3, f"all-on L0 test failed: {on_l0_err}"
    _ok("all-on forced test passed (L0 exact, output finite)")

    # ---------------- default threshold: three references ----------------
    _stage("default threshold: three reference comparisons")

    with torch.no_grad():
        xhat_bf16, l0_bf16, pre_bf16, gate_bf16 = _sae_forward_pytorch(sae, x, torch.bfloat16)
        xhat_fp32, l0_fp32, pre_fp32, gate_fp32 = _sae_forward_pytorch(sae, x, torch.float32)
        xhat_kmath, l0_kmath, pre_kmath, gate_kmath = _kernel_math_emulation(sae, x)

    _threshold_margin_stats(pre_fp32, sae.log_threshold, margin=0.05)
    _threshold_margin_stats(pre_fp32, sae.log_threshold, margin=0.10)
    _threshold_margin_stats(pre_fp32, sae.log_threshold, margin=0.20)

    _stage("compute Triton fused forward (compile + single-config autotune)")
    t0 = time.perf_counter()
    xhat_tri, l0_tri = fused_sae_forward(x, sae)
    torch.cuda.synchronize()
    print(f"[TIME] Triton forward first call (compile): {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")
    _ok("Triton forward compile complete")

    _report("Triton vs PyTorch bf16 autocast", xhat_bf16, l0_bf16, xhat_tri, l0_tri)
    _report("Triton vs PyTorch fp32", xhat_fp32, l0_fp32, xhat_tri, l0_tri)
    _report("Triton vs kernel-math emulation", xhat_kmath, l0_kmath, xhat_tri, l0_tri)

    # Gate agreement statistics would require materialising [B, n_features];
    # skip in this fused-kernel diagnostic.
    _warn("gate agreement not computed (would require full pre materialisation)")

    # ---------------- backward correctness (against bf16 autocast, the real baseline) ----------------
    _stage("backward correctness against bf16 autocast baseline")

    def _run_ref_backward():
        sae.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pre = sae.encode_pre(x)
            feat, gate = sae.jumprelu_with_gate(pre)
            xhat_ref = sae.decode(feat)
            l0_ref = gate.sum(dim=-1, dtype=torch.float32)
            loss = (x - xhat_ref).pow(2).mean() + 1e-3 * l0_ref.mean()
        loss.backward()
        return {n: p.grad.clone() for n, p in sae.named_parameters() if p.grad is not None}

    def _run_tri_backward():
        sae.zero_grad(set_to_none=True)
        xhat_tri, l0_tri = fused_sae_forward(x, sae)
        loss = (x - xhat_tri).pow(2).mean() + 1e-3 * l0_tri.mean()
        loss.backward()
        return {n: p.grad.clone() for n, p in sae.named_parameters() if p.grad is not None}

    t0 = time.perf_counter()
    ref_grads = _run_ref_backward()
    torch.cuda.synchronize()
    print(f"[TIME] ref backward: {(time.perf_counter() - t0) * 1000:.1f} ms")

    t0 = time.perf_counter()
    tri_grads = _run_tri_backward()
    torch.cuda.synchronize()
    print(f"[TIME] Triton backward first call: {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")

    grad_ok = True
    for name in ref_grads:
        err = (tri_grads[name] - ref_grads[name]).abs().max().item()
        rel = err / (ref_grads[name].abs().mean().item() + 1e-8)
        print(f"[BWD] {name:20s} max err={err:.2e}  rel={rel:.2e}")
        if rel > 0.05:
            grad_ok = False
            _warn(f"{name} relative gradient error > 5%")
    if grad_ok:
        _ok("backward gradients within 5% relative error")

    # ---------------- timing ----------------
    _stage("forward+backward timing (single sample)")

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pre = sae.encode_pre(x)
        feat, gate = sae.jumprelu_with_gate(pre)
        xhat_pt = sae.decode(feat)
        l0_pt = gate.sum(dim=-1, dtype=torch.float32)
        loss = (x - xhat_pt).pow(2).mean() + 1e-3 * l0_pt.mean()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    t_py_cuda = start.elapsed_time(end)
    print(f"[PYTORCH  ] cuda={t_py_cuda:.1f} ms  {batch_tokens/t_py_cuda/1000:.1f}k tok/s")

    sae.zero_grad(set_to_none=True)
    start.record()
    xhat_tri, l0_tri = fused_sae_forward(x, sae)
    loss = (x - xhat_tri).pow(2).mean() + 1e-3 * l0_tri.mean()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    t_tri_cuda = start.elapsed_time(end)
    print(f"[TRITON   ] cuda={t_tri_cuda:.1f} ms  {batch_tokens/t_tri_cuda/1000:.1f}k tok/s")
    print(f"[SPEEDUP  ] {t_py_cuda/t_tri_cuda:.2f}x")

    n_rep = 10
    _stage(f"warm Triton path: {n_rep} repeated forward+backward runs")
    times = []
    for i in range(n_rep):
        sae.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        xhat_tri, l0_tri = fused_sae_forward(x, sae)
        loss = (x - xhat_tri).pow(2).mean() + 1e-3 * l0_tri.mean()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        dt = start.elapsed_time(end)
        times.append(dt)
        if (i + 1) % 2 == 0:
            print(f"  warm {i + 1}/{n_rep}: cuda={dt:.1f} ms  avg={sum(times)/len(times):.1f} ms")

    t_rep = sum(times) / n_rep
    print(f"[TRITONx{n_rep}] avg cuda={t_rep:.1f} ms  {batch_tokens/t_rep/1000:.1f}k tok/s")
    print(f"[TRITONx{n_rep}] min={min(times):.1f} ms  max={max(times):.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")

    _ok("diagnostic benchmark complete")


if __name__ == "__main__":
    main()
