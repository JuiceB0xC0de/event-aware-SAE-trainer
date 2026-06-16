"""Micro-benchmark for the Triton fused SAE kernel on CUDA.

Run inside the Modal container (or any CUDA box) with:

    python benchmark_triton_kernel.py

It builds a SAE matching the SmolLM2-135M-instruct atlas geometry
(d_in=576, n_features=18431), checks the fused forward against the PyTorch
path, checks backward gradients, and times forward + backward throughput.
"""
import time
import torch

from sae_trainer_rolling import _make_sae
from triton_sae_kernel import fused_sae_forward


def _stage(msg: str):
    print(f"\n[STAGE] {msg}")


def _ok(msg: str):
    print(f"[OK] {msg}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(0)

    d_in = 576
    n_features = 18431
    batch_tokens = 32768

    _stage("build SAE and random input")
    sae = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    x = torch.randn(batch_tokens, d_in, device=device, dtype=torch.bfloat16)
    _ok(f"SAE on {torch.cuda.get_device_name(device)} | d_in={d_in} n_features={n_features} B={batch_tokens}")

    # ---------------- forward correctness (under bf16 autocast, matching trainer) ----------------
    _stage("compute PyTorch reference forward under bf16 autocast")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pre = sae.encode_pre(x)
        feat, gate = sae.jumprelu_with_gate(pre)
        xhat_ref = sae.decode(feat)
        l0_ref = gate.sum(dim=-1, dtype=torch.float32)
    _ok("reference forward complete")

    _stage("compute Triton fused forward (this triggers autotune compile)")
    xhat_tri, l0_tri = fused_sae_forward(x, sae)
    torch.cuda.synchronize()
    _ok("Triton forward + autotune compile complete")

    recon_err = (xhat_tri - xhat_ref).abs().max().item()
    l0_err = (l0_tri - l0_ref).abs().max().item()
    print(f"[CORRECTNESS] max recon abs err = {recon_err:.2e}")
    print(f"[CORRECTNESS] max L0      abs err = {l0_err:.2e}")
    assert recon_err < 1e-3, f"reconstruction mismatch too large: {recon_err}"
    assert l0_err < 1e-3, f"L0 mismatch too large: {l0_err}"
    _ok("forward correctness passed")

    # ---------------- backward correctness ----------------
    _stage("compute PyTorch reference backward")

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

    ref_grads = _run_ref_backward()
    _ok("reference backward complete")

    _stage("compute Triton fused backward")
    tri_grads = _run_tri_backward()
    torch.cuda.synchronize()
    _ok("Triton backward complete")

    for name in ref_grads:
        err = (tri_grads[name] - ref_grads[name]).abs().max().item()
        print(f"[BWD-CORRECTNESS] {name:20s} max grad err = {err:.2e}")
        assert err < 5e-2, f"{name} gradient mismatch too large: {err}"
    _ok("backward correctness passed")

    # ---------------- PyTorch baseline timing ----------------
    _stage("time PyTorch baseline forward+backward")
    sae.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pre = sae.encode_pre(x)
        feat, gate = sae.jumprelu_with_gate(pre)
        xhat_pt = sae.decode(feat)
        l0_pt = gate.sum(dim=-1, dtype=torch.float32)
        loss = (x - xhat_pt).pow(2).mean() + 1e-3 * l0_pt.mean()
    loss.backward()
    torch.cuda.synchronize()
    t_py = time.perf_counter() - t0
    tok_s_py = batch_tokens / t_py
    print(f"[PYTORCH  ] {t_py*1000:.1f} ms  {tok_s_py/1000:.1f}k tok/s")

    # ---------------- Triton timing ----------------
    _stage("time Triton fused forward+backward")
    sae.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    xhat_tri, l0_tri = fused_sae_forward(x, sae)
    loss = (x - xhat_tri).pow(2).mean() + 1e-3 * l0_tri.mean()
    loss.backward()
    torch.cuda.synchronize()
    t_tri = time.perf_counter() - t0
    tok_s_tri = batch_tokens / t_tri
    print(f"[TRITON   ] {t_tri*1000:.1f} ms  {tok_s_tri/1000:.1f}k tok/s")

    speedup = t_py / t_tri
    print(f"[SPEEDUP  ] {speedup:.2f}x")

    # ---------------- warm repeated runs ----------------
    n_rep = 10
    _stage(f"warm Triton path: {n_rep} repeated forward+backward runs")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_rep):
        sae.zero_grad(set_to_none=True)
        xhat_tri, l0_tri = fused_sae_forward(x, sae)
        loss = (x - xhat_tri).pow(2).mean() + 1e-3 * l0_tri.mean()
        loss.backward()
        if (i + 1) % 2 == 0:
            print(f"  warm {i + 1}/{n_rep} ...")
    torch.cuda.synchronize()
    t_rep = (time.perf_counter() - t0) / n_rep
    print(f"[TRITONx{n_rep}] {t_rep*1000:.1f} ms  {batch_tokens/t_rep/1000:.1f}k tok/s")

    _ok("benchmark complete")


if __name__ == "__main__":
    main()
