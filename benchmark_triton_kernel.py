"""Micro-benchmark for the Triton fused SAE kernel on CUDA.

Run inside the Modal container (or any CUDA box) with:

    python benchmark_triton_kernel.py

It builds a SAE matching the SmolLM2-135M-instruct atlas geometry
(d_in=576, n_features=18431), checks the fused forward against the PyTorch
path, and times forward + backward throughput.
"""
import time
import torch

from sae_trainer_rolling import _make_sae
from triton_sae_kernel import fused_sae_forward


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(0)

    d_in = 576
    n_features = 18431
    batch_tokens = 32768

    sae = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    x = torch.randn(batch_tokens, d_in, device=device, dtype=torch.bfloat16)

    print("=" * 60)
    print(f"Benchmark geometry: d_in={d_in} n_features={n_features} B={batch_tokens}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print("=" * 60)

    # ---------------- correctness ----------------
    with torch.no_grad():
        pre = sae.encode_pre(x)
        feat, gate = sae.jumprelu_with_gate(pre)
        xhat_ref = sae.decode(feat)
        l0_ref = gate.sum(dim=-1, dtype=torch.float32)

    xhat_tri, l0_tri = fused_sae_forward(x, sae)

    recon_err = (xhat_tri - xhat_ref).abs().max().item()
    l0_err = (l0_tri - l0_ref).abs().max().item()
    print(f"[CORRECTNESS] max recon abs err = {recon_err:.2e}")
    print(f"[CORRECTNESS] max L0      abs err = {l0_err:.2e}")
    assert recon_err < 1e-3, f"reconstruction mismatch too large: {recon_err}"
    assert l0_err < 1e-3, f"L0 mismatch too large: {l0_err}"

    # ---------------- PyTorch baseline ----------------
    sae.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
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

    # ---------------- Triton path ----------------
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
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        sae.zero_grad(set_to_none=True)
        xhat_tri, l0_tri = fused_sae_forward(x, sae)
        loss = (x - xhat_tri).pow(2).mean() + 1e-3 * l0_tri.mean()
        loss.backward()
    torch.cuda.synchronize()
    t_rep = (time.perf_counter() - t0) / n_rep
    print(f"[TRITONx{n_rep}] {t_rep*1000:.1f} ms  {batch_tokens/t_rep/1000:.1f}k tok/s")


if __name__ == "__main__":
    main()
