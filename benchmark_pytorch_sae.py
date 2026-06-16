"""Clean SAE-only PyTorch benchmark: fwd+bwd, no data pipeline.

Measures the standard trainer SAE path on random activations, optionally
under torch.compile, so we know the ceiling of PyTorch/cuBLAS SAE math.
"""
import time
import torch

from sae_trainer_rolling import _make_sae


def _stage(msg: str):
    print(f"\n[STAGE] {msg}")


def _ok(msg: str):
    print(f"[OK] {msg}")


def _mem():
    return torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2


def _run_fwd_bwd(sae, x, compile_mode: bool):
    sae.zero_grad(set_to_none=True)
    if compile_mode:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pre = sae.encode_pre(x)
            feat, gate = sae.jumprelu_with_gate(pre)
            xhat = sae.decode(feat)
            l0 = gate.sum(dim=-1, dtype=torch.float32)
            loss = (x - xhat).pow(2).mean() + 1e-3 * l0.mean()
    else:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pre = sae.encode_pre(x)
            feat, gate = sae.jumprelu_with_gate(pre)
            xhat = sae.decode(feat)
            l0 = gate.sum(dim=-1, dtype=torch.float32)
            loss = (x - xhat).pow(2).mean() + 1e-3 * l0.mean()
    loss.backward()
    return loss.item()


def main(compile_sae: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(0)

    d_in = 576
    n_features = 18431
    batch_tokens = 32768
    n_warm = 3
    n_rep = 10

    _stage("build SAE and random input")
    t0 = time.perf_counter()
    sae = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    x = torch.randn(batch_tokens, d_in, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"[TIME] setup: {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")
    _ok(f"SAE on {torch.cuda.get_device_name(device)} | d_in={d_in} n_features={n_features} B={batch_tokens}")

    if compile_sae:
        _stage("torch.compile SAE")
        t0 = time.perf_counter()
        sae = torch.compile(sae, mode="max-autotune")
        # warm compile
        _run_fwd_bwd(sae, x, compile_mode=False)
        torch.cuda.synchronize()
        print(f"[TIME] first compiled step: {(time.perf_counter() - t0) * 1000:.1f} ms")
        print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")

    _stage(f"warm-up: {n_warm} fwd+bwd steps")
    for i in range(n_warm):
        _run_fwd_bwd(sae, x, compile_mode=compile_sae)
        torch.cuda.synchronize()
        print(f"  warm {i + 1}/{n_warm} done")

    _stage(f"timed runs: {n_rep} fwd+bwd steps")
    times = []
    for i in range(n_rep):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss_val = _run_fwd_bwd(sae, x, compile_mode=compile_sae)
        end.record()
        torch.cuda.synchronize()
        dt = start.elapsed_time(end)
        times.append(dt)
        if (i + 1) % 2 == 0:
            print(f"  timed {i + 1}/{n_rep}: cuda={dt:.1f} ms  loss={loss_val:.4f}")

    avg = sum(times) / len(times)
    print(f"\n[RESULT] compile={compile_sae}  avg cuda={avg:.1f} ms  {batch_tokens/avg:.1f}k tok/s")
    print(f"[RESULT] min={min(times):.1f} ms  max={max(times):.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")
    _ok("SAE-only PyTorch benchmark complete")


if __name__ == "__main__":
    import sys
    compile_flag = "--compile" in sys.argv
    main(compile_sae=compile_flag)
