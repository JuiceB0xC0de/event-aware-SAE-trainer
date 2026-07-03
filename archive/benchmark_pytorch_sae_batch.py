"""Batch-scaling SAE-only PyTorch benchmark.

Runs the standard trainer SAE path at several batch sizes to see whether
tok/s scales with batch or saturates. Helps decide if memory should be spent
on larger microbatches vs. model eviction/AuxK.

Run:
    python benchmark_pytorch_sae_batch.py
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


def _run_fwd_bwd(sae, x):
    sae.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pre = sae.encode_pre(x)
        feat, gate = sae.jumprelu_with_gate(pre)
        xhat = sae.decode(feat)
        l0 = gate.sum(dim=-1, dtype=torch.float32)
        loss = (x - xhat).pow(2).mean() + 1e-3 * l0.mean()
    loss.backward()


def benchmark_batch(sae, x_full, batch_tokens, n_warm=2, n_rep=5):
    x = x_full[:batch_tokens]
    torch.cuda.synchronize()

    # warm
    for _ in range(n_warm):
        _run_fwd_bwd(sae, x)
        torch.cuda.synchronize()

    times = []
    for _ in range(n_rep):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_fwd_bwd(sae, x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg = sum(times) / len(times)
    return {
        "B": batch_tokens,
        "avg_ms": avg,
        "tok_s": batch_tokens / avg,
        "tok_s_k": batch_tokens / avg,
        "min_ms": min(times),
        "max_ms": max(times),
        "mem_alloc_mb": _mem()[0],
        "mem_res_mb": _mem()[1],
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(0)

    d_in = 576
    n_features = 18431
    max_batch = 32768
    batch_sizes = [4096, 8192, 16384, 24576, 32768]

    _stage("build SAE and random input")
    t0 = time.perf_counter()
    sae = _make_sae(d_in=d_in, n_features=n_features, seed=0).to(device)
    x = torch.randn(max_batch, d_in, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"[TIME] setup: {(time.perf_counter() - t0) * 1000:.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")
    _ok(f"SAE on {torch.cuda.get_device_name(device)} | d_in={d_in} n_features={n_features}")

    results = []
    for B in batch_sizes:
        if B > max_batch:
            continue
        _stage(f"batch B={B}")
        try:
            r = benchmark_batch(sae, x, B)
            results.append(r)
            print(f"[B={B:>5}] avg={r['avg_ms']:.1f} ms  {r['tok_s_k']:.1f}k tok/s  "
                  f"min={r['min_ms']:.1f} max={r['max_ms']:.1f}  "
                  f"mem_alloc={r['mem_alloc_mb']:.0f}MB res={r['mem_res_mb']:.0f}MB")
        except Exception as e:
            print(f"[B={B:>5}] FAILED: {e}")

    _stage("summary")
    print("B        avg_ms   tok/s   mem_alloc_MB")
    for r in results:
        print(f"{r['B']:<8} {r['avg_ms']:<8.1f} {r['tok_s_k']:<7.1f} {r['mem_alloc_mb']:<8.0f}")

    _ok("batch-scaling benchmark complete")


if __name__ == "__main__":
    main()
