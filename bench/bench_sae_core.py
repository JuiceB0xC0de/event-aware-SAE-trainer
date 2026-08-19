"""Microbenchmark for the SAE training inner loop.

Isolates the forward/backward that dominates VRAM and step time during SAE
training, driven by synthetic activations so it needs no model download, no
token pool, and no disk. Shapes and op order mirror the real loop in
`train_sae_on_activations` (~2545-2660 of sae_trainer_rolling.py).

Why synthetic is fine here: peak VRAM and kernel cost depend on tensor shapes
and dtypes, not on activation content. Quality comparisons still need real
activations -- this harness is explicitly not for those.

Usage:
    python bench/bench_sae_core.py --label baseline
    python bench/bench_sae_core.py --label baseline --microbatch-tokens 8192
    python bench/bench_sae_core.py --json out.json

Reports peak allocated / reserved VRAM, tok/s, and a forward / backward /
optimizer split measured with CUDA events.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import sae_trainer_rolling as T  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_state_clone(sae, reps=3):
    """Cost of snapshotting the whole SAE to host.

    The dead-feature rollback buffer used to hold four of these and refresh one
    every log window; it now holds one and reuses its storage. The copy is
    unpinned, so it blocks. Reported per clone so the buffer-size choice can be
    priced directly.
    """
    sd = sae.state_dict()
    n_bytes = sum(v.numel() * v.element_size() for v in sd.values())
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        snapshot = {k: v.detach().cpu().clone() for k, v in sd.items()}
        del snapshot
    _sync()
    per_clone_ms = (time.perf_counter() - t0) / reps * 1000
    return {
        "state_clone_gb": round(n_bytes / 1024 ** 3, 3),
        "state_clone_ms": round(per_clone_ms, 1),
        "host_ram_legacy_gb": round(n_bytes * 4 / 1024 ** 3, 3),   # 4 rollback slots
        "host_ram_single_gb": round(n_bytes / 1024 ** 3, 3),       # 1 reused slot
    }


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    n_features = args.d_in * args.expansion
    accum_steps = max(1, args.batch_tokens // args.microbatch_tokens)
    micro = args.batch_tokens // accum_steps

    sae = T._make_sae(args.d_in, n_features, seed=args.seed).to(device)
    sae = sae.to(dtype=torch.float32)  # params fp32, activations bf16
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)

    # Synthetic activation batch, regenerated per step so nothing is cached.
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    # Simulate a realistic dead-feature population for the aux path.
    n_dead = int(n_features * args.dead_frac)
    if n_dead > 0:
        dead_indices = torch.randperm(n_features, generator=gen)[:n_dead].to(device)
        eff_k = min(T.AUX_K, n_dead)
    else:
        dead_indices, eff_k = None, 0

    target_l0 = float(args.target_l0)
    lambda_l0 = 1.0
    al_mu = 1.0

    def one_step(measure: bool):
        stats = {"fwd_ms": 0.0, "bwd_ms": 0.0, "opt_ms": 0.0}
        optimizer.zero_grad(set_to_none=True)
        fired_accum = torch.zeros(n_features, dtype=torch.bool, device=device)
        accum_l0 = 0.0

        for _ in range(accum_steps):
            acts_mb = torch.randn(micro, args.d_in, generator=gen,
                                  dtype=torch.float32).to(device=device, dtype=dtype)

            if measure:
                _sync()
                t0 = time.perf_counter()

            pre = sae.encode_pre(acts_mb)
            feat_acts, gate = sae.jumprelu_with_gate(pre)
            gate_counts = gate.sum(dim=-1, dtype=torch.float32)
            active_max = (int(gate_counts.max().item())
                          if T.SAE_SPARSE_DECODE else None)
            x_hat = sae.decode(feat_acts, active_max=active_max)

            residual_float = acts_mb.float() - x_hat.float()
            recon_loss = residual_float.pow(2).mean() / accum_steps

            l0_mb = gate_counts.mean()
            slack = (l0_mb - target_l0).clamp(min=0.0)
            sparsity_loss = (lambda_l0 * slack + 0.5 * al_mu * slack * slack) / accum_steps

            if n_dead > 0:
                # Ref-agnostic: use the sparse helper when the checkout has it,
                # otherwise reproduce the dense formulation so the same benchmark
                # runs against main and against the optimisation branch.
                # Dense by default because that is what the trainer ships. The
                # sparse helper stays importable and is opt-in via --sparse-aux,
                # so its presence in the module does not silently change what
                # this harness measures.
                if args.sparse_aux and hasattr(T, "_dead_feature_aux_recon"):
                    x_aux = T._dead_feature_aux_recon(
                        pre, dead_indices, eff_k, sae.W_dec.weight)
                else:
                    pre_dead = pre[:, dead_indices].relu()
                    topk_vals, topk_idx = pre_dead.topk(eff_k, dim=-1)
                    aux_acts = torch.zeros_like(pre_dead)
                    aux_acts.scatter_(-1, topk_idx, topk_vals)
                    W_dec_dead = sae.W_dec.weight.t()[dead_indices]
                    x_aux = aux_acts @ W_dec_dead
                residual_target = residual_float.detach()
                aux_loss = ((residual_target - x_aux.float()).pow(2).mean()
                            * T.AUX_COEFF) / accum_steps
            else:
                aux_loss = torch.zeros((), device=device, dtype=torch.float32)

            loss = recon_loss + sparsity_loss + aux_loss

            if measure:
                _sync()
                stats["fwd_ms"] += (time.perf_counter() - t0) * 1000
                t1 = time.perf_counter()

            loss.backward()

            if measure:
                _sync()
                stats["bwd_ms"] += (time.perf_counter() - t1) * 1000

            # These two .item() calls are the per-microbatch syncs under review.
            accum_l0 += l0_mb.detach().item()
            with torch.no_grad():
                fired_accum |= (feat_acts.detach() > 0).any(dim=0)

        if measure:
            _sync()
            t2 = time.perf_counter()

        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()

        if measure:
            _sync()
            stats["opt_ms"] = (time.perf_counter() - t2) * 1000
        return stats

    # Warmup: allocator settles, autotune runs, nothing measured.
    for _ in range(args.warmup):
        one_step(measure=False)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync()
    wall0 = time.perf_counter()

    agg = {"fwd_ms": 0.0, "bwd_ms": 0.0, "opt_ms": 0.0}
    for _ in range(args.steps):
        s = one_step(measure=True)
        for k in agg:
            agg[k] += s[k]

    _sync()
    wall = time.perf_counter() - wall0

    peak_alloc = (torch.cuda.max_memory_allocated() / 1024 ** 3
                  if torch.cuda.is_available() else 0.0)
    peak_resv = (torch.cuda.max_memory_reserved() / 1024 ** 3
                 if torch.cuda.is_available() else 0.0)

    result = {
        "label": args.label,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "d_in": args.d_in,
        "expansion": args.expansion,
        "n_features": n_features,
        "batch_tokens": args.batch_tokens,
        "microbatch_tokens": micro,
        "accum_steps": accum_steps,
        "dtype": args.dtype,
        "dead_frac": args.dead_frac,
        "n_dead": n_dead,
        "sparse_decode": bool(T.SAE_SPARSE_DECODE),
        "steps": args.steps,
        "peak_alloc_gb": round(peak_alloc, 3),
        "peak_reserved_gb": round(peak_resv, 3),
        "frag_gb": round(peak_resv - peak_alloc, 3),
        "wall_s": round(wall, 3),
        "tok_per_s": round(args.batch_tokens * args.steps / wall, 1),
        "ms_per_step": round(wall / args.steps * 1000, 2),
        "fwd_ms_per_step": round(agg["fwd_ms"] / args.steps, 2),
        "bwd_ms_per_step": round(agg["bwd_ms"] / args.steps, 2),
        "opt_ms_per_step": round(agg["opt_ms"] / args.steps, 2),
        "sparse_decode_calls": getattr(sae, "_sparse_decode_calls", 0),
        "sparse_decode_fallbacks": getattr(sae, "_sparse_decode_fallbacks", 0),
        # Marks which checkout produced the row, so A/B results are self-labelling.
        # What this row actually ran, not merely what the module exports.
        "aux_path": "sparse" if (args.sparse_aux
                                 and hasattr(T, "_dead_feature_aux_recon")) else "dense",
    }
    if not args.skip_clone:
        result.update(measure_state_clone(sae))

    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps(result) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="unlabeled")
    p.add_argument("--d-in", type=int, default=2304)
    p.add_argument("--expansion", type=int, default=T.EXPANSION)
    p.add_argument("--batch-tokens", type=int, default=T.BATCH_TOKENS)
    p.add_argument("--microbatch-tokens", type=int, default=T.BATCH_TOKENS)
    p.add_argument("--target-l0", type=int, default=T.K)
    p.add_argument("--dead-frac", type=float, default=0.10,
                   help="fraction of features treated as dead, to exercise the aux path")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    p.add_argument("--sparse-aux", action="store_true",
                   help="use the sparse embedding_bag aux helper instead of the "
                        "dense formulation the trainer ships (measured ~21% slower "
                        "on H100; see the comment at the aux call site)")
    p.add_argument("--skip-clone", action="store_true",
                   help="skip the state-dict snapshot cost measurement")
    main_args = p.parse_args()
    run(main_args)


if __name__ == "__main__":
    main()
