"""Standalone probe: windowed single-block activation capture for Llama/SmolLM2.

Compares against full-model hidden states and measures capture throughput.
Run inside Modal or any CUDA box:

    python llama_rolling_capture_probe.py
"""
import time
import torch
from transformers import AutoModelForCausalLM


def _stage(msg):
    print(f"\n[STAGE] {msg}")


def _ok(msg):
    print(f"[OK] {msg}")


def _mem():
    return torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2


def _build_invariants(model, input_ids):
    """Build invariants matching LlamaModel.forward exactly.

    Avoids version-sensitive create_causal_mask kwargs by relying on the
    decoder layer's internal mask creation (attention_mask=None is accepted).
    """
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        B, S, D = inputs_embeds.shape
        cache_position = torch.arange(S, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids=position_ids)
    return {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "attention_mask": None,
        "position_embeddings": position_embeddings,
        "cache_position": cache_position,
    }


def _run_block_rolling(layer, hidden, inv, layer_idx: int = -1):
    """Execute one Llama decoder layer in rolling mode.

    Some models return a tuple (hidden_states, ...), others return the tensor
    directly. Do not use [0] blindly: for a batch-1 tensor it squeezes the
    batch dimension.
    """
    with torch.no_grad():
        out = layer(
            hidden,
            attention_mask=inv["attention_mask"],
            position_ids=inv["position_ids"],
            position_embeddings=inv["position_embeddings"],
            use_cache=False,
        )
    return out[0] if isinstance(out, tuple) else out


def _capture_layer_residual_rolling(model, input_ids, target_layer: int):
    """Run embedding + layers 0..target_layer, return residual."""
    inv = _build_invariants(model, input_ids)
    hidden = inv["inputs_embeds"]
    for i in range(target_layer + 1):
        hidden = _run_block_rolling(model.model.layers[i], hidden, inv, layer_idx=i)
    return hidden


def _reference_hidden(model, input_ids, target_layer: int):
    """Reference: full model forward and hidden_states tuple."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    # hidden_states[0] is embedding, [1] is after layer 0, etc.
    return out.hidden_states[target_layer + 1]


def _print_kwargs(msg, kwargs):
    print(f"  {msg} kwargs:")
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        elif isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v):
            print(f"    {k}: tuple of tensors " + ", ".join(f"{tuple(t.shape)}" for t in v))
        else:
            print(f"    {k}: {v}")


def _detach_kwarg(v):
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    if isinstance(v, tuple):
        return tuple(t.detach().clone() if isinstance(t, torch.Tensor) else t for t in v)
    return v


def _hook_replay_correctness(model, input_ids, test_layers):
    """Capture exact layer inputs/kwargs via forward_pre_hook and replay them standalone.

    Reference forward uses use_cache=False so captured kwargs are stateless and
    layer replay should be exactly bit-for-bit identical.
    """
    captured = {}

    def make_hook(idx):
        def hook(module, args, kwargs):
            hidden = args[0].detach().clone() if args else kwargs["hidden_states"].detach().clone()
            captured[idx] = {
                "hidden": hidden,
                "kwargs": {k: _detach_kwarg(v) for k, v in kwargs.items()},
            }

        return hook

    handles = [
        layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    for h in handles:
        h.remove()

    refs = out.hidden_states
    ok = True
    for L in test_layers:
        rec = captured[L]
        hidden = rec["hidden"]
        kwargs = dict(rec["kwargs"])
        kwargs.pop("hidden_states", None)
        kwargs.pop("past_key_values", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        _print_kwargs(f"layer {L} captured", kwargs)
        out = model.model.layers[L](hidden, use_cache=False, **kwargs)
        rolled = out[0] if isinstance(out, tuple) else out
        err = (refs[L + 1] - rolled).abs().max().item()
        print(f"  hook replay layer {L:2d}: max abs diff = {err:.4f}")
        if err > 1e-2:
            ok = False
            print(f"    [FAIL] hook replay layer {L} diff exceeds 1e-2")
    return ok


def main(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    test_layers=(0, 1, 15, 28),
    speed_targets=(0, 1, 15, 28),
    n_batches_for_speed: int = 50,
    batch_size: int = 16,
    seq_len: int = 2048,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    _stage(f"load {model_id}")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    vocab_size = model.config.vocab_size
    torch.cuda.synchronize()
    print(f"[TIME] load to CPU: {(time.perf_counter() - t0) * 1000:.1f} ms")

    _stage("move model to GPU for correctness reference")
    model = model.to(device)
    torch.cuda.synchronize()
    print(f"[MEM after model to GPU] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")

    _stage(f"correctness check on B={batch_size} x S={seq_len}")
    sample_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    _stage("hook replay correctness (uses HF's exact kwargs)")
    hook_ok = _hook_replay_correctness(model, sample_ids, test_layers)
    if hook_ok:
        _ok("hook replay residuals match reference")
    else:
        print("[WARN] hook replay residuals differ; the layer is not deterministic under exact kwargs")

    _stage("manual invariant correctness")
    manual_ok = True
    for L in test_layers:
        ref = _reference_hidden(model, sample_ids, L)
        rolling = _capture_layer_residual_rolling(model, sample_ids, L)
        err = (ref - rolling).abs().max().item()
        print(f"  layer {L:2d}: max abs diff = {err:.4f}")
        if err > 1e-2:
            manual_ok = False
            print(f"    [FAIL] layer {L} diff exceeds 1e-2")

    if manual_ok:
        _ok("manual rolling residuals match reference within tolerance")
    else:
        print("[WARN] manual rolling residuals differ from reference; see diffs above")

    # chain speed run: time each target depth independently (triangular ok for probe)
    _stage(f"chain speed run: timing targets {list(speed_targets)}")
    for target in speed_targets:
        times = []
        for b in range(n_batches_for_speed + 5):  # 5 warmup
            ids_b = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            _ = _capture_layer_residual_rolling(model, ids_b, target)
            end_ev.record()
            torch.cuda.synchronize()
            cuda = start_ev.elapsed_time(end_ev)
            if b >= 5:
                times.append(cuda)
            if (b + 1) % 100 == 0:
                print(f"  target {target:2d} batch {b + 1}/{n_batches_for_speed + 5}: cuda={cuda:.1f} ms")
        avg = sum(times) / len(times)
        tok_s = (batch_size * seq_len) / (avg / 1000.0)
        print(f"[SPEED] target layer {target:2d}: avg={avg:.1f} ms  {tok_s/1000:.1f}k tok/s  min={min(times):.1f} ms  max={max(times):.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")


if __name__ == "__main__":
    main()
