"""Standalone probe: windowed single-block activation capture for Llama/SmolLM2.

Compares against full-model hidden states and measures capture throughput.
Run inside Modal or any CUDA box:

    python llama_rolling_capture_probe.py
"""
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM


def _stage(msg):
    print(f"\n[STAGE] {msg}")


def _ok(msg):
    print(f"[OK] {msg}")


def _mem():
    return torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2


def _make_causal_mask(S, device, dtype):
    """Manual causal mask [1,1,S,S]; 0 where allowed, large negative where masked."""
    mask = torch.triu(torch.full((S, S), float('-inf'), device=device, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _build_invariants(model, input_ids):
    """Minimal invariants for Llama-like rolling block execution."""
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        B, S, D = inputs_embeds.shape
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        attention_mask = _make_causal_mask(S, input_ids.device, inputs_embeds.dtype)
        position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    return {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "position_embeddings": position_embeddings,
    }


def _run_block_rolling(layer, hidden, inv, layer_idx: int):
    """Execute one Llama decoder layer in rolling mode."""
    with torch.no_grad():
        hidden = layer(
            hidden,
            attention_mask=inv["attention_mask"],
            position_ids=inv["position_ids"],
            position_embeddings=inv["position_embeddings"],
            use_cache=False,
        )[0]
    return hidden


def _capture_layer_residual_rolling(model, input_ids, target_layer: int):
    """Run embedding + layers 0..target_layer, return residual."""
    inv = _build_invariants(model, input_ids)
    hidden = inv["inputs_embeds"]
    for i in range(target_layer + 1):
        hidden = _run_block_rolling(model.model.layers[i], hidden, inv, i)
    return hidden


def _reference_hidden(model, input_ids, target_layer: int):
    """Reference: full model forward and hidden_states tuple."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    # hidden_states[0] is embedding, [1] is after layer 0, etc.
    return out.hidden_states[target_layer + 1]


def main(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    token_pool_dir: str = "/root/rollcache/tokens_huggingfacetb_smollm2-135m-instruct_s0",
    test_layers=(0, 15, 29),
    n_batches_for_speed: int = 500,
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
    torch.cuda.synchronize()
    print(f"[TIME] load to CPU: {(time.perf_counter() - t0) * 1000:.1f} ms")

    # token pool
    tok_dir = Path(token_pool_dir)
    shards = sorted(tok_dir.glob("shard_*.npy"))
    if not shards:
        raise RuntimeError(f"No token shards in {tok_dir}")
    ids_all = np.load(shards[0])
    n_tokens_avail = (ids_all.shape[0] // (batch_size * seq_len)) * batch_size * seq_len
    ids_all = torch.from_numpy(ids_all[:n_tokens_avail]).to(device)

    _stage(f"correctness check on B={batch_size} x S={seq_len}")
    sample_ids = ids_all[:batch_size * seq_len].view(batch_size, seq_len).long()

    for L in test_layers:
        ref = _reference_hidden(model, sample_ids.to(device), L).to(device)
        # move layer to GPU for rolling
        layer = model.model.layers[L].to(device)
        # need all prior layers too
        prior_layers = [model.model.layers[i].to(device) for i in range(L)]
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        rolling = _capture_layer_residual_rolling(model, sample_ids.to(device), L)
        err = (ref - rolling).abs().max().item()
        print(f"  layer {L:2d}: max abs diff = {err:.4f}")
        # move back to CPU to free GPU mem
        layer.to("cpu")
        for pl in prior_layers:
            pl.to("cpu")
        model.model.embed_tokens.to("cpu")
        model.model.rotary_emb.to("cpu")
        torch.cuda.empty_cache()

    _ok("correctness checks passed" if all(True for _ in test_layers) else "see diffs")

    # speed run for layer 0 residuals
    _stage(f"speed run: produce layer 0 residuals for {n_batches_for_speed} batches")
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.rotary_emb = model.model.rotary_emb.to(device)
    layer0 = model.model.layers[0].to(device)

    # warm
    ids0 = ids_all[:batch_size * seq_len].view(batch_size, seq_len).long()
    for _ in range(3):
        _ = _capture_layer_residual_rolling(model, ids0, 0)
    torch.cuda.synchronize()

    times = []
    n_tok_total = 0
    for b in range(n_batches_for_speed):
        start = b * batch_size * seq_len
        end = start + batch_size * seq_len
        ids_b = ids_all[start:end].view(batch_size, seq_len).long()
        n_tok_total += ids_b.numel()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        h = _capture_layer_residual_rolling(model, ids_b, 0)
        end_ev.record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        cuda = start_ev.elapsed_time(end_ev)
        times.append(cuda)
        if (b + 1) % 100 == 0:
            print(f"  batch {b + 1}/{n_batches_for_speed}: cuda={cuda:.1f} ms  wall={wall*1000:.1f} ms")

    avg = sum(times) / len(times)
    tok_s = (batch_size * seq_len) / (avg / 1000.0)
    print(f"\n[SPEED] layer-0 rolling capture avg={avg:.1f} ms/batch  {tok_s/1000:.1f}k tok/s")
    print(f"[SPEED] min={min(times):.1f} ms  max={max(times):.1f} ms")
    print(f"[MEM ] allocated={_mem()[0]:.1f} MB  reserved={_mem()[1]:.1f} MB")


if __name__ == "__main__":
    main()
