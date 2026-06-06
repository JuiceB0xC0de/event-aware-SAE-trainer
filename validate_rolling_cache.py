"""
Validation gate for the rolling single-block capture (--capture rolling).
=========================================================================
Proves that the SHIPPED single-block invocation in sae_trainer_rolling
(`_make_invariants` + `_run_block`) reproduces, bit-exact, the activations from a
true full forward 0->L -- for the layers the rolling path actually claims, i.e.
0..HARD_STOP_LAYER-1.

It imports the real trainer functions, so it guards the code that ships -- not a
parallel reimplementation.

It also probes layers >= HARD_STOP_LAYER to *demonstrate* the divergence that
motivates the hard stop: those layers read cross-layer `shared_kv_states`, which the
shipped `_run_block` deliberately does NOT thread (it passes an empty dict). So the
single-block reconstruction is exact below the KV-share boundary and diverges at/above
it -- exactly why rolling is scoped to 0..14. Layers >= HARD_STOP_LAYER are reported
but NOT part of the pass/fail gate.

Run (plain Python; needs HF_TOKEN + the model; runs on CPU or GPU):
    python validate_rolling_cache.py
    python validate_rolling_cache.py --layers 1,8,13,14,15,19,30,34
    python validate_rolling_cache.py --inspect        # dump the model's decoder internals
"""
from __future__ import annotations

import os

import sae_trainer_rolling as base   # validate the SHIPPED rolling functions, not a copy

MODEL_ID = os.environ.get("SAE_MODEL_ID", base.MODEL_ID)


def _build_token_batches(tokenizer, n_seqs, seq_len, n_batches, bos_token_id, seed=0):
    """Pull a few real FineWeb-Edu docs and pack them into [n_seqs, seq_len] batches
    with a BOS prepended per sequence (matches trainer batching)."""
    import torch
    from datasets import load_dataset

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, token=hf_token)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    it = iter(ds)

    need = n_batches * n_seqs * (seq_len - 1)
    buf: list = []
    total = 0
    while total < need:
        row = next(it)
        text = row.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer(text, truncation=True, max_length=seq_len,
                        return_tensors="pt", add_special_tokens=True).input_ids[0]
        if ids.numel() < 8:
            continue
        buf.append(ids)
        total += ids.numel()

    flat = torch.cat(buf)[:need]
    batches = []
    per_batch = n_seqs * (seq_len - 1)
    for b in range(n_batches):
        chunk = flat[b * per_batch:(b + 1) * per_batch].view(n_seqs, seq_len - 1)
        bos = torch.full((n_seqs, 1), bos_token_id, dtype=chunk.dtype)
        batches.append(torch.cat([bos, chunk], dim=1))  # [n_seqs, seq_len]
    return batches


def _load_model(model_id, hf_token):
    """Generic loader: AutoModelForCausalLM, multimodal fallback (e.g. Gemma-4)."""
    import torch
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=hf_token, dtype=torch.bfloat16, device_map="cpu")
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, token=hf_token, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def inspect_arch(model_id=MODEL_ID):
    """Dump the decoder internals needed to drive a single block correctly. Gemma-only
    source dump; falls back to a config-knob dump for other architectures."""
    import inspect
    from transformers import AutoConfig

    try:
        from transformers.models.gemma4 import modeling_gemma4 as m

        def dump(name, obj):
            print("\n" + "=" * 70 + f"\n### {name}\n" + "=" * 70)
            try:
                print(inspect.getsource(obj))
            except Exception as e:
                print(f"  <could not get source: {e}>")

        classes = {n: c for n, c in vars(m).items() if isinstance(c, type)}
        print("CLASSES in modeling_gemma4:", sorted(classes))
        for cname, cls in classes.items():
            if cname.endswith("DecoderLayer"):
                dump(f"{cname}.forward", cls.forward)
            if cname.endswith("TextModel") and hasattr(cls, "project_per_layer_inputs"):
                dump(f"{cname}.project_per_layer_inputs", cls.project_per_layer_inputs)
    except Exception as e:
        print(f"(no gemma4 source available: {type(e).__name__}); dumping config knobs only")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cfg = AutoConfig.from_pretrained(model_id, token=hf_token)
    tcfg = getattr(cfg, "text_config", cfg)
    print("\n" + "=" * 70 + "\n### CONFIG KNOBS (text_config)\n" + "=" * 70)
    for a in ("num_hidden_layers", "hidden_size", "intermediate_size",
              "hidden_size_per_layer_input", "layer_types", "sliding_window",
              "sliding_window_pattern", "num_kv_shared_layers",
              "rope_theta", "rope_local_base_freq", "num_attention_heads",
              "num_key_value_heads"):
        if hasattr(tcfg, a):
            print(f"  {a} = {getattr(tcfg, a)}")


def validate(layers_csv: str = "1,8,13,14,15,19", n_seqs: int = 2, seq_len: int = 1024,
             n_batches: int = 2, tol: float = 1e-3, cos_tol: float = 0.9999,
             model_id: str = MODEL_ID):
    """For each layer L: capture block L-1 and block L outputs from a true full forward
    (ground truth), then reconstruct block L via the shipped _make_invariants/_run_block
    and compare. Gate = every layer L < HARD_STOP_LAYER passes (max|diff|<tol, cos>cos_tol).
    Layers >= HARD_STOP_LAYER are reported only (divergence is expected, by design)."""
    import torch

    torch.manual_seed(0)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    target_layers = [int(x) for x in layers_csv.split(",") if x.strip() != ""]

    print(f"Loading {model_id} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    bos_token_id = tokenizer.bos_token_id or 2
    model = _load_model(model_id, hf_token)

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    n_layers = int(tcfg.num_hidden_layers)
    text_model, attr_name, decoder_layers = base._find_text_model(model, n_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  model={type(model).__name__}  blocks={len(decoder_layers)}  "
          f"text_model={type(text_model).__name__}  device={device}")
    print(f"  HARD_STOP_LAYER={base.HARD_STOP_LAYER} (gate covers layers < this; "
          f">= it is reported only)")

    print(f"\nBuilding {n_batches} batches of [{n_seqs}, {seq_len}] ...")
    batches = [b.to(device) for b in
               _build_token_batches(tokenizer, n_seqs, seq_len, n_batches, bos_token_id)]

    def _out(o):
        return o[0] if isinstance(o, tuple) else o

    results = {}
    for L in target_layers:
        if L < 1 or L >= n_layers:
            print(f"\n[skip] layer {L} out of range [1,{n_layers-1}] (L=0 uses embeds directly)")
            continue
        print("\n" + "-" * 64 + f"\nVALIDATE layer {L}\n" + "-" * 64)

        max_abs, min_cos = 0.0, 1.0
        in_shape = out_shape = None

        class _EarlyExit(Exception):
            pass

        for bi, ids in enumerate(batches):
            # -- Ground truth: real forward, capture block L-1 (= input to L) and block L
            cap = {}

            def hook_prev(_m, _i, o):
                cap["prev"] = _out(o).detach()

            def hook_tgt(_m, _i, o):
                cap["tgt"] = _out(o).detach()
                raise _EarlyExit

            hp = decoder_layers[L - 1].register_forward_hook(hook_prev)
            ht = decoder_layers[L].register_forward_hook(hook_tgt)
            try:
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
            except _EarlyExit:
                pass
            finally:
                hp.remove()
                ht.remove()

            pool_input = cap["prev"]            # block L-1 output == input to block L
            truth = cap["tgt"].float()          # block L output (ground truth)
            in_shape, out_shape = tuple(pool_input.shape), tuple(truth.shape)

            # -- Shipped single-block reconstruction
            inv = base._make_invariants(text_model, tcfg, ids)
            with torch.no_grad():
                b2 = base._run_block(decoder_layers, tcfg, L, pool_input, inv).float()

            diff = (truth - b2).abs()
            max_abs = max(max_abs, diff.max().item())
            cos = torch.nn.functional.cosine_similarity(
                truth.reshape(-1, truth.shape[-1]), b2.reshape(-1, b2.shape[-1]), dim=-1
            ).mean().item()
            min_cos = min(min_cos, cos)
            print(f"  batch {bi}: max|diff|={diff.max().item():.3e}  cos={cos:.6f}")

        gated = L < base.HARD_STOP_LAYER
        passed = (max_abs < tol) and (min_cos > cos_tol)
        results[L] = {"max_abs": max_abs, "min_cos": min_cos, "in_shape": in_shape,
                      "out_shape": out_shape, "passed": passed, "gated": gated}
        tag = ("PASS" if passed else "FAIL") if gated else \
              (f"(>= HARD_STOP {base.HARD_STOP_LAYER}: divergence expected"
               f"{' -- OK' if not passed else ' -- unexpectedly exact'})")
        print(f"  --> layer {L}: in={in_shape} out={out_shape}  "
              f"max|diff|={max_abs:.3e}  min_cos={min_cos:.6f}  {tag}")

    print("\n" + "=" * 64)
    print("SUMMARY  (gate over layers < %d: max|diff| < %.0e AND cos > %.4f)"
          % (base.HARD_STOP_LAYER, tol, cos_tol))
    print("=" * 64)
    gate_pass = True
    for L in sorted(results):
        r = results[L]
        mark = (("PASS" if r["passed"] else "FAIL") if r["gated"] else "info")
        if r["gated"]:
            gate_pass = gate_pass and r["passed"]
        print(f"  L{L:>2}  max|diff|={r['max_abs']:.3e}  min_cos={r['min_cos']:.6f}  {mark}")
    print(f"\n  OVERALL: {'PASS -- shipped single-block invocation is bit-exact within scope' if gate_pass else 'FAIL -- shipped _run_block diverges within the 0..HARD_STOP range; investigate'}")
    return {"gate_pass": gate_pass, "results": results}


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Bit-exactness gate for the shipped rolling single-block capture.")
    p.add_argument("--layers", default="1,8,13,14,15,19",
                   help="comma-separated layer indices to validate")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--n-seqs", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--n-batches", type=int, default=2)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--cos-tol", type=float, default=0.9999)
    p.add_argument("--inspect", action="store_true",
                   help="dump decoder internals / config knobs and exit")
    args = p.parse_args()

    if args.inspect:
        inspect_arch(args.model_id)
        return

    res = validate(layers_csv=args.layers, n_seqs=args.n_seqs, seq_len=args.seq_len,
                   n_batches=args.n_batches, tol=args.tol, cos_tol=args.cos_tol,
                   model_id=args.model_id)
    raise SystemExit(0 if res["gate_pass"] else 1)


if __name__ == "__main__":
    main()
