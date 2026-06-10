"""
Load a trained SAE and run inference on sample text.

Usage:
    python examples/use_trained_sae.py \
        --sae-dir ./data/google_gemma-4-e2b-it/layer_0 \
        --text "The quick brown fox jumps over the lazy dog."

Or programmatically:
    from sae_trainer_rolling import SparseAutoencoder
    sae = SparseAutoencoder.load_from_dir("./data/google_gemma-4-e2b-it/layer_0")
    features = sae.encode(activations)
    reconstruction = sae.decode(features)
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_sae(sae_dir: str):
    """Load encoder and decoder weights from a trained SAE state dict."""
    sae_dir = Path(sae_dir)

    # Load metadata
    with open(sae_dir / "meta.json") as f:
        meta = json.load(f)

    # Load SAE weights saved by the trainer
    state = torch.load(sae_dir / "sae.pt", map_location="cpu", weights_only=True)

    d_in = meta["d_in"]
    n_features = meta["n_features"]
    W_enc = state["W_enc.weight"]
    b_enc = state.get("W_enc.bias", torch.zeros(n_features, dtype=W_enc.dtype))
    W_dec = state["W_dec.weight"]
    b_dec = state.get("b_dec", torch.zeros(d_in, dtype=W_dec.dtype))
    threshold = state.get("log_threshold", torch.zeros(n_features, dtype=W_enc.dtype)).exp()

    print(f"Loaded SAE from {sae_dir}")
    print(f"  d_in: {d_in}, n_features: {n_features}")
    print(f"  threshold: {threshold.mean().item():.4f} (vector)")
    print(f"  training steps: {meta.get('n_steps', 'N/A')}")

    return {
        "W_enc": W_enc,
        "b_enc": b_enc,
        "W_dec": W_dec,
        "b_dec": b_dec,
        "threshold": threshold,
        "meta": meta,
    }


def encode(activations: torch.Tensor, sae: dict) -> torch.Tensor:
    """Encode activations into sparse feature activations."""
    W_enc = sae["W_enc"].to(activations.device)
    b_enc = sae["b_enc"].to(activations.device)
    b_dec = sae["b_dec"].to(activations.device)
    threshold = sae["threshold"].to(activations.device)

    # Pre-activation: subtract decoder bias before encoder linear projection
    pre_act = F.linear(activations - b_dec, W_enc, b_enc)
    feature_acts = F.relu(pre_act - threshold)
    return feature_acts


def decode(feature_acts: torch.Tensor, sae: dict) -> torch.Tensor:
    """Decode sparse features back to activation space."""
    W_dec = sae["W_dec"].to(feature_acts.device)
    b_dec = sae["b_dec"].to(feature_acts.device)

    return F.linear(feature_acts, W_dec, b_dec)


def get_top_features(feature_acts: torch.Tensor, k: int = 10) -> list:
    """Get the top-k firing features and their activation values."""
    # Sum over sequence dimension if present
    if feature_acts.dim() > 2:
        summed = feature_acts.sum(dim=tuple(range(feature_acts.dim() - 1)))
    elif feature_acts.dim() == 2:
        summed = feature_acts.sum(dim=0)
    else:
        summed = feature_acts

    k_actual = min(k, summed.size(0))
    top_vals, top_indices = torch.topk(summed, k=k_actual)

    return [
        {"feature": idx.item(), "activation": val.item()}
        for idx, val in zip(top_indices, top_vals)
    ]


def main():
    parser = argparse.ArgumentParser(description="Load and use a trained SAE")
    parser.add_argument("--sae-dir", required=True, help="Path to trained SAE directory")
    parser.add_argument("--text", default="Hello, world!", help="Input text to encode")
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it", help="Model to get activations from")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to extract activations from")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top features to show")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load the SAE
    sae = load_sae(args.sae_dir)

    # Load model and tokenizer
    print(f"\nLoading model {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    # Tokenize and get activations
    inputs = tokenizer(args.text, return_tensors="pt").to(args.device)
    input_ids = inputs["input_ids"]

    print(f"\nInput text: {args.text}")
    print(f"Tokenized length: {input_ids.shape[1]} tokens")

    # Get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[args.layer + 1]  # +1 because embed is layer 0

    print(f"Activations shape: {hidden_states.shape}")

    # Encode
    feature_acts = encode(hidden_states, sae)

    # Decode / reconstruct
    reconstructed = decode(feature_acts, sae)

    # Compute reconstruction quality
    mse = F.mse_loss(reconstructed, hidden_states)
    variance = hidden_states.var()
    explained_var = 1 - (reconstructed - hidden_states).var() / variance

    print("\n=== Results ===")
    print(f"Reconstruction MSE: {mse.item():.6f}")
    print(f"Explained variance: {explained_var.item():.2%}")

    # Show top features
    top_features = get_top_features(feature_acts, args.top_k)
    print(f"\nTop {args.top_k} firing features:")
    for i, feat in enumerate(top_features):
        print(f"  {i+1}. Feature {feat['feature']:6d}  activation: {feat['activation']:8.2f}")

    # Sparsity stats
    total_elements = feature_acts.numel()
    active_elements = (feature_acts > 0).sum().item()
    sparsity_pct = 100 * (1 - active_elements / total_elements)

    print(f"\nSparsity: {sparsity_pct:.2f}% zero ({active_elements}/{total_elements} active)")


if __name__ == "__main__":
    main()
