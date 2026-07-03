"""Debug helper: print Triton SAE intermediates vs PyTorch reference.

Run on CUDA only.  Tiny SAE so the tensors are small enough to eyeball.
"""
import torch
from sae_trainer_rolling import _make_sae
from triton_sae_kernel import fused_sae_forward


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.manual_seed(0)
    d_in = 16
    n_features = 32
    B = 4

    sae = _make_sae(d_in=d_in, n_features=n_features, seed=0).cuda()
    x = torch.randn(B, d_in, device="cuda", dtype=torch.bfloat16) * 0.5

    # Reference under bf16 autocast (matching trainer / benchmark)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pre_ref = sae.encode_pre(x)
        feat_ref, gate_ref = sae.jumprelu_with_gate(pre_ref)
        xhat_ref = sae.decode(feat_ref)
        l0_ref = gate_ref.sum(dim=-1, dtype=torch.float32)

    print("Reference pre-activations (first 2 tokens, all features):")
    print(pre_ref[:2].float())
    print("Reference gate (first 2 tokens):")
    print(gate_ref[:2].float())
    print("Reference xhat (first 2 tokens):")
    print(xhat_ref[:2].float())
    print("Reference L0:", l0_ref.float())

    # Triton fused
    xhat_tri, l0_tri = fused_sae_forward(x, sae)

    print("\nTriton xhat (first 2 tokens):")
    print(xhat_tri[:2].float())
    print("Triton L0:", l0_tri)

    print("\nDiff xhat (abs, first 2 tokens):")
    print((xhat_tri[:2] - xhat_ref[:2]).abs().float())
    print("Max recon abs err:", (xhat_tri - xhat_ref).abs().max().item())

    # Re-run with a *tiny* x of ones to isolate the matmul.
    print("\n--- ones input sanity ---")
    x1 = torch.ones(1, d_in, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pre1_ref = sae.encode_pre(x1)
        feat1_ref, _ = sae.jumprelu_with_gate(pre1_ref)
        xhat1_ref = sae.decode(feat1_ref)
    xhat1_tri, _ = fused_sae_forward(x1, sae)
    print("ones ref xhat first 5 dims:", xhat1_ref[0, :5].float())
    print("ones tri xhat first 5 dims:", xhat1_tri[0, :5].float())


if __name__ == "__main__":
    main()
