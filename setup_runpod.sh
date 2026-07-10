#!/bin/bash
# One-time setup for a fresh RunPod pod with /workspace mounted.
# Run once after SSHing in:   bash setup_runpod.sh
#
# Creates an isolated persistent venv at /workspace/venv (NO --system-site-packages).
# RunPod base images often ship torchaudio/torchvision binaries that are ABI-incompatible
# with newer torch/transformers; an isolated venv prevents those broken system packages
# from leaking in. The venv survives pod stop/start since it lives on /workspace.
#
# After this finishes:
#     source /workspace/venv/bin/activate
#     export HF_TOKEN=hf_...
#     export WANDB_API_KEY=...      # optional
#     python pretokenize_runpod.py --model-id HuggingFaceTB/SmolLM2-135M-Instruct
#     python train_runpod.py --layer-range 0,14 --capture rolling \
#         --target-l0 50 --expansion 32 --microbatch-tokens 32768
set -e

MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolLM2-135M-Instruct}"

cd /workspace
mkdir -p /workspace/data /workspace/code

# Clone the trainer if not already present
if [ ! -d "/workspace/code/event-aware-SAE-trainer" ]; then
    git clone --depth 1 --branch main https://github.com/JuiceB0xC0de/event-aware-SAE-trainer.git /workspace/code/event-aware-SAE-trainer
fi

cd /workspace/code/event-aware-SAE-trainer

# Persistent isolated venv on the volume. We do NOT inherit system-site packages
# because pod images ship torchaudio/torchvision wheels built against a different
# torch ABI, which causes `undefined symbol` import errors with transformers 5.x.
if [ ! -d "/workspace/venv" ]; then
    python3 -m venv /workspace/venv
fi
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

# Upgrade pip and install an ABI-matched CUDA 12.4 PyTorch stack first.
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install the trainer + its dependencies (transformers 5.x, datasets, wandb, etc.)
pip install -e .

# Pre-download the target tokenizer so nothing streams mid-run.
python - <<'PY'
import os
from transformers import AutoTokenizer
model_id = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolLM2-135M-Instruct")
tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
print(f"Tokenizer cached for {model_id}.")
PY

echo ""
echo "=================================================================="
echo "Setup complete. Venv: /workspace/venv"
echo "Next (drop in and go):"
echo "  source /workspace/venv/bin/activate"
echo "  export HF_TOKEN=hf_...        # gated models + push to your SAE repo"
echo "  export WANDB_API_KEY=...      # optional"
echo "  python pretokenize_runpod.py --model-id ${MODEL_ID}"
echo "  python train_runpod.py --layer-range 0,14 --capture rolling \\"
echo "      --target-l0 50 --expansion 32 --microbatch-tokens 32768"
echo "=================================================================="
