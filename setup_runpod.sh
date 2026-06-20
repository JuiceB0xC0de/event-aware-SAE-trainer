#!/bin/bash
# One-time setup for a fresh RunPod pod (runpod/pytorch image) with /workspace mounted.
# Run once after SSHing in:   bash setup_runpod.sh
#
# Creates a persistent venv at /workspace/venv that INHERITS the image's system
# torch (no 2GB re-download) and survives pod stop/start since it lives on the
# mounted volume. After this finishes:
#     source /workspace/venv/bin/activate
#     export HF_TOKEN=hf_...
#     python pretokenize_runpod.py            # builds E2B shards
#     python train_runpod.py --layer-range 0,14 --capture rolling \
#         --target-l0 50 --expansion 32 --microbatch-tokens 32768
set -e

cd /workspace
mkdir -p /workspace/data /workspace/code

# Clone the trainer if not already present
if [ ! -d "/workspace/code/event-aware-SAE-trainer" ]; then
    git clone --depth 1 --branch main https://github.com/JuiceB0xC0de/event-aware-SAE-trainer.git /workspace/code/event-aware-SAE-trainer
fi

cd /workspace/code/event-aware-SAE-trainer

# Persistent venv on the volume, inheriting system-site torch/cuda from the image
if [ ! -d "/workspace/venv" ]; then
    python3 -m venv --system-site-packages /workspace/venv
fi
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

# Upgrade pip and install the package in editable mode into the venv
python -m pip install --upgrade pip
pip install -e .

# Pre-download the E2B tokenizer so nothing streams mid-run (needs HF_TOKEN if gated)
python - <<'PY'
import os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it", token=os.environ.get("HF_TOKEN"))
print("Tokenizer cached.")
PY

echo ""
echo "=================================================================="
echo "Setup complete. Venv: /workspace/venv"
echo "Next (drop in and go):"
echo "  source /workspace/venv/bin/activate"
echo "  export HF_TOKEN=hf_...        # gated E2B + push to your SAE repo"
echo "  export WANDB_API_KEY=...      # optional"
echo "  python pretokenize_runpod.py --model-id google/gemma-4-E2B-it"
echo "  python train_runpod.py --layer-range 0,14 --capture rolling \\"
echo "      --target-l0 50 --expansion 32 --microbatch-tokens 32768"
echo "=================================================================="
