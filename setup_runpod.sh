#!/bin/bash
# One-time setup for a fresh RunPod Ubuntu/Debian pod with /workspace mounted.
# Run: bash setup_runpod.sh
set -e

cd /workspace
mkdir -p /workspace/data /workspace/code

# Clone the trainer if not already present
if [ ! -d "/workspace/code/event-aware-SAE-trainer" ]; then
    git clone --depth 1 --branch main https://github.com/JuiceB0xC0de/event-aware-SAE-trainer.git /workspace/code/event-aware-SAE-trainer
fi

cd /workspace/code/event-aware-SAE-trainer

# Upgrade pip and install the package in editable mode
python -m pip install --upgrade pip
pip install -e .

# Pre-download tokenizer so we don't stream it mid-run
python - <<'PY'
import os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it", token=os.environ.get("HF_TOKEN"))
print("Tokenizer cached.")
PY

echo "Setup complete. Next:"
echo "  python pretokenize_runpod.py"
echo "  python train_runpod.py"
