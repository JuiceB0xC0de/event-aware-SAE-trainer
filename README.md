# Event-Aware SAE Trainer

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers%205.x-FFD21E?style=for-the-badge" alt="HuggingFace"></a>
  <a href="https://wandb.ai/"><img src="https://img.shields.io/badge/Weights_%26_Biases-Monitored-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black" alt="W&B"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.style?style=for-the-badge" alt="Code Style"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License"></a>
</p>

---

## 📌 Executive Overview

The **Event-Aware SAE Trainer** is an autonomous, self-tuning training system for **JumpReLU Sparse Autoencoders (SAEs)** across transformer language models. 

Standard SAE training pipelines suffer from severe per-layer hyperparameter sensitivity: a set of learning rates and $L_0$ sparsity coefficients ($\lambda$) that work for layer 4 will collapse or fail to converge at layer 20, forcing tedious manual retuning and restart loops. 

This repository solves that with a **Dual-Loop Adaptive Control System**:
1. **AECS Mode Controller**: A 4-mode finite state machine (`BASELINE`, `RECOVERY`, `EXPLORE`, `STABILIZE`) driven by real-time loss and gradient dynamics.
2. **Augmented-Lagrangian $L_0$ Dual Integrator**: Dynamically controls $L_0$ sparsity as a hard mathematical constraint ($L_0 \le K$) using projected dual ascent.

The result is a **model-agnostic, unattended layer-by-layer sweep**: you set an $L_0$ target, and the trainer sweeps all decoder layers in a single execution without human intervention.

---

## 📐 Architecture & Core Innovations

```
                       ┌────────────────────────────────────────┐
                       │       Language Model / Activation      │
                       │             Stream Provider            │
                       └───────────────────┬────────────────────┘
                                           │ Activations (x)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           JumpReLU Sparse Autoencoder                          │
│                                                                                │
│    Encoder:  pre = (x - b_dec) @ W_enc + b_enc                                 │
│    Gating:   f = ReLU(pre - θ)               [STE gradient via band]       │
│    Decoder:  x_hat = f @ W_dec + b_dec         [Normalized ||W_dec||_2 = 1]     │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
┌──────────────────────────┐             ┌──────────────────────────┐
│  AECS Mode Controller    │             │   Augmented-Lagrangian   │
│  (LR & Optimization)     │             │     Sparsity Controller  │
├──────────────────────────┤             ├──────────────────────────┤
│ • BASELINE               │             │ • Dual ascent: λ_(t+1)   │
│ • RECOVERY (Loss Spike)  │             │ • Targets L0 <= K        │
│ • EXPLORE  (Plateaus)    │             │ • EV Floor Protection    │
│ • STABILIZE (Dead Resamp)│             │ • Emergency Dead Reset   │
└──────────────────────────┘             └──────────────────────────┘
```

### 1. Augmented-Lagrangian $L_0$ Dual Integrator
Instead of treating $\lambda$ as a static hyperparameter, sparsity is formulated as an inequality constraint:
$$\min_{\theta} \mathcal{L}_{\text{recon}} \quad \text{subject to} \quad L_0(\theta) \le K$$

The dual integrator updates $\lambda$ on every step via projected dual ascent:
$$\lambda_{t+1} = \max\left(0, \lambda_t + \alpha \cdot (\overline{L}_0 - K)\right)$$
Combined with a quadratic penalty parameter $\mu$, $\lambda$ builds up exact steady-state pressure near the target $K$ while preventing overshooting or dead feature traps.

### 2. Event-Aware AECS Scheduler (`sae_scheduler.py`)
Combines loss-variance tracking, gradient norm monitoring, and z-score anomaly detection to transition automatically between four operational modes:
- **`BASELINE`**: Nominal AdamW optimization.
- **`RECOVERY`**: Triggered on loss spikes or EV drops; throttles learning rate and clips gradients to restore numerical stability.
- **`EXPLORE`**: Inject noise and boost LR during loss plateaus to break out of suboptimal local minima.
- **`STABILIZE`**: Activated during dead-feature resampling or emergency dead-neuron resets.

### 3. Pluggable Activation Capture Backends
Activation extraction is completely decoupled from trainer logic (`--capture` options):
| Capture Mode | Target Model / Compatibility | Compute & Memory Profile |
| :--- | :--- | :--- |
| **`auto`** *(Default)* | Any `AutoModelForCausalLM` | Forward-hook on residual stream. 1x disk pool, $N \times$ compute. |
| **`rolling`** | Gemma-3n / Gemma-4 | Single-block walk. Fast path for Gemma-specific invariants. |
| **`rolling-float`** | Gemma-3n / Gemma-4 (CPU/GPU hybrid) | Keeps full model on CPU; hoists active blocks to GPU. |
| **`rolling-hf`** | Llama 3.2, SmolLM2, Qwen3 | Generic single-block walk using HF `rotary_emb` & position embeddings. |
| **`rolling-hf-float`**| Llama 3.2, SmolLM2, Qwen3 (CPU offload) | Single-block walk with CPU offloading for large parameter models. |

### 4. Quasi-Orthogonality Diagnostics (`sae_diagnostics.py`)
Tracks $||z||_2 / ||x||_2$ ratios and off-diagonal Gram mass gaps ($| ||z||_2 - ||\hat{x}||_2 | / ||\hat{x}||_2$). Detects dictionary degeneracy, feature redundancy, and collapse risk *before* reconstruction EV degrades.

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.6+ with CUDA 12.4 support
- CUDA-capable GPU (NVIDIA H100 / A100 recommended for large runs)

### Setup
```bash
# Clone repository
git clone https://github.com/JuiceB0xC0de/event-aware-SAE-trainer.git
cd event-aware-SAE-trainer

# Install core dependencies
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124

# Install development dependencies (for testing & linting)
pip install -r requirements-dev.txt
```

---

## 💻 CLI Usage

### Direct Trainer Execution (`sae_trainer_rolling.py`)

#### 1. Llama-3.2-1B Layer Sweep
```bash
python sae_trainer_rolling.py \
  --model-id meta-llama/Llama-3.2-1B \
  --capture rolling-hf \
  --start-layer 0 \
  --end-layer 16 \
  --target-l0 32
```

#### 2. Gemma-4 E2B Fast Path with Hugging Face Hub Upload
```bash
export HF_TOKEN="your_hf_token_here"
export WANDB_API_KEY="your_wandb_key_here"

python sae_trainer_rolling.py \
  --model-id google/gemma-4-E2B-it \
  --capture rolling \
  --hub-id your-org/gemma-4-saes \
  --wandb-project gemma4-sae-sweep
```

---

### Config-Driven Atlas Runner (`run_atlas.py`)

For multi-model family sweeps, use `run_atlas.py` driven by YAML configurations in `configs/`:

```bash
# List available model family configs
python run_atlas.py --list

# Run SmolLM2 configuration
python run_atlas.py --model smollm2

# Override target L0 and max steps on the fly
python run_atlas.py --model qwen3 --target-l0 64 --max-steps 8000

# Custom YAML file
python run_atlas.py --config ./configs/custom_experiment.yaml
```

---

## 🐍 Python API & Programmatic Usage

You can load and query trained SAE checkpoints programmatically in Python:

```python
import torch
from examples.use_trained_sae import load_sae, encode, decode, get_top_features

# 1. Load trained SAE directory (sae.pt + meta.json)
sae = load_sae("./data/saes/google_gemma-4-e2b-it/layer_0")

# 2. Encode activations into sparse feature activations z
# activations shape: [batch_size, sequence_length, d_in]
activations = torch.randn(2, 16, sae["meta"]["d_in"])
features = encode(activations, sae)

# 3. Decode sparse features back to embedding space
reconstructed = decode(features, sae)

# 4. Extract top-firing features
top_features = get_top_features(features, k=5)
for feat in top_features:
    print(f"Feature ID: {feat['feature']} | Activation Sum: {feat['activation']:.4f}")
```

---

## 📂 Repository Structure

```
event-aware-SAE-trainer/
├── sae_trainer_rolling.py  # Core JumpReLU SAE architecture & training loop
├── sae_scheduler.py        # Dual-Loop AECS & Augmented Lagrangian L0 scheduler
├── run_atlas.py            # Config-driven multi-model family Atlas runner
├── sae_diagnostics.py      # Quasi-orthogonality & dictionary geometry metrics
├── configs/                # Model family YAML configurations (SmolLM2, Qwen3, etc.)
├── examples/               # Usage examples & programmatic API helpers
│   ├── use_trained_sae.py  # Programmatic inference & feature extraction
│   └── configs.py          # Example Python configuration dictionaries
├── tests/                  # Comprehensive pytest test suite
│   ├── test_sae_model.py   # SAE architecture & parameter tests
│   ├── test_sparse_decode.py # High-performance sparse decode GEMM tests
│   ├── test_scheduler.py   # AECS dual-loop state machine & integrator tests
│   └── test_smoke_gpu.py   # CUDA smoke tests
├── pyproject.toml          # Package build & tool configs
├── requirements.txt        # Core requirements (PyTorch, Transformers 5.x)
└── pytest.ini              # Pytest configuration
```

---

## ⚙️ Environment Variables

The trainer respects standard environment variables for seamless deployment:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SAE_DATA_DIR` | Directory for persistent output data & trained SAE checkpoints | `./data` |
| `SAE_SCRATCH_DIR` | Fast local scratch space for activation pools | `$SAE_DATA_DIR/rollcache` |
| `SAE_MODEL_ID` | Default Hugging Face model ID | `google/gemma-4-E2B-it` |
| `SAE_HUB_ID` | Hugging Face Hub upload target (`org/repo`) | *None (disabled)* |
| `WANDB_PROJECT` | Weights & Biases project name | *None (disabled)* |
| `HF_TOKEN` | Hugging Face authentication token | Read from `hf auth login` |

---

## 🧪 Testing

Run the full unit test suite using `pytest`:

```bash
# Run all CPU tests
pytest

# Run GPU smoke tests (requires CUDA)
pytest -m gpu

# Run tests with coverage report
pytest --cov=.
```

---

## 📄 License & Citation

This project is licensed under the **MIT License**.

If you use this trainer or the AECS scheduler in your research, please cite:

```bibtex
@software{event_aware_sae_trainer2026,
  author = {Rick Holmberg},
  title = {Event-Aware SAE Trainer: Autonomous Dual-Loop Adaptive Control for JumpReLU SAEs},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/JuiceB0xC0de/event-aware-SAE-trainer}
}
```

