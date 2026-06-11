import sys
from pathlib import Path
import json

import torch
import torch.nn.functional as F

# Add examples to sys.path so we can import use_trained_sae
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.use_trained_sae import encode, decode, get_top_features, load_sae

def test_encode():
    d_in = 4
    n_features = 8

    # In examples/use_trained_sae.py: F.linear(activations - b_dec, W_enc, b_enc)
    # F.linear expects W_enc to be [n_features, d_in]
    sae = {
        "W_enc": torch.randn(n_features, d_in),
        "b_enc": torch.randn(n_features),
        "b_dec": torch.randn(d_in),
        "threshold": torch.randn(n_features)
    }

    activations = torch.randn(5, d_in)

    # Replicate the exact logic in the function to test mathematically
    pre_act = F.linear(activations - sae["b_dec"], sae["W_enc"], sae["b_enc"])
    expected = F.relu(pre_act - sae["threshold"])

    result = encode(activations, sae)

    assert result.shape == (5, n_features)
    assert torch.allclose(result, expected, atol=1e-5)

def test_decode():
    d_in = 4
    n_features = 8

    # In examples/use_trained_sae.py: F.linear(feature_acts, W_dec, b_dec)
    # F.linear expects W_dec to be [d_in, n_features]
    sae = {
        "W_dec": torch.randn(d_in, n_features),
        "b_dec": torch.randn(d_in)
    }

    feature_acts = torch.randn(5, n_features)

    expected = F.linear(feature_acts, sae["W_dec"], sae["b_dec"])

    result = decode(feature_acts, sae)

    assert result.shape == (5, d_in)
    assert torch.allclose(result, expected, atol=1e-5)

def test_get_top_features_2d():
    feature_acts = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 1.0, 0.0]
    ])

    result = get_top_features(feature_acts, k=2)
    assert len(result) == 2
    assert result[0] == {"feature": 0, "activation": 5.0}
    assert result[1] == {"feature": 1, "activation": 3.0}

def test_get_top_features_3d():
    feature_acts = torch.tensor([
        [[1.0, 2.0], [0.0, 1.0]],
        [[3.0, 0.0], [0.0, 0.0]]
    ])

    result = get_top_features(feature_acts, k=1)
    assert len(result) == 1
    assert result[0] == {"feature": 0, "activation": 4.0}

def test_get_top_features_1d():
    feature_acts = torch.tensor([1.5, 4.5, 0.5])

    result = get_top_features(feature_acts, k=2)
    assert len(result) == 2
    assert result[0] == {"feature": 1, "activation": 4.5}
    assert result[1] == {"feature": 0, "activation": 1.5}

def test_load_sae(tmp_path):
    sae_dir = tmp_path / "layer_0"
    sae_dir.mkdir()

    meta = {
        "d_in": 4,
        "n_features": 8,
        "n_steps": 100
    }
    with open(sae_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    state = {
        "W_enc.weight": torch.ones(8, 4),
        "W_enc.bias": torch.zeros(8),
        "W_dec.weight": torch.ones(4, 8),
        "b_dec": torch.zeros(4),
        "log_threshold": torch.zeros(8)
    }
    torch.save(state, sae_dir / "sae.pt")

    sae = load_sae(str(sae_dir))

    assert "W_enc" in sae
    assert sae["W_enc"].shape == (8, 4)
    assert "b_enc" in sae
    assert sae["b_enc"].shape == (8,)
    assert "W_dec" in sae
    assert sae["W_dec"].shape == (4, 8)
    assert "b_dec" in sae
    assert sae["b_dec"].shape == (4,)
    assert "threshold" in sae
    assert sae["threshold"].shape == (8,)
    assert "meta" in sae
    assert sae["meta"]["d_in"] == 4
