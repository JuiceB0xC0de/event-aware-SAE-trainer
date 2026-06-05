"""Shared pytest fixtures/helpers.

All tests here run on CPU with no network access. Anything that genuinely needs
a GPU is marked with @pytest.mark.gpu and skipped when CUDA is unavailable.
"""
import sys
from pathlib import Path

import pytest

# Make the repo root importable (sae_trainer_rolling, sae_scheduler live there).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a CUDA device")


@pytest.fixture
def tiny_sae():
    """A small JumpReLU SAE (d_in=4, n_features=8) for fast CPU tests."""
    import sae_trainer_rolling as t
    return t._make_sae(d_in=4, n_features=8, seed=0)


def make_dummy_optimizer(lr=2e-4):
    """A real optimizer over one dummy param -- enough for the scheduler, which
    only reads/writes optimizer.param_groups."""
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.Adam([p], lr=lr, betas=(0.9, 0.999))
