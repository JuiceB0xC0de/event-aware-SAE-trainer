from pathlib import Path
import tomllib

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_packaging_declares_the_runtime_it_imports():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = "\n".join(pyproject["project"]["dependencies"]).lower()
    modules = set(pyproject["tool"]["setuptools"]["py-modules"])

    assert "pyyaml" in dependencies
    assert "hf_transfer" not in dependencies
    assert {"gemma_attention", "run_atlas", "sae_diagnostics"} <= modules

    requirements = (ROOT / "requirements.txt").read_text().lower()
    assert "pyyaml" in requirements
    assert "hf_transfer" not in requirements
    assert "torch>=2.6.0,<3.0.0" in requirements


def test_gemma_config_matches_the_proven_pool_fusion_setting():
    config = yaml.safe_load((ROOT / "configs" / "gemma4.yaml").read_text())
    assert config["defaults"]["capture"] == "rolling-float"
    assert config["defaults"]["pool_forward_fusion"] == 2
    assert config["hf_repo_type"] == "dataset"


def test_readme_points_at_runnable_and_published_paths():
    readme = (ROOT / "README.md").read_text()

    assert "ReLU(pre - θ)" not in readme
    assert "--model smollm2" not in readme
    assert "--model qwen3" not in readme
    assert "layer_00_s0" in readme
    assert "juiceboxdocks/gemma-4-e2b-it-base:cu128-torch291-py311" in readme
    assert "https://huggingface.co/datasets/juiceb0xc0de/gemma-4-e2b-it-SAE" in readme
