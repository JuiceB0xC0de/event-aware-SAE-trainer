"""Sanity checks on module import and headline constants."""
import sae_trainer_rolling as t


def test_module_imports_without_gpu_or_network():
    # importing must not require torch CUDA, transformers, datasets, or HF tokens
    assert t.MODEL_ID == "google/gemma-4-E2B-it"


def test_expansion_and_targets():
    assert t.EXPANSION == 32
    assert t.N_FEATURES == t.EXPANSION * t.D_IN == 49152
    assert t.D_IN == 1536
    assert t.K == 500
    assert t.BATCH_TOKENS == 32_768
    assert t.SEQ_LEN == 2_048


def test_rolling_hard_stop_only_relevant_to_gemma_path():
    assert t.HARD_STOP_LAYER == 15


def test_no_org_pii_defaults():
    # nothing pushes/logs anywhere by default; no hardcoded org in the source
    assert t.SAE_HUB_ID is None
    assert t.WANDB_PROJECT is None


def test_slug():
    assert t._slug("google/gemma-4-E2B-it") == "google_gemma-4-e2b-it"
    assert t._slug("meta-llama/Llama-3.2-1B") == "meta-llama_llama-3.2-1b"


def test_sae_dir_composes_under_data_dir():
    assert str(t.DATA_DIR) in t.SAE_DIR
    # output dir is derived from the model id slug, not a hardcoded name
    assert t.SAE_DIR.replace("\\", "/").endswith("saes/" + t._slug(t.MODEL_ID))


def test_data_dir_is_configurable_via_env(tmp_path):
    # SAE_DATA_DIR is read at import time. Verify the wiring in a fresh interpreter
    # so we don't mutate the module already imported by the rest of the suite.
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ, SAE_DATA_DIR=str(tmp_path))
    code = "import sae_trainer_rolling as t; print(t.SAE_DIR)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert str(tmp_path) in out
