import sys
import types

import run_atlas


def test_push_layer_uses_explicit_dataset_repo_type(tmp_path, monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, **kwargs):
            calls.append(("api", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("summary", kwargs))

    fake_hub = types.SimpleNamespace(
        HfApi=FakeApi,
        create_repo=lambda *args, **kwargs: calls.append(("create", kwargs)),
        upload_folder=lambda *args, **kwargs: calls.append(("upload", kwargs)),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(run_atlas, "resolve_hf_token", lambda: "test-token")

    (tmp_path / "layer_00_s0").mkdir()
    assert run_atlas.push_layer(
        tmp_path, "juiceb0xc0de/gemma-4-e2b-it-SAE", 0, repo_type="dataset"
    )

    assert calls[0][1]["repo_type"] == "dataset"
    assert calls[1][1]["repo_type"] == "dataset"

    run_atlas.push_summary(
        tmp_path,
        "juiceb0xc0de/gemma-4-e2b-it-SAE",
        {"ok": True},
        repo_type="dataset",
    )
    summary_call = next(kwargs for name, kwargs in calls if name == "summary")
    assert summary_call["repo_type"] == "dataset"
