#!/usr/bin/env python3
"""Copy the Gemma-4 SAE atlas from a model repo to a dataset repo layer by layer."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


LAYER_FILES = ("checkpoint_full.pt", "meta.json", "sae.pt")


def layer_paths(layer: int) -> list[str]:
    stem = f"layer_{layer:02d}_s0"
    return [f"{stem}/{name}" for name in LAYER_FILES]


def remote_sizes(api: HfApi, repo_id: str, repo_type: str, paths: list[str]) -> dict[str, int]:
    return {
        item.path: int(item.size)
        for item in api.get_paths_info(repo_id, paths=paths, repo_type=repo_type)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="juiceb0xc0de/gemma-4-e2b-it-SAE")
    parser.add_argument("--destination", default="juiceb0xc0de/gemma-4-e2b-it-SAE")
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=34)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not 0 <= args.first_layer <= args.last_layer <= 34:
        raise ValueError("layer range must satisfy 0 <= first <= last <= 34")

    api = HfApi(token=token)
    api.repo_info(args.source, repo_type="model")
    api.repo_info(args.destination, repo_type="dataset")

    print(
        f"Migrating {args.source} (model) -> {args.destination} (dataset), "
        f"layers {args.first_layer:02d}-{args.last_layer:02d}",
        flush=True,
    )

    for layer in range(args.first_layer, args.last_layer + 1):
        stem = f"layer_{layer:02d}_s0"
        paths = layer_paths(layer)
        source_sizes = remote_sizes(api, args.source, "model", paths)
        if set(source_sizes) != set(paths):
            missing = sorted(set(paths) - set(source_sizes))
            raise RuntimeError(f"source layer {layer:02d} is incomplete: {missing}")

        destination_sizes = remote_sizes(api, args.destination, "dataset", paths)
        if destination_sizes == source_sizes:
            print(f"[L{layer:02d}] already complete and byte-matched; skipping", flush=True)
            continue

        required = sum(source_sizes.values())
        free = shutil.disk_usage(tempfile.gettempdir()).free
        if free < required + 2_000_000_000:
            raise RuntimeError(
                f"insufficient scratch for L{layer:02d}: need {required} bytes plus 2 GB, "
                f"have {free} bytes"
            )

        with tempfile.TemporaryDirectory(prefix=f"gemma-sae-L{layer:02d}-") as scratch:
            print(
                f"[L{layer:02d}] downloading {required / 1e9:.2f} GB into {scratch}",
                flush=True,
            )
            snapshot_download(
                repo_id=args.source,
                repo_type="model",
                allow_patterns=[f"{stem}/*"],
                local_dir=scratch,
                token=token,
            )

            local_layer = Path(scratch) / stem
            local_sizes = {
                f"{stem}/{name}": (local_layer / name).stat().st_size
                for name in LAYER_FILES
            }
            if local_sizes != source_sizes:
                raise RuntimeError(
                    f"downloaded L{layer:02d} sizes do not match source: "
                    f"source={source_sizes}, local={local_sizes}"
                )

            print(f"[L{layer:02d}] uploading to dataset", flush=True)
            api.upload_folder(
                repo_id=args.destination,
                repo_type="dataset",
                folder_path=local_layer,
                path_in_repo=stem,
                commit_message=f"Copy SAE layer {layer:02d} from model repo",
            )

            uploaded_sizes = remote_sizes(api, args.destination, "dataset", paths)
            if uploaded_sizes != source_sizes:
                raise RuntimeError(
                    f"uploaded L{layer:02d} failed byte verification: "
                    f"source={source_sizes}, destination={uploaded_sizes}"
                )
            print(f"[L{layer:02d}] verified; scratch will now be deleted", flush=True)

        if layer < args.last_layer:
            print(f"[L{layer:02d}] sleeping {args.sleep_seconds} seconds", flush=True)
            time.sleep(args.sleep_seconds)

    print("All requested layers are present and byte-matched in the dataset repo.", flush=True)


if __name__ == "__main__":
    main()
