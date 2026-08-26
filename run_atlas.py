#!/usr/bin/env python3
"""Config-driven SAE atlas runner.

Replaces the per-model run_*_runpod.py scripts. One runner, one YAML per
model family:

    python run_atlas.py --model smollm2
    python run_atlas.py --model qwen3 --model-id Qwen/Qwen3-8B
    python run_atlas.py --model minicpm5 --target-l0 100 --max-steps 8000
    python run_atlas.py --config ./my-experiment.yaml
    python run_atlas.py --list

Precedence is CLI > config file > trainer defaults, so anything in a config
can be overridden ad hoc without editing the file.

The core trainer (sae_trainer_rolling.py / sae_scheduler.py) is untouched;
this only assembles arguments and handles upload.

Environment:
    HF_TOKEN        required unless --no-push
    SAE_HUB_ID      full "org/name" upload target (highest precedence)
    SAE_HUB_REPO_TYPE  upload target kind: model or dataset
    SAE_HUB_ORG     org to derive the upload target under, e.g. "myorg"
    SAE_DATA_DIR    output root, defaults to ./data
    SAE_SCRATCH_DIR activation pool scratch, defaults to $SAE_DATA_DIR/rollcache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs"

# Capture backends, mirrored from sae_trainer_rolling.main().
CAPTURE_CHOICES = ["auto", "rolling", "rolling-float", "rolling-hf", "rolling-hf-float"]


def resolve_hf_token():
    """Explicit env vars first, then the token stored by `hf auth login`.

    Being logged in through the CLI is the normal case; requiring HF_TOKEN in the
    environment on top of that is a papercut, not a security boundary.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


# --------------------------------------------------------------------------
# config loading
# --------------------------------------------------------------------------
def _available_configs():
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def load_config(model: str | None, config_path: str | None) -> tuple[dict, Path]:
    """Resolve --model <name> or --config <path> to a parsed config dict."""
    try:
        import yaml
    except ImportError:  # pragma: no cover
        raise SystemExit("pyyaml is required: pip install pyyaml")

    if config_path:
        path = Path(config_path).expanduser()
    else:
        path = CONFIG_DIR / f"{model}.yaml"

    if not path.is_file():
        avail = ", ".join(_available_configs()) or "(none found)"
        raise SystemExit(f"config not found: {path}\navailable --model names: {avail}")

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if not cfg.get("model_id"):
        raise SystemExit(f"{path}: 'model_id' is required")
    return cfg, path


def _slug(model_id: str) -> str:
    """Match sae_trainer_rolling._slug so output paths line up."""
    return model_id.replace("/", "_").lower()


def resolve_n_layers(model_id: str, pinned, trust_remote_code: bool = False) -> int:
    """Layer count from the config if pinned, else the model's own HF config.

    Auto-detection is what makes one config cover a whole model family.
    """
    if pinned:
        return int(pinned)

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    for obj in (cfg, getattr(cfg, "text_config", None)):
        n = getattr(obj, "num_hidden_layers", None) if obj is not None else None
        if n:
            return int(n)
    raise SystemExit(
        f"could not read num_hidden_layers for {model_id}; "
        f"pin architecture.n_layers in the config or pass --layer-range"
    )


def resolve_hub_id(cfg_repo, model_id: str, push: bool) -> str | None:
    """Upload target. Explicit beats derived; never assume someone's namespace."""
    hub = os.environ.get("SAE_HUB_ID") or cfg_repo
    if hub:
        return hub

    org = os.environ.get("SAE_HUB_ORG")
    if org:
        name = model_id.split("/")[-1].lower()
        return f"{org}/{name}-sae"

    if push:
        raise SystemExit(
            "no upload target. Either set hf_sae_repo in the config, export\n"
            "SAE_HUB_ID=org/name (or SAE_HUB_ORG=org to derive one), pass\n"
            "--hf-sae-repo, or run with --no-push."
        )
    return None


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------
def push_layer(sae_dir: Path, hub_id: str, layer: int, seed: int = 0,
               repo_type: str = "model") -> bool:
    from huggingface_hub import HfApi, create_repo, upload_folder

    token = resolve_hf_token()
    if not token:
        print(f"[HF push L{layer:02d}] no HF credentials (env or `hf auth login`), skipping")
        return False

    layer_dir = sae_dir / f"layer_{layer:02d}_s{seed}"
    if not layer_dir.exists():
        print(f"[HF push L{layer:02d}] {layer_dir} not found, skipping")
        return False

    try:
        create_repo(hub_id, repo_type=repo_type, private=False, exist_ok=True, token=token)
    except Exception as e:
        print(f"  [HF push L{layer:02d}] repo creation note: {e}")

    upload_folder(
        folder_path=str(layer_dir),
        repo_id=hub_id,
        repo_type=repo_type,
        path_in_repo=f"layer_{layer:02d}_s{seed}",
        token=token,
    )
    print(f"[HF push L{layer:02d}] uploaded -> {hub_id}/tree/main/layer_{layer:02d}_s{seed}")
    return True


def push_summary(sae_dir: Path, hub_id: str, summary: dict,
                 repo_type: str = "model"):
    from huggingface_hub import HfApi

    sae_dir.mkdir(parents=True, exist_ok=True)
    path = sae_dir / "run_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    HfApi(token=resolve_hf_token()).upload_file(
        path_or_fileobj=str(path),
        path_in_repo="run_summary.json",
        repo_id=hub_id,
        repo_type=repo_type,
    )
    print("[HF] uploaded run_summary.json")


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Config-driven SAE atlas runner. CLI overrides the config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("config source")
    src.add_argument("--model", type=str, help=f"config name from configs/ ({', '.join(_available_configs()) or 'none'})")
    src.add_argument("--config", type=str, help="path to an arbitrary config YAML")
    src.add_argument("--list", action="store_true", help="list available model configs and exit")

    ov = p.add_argument_group("overrides (default: from config)")
    ov.add_argument("--model-id", type=str, default=None)
    ov.add_argument("--hf-sae-repo", type=str, default=None)
    ov.add_argument("--hf-repo-type", choices=["model", "dataset"], default=None,
                    help="Hugging Face repository kind; default from config or model")
    ov.add_argument("--wandb-project", type=str, default=None)
    ov.add_argument("--layer-range", type=str, default=None, help="inclusive 'start,end'; default 0,n_layers-1")
    ov.add_argument("--capture", type=str, default=None, choices=CAPTURE_CHOICES)
    ov.add_argument("--expansion", type=int, default=None)
    ov.add_argument("--pool-batches", type=int, default=None)
    ov.add_argument("--pool-forward-fusion", type=int, default=None,
                    help="pool-production shards fused per model forward; default 1")
    ov.add_argument("--max-steps", type=int, default=None)
    ov.add_argument("--target-l0", type=int, default=None)
    ov.add_argument("--microbatch-tokens", type=int, default=None)
    ov.add_argument("--timing-every", type=int, default=None,
                    help="[STEP-TIME] print cadence in steps; 0 disables. Overrides config env and $SAE_TIMING.")
    ov.add_argument("--seed", type=int, default=0)
    ov.add_argument("--bdec-batches", type=int, default=50)
    ov.add_argument("--resume-from", type=str, default=None)
    ov.add_argument("--pool-retention", type=int, default=None)
    ov.add_argument("--corpus", type=str, default=None)
    ov.add_argument("--corpus-text-field", type=str, default=None)
    ov.add_argument("--corpus-prefix", type=str, default=None)
    ov.add_argument("--norm-ref", type=float, default=None,
                    help="pin activation_norm_ref (L0 probe norm) when retraining "
                         "a mid-chain layer in a fresh process")

    fl = p.add_argument_group("flags")
    fl.add_argument("--trust-remote-code", action="store_true")
    fl.add_argument("--no-pretok", action="store_true")
    fl.add_argument("--no-push", action="store_true")
    fl.add_argument("--no-model-evict", action="store_true")
    fl.add_argument("--cpu", action="store_true", help="force CPU training (debug only, very slow)")
    fl.add_argument("--dry-run", action="store_true",
                    help="resolve config and print the plan without training")
    return p


def main():
    args = build_parser().parse_args()

    if args.list:
        print("available model configs:")
        for name in _available_configs():
            print(f"  {name}")
        return

    if not args.model and not args.config:
        raise SystemExit("one of --model or --config is required (see --list)")

    cfg, cfg_path = load_config(args.model, args.config)
    defaults = cfg.get("defaults") or {}
    arch = cfg.get("architecture") or {}

    def pick(cli_value, key, fallback=None):
        """CLI > config > fallback."""
        if cli_value is not None:
            return cli_value
        v = defaults.get(key)
        return fallback if v is None else v

    model_id = args.model_id or cfg["model_id"]
    # Gemma-4 on Transformers 5.5.4 needs the kernel package's compatibility
    # patch before AutoConfig imports the Gemma configuration module.
    from gemma_attention import prepare_gemma4_attention
    prepare_gemma4_attention(model_id)
    push = not args.no_push
    hub_id = args.hf_sae_repo or resolve_hub_id(cfg.get("hf_sae_repo"), model_id, push)
    hub_repo_type = args.hf_repo_type or cfg.get("hf_repo_type") or "model"
    wandb_project = args.wandb_project or cfg.get("wandb_project") or f"{_slug(model_id)}-sae"

    capture = pick(args.capture, "capture", "auto")
    expansion = pick(args.expansion, "expansion", 32)
    pool_batches = pick(args.pool_batches, "pool_batches", 500)
    pool_forward_fusion = pick(args.pool_forward_fusion, "pool_forward_fusion", 1)
    if pool_forward_fusion < 1:
        raise SystemExit("--pool-forward-fusion must be >= 1")
    max_steps = pick(args.max_steps, "max_steps", 5000)
    target_l0 = pick(args.target_l0, "target_l0", None)
    microbatch = pick(args.microbatch_tokens, "microbatch_tokens", None)
    pool_retention = pick(args.pool_retention, "pool_retention", 3)

    corpus_cfg = cfg.get("corpus") or {}
    corpus = args.corpus or corpus_cfg.get("id")
    corpus_text_field = args.corpus_text_field or corpus_cfg.get("text_field")
    corpus_prefix = args.corpus_prefix or corpus_cfg.get("prefix")
    trust_remote_code = args.trust_remote_code or bool(cfg.get("trust_remote_code"))

    if push and not resolve_hf_token():
        raise SystemExit(
            "no HF credentials: run `hf auth login`, set HF_TOKEN, or pass --no-push")

    # ---- environment -----------------------------------------------------
    data_dir = Path(os.environ.setdefault("SAE_DATA_DIR", "./data"))
    os.environ.setdefault("SAE_SCRATCH_DIR", str(data_dir / "rollcache"))
    os.environ.setdefault("SAE_MODEL_ID", model_id)
    if hub_id:
        os.environ.setdefault("SAE_HUB_ID", hub_id)
    os.environ["SAE_HUB_REPO_TYPE"] = hub_repo_type
    os.environ.setdefault("SAE_BATCH_TOKENS", "32768")
    os.environ.setdefault("SAE_MICROBATCH_TOKENS", str(microbatch or 32768))
    # The Triton path is dead (measured ~100x slower than PyTorch); keep it off.
    os.environ.setdefault("SAE_USE_TRITON", "0")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # hf_transfer is retired; Xet is the current fast-download path.
    # (HF_HUB_ENABLE_HF_TRANSFER is deprecated and now only emits a warning.)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    # The config env block is an explicit user choice, so it overrides whatever the
    # shell happens to have exported -- setdefault here would invert the documented
    # CLI > config > defaults precedence.
    for k, v in (cfg.get("env") or {}).items():
        os.environ[str(k)] = str(v)
    # CLI beats both the config env block and the inherited shell value.
    if args.timing_every is not None:
        os.environ["SAE_TIMING"] = str(max(0, args.timing_every))
    os.environ["WANDB_PROJECT"] = wandb_project

    # ---- layer range -----------------------------------------------------
    if args.layer_range:
        start, end = map(int, args.layer_range.split(","))
    else:
        n_layers = resolve_n_layers(model_id, arch.get("n_layers"), trust_remote_code)
        start, end = 0, n_layers - 1

    sae_dir = data_dir / "saes" / _slug(model_id)

    # ---- banner ----------------------------------------------------------
    print(f"\n{'=' * 68}")
    print("SAE atlas training")
    print(f"  Config:      {cfg_path.name}")
    print(f"  Model:       {model_id}")
    print(f"  Layers:      {start}-{end} ({end - start + 1} layers)")
    print(f"  Capture:     {capture}")
    if corpus:
        print(f"  Corpus:      {corpus} (field={corpus_text_field or 'text'}, "
              f"prefix={corpus_prefix!r})")
    if trust_remote_code:
        print("  Remote code: trusted")
    print(f"  Expansion:   {expansion}x")
    print(f"  Pool batches:{pool_batches}")
    print(f"  Pool fusion: {pool_forward_fusion}x")
    print(f"  Max steps:   {max_steps}")
    print(f"  Target L0:   {target_l0}")
    print(f"  Microbatch:  {microbatch} tokens")
    print(f"  SAE dir:     {sae_dir}")
    print(f"  HF repo:     {hub_id or '(push disabled)'} ({hub_repo_type})")
    wandb_disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled"
    print(f"  W&B project: {wandb_project}{' (disabled)' if wandb_disabled else ''}")
    if not wandb_disabled:
        print("  W&B auth:    checked when each layer run starts")
    if cfg.get("notes"):
        print(f"{'-' * 68}")
        for line in str(cfg["notes"]).rstrip().splitlines():
            print(f"  {line}")
    print(f"{'=' * 68}\n")

    if args.dry_run:
        print("[dry-run] config resolved; not training.")
        return

    sys.path.insert(0, str(REPO_ROOT))
    from sae_trainer_rolling import run_atlas_rolling

    results = run_atlas_rolling(
        start_layer=start,
        end_layer=end,
        seed=args.seed,
        pool_batches=pool_batches,
        pool_forward_fusion=pool_forward_fusion,
        microbatch_tokens=microbatch,
        use_pretok=not args.no_pretok,
        max_steps=max_steps,
        bdec_batches=args.bdec_batches,
        resume_from=args.resume_from,
        push=push,
        capture=capture,
        model_id=model_id,
        hub_id=hub_id,
        hub_repo_type=hub_repo_type,
        wandb_project=wandb_project,
        expansion=expansion,
        evict_model=not args.no_model_evict,
        target_l0=target_l0,
        cpu=args.cpu,
        norm_ref=args.norm_ref,
        corpus=corpus,
        corpus_text_field=corpus_text_field,
        corpus_prefix=corpus_prefix,
        trust_remote_code=trust_remote_code,
        pool_retention=pool_retention,
    )

    if push and hub_id:
        for layer in range(start, end + 1):
            push_layer(sae_dir, hub_id, layer, seed=args.seed,
                       repo_type=hub_repo_type)

        summary = {
            "model_id": model_id,
            "config": cfg_path.name,
            "layers": f"{start},{end}",
            "capture": capture,
            "expansion": expansion,
            "pool_batches": pool_batches,
            "pool_forward_fusion": pool_forward_fusion,
            "max_steps": max_steps,
            "target_l0": target_l0,
            "hf_repo_type": hub_repo_type,
            "results": results,
        }
        summary.update(cfg.get("caveats") or {})
        push_summary(sae_dir, hub_id, summary, repo_type=hub_repo_type)

    print("\nTraining complete.")
    print(f"  Results: {results}")


if __name__ == "__main__":
    main()
