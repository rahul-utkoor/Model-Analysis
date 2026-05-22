#!/usr/bin/env python
"""Download configured Hugging Face models for local analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model_analysis.hf_utils import get_model_class, load_tokenizer_or_processor
from model_analysis.paths import ensure_dir, get_project_root
from model_analysis.registry import get_model_config, load_model_registry


def _is_downloaded(local_dir: Path) -> bool:
    return (local_dir / "config.json").exists()


def download_one(config: dict, force: bool = False, cache_dir: str | None = None) -> None:
    local_dir = get_project_root() / config["local_dir"]
    ensure_dir(local_dir)

    if _is_downloaded(local_dir) and not force:
        print(f"[skip] {config['name']} already exists at {local_dir}. Use --force to redownload.")
        return

    print(f"[download] {config['name']} ({config['hf_id']}) -> {local_dir}")
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    model = get_model_class(config["task"]).from_pretrained(config["hf_id"], **kwargs)
    model.save_pretrained(local_dir)

    tokenizer_or_processor = load_tokenizer_or_processor(config, cache_dir=cache_dir)
    tokenizer_or_processor.save_pretrained(local_dir)
    print(f"[ok] saved {config['name']} to {local_dir}")


def select_models(model_arg: str) -> list[dict]:
    if model_arg == "all":
        return load_model_registry()
    return [get_model_config(model_arg)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download configured Hugging Face models.")
    parser.add_argument("--model", required=True, help="Configured model name, Hugging Face ID, or 'all'.")
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite existing local files.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = 0

    try:
        configs = select_models(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    for config in configs:
        try:
            download_one(config, force=args.force, cache_dir=args.cache_dir)
        except Exception as exc:
            failures += 1
            print(f"[error] failed to download {config.get('name', '<unknown>')}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
