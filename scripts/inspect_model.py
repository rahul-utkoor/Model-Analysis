#!/usr/bin/env python
"""Inspect locally downloaded PyTorch models."""

from __future__ import annotations

import argparse
import sys

import torch

from model_analysis.graph_summary import (
    generate_markdown_summary,
    list_attention_like_modules,
    list_named_modules_by_type,
)
from model_analysis.hf_utils import load_model
from model_analysis.paths import ensure_dir, get_project_root
from model_analysis.torch_utils import count_parameters, count_trainable_parameters
from model_analysis.registry import get_model_config, load_model_registry


def select_models(model_arg: str) -> list[dict]:
    if model_arg == "all":
        return load_model_registry()
    return [get_model_config(model_arg)]


def inspect_one(config: dict) -> None:
    source_dir = get_project_root() / config["local_dir"]
    if not source_dir.exists() or not (source_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Local model not found at {source_dir}. Run: python scripts/download_models.py --model {config['name']}"
        )

    model = load_model(config, source=source_dir)
    model.eval()

    parameter_count = count_parameters(model)
    trainable_count = count_trainable_parameters(model)
    top_level = list(model.named_children())
    linear_layers = list_named_modules_by_type(model, torch.nn.Linear)
    attention_like = list_attention_like_modules(model)
    embedding_layers = list_named_modules_by_type(model, torch.nn.Embedding)

    print(f"\n{config['name']}")
    print(f"  parameters: {parameter_count:,}")
    print(f"  trainable parameters: {trainable_count:,}")
    print("  top-level modules:")
    for name, module in top_level:
        print(f"    - {name}: {module.__class__.__name__}")
    print(f"  linear layers: {len(linear_layers)}")
    print(f"  attention-like modules: {len(attention_like)}")
    print(f"  embedding layers: {len(embedding_layers)}")

    report_dir = ensure_dir(get_project_root() / "reports" / "model_summaries")
    report_path = report_dir / f"{config['name'].replace('/', '__')}.md"
    report_path.write_text(generate_markdown_summary(model, config), encoding="utf-8")
    print(f"  summary: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect configured local PyTorch models.")
    parser.add_argument("--model", required=True, help="Configured model name, Hugging Face ID, or 'all'.")
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
            inspect_one(config)
        except FileNotFoundError as exc:
            failures += 1
            print(f"[missing] {exc}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"[error] failed to inspect {config.get('name', '<unknown>')}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
