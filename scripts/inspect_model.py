#!/usr/bin/env python
"""Inspect locally downloaded PyTorch models."""

from __future__ import annotations

import argparse
import sys

from model_analysis.hf_utils import load_model
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, load_model_registry
from model_analysis.reporting import structural_inventory_to_markdown, write_markdown
from model_analysis.structural_inventory import summarize_torch_model


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

    summary = summarize_torch_model(model, config["name"], config)
    parameters = summary["parameter_summary"]

    print(f"\n{config['name']}")
    print(f"  parameters: {parameters['total_parameters']:,}")
    print(f"  trainable parameters: {parameters['trainable_parameters']:,}")
    print(f"  modules: {summary['module_summary']['total_modules']:,}")
    print(f"  linear layers: {len(summary['linear_layers'])}")
    print(f"  attention-like modules: {len(summary['attention_like_modules'])}")
    print(f"  embedding layers: {len(summary['embedding_layers'])}")
    print(f"  pruning-relevant groups: {len(summary['pruning_relevant_groups'])}")

    report_path = get_project_root() / "reports" / "model_summaries" / f"{safe_model_name(config['hf_id'])}.md"
    write_markdown(structural_inventory_to_markdown(summary), report_path)
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
