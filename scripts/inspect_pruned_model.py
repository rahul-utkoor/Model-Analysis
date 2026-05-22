#!/usr/bin/env python
"""Inspect a saved pruned model artifact directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoConfig, AutoModel

from model_analysis.paths import get_project_root
from model_analysis.reporting import structural_inventory_to_markdown, write_json, write_markdown
from model_analysis.structural_inventory import summarize_torch_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a pruned model artifact directory.")
    parser.add_argument("--model-dir", required=True, help="Path to artifacts/pruned_models/<safe-name>/<execution-id>.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"[missing] Model directory does not exist: {model_dir}", file=sys.stderr)
        return 1

    try:
        config = AutoConfig.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir, config=config)
    except Exception as exc:
        print(f"[error] could not load pruned model with AutoModel: {exc}", file=sys.stderr)
        return 1

    execution_id = model_dir.name
    summary = summarize_torch_model(model, execution_id, {"hf_id": str(model_dir), "task": "pruned-inspection"})
    root = get_project_root()
    json_path = root / "reports" / "pruning_execution" / f"{execution_id}__post_inspection.json"
    md_path = root / "reports" / "pruning_execution" / f"{execution_id}__post_inspection.md"
    write_json(summary, json_path)
    write_markdown(structural_inventory_to_markdown(summary), md_path)
    print(f"[ok] wrote post-pruning inspection to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
