#!/usr/bin/env python
"""List detected BERT-style MLP pruning targets."""

from __future__ import annotations

import argparse
import sys

from model_analysis.bert_mlp_pruning import (
    bert_mlp_target_to_dict,
    bert_mlp_targets_to_markdown,
    detect_bert_mlp_block_targets,
)
from model_analysis.hf_utils import load_model
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List BERT-style MLP block pruning targets.")
    parser.add_argument("--model", required=True, help="Configured model name or Hugging Face ID.")
    return parser.parse_args()


def _print_table(rows: list[dict]) -> None:
    columns = ["layer_index", "intermediate_module", "output_module", "hidden_size", "intermediate_size", "confidence"]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(column, "")) for column in columns))


def main() -> int:
    args = parse_args()
    try:
        config = get_model_config(args.model)
        root = get_project_root()
        safe_name = safe_model_name(config["hf_id"])
        source_dir = root / config["local_dir"]
        if not source_dir.exists():
            print(f"[missing] Local model missing. Run: python scripts/download_models.py --model {config['name']}", file=sys.stderr)
            return 1
        model = load_model(config, source=source_dir)
        targets = detect_bert_mlp_block_targets(model, config["name"])
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    rows = [bert_mlp_target_to_dict(target) for target in targets]
    _print_table(rows)
    write_json(
        {"model_name": config["name"], "targets": rows},
        root / "reports" / "block_pruning" / f"{safe_name}__bert_mlp_targets.json",
    )
    write_markdown(
        bert_mlp_targets_to_markdown(config["name"], targets),
        root / "reports" / "block_pruning" / f"{safe_name}__bert_mlp_targets.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
