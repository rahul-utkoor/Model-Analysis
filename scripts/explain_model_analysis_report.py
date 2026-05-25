#!/usr/bin/env python
"""Print focused excerpts from a generated full-model analysis report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--section", choices=["feedforward", "attention", "safe", "constrained", "blocked", "auxiliary"])
    parser.add_argument("--layer", type=int)
    parser.add_argument("--contains")
    parser.add_argument("--class", dest="pruning_class")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-root", default="reports/model_analysis_reports")
    return parser.parse_args()


def _load_report(root: Path, model: str, output_root: str) -> dict:
    config = get_model_config(model)
    safe = safe_model_name(config["hf_id"])
    path = root / output_root / safe / "index.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing model analysis report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _print_rows(title: str, rows: list[dict], limit: int) -> None:
    print(f"# {title}")
    for row in rows[:limit]:
        print(
            "- {name} | category={category} | class={cls} | plan={plan} | validation={validation}".format(
                name=row.get("display_name") or row.get("region_name"),
                category=row.get("semantic_category"),
                cls=row.get("pruning_class", row.get("Class", "")),
                plan=row.get("plan_status", row.get("plan_id", "")),
                validation=row.get("validation_status", ""),
            )
        )


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        report = _load_report(root, args.model, args.output_root)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    rows: list[dict] = []
    if args.layer is not None:
        for layer in report.get("layers", []):
            if layer.get("layer_index") == args.layer:
                rows.extend(layer.get("subgraphs", []))
                break
    else:
        for layer in report.get("layers", []):
            rows.extend(layer.get("subgraphs", []))
    if args.section == "feedforward":
        rows = [row for row in rows if "ffn" in str(row.get("semantic_category", "")) or "feed" in str(row.get("semantic_category", ""))]
    elif args.section == "attention":
        rows = [row for row in rows if "attention" in str(row.get("semantic_category", ""))]
    elif args.section in {"safe", "constrained", "blocked", "auxiliary"}:
        rows = [row for row in rows if row.get("pruning_class") == args.section]
    if args.pruning_class:
        rows = [row for row in rows if row.get("pruning_class") == args.pruning_class]
    if args.contains:
        needle = args.contains.lower()
        rows = [row for row in rows if needle in str(row.get("display_name", "")).lower()]
    _print_rows("Model Analysis Report Excerpt", rows, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
