#!/usr/bin/env python
"""Validate symbolic pruning plans against static semantics artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.pruning_plan_validation import pruning_plan_validation_to_markdown, validate_pruning_plans, write_pruning_plan_validation_json
from model_analysis.pruning_plan_validation_text import write_pruning_plan_validation_text
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate symbolic pruning plans.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _configs(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def build_one(root: Path, config: dict, verbose: bool) -> bool:
    safe = safe_model_name(config["hf_id"])
    plan_path = root / "reports" / "pruning_plans" / f"{safe}.json"
    ranking_path = root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json"
    region_path = root / "reports" / "region_pruning_semantics" / f"{safe}.json"
    op_path = root / "reports" / "op_semantics" / f"{safe}.json"
    if not plan_path.exists():
        print(f"[missing] Pruning plans missing. Run: python scripts/synthesize_pruning_plans.py --model {config['name']}", file=sys.stderr)
        return False
    if not ranking_path.exists():
        print(f"[missing] Pruning opportunity ranking missing. Run: python scripts/rank_pruning_opportunities.py --model {config['name']}", file=sys.stderr)
        return False
    if not region_path.exists():
        print(f"[missing] Region Pruning Semantics missing. Run: python scripts/build_region_pruning_semantics.py --model {config['name']}", file=sys.stderr)
        return False
    if not op_path.exists():
        print(f"[missing] Op Semantics missing. Run: python scripts/build_op_semantics.py --model {config['name']}", file=sys.stderr)
        return False
    validation = validate_pruning_plans(
        _load_json(plan_path),
        _load_json(ranking_path),
        _load_json(region_path),
        _load_json(op_path),
        source_plan_path=str(plan_path),
        source_ranking_path=str(ranking_path),
        source_region_pruning_semantics_path=str(region_path),
        source_op_semantics_path=str(op_path),
    )
    json_path = root / "reports" / "pruning_plan_validation" / f"{safe}.json"
    text_path = root / "reports" / "pruning_plan_validation_dumps" / f"{safe}.pvalid"
    md_path = root / "reports" / "pruning_plan_validation_explanations" / f"{safe}.md"
    write_pruning_plan_validation_json(validation, json_path)
    write_pruning_plan_validation_text(validation, text_path)
    write_markdown(pruning_plan_validation_to_markdown(validation), md_path)
    if verbose:
        summary = validation.summary
        print(f"[pruning-plan-validation] {validation.model_name}")
        print(f"  validations: {summary.get('total_plans', 0)}")
        print(f"  valid: {summary.get('valid_plans', 0)}")
        print(f"  warning: {summary.get('warning_plans', 0)}")
        print(f"  invalid: {summary.get('invalid_plans', 0)}")
        print(f"  unknown: {summary.get('unknown_plans', 0)}")
        print(f"  json: {json_path}")
        print(f"  text: {text_path}")
        print(f"  markdown: {md_path}")
    return True


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        configs = _configs(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    ok = True
    for config in configs:
        ok = build_one(root, config, args.verbose) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
