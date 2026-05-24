#!/usr/bin/env python
"""Synthesize symbolic pruning plans from safe ranked opportunities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.pruning_plan_synthesis import pruning_plan_set_to_markdown, synthesize_pruning_plans, write_pruning_plan_set_json
from model_analysis.pruning_plan_text import write_pruning_plan_text
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize symbolic pruning plans for safe opportunities.")
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
    ranking_path = root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json"
    region_path = root / "reports" / "region_pruning_semantics" / f"{safe}.json"
    op_path = root / "reports" / "op_semantics" / f"{safe}.json"
    if not ranking_path.exists():
        print(f"[missing] Pruning opportunity ranking missing. Run: python scripts/rank_pruning_opportunities.py --model {config['name']}", file=sys.stderr)
        return False
    if not region_path.exists():
        print(f"[missing] Region Pruning Semantics missing. Run: python scripts/build_region_pruning_semantics.py --model {config['name']}", file=sys.stderr)
        return False
    if not op_path.exists():
        print(f"[missing] Op Semantics missing. Run: python scripts/build_op_semantics.py --model {config['name']}", file=sys.stderr)
        return False
    plan_set = synthesize_pruning_plans(
        _load_json(ranking_path),
        _load_json(region_path),
        _load_json(op_path),
        source_ranking_path=str(ranking_path),
        source_region_pruning_semantics_path=str(region_path),
        source_op_semantics_path=str(op_path),
    )
    json_path = root / "reports" / "pruning_plans" / f"{safe}.json"
    text_path = root / "reports" / "pruning_plan_dumps" / f"{safe}.plan"
    md_path = root / "reports" / "pruning_plan_explanations" / f"{safe}.md"
    write_pruning_plan_set_json(plan_set, json_path)
    write_pruning_plan_text(plan_set, text_path)
    write_markdown(pruning_plan_set_to_markdown(plan_set), md_path)
    if verbose:
        summary = plan_set.summary
        print(f"[pruning-plan-synthesis] {plan_set.model_name}")
        print(f"  plans: {summary.get('total_plans', 0)}")
        print(f"  ready_symbolic: {summary.get('ready_symbolic', 0)}")
        print(f"  incomplete: {summary.get('incomplete', 0)}")
        print(f"  blocked: {summary.get('blocked', 0)}")
        print(f"  unknown: {summary.get('unknown', 0)}")
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

