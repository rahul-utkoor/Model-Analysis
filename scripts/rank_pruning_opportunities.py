#!/usr/bin/env python
"""Rank static pruning opportunities from region and op semantics reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.pruning_opportunity_ranking import (
    build_pruning_opportunity_ranking,
    pruning_opportunity_ranking_to_markdown,
    write_pruning_opportunity_ranking_json,
)
from model_analysis.pruning_opportunity_ranking_text import write_pruning_opportunity_ranking_text
from model_analysis.registry import get_model_config, list_models
from model_analysis.reporting import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank static pruning opportunities.")
    parser.add_argument("--model", required=True, help="Configured model name/HF ID or 'all'.")
    parser.add_argument("--include-auxiliary-details", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe(path: Path) -> dict | None:
    return _load_json(path) if path.exists() else None


def _configs(value: str) -> list[dict]:
    if value == "all":
        return [get_model_config(name) for name in list_models()]
    return [get_model_config(value)]


def build_one(root: Path, config: dict, include_auxiliary_details: bool, verbose: bool) -> bool:
    safe = safe_model_name(config["hf_id"])
    region_path = root / "reports" / "region_pruning_semantics" / f"{safe}.json"
    op_path = root / "reports" / "op_semantics" / f"{safe}.json"
    if not region_path.exists():
        print(f"[missing] Region Pruning Semantics missing. Run: python scripts/build_region_pruning_semantics.py --model {config['name']}", file=sys.stderr)
        return False
    ranking = build_pruning_opportunity_ranking(
        _load_json(region_path),
        op_semantics=_maybe(op_path),
        source_region_pruning_semantics_path=str(region_path),
        source_op_semantics_path=str(op_path) if op_path.exists() else None,
    )
    json_path = root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json"
    text_path = root / "reports" / "pruning_opportunity_ranking_dumps" / f"{safe}.rank"
    md_path = root / "reports" / "pruning_opportunity_explanations" / f"{safe}.md"
    write_pruning_opportunity_ranking_json(ranking, json_path)
    write_pruning_opportunity_ranking_text(ranking, text_path)
    write_markdown(pruning_opportunity_ranking_to_markdown(ranking, include_auxiliary_details=include_auxiliary_details), md_path)
    if verbose:
        summary = ranking.summary
        print(f"[pruning-opportunity-ranking] {ranking.model_name}")
        print(f"  total: {summary.get('total_candidates', 0)}")
        print(f"  safe: {summary.get('safe_candidates', 0)}")
        print(f"  constrained: {summary.get('constrained_candidates', 0)}")
        print(f"  blocked: {summary.get('blocked_candidates', 0)}")
        print(f"  auxiliary: {summary.get('auxiliary_candidates', 0)}")
        print(f"  unknown: {summary.get('unknown_candidates', 0)}")
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
        ok = build_one(root, config, args.include_auxiliary_details, args.verbose) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

