#!/usr/bin/env python
"""Explain selected ranked pruning opportunity candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain selected pruning opportunity candidates.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--class", dest="pruning_class", choices=["safe", "constrained", "blocked", "auxiliary", "unknown"])
    parser.add_argument("--contains")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def _matches(item: dict, args: argparse.Namespace) -> bool:
    if args.pruning_class and item.get("pruning_class") != args.pruning_class:
        return False
    if args.contains:
        needle = args.contains.lower()
        haystack = " ".join(
            [
                str(item.get("region_name", "")),
                str(item.get("semantic_category", "")),
                str(item.get("candidate_kind", "")),
                str(item.get("reason", "")),
            ]
        ).lower()
        if needle not in haystack:
            return False
    return True


def _markdown(model: str, rows: list[dict]) -> str:
    lines = [
        f"# Pruning Opportunity Selection: {model}",
        "",
        "| candidate | class | score | confidence | kind | target | blockers | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        blockers = ", ".join(blocker.get("blocker_type", "") for blocker in item.get("blockers", []))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("region_name", "")).replace("|", "\\|"),
                    str(item.get("pruning_class", "")),
                    str(item.get("rank_score", "")),
                    str(item.get("confidence", "")),
                    str(item.get("candidate_kind", "")),
                    str(item.get("target_dimension", "")),
                    blockers,
                    str(item.get("reason", "")).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.extend(["", "This is a static explanation over pruning opportunity rankings. It does not modify models.", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    safe = safe_model_name(config["hf_id"])
    path = root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json"
    if not path.exists():
        print(f"[missing] Pruning opportunity ranking missing. Run: python scripts/rank_pruning_opportunities.py --model {config['name']}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [item for item in data.get("candidates", []) if _matches(item, args)][: args.limit]
    suffix_parts = []
    if args.pruning_class:
        suffix_parts.append(f"class_{args.pruning_class}")
    if args.contains:
        suffix_parts.append("contains_" + "_".join(args.contains.lower().split())[:60])
    suffix = "__selection__" + "__".join(suffix_parts) if suffix_parts else "__selection"
    out_json = root / "reports" / "pruning_opportunity_explanations" / f"{safe}{suffix}.json"
    out_md = root / "reports" / "pruning_opportunity_explanations" / f"{safe}{suffix}.md"
    write_json({"model_name": data.get("model_name"), "matches": rows}, out_json)
    write_markdown(_markdown(data.get("model_name", safe), rows), out_md)
    print(f"[pruning-opportunity-explain] matches={len(rows)}")
    for item in rows:
        print(f"- {item.get('region_name')} [{item.get('pruning_class')} score={item.get('rank_score')}] {item.get('candidate_kind')}")
    print(f"[pruning-opportunity-explain] json={out_json}")
    print(f"[pruning-opportunity-explain] markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

