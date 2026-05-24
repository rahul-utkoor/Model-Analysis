#!/usr/bin/env python
"""Explain selected symbolic pruning plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain selected symbolic pruning plans.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--contains")
    parser.add_argument("--status", choices=["ready_symbolic", "incomplete", "blocked", "unknown"])
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def _matches(plan: dict, args: argparse.Namespace) -> bool:
    if args.status and plan.get("plan_status") != args.status:
        return False
    if args.contains:
        needle = args.contains.lower()
        haystack = " ".join(
            [
                str(plan.get("candidate_region_name", "")),
                str(plan.get("plan_kind", "")),
                str(plan.get("target_dimension", "")),
                str(plan.get("symbolic_index_set", {}).get("name", "")),
            ]
        ).lower()
        if needle not in haystack:
            return False
    return True


def _markdown(model: str, rows: list[dict]) -> str:
    lines = ["# Pruning Plan Selection: " + model, "", "| plan | status | score | index set | actions | warnings |", "| --- | --- | --- | --- | --- | --- |"]
    for plan in rows:
        actions = ", ".join(action.get("action_type", "") for action in plan.get("actions", []) if action.get("required"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(plan.get("candidate_region_name", "")).replace("|", "\\|"),
                    str(plan.get("plan_status", "")),
                    str(plan.get("rank_score", "")),
                    str(plan.get("symbolic_index_set", {}).get("name", "")),
                    actions,
                    ", ".join(plan.get("warnings", [])),
                ]
            )
            + " |"
        )
    lines.extend(["", "This is a static symbolic pruning plan explanation. It does not modify models.", ""])
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
    path = root / "reports" / "pruning_plans" / f"{safe}.json"
    if not path.exists():
        print(f"[missing] Pruning plans missing. Run: python scripts/synthesize_pruning_plans.py --model {config['name']}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [plan for plan in data.get("plans", []) if _matches(plan, args)][: args.limit]
    suffix_parts = []
    if args.status:
        suffix_parts.append(f"status_{args.status}")
    if args.contains:
        suffix_parts.append("contains_" + "_".join(args.contains.lower().split())[:60])
    suffix = "__selection__" + "__".join(suffix_parts) if suffix_parts else "__selection"
    out_json = root / "reports" / "pruning_plan_explanations" / f"{safe}{suffix}.json"
    out_md = root / "reports" / "pruning_plan_explanations" / f"{safe}{suffix}.md"
    write_json({"model_name": data.get("model_name"), "matches": rows}, out_json)
    write_markdown(_markdown(data.get("model_name", safe), rows), out_md)
    print(f"[pruning-plan-explain] matches={len(rows)}")
    for plan in rows:
        actions = ", ".join(action.get("action_type", "") for action in plan.get("actions", []) if action.get("required"))
        print(f"- {plan.get('candidate_region_name')} [{plan.get('plan_status')}] {plan.get('symbolic_index_set', {}).get('name')} actions={actions}")
    print(f"[pruning-plan-explain] json={out_json}")
    print(f"[pruning-plan-explain] markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

