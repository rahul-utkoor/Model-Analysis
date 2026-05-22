#!/usr/bin/env python
"""Simulate one dry-run pruning action through a dependency graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.dependency_graph import DependencyGraph
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.pruning_action import (
    PruningAction,
    load_pruning_action_json,
    make_action_id,
    pruning_plan_to_dict,
    write_pruning_plan_json,
)
from model_analysis.pruning_plan_reporting import pruning_plan_to_markdown
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def _load_graph(path: Path) -> DependencyGraph:
    return DependencyGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _parse_indices(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _safe_action_id(action_id: str) -> str:
    return action_id.replace("/", "__").replace(":", "_").replace(" ", "_")


def _construct_action(args: argparse.Namespace, graph: DependencyGraph) -> PruningAction:
    if args.action_json:
        return load_pruning_action_json(Path(args.action_json))

    if not args.target_unit or not args.dim or args.indices is None:
        raise ValueError("Direct action mode requires --target-unit, --dim, and --indices.")

    indices = _parse_indices(args.indices)
    action_id = make_action_id(args.target_unit, args.dim, indices, args.strategy)
    return PruningAction(
        action_id=action_id,
        model_name=graph.model_name,
        target_unit_id=args.target_unit,
        target_unit_name=None,
        target_unit_type=None,
        prune_dim=args.dim,
        indices=indices,
        amount=len(indices),
        fraction=None,
        strategy=args.strategy,
        reason=args.reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a dry-run pruning action.")
    parser.add_argument("--model", required=True, help="Configured model name or Hugging Face ID.")
    parser.add_argument("--target-unit", help="Dependency graph unit ID to prune.")
    parser.add_argument("--dim", help="Dimension to prune, such as out_features.")
    parser.add_argument("--indices", help="Comma-separated indices to prune, such as 0,1,2,3.")
    parser.add_argument("--action-json", help="Path to a pruning action JSON file.")
    parser.add_argument("--reason", default=None, help="Optional rationale for the requested action.")
    parser.add_argument("--strategy", default="manual_indices", help="Action strategy label. Defaults to manual_indices.")
    parser.add_argument("--verbose", action="store_true", help="Print plan summary.")
    parser.add_argument("--allow-ambiguous", action="store_true", help="Exit 0 for ambiguous plans.")
    parser.add_argument("--fail-on-ambiguous", action="store_true", help="Exit nonzero if status is ambiguous.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = get_model_config(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    root = get_project_root()
    safe_name = safe_model_name(config["hf_id"])
    graph_path = root / "reports" / "dependency_graphs" / f"{safe_name}.json"
    if not graph_path.exists():
        print(
            f"[missing] Dependency graph missing. Run: python scripts/build_dependency_graph.py --model {config['name']}",
            file=sys.stderr,
        )
        return 1

    try:
        graph = _load_graph(graph_path)
        action = _construct_action(args, graph)
        plan = simulate_pruning_action(graph, action)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    action_part = _safe_action_id(action.action_id)
    stem = f"{safe_name}__{action_part}"
    write_pruning_plan_json(plan, root / "reports" / "pruning_plans" / f"{stem}.json")
    write_markdown(pruning_plan_to_markdown(plan), root / "reports" / "pruning_plans" / f"{stem}.md")
    write_json(
        {
            "plan_id": plan.plan_id,
            "status": plan.status,
            "action": plan.action.__dict__,
            "propagation_steps": [step.__dict__ for step in plan.propagation_steps],
        },
        root / "reports" / "propagation_traces" / f"{stem}.json",
    )
    write_json(
        {
            "plan_id": plan.plan_id,
            "status": plan.status,
            "summary": plan.summary,
            "constraints": plan.constraints,
            "conflicts": plan.conflicts,
            "manual_review_items": plan.manual_review_items,
        },
        root / "reports" / "pruning_action_checks" / f"{stem}.json",
    )

    if args.verbose:
        print(f"[plan] {plan.plan_id}")
        print(f"  status: {plan.status}")
        print(f"  affected units: {len(plan.affected_units)}")
        print(f"  propagation steps: {len(plan.propagation_steps)}")
        print(f"  constraints: {len(plan.constraints)}")
        print(f"  conflicts: {len(plan.conflicts)}")
        print(f"  manual review items: {len(plan.manual_review_items)}")

    if plan.status == "rejected":
        return 1
    if plan.status == "ambiguous" and (args.fail_on_ambiguous or not args.allow_ambiguous):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
