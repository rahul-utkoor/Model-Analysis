#!/usr/bin/env python
"""Generate small dry-run pruning action candidates from dependency graphs."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.action_generation import generate_candidate_actions
from model_analysis.dependency_graph import DependencyGraph
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.pruning_action import pruning_action_to_dict, write_pruning_plan_json
from model_analysis.pruning_plan_reporting import candidate_actions_to_markdown, pruning_plan_to_markdown
from model_analysis.registry import get_model_config, load_model_registry
from model_analysis.reporting import write_json, write_markdown


def select_models(model_arg: str) -> list[dict]:
    if model_arg == "all":
        return load_model_registry()
    return [get_model_config(model_arg)]


def _load_graph(path):
    return DependencyGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _safe_action_id(action_id: str) -> str:
    return action_id.replace("/", "__").replace(":", "_").replace(" ", "_")


def generate_for_model(config: dict, max_actions_per_unit: int, simulate: bool, limit: int | None) -> None:
    root = get_project_root()
    safe_name = safe_model_name(config["hf_id"])
    graph_path = root / "reports" / "dependency_graphs" / f"{safe_name}.json"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Dependency graph missing. Run: python scripts/build_dependency_graph.py --model {config['name']}"
        )

    graph = _load_graph(graph_path)
    actions = generate_candidate_actions(graph, max_actions_per_unit=max_actions_per_unit)
    if limit is not None:
        actions = actions[:limit]

    checks_dir = root / "reports" / "pruning_action_checks"
    write_json(
        {"model_name": graph.model_name, "candidate_actions": [pruning_action_to_dict(action) for action in actions]},
        checks_dir / f"{safe_name}__candidate_actions.json",
    )
    write_markdown(candidate_actions_to_markdown(actions), checks_dir / f"{safe_name}__candidate_actions.md")

    if simulate:
        for action in actions:
            plan = simulate_pruning_action(graph, action)
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
                checks_dir / f"{stem}.json",
            )

    print(f"[ok] generated {len(actions)} candidate actions for {config['name']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dry-run candidate pruning actions.")
    parser.add_argument("--model", required=True, help="Configured model name, Hugging Face ID, or 'all'.")
    parser.add_argument("--max-actions-per-unit", type=int, default=3, help="Maximum candidate actions per unit.")
    parser.add_argument("--simulate", action="store_true", help="Simulate each generated candidate and save plans.")
    parser.add_argument("--limit", type=int, default=None, help="Limit total candidate actions per model.")
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
            generate_for_model(config, args.max_actions_per_unit, args.simulate, args.limit)
        except FileNotFoundError as exc:
            failures += 1
            print(f"[missing] {exc}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"[error] failed for {config.get('name', '<unknown>')}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
