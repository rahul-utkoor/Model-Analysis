#!/usr/bin/env python
"""Execute a conservative Linear-only pruning plan."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_analysis.correspondence import load_correspondence_json
from model_analysis.dependency_graph import DependencyGraph
from model_analysis.hf_utils import load_model, load_tokenizer_or_processor
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.pruning_action import PruningAction, make_action_id, pruning_plan_from_dict
from model_analysis.pruning_diff import pruning_diff_to_markdown
from model_analysis.pruning_execution import (
    pruning_execution_report_to_markdown,
    write_pruning_execution_report_json,
)
from model_analysis.pruning_plan_executor import execute_linear_pruning_plan
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown
from model_analysis.rollback import create_rollback_manifest, rollback_manifest_to_markdown, write_rollback_manifest
from model_analysis.shape_evidence import load_shape_evidence_json


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_indices(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _safe_stem(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(" ", "_")


def _load_graph(root: Path, safe_name: str) -> DependencyGraph:
    graph_path = root / "reports" / "dependency_graphs" / f"{safe_name}.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Dependency graph missing. Run: python scripts/build_dependency_graph.py --model {safe_name}")
    return DependencyGraph.from_dict(_load_json(graph_path))


def _load_evidence(root: Path, safe_name: str, model_name: str):
    correspondence_path = root / "reports" / "correspondence" / f"{safe_name}.json"
    shape_path = root / "reports" / "shape_evidence" / f"{safe_name}.json"
    validation_path = root / "reports" / "validated_dependency_graphs" / f"{safe_name}.json"
    if not correspondence_path.exists() or not shape_path.exists():
        raise FileNotFoundError(f"Evidence reports missing. Run: python scripts/build_correspondence.py --model {model_name}")
    validation = _load_json(validation_path) if validation_path.exists() else None
    return load_correspondence_json(correspondence_path), load_shape_evidence_json(shape_path), validation


def _construct_action(args: argparse.Namespace, graph: DependencyGraph) -> PruningAction:
    if not args.target_unit or not args.dim or args.indices is None:
        raise ValueError("Direct action execution requires --target-unit, --dim, and --indices.")
    indices = _parse_indices(args.indices)
    return PruningAction(
        action_id=make_action_id(args.target_unit, args.dim, indices, "manual_indices"),
        model_name=graph.model_name,
        target_unit_id=args.target_unit,
        target_unit_name=None,
        target_unit_type=None,
        prune_dim=args.dim,
        indices=indices,
        amount=len(indices),
        fraction=None,
        strategy="manual_indices",
        reason=args.reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Linear-only pruning from a plan or direct action.")
    parser.add_argument("--model", required=True, help="Configured model name or Hugging Face ID.")
    parser.add_argument("--plan-json", help="Path to pruning plan JSON.")
    parser.add_argument("--target-unit", help="Dependency graph unit ID for direct action mode.")
    parser.add_argument("--dim", help="Dimension to prune.")
    parser.add_argument("--indices", help="Comma-separated indices to prune.")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify or save model.")
    parser.add_argument("--only-target", action="store_true", help="Prune only the target Linear module.")
    parser.add_argument("--allow-ambiguous", action="store_true", help="Allow execution of ambiguous plans.")
    parser.add_argument("--use-evidence", action="store_true", help="Use correspondence/shape evidence during action simulation.")
    parser.add_argument("--reason", default=None, help="Optional action reason.")
    parser.add_argument("--verbose", action="store_true", help="Print execution summary.")
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
    source_dir = root / config["local_dir"]
    if not source_dir.exists():
        print(f"[missing] Local model missing. Run: python scripts/download_models.py --model {config['name']}", file=sys.stderr)
        return 1

    try:
        graph = _load_graph(root, safe_name)
        if args.plan_json:
            plan = pruning_plan_from_dict(_load_json(Path(args.plan_json)))
        else:
            action = _construct_action(args, graph)
            correspondence = shape = validation = None
            if args.use_evidence:
                correspondence, shape, validation = _load_evidence(root, safe_name, config["name"])
            plan = simulate_pruning_action(graph, action, correspondence, shape, validation)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if plan.status == "rejected":
        print("[rejected] Pruning plan is rejected; refusing execution.", file=sys.stderr)
        return 1
    if plan.status == "ambiguous" and not args.allow_ambiguous and not args.dry_run:
        print("[rejected] Pruning plan is ambiguous. Re-run with --allow-ambiguous to execute Linear-only surgery.", file=sys.stderr)
        return 1

    execution_id = f"{_safe_stem(plan.action.action_id)}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = root / "artifacts" / "pruned_models" / safe_name / execution_id

    try:
        model = load_model(config, source=source_dir)
        tokenizer_or_processor = None
        try:
            tokenizer_or_processor = load_tokenizer_or_processor(config, source=source_dir)
        except Exception:
            tokenizer_or_processor = None
        report = execute_linear_pruning_plan(
            model=model,
            model_name=config["name"],
            source_model_dir=source_dir,
            output_model_dir=output_dir,
            plan=plan,
            dependency_graph=graph,
            tokenizer_or_processor=tokenizer_or_processor,
            allow_ambiguous=args.allow_ambiguous or args.dry_run,
            only_target=args.only_target,
            dry_run=args.dry_run,
        )
        report.execution_id = execution_id
        report.output_model_dir = str(output_dir)
    except Exception as exc:
        print(f"[error] execution failed: {exc}", file=sys.stderr)
        return 1

    stem = f"{safe_name}__{execution_id}"
    execution_json = root / "reports" / "pruning_execution" / f"{stem}.json"
    execution_md = root / "reports" / "pruning_execution" / f"{stem}.md"
    diff_json = root / "reports" / "pruning_diffs" / f"{stem}.json"
    diff_md = root / "reports" / "pruning_diffs" / f"{stem}.md"
    rollback_json = root / "reports" / "rollback_manifests" / f"{stem}.json"
    rollback_md = root / "reports" / "rollback_manifests" / f"{stem}.md"

    manifest = create_rollback_manifest(report, source_dir, output_dir, rollback_json)
    report.rollback_manifest_path = str(rollback_json)
    write_pruning_execution_report_json(report, execution_json)
    write_markdown(pruning_execution_report_to_markdown(report), execution_md)
    write_json(report.diff_summary, diff_json)
    write_markdown(pruning_diff_to_markdown(report.diff_summary), diff_md)
    write_rollback_manifest(manifest, rollback_json)
    write_markdown(rollback_manifest_to_markdown(manifest), rollback_md)

    if args.verbose:
        print(f"[execution] {execution_id}")
        print(f"  status: {report.status}")
        print(f"  dry_run: {args.dry_run}")
        print(f"  applied: {len(report.applied_records)}")
        print(f"  skipped: {len(report.skipped_records)}")
        print(f"  rejected: {len(report.rejected_records)}")
        print(f"  output: {output_dir}")

    return 0 if report.status in {"success", "partial"} or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
