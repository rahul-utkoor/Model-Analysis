#!/usr/bin/env python
"""Execute conservative Linear-only pruning plans."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_analysis.correspondence import load_correspondence_json
from model_analysis.dependency_graph import DependencyGraph
from model_analysis.forward_validation import (
    forward_smoke_result_to_dict,
    forward_smoke_result_to_markdown,
    run_forward_smoke_test,
)
from model_analysis.hf_utils import load_model, load_tokenizer_or_processor
from model_analysis.paired_linear_pruning import apply_repair_plan
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.pruning_action import PruningAction, make_action_id, pruning_plan_from_dict
from model_analysis.pruning_diff import compute_structural_diff, pruning_diff_to_markdown
from model_analysis.pruning_execution import (
    PruningExecutionReport,
    pruning_execution_report_to_markdown,
    write_pruning_execution_report_json,
)
from model_analysis.pruning_plan_executor import execute_linear_pruning_plan
from model_analysis.registry import get_model_config
from model_analysis.repair_detection import detect_linear_repair_plan
from model_analysis.repair_plan import (
    repair_plan_from_dict,
    repair_plan_to_dict,
    repair_plan_to_markdown,
    repair_transaction_records_to_dict,
    repair_transaction_records_to_markdown,
)
from model_analysis.reporting import write_json, write_markdown
from model_analysis.rollback import create_rollback_manifest, rollback_manifest_to_markdown, write_rollback_manifest
from model_analysis.shape_evidence import load_shape_evidence_json
from model_analysis.structural_inventory import summarize_torch_model


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


def _load_tokenizer_safely(config: dict, source_dir: Path):
    try:
        return load_tokenizer_or_processor(config, source=source_dir)
    except Exception:
        return None


def _write_smoke_report(root: Path, stem: str, phase: str, result) -> None:
    json_path = root / "reports" / "forward_smoke_tests" / f"{stem}__{phase}.json"
    md_path = root / "reports" / "forward_smoke_tests" / f"{stem}__{phase}.md"
    write_json(forward_smoke_result_to_dict(result), json_path)
    write_markdown(forward_smoke_result_to_markdown(result), md_path)


def _save_standard_reports(
    root: Path,
    safe_name: str,
    execution_id: str,
    report: PruningExecutionReport,
    source_dir: Path,
    output_dir: Path,
) -> None:
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


def _write_repair_reports(root: Path, safe_name: str, execution_id: str, repair_plan, transactions=None) -> None:
    stem = f"{safe_name}__{execution_id}"
    write_json(repair_plan_to_dict(repair_plan), root / "reports" / "repair_plans" / f"{stem}.json")
    write_markdown(repair_plan_to_markdown(repair_plan), root / "reports" / "repair_plans" / f"{stem}.md")
    if transactions is not None:
        rows = repair_transaction_records_to_dict(transactions)
        write_json({"transactions": rows}, root / "reports" / "repair_transactions" / f"{stem}.json")
        write_markdown(repair_transaction_records_to_markdown(transactions), root / "reports" / "repair_transactions" / f"{stem}.md")


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
    parser.add_argument("--repair-pairs", action="store_true", help="Detect and apply executable paired Linear repairs.")
    parser.add_argument("--repair-plan-json", help="Path to an existing repair plan JSON.")
    parser.add_argument("--write-repair-plan-only", action="store_true", help="Write the repair plan and stop before pruning.")
    parser.add_argument("--strict-repairs", action="store_true", help="Stop paired repair execution after the first rejected transaction.")
    parser.add_argument("--smoke-test-before", action="store_true", help="Run a forward smoke test before pruning.")
    parser.add_argument("--smoke-test-after", action="store_true", help="Run a forward smoke test after pruning.")
    parser.add_argument("--smoke-test-device", choices=["cpu", "cuda", "auto"], default="cpu", help="Device for forward smoke tests.")
    parser.add_argument("--reason", default=None, help="Optional action reason.")
    parser.add_argument("--verbose", action="store_true", help="Print execution summary.")
    return parser.parse_args()


def _load_or_build_plan(args: argparse.Namespace, root: Path, safe_name: str, config: dict, graph: DependencyGraph):
    if args.plan_json:
        return pruning_plan_from_dict(_load_json(Path(args.plan_json)))

    action = _construct_action(args, graph)
    correspondence = shape = validation = None
    if args.use_evidence:
        correspondence, shape, validation = _load_evidence(root, safe_name, config["name"])
    return simulate_pruning_action(graph, action, correspondence, shape, validation)


def _execute_repair_mode(args, root: Path, safe_name: str, config: dict, source_dir: Path, graph: DependencyGraph, plan, execution_id: str, output_dir: Path):
    if args.repair_plan_json:
        repair_plan = repair_plan_from_dict(_load_json(Path(args.repair_plan_json)))
    else:
        repair_plan = detect_linear_repair_plan(plan, graph, allow_ambiguous=args.allow_ambiguous)
    _write_repair_reports(root, safe_name, execution_id, repair_plan)

    if args.write_repair_plan_only:
        return None, repair_plan, [], {}

    if repair_plan.status == "rejected":
        raise ValueError("No executable paired Linear repairs were detected.")
    if repair_plan.status == "ambiguous" and not args.allow_ambiguous:
        raise ValueError("Repair plan is ambiguous. Re-run with --allow-ambiguous or inspect the repair plan.")

    model = load_model(config, source=source_dir)
    tokenizer_or_processor = _load_tokenizer_safely(config, source_dir)
    smoke_results = {}
    stem = f"{safe_name}__{execution_id}"
    smoke_config = dict(config)
    smoke_config["model_dir"] = str(source_dir)
    if args.smoke_test_before:
        before_smoke = run_forward_smoke_test(model, smoke_config, tokenizer_or_processor, device=args.smoke_test_device)
        smoke_results["before"] = forward_smoke_result_to_dict(before_smoke)
        _write_smoke_report(root, stem, "before", before_smoke)

    before_summary = summarize_torch_model(model, config["name"], config)
    transactions = apply_repair_plan(model, repair_plan, dry_run=args.dry_run, strict=args.strict_repairs)
    after_summary = summarize_torch_model(model, config["name"], config)
    diff_summary = compute_structural_diff(before_summary, after_summary)
    applied = [record for record in transactions if record.status == "applied"]
    rejected = [record for record in transactions if record.status == "rejected"]
    skipped = [record for record in transactions if record.status == "skipped"]

    if args.smoke_test_after:
        after_config = dict(config)
        after_config["model_dir"] = str(output_dir)
        after_smoke = run_forward_smoke_test(model, after_config, tokenizer_or_processor, device=args.smoke_test_device)
        smoke_results["after"] = forward_smoke_result_to_dict(after_smoke)
        _write_smoke_report(root, stem, "after", after_smoke)

    if args.dry_run:
        status = "success" if skipped and not rejected else "rejected" if rejected else "success"
    else:
        status = "success" if applied and not rejected else "partial" if applied else "rejected"
        if smoke_results.get("after", {}).get("status") == "failed" and applied:
            status = "partial"

    if not args.dry_run and applied and hasattr(model, "save_pretrained"):
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        if tokenizer_or_processor is not None and hasattr(tokenizer_or_processor, "save_pretrained"):
            tokenizer_or_processor.save_pretrained(output_dir)

    report = PruningExecutionReport(
        execution_id=execution_id,
        model_name=config["name"],
        source_model_dir=str(source_dir),
        output_model_dir=str(output_dir),
        action_id=plan.action.action_id,
        plan_id=plan.plan_id,
        status=status,
        before_summary=before_summary,
        after_summary=after_summary,
        diff_summary=diff_summary,
        metadata={
            "dry_run": args.dry_run,
            "only_target": args.only_target,
            "repair_pairs": True,
            "repair_plan": repair_plan_to_dict(repair_plan),
            "repair_transactions": repair_transaction_records_to_dict(transactions),
            "forward_smoke_tests": smoke_results,
        },
    )
    _write_repair_reports(root, safe_name, execution_id, repair_plan, transactions)
    return report, repair_plan, transactions, smoke_results


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
        plan = _load_or_build_plan(args, root, safe_name, config, graph)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if plan.status == "rejected":
        print("[rejected] Pruning plan is rejected; refusing execution.", file=sys.stderr)
        return 1
    if plan.status == "ambiguous" and not args.allow_ambiguous and not args.dry_run and not args.write_repair_plan_only:
        print("[rejected] Pruning plan is ambiguous. Re-run with --allow-ambiguous to execute Linear-only surgery.", file=sys.stderr)
        return 1

    execution_id = f"{_safe_stem(plan.action.action_id)}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = root / "artifacts" / "pruned_models" / safe_name / execution_id

    try:
        if args.repair_pairs or args.repair_plan_json:
            report, repair_plan, transactions, _ = _execute_repair_mode(args, root, safe_name, config, source_dir, graph, plan, execution_id, output_dir)
            if args.write_repair_plan_only:
                if args.verbose:
                    print(f"[repair-plan] {repair_plan.repair_plan_id}")
                    print(f"  status: {repair_plan.status}")
                    print(f"  repairs: {len(repair_plan.repair_specs)}")
                return 0 if repair_plan.status in {"executable", "partial", "ambiguous"} else 1
        else:
            model = load_model(config, source=source_dir)
            tokenizer_or_processor = _load_tokenizer_safely(config, source_dir)
            smoke_results = {}
            stem = f"{safe_name}__{execution_id}"
            if args.smoke_test_before:
                before_smoke = run_forward_smoke_test(model, {**config, "model_dir": str(source_dir)}, tokenizer_or_processor, device=args.smoke_test_device)
                smoke_results["before"] = forward_smoke_result_to_dict(before_smoke)
                _write_smoke_report(root, stem, "before", before_smoke)
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
            if args.smoke_test_after:
                after_smoke = run_forward_smoke_test(model, {**config, "model_dir": str(output_dir)}, tokenizer_or_processor, device=args.smoke_test_device)
                smoke_results["after"] = forward_smoke_result_to_dict(after_smoke)
                _write_smoke_report(root, stem, "after", after_smoke)
                if after_smoke.status == "failed" and report.status == "success":
                    report.status = "partial"
            report.metadata["forward_smoke_tests"] = smoke_results
    except Exception as exc:
        print(f"[error] execution failed: {exc}", file=sys.stderr)
        return 1

    _save_standard_reports(root, safe_name, execution_id, report, source_dir, output_dir)

    if args.verbose:
        print(f"[execution] {execution_id}")
        print(f"  status: {report.status}")
        print(f"  dry_run: {args.dry_run}")
        print(f"  output: {output_dir}")
        if args.repair_pairs or args.repair_plan_json:
            print(f"  repair_transactions: {len(transactions)}")
        else:
            print(f"  applied: {len(report.applied_records)}")
            print(f"  skipped: {len(report.skipped_records)}")
            print(f"  rejected: {len(report.rejected_records)}")

    return 0 if report.status in {"success", "partial"} or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
