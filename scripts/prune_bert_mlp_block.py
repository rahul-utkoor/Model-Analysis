#!/usr/bin/env python
"""Prune a BERT-style MLP intermediate dimension."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_analysis.bert_mlp_pruning import (
    bert_mlp_pruning_report_to_dict,
    bert_mlp_pruning_report_to_markdown,
    execute_bert_mlp_pruning,
    get_bert_mlp_block_target,
    make_bert_mlp_prune_spec,
)
from model_analysis.forward_validation import forward_smoke_result_to_markdown
from model_analysis.hf_utils import load_model, load_tokenizer_or_processor
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.pruning_diff import pruning_diff_to_markdown
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown
from model_analysis.rollback import create_rollback_manifest, rollback_manifest_to_markdown, write_rollback_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute BERT MLP block-level intermediate pruning.")
    parser.add_argument("--model", required=True, help="Configured model name or Hugging Face ID.")
    parser.add_argument("--layer", required=True, type=int, help="BERT encoder layer index.")
    parser.add_argument("--indices", help="Comma-separated intermediate-dimension indices.")
    parser.add_argument("--count", type=int, help="Number of intermediate features to prune.")
    parser.add_argument("--fraction", type=float, help="Fraction of intermediate features to prune.")
    parser.add_argument("--strategy", choices=["first_n", "last_n", "every_other"], default="first_n")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test-before", action="store_true")
    parser.add_argument("--smoke-test-after", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--output-dir", help="Optional output model directory.")
    parser.add_argument("--reason", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_tokenizer_safely(config: dict, source_dir: Path):
    try:
        return load_tokenizer_or_processor(config, source=source_dir)
    except Exception:
        return None


def _write_smoke_report(root: Path, safe_name: str, layer: int, execution_id: str, phase: str, smoke: dict | None) -> None:
    if not smoke:
        return
    stem = f"{safe_name}__layer_{layer}__{execution_id}__{phase}"
    write_json(smoke, root / "reports" / "block_validation" / f"{stem}.json")
    from model_analysis.forward_validation import ForwardSmokeResult

    result = ForwardSmokeResult(**smoke)
    write_markdown(forward_smoke_result_to_markdown(result), root / "reports" / "block_validation" / f"{stem}.md")


def main() -> int:
    args = parse_args()
    try:
        config = get_model_config(args.model)
        root = get_project_root()
        safe_name = safe_model_name(config["hf_id"])
        source_dir = root / config["local_dir"]
        if not source_dir.exists():
            print(f"[missing] Local model missing. Run: python scripts/download_models.py --model {config['name']}", file=sys.stderr)
            return 1
        model = load_model(config, source=source_dir)
        tokenizer_or_processor = _load_tokenizer_safely(config, source_dir)
        target = get_bert_mlp_block_target(model, config["name"], args.layer)
        spec = make_bert_mlp_prune_spec(
            target,
            indices=_parse_indices(args.indices),
            count=args.count,
            fraction=args.fraction,
            strategy=args.strategy,
            reason=args.reason,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        execution_id = f"bert_mlp_layer_{args.layer}_{timestamp}"
        output_dir = Path(args.output_dir) if args.output_dir else root / "artifacts" / "pruned_models" / safe_name / execution_id
        report = execute_bert_mlp_pruning(
            model=model,
            model_name=config["name"],
            source_model_dir=source_dir,
            output_model_dir=output_dir,
            spec=spec,
            tokenizer_or_processor=tokenizer_or_processor,
            model_config={**config, "model_dir": str(source_dir)},
            dry_run=args.dry_run,
            smoke_test_before=args.smoke_test_before,
            smoke_test_after=args.smoke_test_after,
            device=args.device,
        )
        report.execution_id = execution_id
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    stem = f"{safe_name}__layer_{args.layer}__{execution_id}"
    report_json = root / "reports" / "block_pruning" / f"{stem}.json"
    report_md = root / "reports" / "block_pruning" / f"{stem}.md"
    diff_json = root / "reports" / "block_pruning_diffs" / f"{stem}.json"
    diff_md = root / "reports" / "block_pruning_diffs" / f"{stem}.md"
    rollback_json = root / "reports" / "rollback_manifests" / f"{safe_name}__bert_mlp_layer_{args.layer}__{execution_id}.json"
    rollback_md = root / "reports" / "rollback_manifests" / f"{safe_name}__bert_mlp_layer_{args.layer}__{execution_id}.md"

    if output_dir.exists():
        manifest = create_rollback_manifest(bert_mlp_pruning_report_to_dict(report), source_dir, output_dir, rollback_json)
        report.rollback_manifest_path = str(rollback_json)
        write_rollback_manifest(manifest, rollback_json)
        write_markdown(rollback_manifest_to_markdown(manifest), rollback_md)

    write_json(bert_mlp_pruning_report_to_dict(report), report_json)
    write_markdown(bert_mlp_pruning_report_to_markdown(report), report_md)
    if report.diff_summary is not None:
        write_json(report.diff_summary, diff_json)
        write_markdown(pruning_diff_to_markdown(report.diff_summary), diff_md)
    _write_smoke_report(root, safe_name, args.layer, execution_id, "before", report.before_forward_smoke)
    _write_smoke_report(root, safe_name, args.layer, execution_id, "after", report.after_forward_smoke)

    if args.verbose:
        print(f"[bert-mlp] {execution_id}")
        print(f"  status: {report.status}")
        print(f"  layer: {args.layer}")
        print(f"  indices: {spec.prune_indices}")
        print(f"  output: {output_dir}")
        if report.before_forward_smoke:
            print(f"  before_smoke: {report.before_forward_smoke.get('status')}")
        if report.after_forward_smoke:
            print(f"  after_smoke: {report.after_forward_smoke.get('status')}")

    if args.dry_run:
        return 0
    return 0 if report.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
