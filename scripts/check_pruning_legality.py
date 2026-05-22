#!/usr/bin/env python
"""Run static pruning legality checks over Dimension IR."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from model_analysis.ir_analysis import (
    check_pruning_legality,
    legality_check_result_to_dict,
    legality_check_result_to_markdown,
    make_symbolic_pruning_request,
    propagation_slice_to_dict,
    propagation_slice_to_markdown,
    repair_set_to_markdown,
    write_legality_check_json,
)
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check static pruning legality for a Dimension IR variable.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dimension-var", required=True)
    parser.add_argument("--indices")
    parser.add_argument("--count", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--strategy", choices=["symbolic", "explicit_indices", "first_n", "last_n", "fraction"])
    parser.add_argument("--reason")
    parser.add_argument("--fail-on-ambiguous", action="store_true")
    parser.add_argument("--fail-on-rejected", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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
    ir_path = root / "reports" / "dimension_ir" / f"{safe_name}.json"
    if not ir_path.exists():
        print(f"[missing] Dimension IR missing. Run: python scripts/build_dimension_ir.py --model {config['name']}", file=sys.stderr)
        return 1
    try:
        request = make_symbolic_pruning_request(
            model_name=config["name"],
            dimension_var_id=args.dimension_var,
            indices=_parse_indices(args.indices),
            count=args.count,
            fraction=args.fraction,
            strategy=args.strategy,
            reason=args.reason,
        )
        result = check_pruning_legality(_load_json(ir_path), request)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    request_stem = request.request_id.replace("/", "__").replace(":", "_").replace(" ", "_")
    stem = f"{safe_name}__{request_stem}"
    write_legality_check_json(result, root / "reports" / "legality_checks" / f"{stem}.json")
    write_markdown(legality_check_result_to_markdown(result), root / "reports" / "legality_checks" / f"{stem}.md")
    write_json(propagation_slice_to_dict(result.forward_slice), root / "reports" / "propagation_slices" / f"{stem}__forward.json")
    write_markdown(propagation_slice_to_markdown(result.forward_slice), root / "reports" / "propagation_slices" / f"{stem}__forward.md")
    write_json(propagation_slice_to_dict(result.backward_slice), root / "reports" / "propagation_slices" / f"{stem}__backward.json")
    write_markdown(propagation_slice_to_markdown(result.backward_slice), root / "reports" / "propagation_slices" / f"{stem}__backward.md")
    repairs = [asdict(item) for item in result.minimal_repair_set]
    write_json({"request_id": request.request_id, "repairs": repairs}, root / "reports" / "repair_sets" / f"{stem}.json")
    write_markdown(repair_set_to_markdown(request.request_id, result.minimal_repair_set), root / "reports" / "repair_sets" / f"{stem}.md")
    if args.verbose:
        print(f"[legality] {request.request_id}")
        print(f"  status: {result.status}")
        print(f"  repairs: {len(result.minimal_repair_set)}")
        print(f"  unresolved: {len(result.unresolved_items)}")
        print(f"  blockers: {len(result.blocking_reasons)}")
    if result.status == "rejected":
        return 1 if args.fail_on_rejected or True else 0
    if result.status == "ambiguous" and args.fail_on_ambiguous:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
