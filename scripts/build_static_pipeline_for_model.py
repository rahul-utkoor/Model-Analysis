#!/usr/bin/env python
"""Build/check the static analysis pipeline status for one model."""

from __future__ import annotations

import argparse
import sys

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.static_pipeline_orchestrator import build_static_pipeline_for_model
from model_analysis.static_pipeline_status import write_model_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--build-missing-analysis", action="store_true")
    parser.add_argument("--build-layer-packs", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
        status = build_static_pipeline_for_model(
            root=root,
            model_name=args.model,
            build_missing_analysis=args.build_missing_analysis,
            build_layer_packs=args.build_layer_packs,
            strict=args.strict,
            verbose=args.verbose,
        )
        safe = safe_model_name(config["hf_id"])
        out = root / "reports" / "static_pipeline_status"
        write_model_status(status, out / f"{safe}.json", out / f"{safe}.md")
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if args.verbose:
        summary = status.get("summary", {})
        print(f"[static-pipeline-status] {status.get('model_name')}: {status.get('final_status')}")
        print(f"  completed/skipped/failed: {summary.get('completed_stages', 0)}/{summary.get('skipped_stages', 0)}/{summary.get('failed_stages', 0)}")
        print(f"  missing artifacts: {len(summary.get('missing_artifacts', []))}")
        print(f"  status: {out / f'{safe}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
