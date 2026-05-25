#!/usr/bin/env python
"""Explain one model's static pipeline status manifest."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
        safe = safe_model_name(config["hf_id"])
        path = root / "reports" / "static_pipeline_status" / f"{safe}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing status manifest: {path}")
        status = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"# Static Pipeline Status: {status.get('model_name')}")
    print(f"final_status: {status.get('final_status')}")
    print()
    print("## Stages")
    for stage in status.get("stages", []):
        print(f"- {stage.get('stage_name')}: {stage.get('status')}")
        if stage.get("missing_inputs"):
            print(f"  missing: {', '.join(stage.get('missing_inputs', []))}")
        if stage.get("error"):
            print(f"  error: {stage.get('error')}")
        if stage.get("command_hint"):
            print(f"  next: {stage.get('command_hint')}")
    print()
    print("## Missing artifacts")
    missing = status.get("summary", {}).get("missing_artifacts", [])
    if missing:
        for item in missing:
            print(f"- {item}")
    else:
        print("- none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
