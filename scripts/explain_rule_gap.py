#!/usr/bin/env python
"""Explain one model's rule-gap diagnosis."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    root = get_project_root()
    try:
        model = get_model_config(args.model)["hf_id"]
        path = root / "reports" / "rule_gap_diagnosis" / f"{safe_model_name(model)}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"# Rule-Gap Diagnosis: {data.get('model_name')}")
    print(f"family: {data.get('detected_model_family')}")
    print(f"conclusion: {data.get('conclusion')}")
    print()
    for gap in data.get("gaps", []):
        print(f"- {gap.get('gap_type')} [{gap.get('severity')}] stage={gap.get('affected_stage')} count={gap.get('affected_count')}")
        print(f"  {gap.get('explanation')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
