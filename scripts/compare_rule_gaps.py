#!/usr/bin/env python
"""Compare generated rule-gap diagnosis reports."""

from __future__ import annotations

import argparse
import json

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.rule_gap_diagnosis import compare_rule_gap_diagnoses
from model_analysis.rule_gap_diagnosis_text import rule_gap_compare_to_markdown


def _models(value: str) -> list[str]:
    if value == "all":
        return [get_model_config(name)["hf_id"] for name in list_models()]
    return [get_model_config(item.strip())["hf_id"] for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    args = parser.parse_args()
    root = get_project_root()
    diagnoses = []
    for model in _models(args.models):
        path = root / "reports" / "rule_gap_diagnosis" / f"{safe_model_name(model)}.json"
        if path.exists():
            diagnoses.append(json.loads(path.read_text(encoding="utf-8")))
    comparison = compare_rule_gap_diagnoses(diagnoses)
    out = root / "reports" / "rule_gap_diagnosis_compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (out / "index.md").write_text(rule_gap_compare_to_markdown(comparison), encoding="utf-8")
    print(f"[rule-gap-compare] models={len(diagnoses)}")
    print(f"[rule-gap-compare] report={out / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
