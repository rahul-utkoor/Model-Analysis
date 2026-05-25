#!/usr/bin/env python
"""Diagnose cross-model static-analysis rule gaps."""

from __future__ import annotations

import argparse
import json

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config, list_models
from model_analysis.rule_gap_diagnosis import compare_rule_gap_diagnoses, diagnose_rule_gaps_for_model, diagnosis_to_dict, write_rule_gap_diagnosis
from model_analysis.rule_gap_diagnosis_text import rule_gap_compare_to_markdown, rule_gap_diagnosis_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _models(value: str) -> list[str]:
    if value == "all":
        return [get_model_config(name)["hf_id"] for name in list_models()]
    return [get_model_config(item.strip())["hf_id"] for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    root = get_project_root()
    out = root / "reports" / "rule_gap_diagnosis"
    diagnoses = []
    for model in _models(args.models):
        diagnosis = diagnose_rule_gaps_for_model(root, model)
        data = diagnosis_to_dict(diagnosis)
        safe = safe_model_name(model)
        write_rule_gap_diagnosis(data, out / f"{safe}.json")
        (out / f"{safe}.md").write_text(rule_gap_diagnosis_to_markdown(data), encoding="utf-8")
        diagnoses.append(data)
        if args.verbose:
            print(f"[rule-gap] {model}: family={data['detected_model_family']} gaps={len(data['gaps'])}")
    comparison = compare_rule_gap_diagnoses(diagnoses)
    cmp_out = root / "reports" / "rule_gap_diagnosis_compare"
    cmp_out.mkdir(parents=True, exist_ok=True)
    (cmp_out / "index.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (cmp_out / "index.md").write_text(rule_gap_compare_to_markdown(comparison), encoding="utf-8")
    if args.verbose:
        print(f"[rule-gap] comparison={cmp_out / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
