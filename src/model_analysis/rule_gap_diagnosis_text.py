"""Markdown rendering for rule-gap diagnosis reports."""

from __future__ import annotations

import json
from typing import Any

from model_analysis.rule_gap_diagnosis import diagnosis_to_dict


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def rule_gap_diagnosis_to_markdown(value: dict[str, Any]) -> str:
    data = diagnosis_to_dict(value)
    gaps = data.get("gaps", [])
    repairs = data.get("candidate_repairs", [])
    lines = [
        f"# Rule-Gap Diagnosis: {data.get('model_name')}",
        "",
        f"- Final status: `{data.get('final_status')}`",
        f"- Detected family: `{data.get('detected_model_family')}`",
        f"- Plans: `{data.get('plan_summary', {}).get('total_plans', 0)}`",
        f"- Ready/incomplete: `{data.get('plan_summary', {}).get('ready_symbolic', 0)}` / `{data.get('plan_summary', {}).get('incomplete', 0)}`",
        f"- Valid/invalid: `{data.get('validation_summary', {}).get('valid_plans', 0)}` / `{data.get('validation_summary', {}).get('invalid_plans', 0)}`",
        "",
        "## Gaps",
        "",
        _table(gaps, ["gap_type", "severity", "affected_stage", "affected_count", "explanation"]),
        "",
        "## Candidate repairs",
        "",
        _table(repairs, ["repair_type", "target_models", "expected_effect", "priority"]),
        "",
        "## Evidence summary",
        "",
        "```json",
        json.dumps(data.get("evidence_summary", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Conclusion",
        "",
        data.get("conclusion", ""),
        "",
        "This is static diagnosis/reporting only.",
        "",
    ]
    return "\n".join(lines)


def rule_gap_compare_to_markdown(data: dict[str, Any]) -> str:
    rows = []
    for item in data.get("diagnoses", []):
        rows.append(
            {
                "model": item.get("model_name"),
                "family": item.get("detected_model_family"),
                "gaps": len(item.get("gaps", [])),
                "plans": item.get("plan_summary", {}).get("total_plans", 0),
                "valid": item.get("validation_summary", {}).get("valid_plans", 0),
                "conclusion": item.get("conclusion", ""),
            }
        )
    return "\n".join(
        [
            "# Rule-Gap Diagnosis Comparison",
            "",
            "## Models",
            "",
            _table(rows, ["model", "family", "gaps", "plans", "valid", "conclusion"]),
            "",
            "## Gap type counts",
            "",
            _table([{"gap_type": k, "count": v} for k, v in data.get("gap_type_counts", {}).items()], ["gap_type", "count"]),
            "",
            "Static diagnosis/reporting only.",
            "",
        ]
    )
