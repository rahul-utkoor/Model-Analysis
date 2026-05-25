"""Comparison helpers for pruning plan validation reports."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _model_name(report: dict[str, Any]) -> str:
    return str(report.get("model_name") or "unknown")


def _counts(report: dict[str, Any], key: str) -> Counter:
    return Counter(item.get(key, "unknown") for item in report.get("validations", []))


def _matrix(reports: list[dict[str, Any]], counter_fn) -> dict[str, dict[str, int]]:
    models = [_model_name(report) for report in reports]
    observed = sorted({key for report in reports for key in counter_fn(report)})
    matrix = {key: {} for key in observed}
    for report in reports:
        model = _model_name(report)
        counts = counter_fn(report)
        for key in observed:
            matrix[key][model] = counts.get(key, 0)
    for row in matrix.values():
        for model in models:
            row.setdefault(model, 0)
    return matrix


def compare_pruning_plan_validations(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_models": len(reports),
        "models": [_model_name(report) for report in reports],
        "validation_status_matrix": _matrix(reports, lambda report: _counts(report, "validation_status")),
        "plan_kind_matrix": _matrix(reports, lambda report: _counts(report, "plan_kind")),
        "summary": {
            "total_validations": sum(len(report.get("validations", [])) for report in reports),
            "total_valid": sum(report.get("summary", {}).get("valid_plans", 0) for report in reports),
            "total_warning": sum(report.get("summary", {}).get("warning_plans", 0) for report in reports),
            "total_invalid": sum(report.get("summary", {}).get("invalid_plans", 0) for report in reports),
            "total_unknown": sum(report.get("summary", {}).get("unknown_plans", 0) for report in reports),
        },
    }


def comparison_to_markdown(comparison: dict[str, Any]) -> str:
    def table(matrix: dict[str, dict[str, int]]) -> str:
        if not matrix:
            return "_None._"
        models = comparison.get("models", [])
        lines = ["| item | " + " | ".join(models) + " |", "|---|" + "|".join("---" for _ in models) + "|"]
        for item, row in sorted(matrix.items()):
            lines.append("| " + item + " | " + " | ".join(str(row.get(model, 0)) for model in models) + " |")
        return "\n".join(lines)

    summary = comparison.get("summary", {})
    return "\n".join(
        [
            "# Pruning Plan Validation Comparison",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Total validations: `{summary.get('total_validations', 0)}`",
            f"- Valid: `{summary.get('total_valid', 0)}`",
            f"- Warning: `{summary.get('total_warning', 0)}`",
            f"- Invalid: `{summary.get('total_invalid', 0)}`",
            f"- Unknown: `{summary.get('total_unknown', 0)}`",
            "",
            "## Validation Status",
            "",
            table(comparison.get("validation_status_matrix", {})),
            "",
            "## Plan Kinds",
            "",
            table(comparison.get("plan_kind_matrix", {})),
            "",
            "## Interpretation",
            "",
            "This comparison summarizes static pruning-plan validation reports. It does not execute pruning or modify models.",
            "",
        ]
    )
