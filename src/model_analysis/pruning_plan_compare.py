"""Comparison helpers for symbolic pruning plan sets."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _model_name(report: dict[str, Any]) -> str:
    return str(report.get("model_name") or "unknown")


def _counts(report: dict[str, Any], key: str) -> Counter:
    return Counter(plan.get(key, "unknown") for plan in report.get("plans", []))


def _matrix(reports: list[dict[str, Any]], counter_fn) -> dict[str, dict[str, int]]:
    models = [_model_name(report) for report in reports]
    observed = sorted({key for report in reports for key in counter_fn(report)})
    out = {key: {} for key in observed}
    for report in reports:
        model = _model_name(report)
        counts = counter_fn(report)
        for key in observed:
            out[key][model] = counts.get(key, 0)
    for row in out.values():
        for model in models:
            row.setdefault(model, 0)
    return out


def compare_pruning_plan_sets(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_models": len(reports),
        "models": [_model_name(report) for report in reports],
        "plan_status_matrix": _matrix(reports, lambda report: _counts(report, "plan_status")),
        "plan_kind_matrix": _matrix(reports, lambda report: _counts(report, "plan_kind")),
        "target_dimension_matrix": _matrix(reports, lambda report: _counts(report, "target_dimension")),
        "summary": {
            "total_plans": sum(len(report.get("plans", [])) for report in reports),
            "total_ready_symbolic": sum(report.get("summary", {}).get("ready_symbolic", 0) for report in reports),
            "total_incomplete": sum(report.get("summary", {}).get("incomplete", 0) for report in reports),
            "total_blocked": sum(report.get("summary", {}).get("blocked", 0) for report in reports),
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

    return "\n".join(
        [
            "# Pruning Plan Comparison",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Total plans: `{comparison.get('summary', {}).get('total_plans', 0)}`",
            f"- Ready symbolic: `{comparison.get('summary', {}).get('total_ready_symbolic', 0)}`",
            "",
            "## Plan Status",
            "",
            table(comparison.get("plan_status_matrix", {})),
            "",
            "## Plan Kinds",
            "",
            table(comparison.get("plan_kind_matrix", {})),
            "",
            "## Target Dimensions",
            "",
            table(comparison.get("target_dimension_matrix", {})),
            "",
            "## Interpretation",
            "",
            "This comparison summarizes symbolic pruning plan reports. It does not execute pruning or modify models.",
            "",
        ]
    )

