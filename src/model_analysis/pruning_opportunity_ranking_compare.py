"""Comparison helpers for pruning opportunity rankings."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _model_name(report: dict[str, Any]) -> str:
    return str(report.get("model_name") or "unknown")


def _counts(report: dict[str, Any], key: str) -> Counter:
    return Counter(item.get(key, "unknown") for item in report.get("candidates", []))


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


def compare_pruning_opportunity_rankings(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_models": len(reports),
        "models": [_model_name(report) for report in reports],
        "pruning_class_matrix": _matrix(reports, lambda report: _counts(report, "pruning_class")),
        "candidate_kind_matrix": _matrix(reports, lambda report: _counts(report, "candidate_kind")),
        "semantic_category_matrix": _matrix(reports, lambda report: _counts(report, "semantic_category")),
        "summary": {
            "total_candidates": sum(len(report.get("candidates", [])) for report in reports),
            "total_safe_candidates": sum(report.get("summary", {}).get("safe_candidates", 0) for report in reports),
            "total_constrained_candidates": sum(report.get("summary", {}).get("constrained_candidates", 0) for report in reports),
            "total_blocked_candidates": sum(report.get("summary", {}).get("blocked_candidates", 0) for report in reports),
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
            "# Pruning Opportunity Ranking Comparison",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Total candidates: `{comparison.get('summary', {}).get('total_candidates', 0)}`",
            f"- Safe candidates: `{comparison.get('summary', {}).get('total_safe_candidates', 0)}`",
            "",
            "## Pruning Classes",
            "",
            table(comparison.get("pruning_class_matrix", {})),
            "",
            "## Candidate Kinds",
            "",
            table(comparison.get("candidate_kind_matrix", {})),
            "",
            "## Semantic Categories",
            "",
            table(comparison.get("semantic_category_matrix", {})),
            "",
            "## Interpretation",
            "",
            "This comparison summarizes static pruning opportunity rankings. It does not execute pruning or modify models.",
            "",
        ]
    )

