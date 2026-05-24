"""Comparison helpers for Region Pruning Semantics reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _model_name(report: dict[str, Any]) -> str:
    return str(report.get("model_name") or report.get("metadata", {}).get("model_name") or "unknown")


def _counts(report: dict[str, Any], key: str) -> Counter:
    return Counter(item.get(key, "unknown") for item in report.get("regions", []))


def _nested_counts(report: dict[str, Any], list_key: str, item_key: str) -> Counter:
    counts: Counter = Counter()
    for region in report.get("regions", []):
        for item in region.get(list_key, []):
            counts[item.get(item_key, "unknown")] += 1
    return counts


def _matrix(reports: list[dict[str, Any]], counter_fn) -> dict[str, dict[str, int]]:
    keys = sorted({_model_name(report) for report in reports})
    observed = sorted({key for report in reports for key in counter_fn(report)})
    out = {key: {} for key in observed}
    for report in reports:
        model = _model_name(report)
        counts = counter_fn(report)
        for key in observed:
            out[key][model] = counts.get(key, 0)
    for row in out.values():
        for model in keys:
            row.setdefault(model, 0)
    return out


def compare_region_pruning_semantics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    models = [_model_name(report) for report in reports]
    pattern_sets = {
        _model_name(report): {f"{item.get('region_type')}::{item.get('pruning_role')}" for item in report.get("regions", [])}
        for report in reports
    }
    common = sorted(set.intersection(*pattern_sets.values())) if pattern_sets else []
    specific = {
        model: sorted(patterns - set().union(*(other for other_model, other in pattern_sets.items() if other_model != model)))
        for model, patterns in pattern_sets.items()
    }
    return {
        "num_models": len(reports),
        "models": models,
        "pruning_role_matrix": _matrix(reports, lambda report: _counts(report, "pruning_role")),
        "region_type_matrix": _matrix(reports, lambda report: _counts(report, "region_type")),
        "blocker_type_matrix": _matrix(reports, lambda report: _nested_counts(report, "blockers", "blocker_type")),
        "repair_obligation_matrix": _matrix(reports, lambda report: _nested_counts(report, "repair_obligations", "obligation_type")),
        "dimension_status_matrix": _matrix(reports, lambda report: _nested_counts(report, "dimensions", "status")),
        "common_region_semantics": common,
        "model_specific_region_semantics": specific,
        "summary": {
            "total_regions": sum(len(report.get("regions", [])) for report in reports),
            "total_blockers": sum(sum(1 for region in report.get("regions", []) for _ in region.get("blockers", [])) for report in reports),
            "total_repairs": sum(sum(1 for region in report.get("regions", []) for _ in region.get("repair_obligations", [])) for report in reports),
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
            "# Region Pruning Semantics Comparison",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Total regions: `{comparison.get('summary', {}).get('total_regions', 0)}`",
            "",
            "## Pruning Roles",
            "",
            table(comparison.get("pruning_role_matrix", {})),
            "",
            "## Region Types",
            "",
            table(comparison.get("region_type_matrix", {})),
            "",
            "## Blockers",
            "",
            table(comparison.get("blocker_type_matrix", {})),
            "",
            "## Repair Obligations",
            "",
            table(comparison.get("repair_obligation_matrix", {})),
            "",
            "## Interpretation",
            "",
            "This comparison summarizes static region-level pruning semantics. It does not execute pruning or modify models.",
            "",
        ]
    )
