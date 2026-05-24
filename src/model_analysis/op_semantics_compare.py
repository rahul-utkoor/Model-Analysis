"""Comparison helpers for Op Semantics IR reports."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _model_name(report: dict[str, Any]) -> str:
    return str(report.get("model_name") or "unknown")


def _counts(report: dict[str, Any], key: str) -> Counter:
    return Counter(op.get(key, "unknown") for op in report.get("ops", []))


def _effect_counts(report: dict[str, Any], key: str) -> Counter:
    counts: Counter = Counter()
    for op in report.get("ops", []):
        effect = op.get("pruning_effect", {})
        for item in effect.get(key, []):
            counts[item] += 1
    return counts


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


def compare_op_semantics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_models": len(reports),
        "models": [_model_name(report) for report in reports],
        "semantic_kind_matrix": _matrix(reports, lambda report: _counts(report, "semantic_kind")),
        "semantic_category_matrix": _matrix(reports, lambda report: _counts(report, "semantic_category")),
        "index_behavior_matrix": _matrix(reports, lambda report: _counts(report, "index_behavior")),
        "blocker_matrix": _matrix(reports, lambda report: _effect_counts(report, "blockers")),
        "repair_matrix": _matrix(reports, lambda report: _effect_counts(report, "required_repairs")),
        "summary": {
            "total_ops": sum(len(report.get("ops", [])) for report in reports),
            "total_unknown_ops": sum(report.get("summary", {}).get("unknown_ops", 0) for report in reports),
            "total_parameterized_ops": sum(report.get("summary", {}).get("parameterized_ops", 0) for report in reports),
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
            "# Op Semantics Comparison",
            "",
            f"- Models: `{comparison.get('num_models', 0)}`",
            f"- Total ops: `{comparison.get('summary', {}).get('total_ops', 0)}`",
            f"- Unknown ops: `{comparison.get('summary', {}).get('total_unknown_ops', 0)}`",
            "",
            "## Semantic Kinds",
            "",
            table(comparison.get("semantic_kind_matrix", {})),
            "",
            "## Semantic Categories",
            "",
            table(comparison.get("semantic_category_matrix", {})),
            "",
            "## Index Behavior",
            "",
            table(comparison.get("index_behavior_matrix", {})),
            "",
            "## Blockers",
            "",
            table(comparison.get("blocker_matrix", {})),
            "",
            "## Repairs",
            "",
            table(comparison.get("repair_matrix", {})),
            "",
            "## Interpretation",
            "",
            "This comparison summarizes static primitive-op pruning semantics. It does not execute pruning or modify models.",
            "",
        ]
    )

