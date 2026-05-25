"""Markdown rendering for cross-model analysis reports."""

from __future__ import annotations

from typing import Any


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def cross_model_report_to_markdown(report: dict[str, Any], section: str | None = None) -> str:
    models = report.get("model_summaries", [])
    model_columns = ["model_name", "status", "layers", "safe", "constrained", "blocked", "auxiliary", "unknown", "plans", "valid_plans"]
    opportunity_columns = ["model", "ffn_safe_plans", "attention_constrained", "residual_blocked", "layernorm_blocked", "unknown_candidates"]
    coverage_columns = ["model", "parameterized_projections", "attention_contractions", "residuals", "layernorms", "unknown_ops"]
    if section == "models":
        return _table(models, model_columns) + "\n"
    if section == "opportunity":
        return _table(report.get("opportunity_comparison", []), opportunity_columns) + "\n"
    if section == "validation":
        return _table(report.get("semantic_coverage", []), coverage_columns) + "\n"
    lines = [
        "# Cross-Model Static Analysis Summary",
        "",
        "## Models analyzed",
        "",
        _table(models, model_columns),
        "",
        "## Opportunity comparison",
        "",
        _table(report.get("opportunity_comparison", []), opportunity_columns),
        "",
        "## Semantic coverage",
        "",
        _table(report.get("semantic_coverage", []), coverage_columns),
        "",
        "## Generalization notes",
        "",
        "- Models with validated FFN plans expose the current clean pruning opportunity.",
        "- Models with missing reports are skipped for detailed comparison but kept visible.",
        "- High unknown semantic counts indicate where model-specific rules need improvement.",
        "",
        "## Research conclusion",
        "",
        "This cross-model report compares static analysis artifacts only. It does not execute pruning or modify models.",
        "",
    ]
    for conclusion in report.get("conclusions", []):
        lines.append(f"- {conclusion}")
    lines.append("")
    return "\n".join(lines)

