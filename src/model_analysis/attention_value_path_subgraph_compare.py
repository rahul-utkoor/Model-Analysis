"""Cross-model summaries for attention value-path artifact reports."""

from __future__ import annotations

from typing import Any


def compare_attention_value_path_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    models = [
        {
            "model_name": report.get("model_name"),
            **{key: report.get(key, 0) for key in ("total_paths", "exported", "skipped", "failed", "seedable", "partial", "blocked", "unknown")},
        }
        for report in reports
    ]
    return {
        "num_models": len(models),
        "models": models,
        "summary": {key: sum(model.get(key, 0) for model in models) for key in ("total_paths", "exported", "skipped", "failed", "seedable", "partial", "blocked", "unknown")},
    }


def attention_value_path_compare_to_markdown(data: dict[str, Any]) -> str:
    columns = ["model_name", "total_paths", "exported", "skipped", "failed", "seedable", "partial", "blocked", "unknown"]
    lines = ["# Attention Value-Path Subgraph Comparison", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in data.get("models", []):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.extend(["", "This is static artifact/evidence generation only.", ""])
    return "\n".join(lines)
