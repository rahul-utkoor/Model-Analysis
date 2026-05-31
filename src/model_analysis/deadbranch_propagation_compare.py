"""Cross-model summaries for static deadbranch propagation reports."""

from __future__ import annotations

from typing import Any


def compare_deadbranch_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for report in reports:
        summary = report.get("summary", {})
        rows.append({"model_name": report.get("model_name"), **summary})
    return {
        "num_models": len(rows),
        "models": rows,
        "summary": {
            "total_pairs": sum(row.get("total_pairs", 0) for row in rows),
            "ffn_pairs": sum(row.get("ffn_pairs", 0) for row in rows),
            "attention_value_pairs": sum(row.get("attention_value_pairs", 0) for row in rows),
            "query_key_blocked_pairs": sum(row.get("query_key_blocked_pairs", 0) for row in rows),
        },
    }


def deadbranch_compare_to_markdown(data: dict[str, Any]) -> str:
    columns = ["model_name", "total_pairs", "ffn_pairs", "attention_value_pairs", "query_key_blocked_pairs", "sparsegpt_alignment_status"]
    rows = data.get("models", [])
    lines = ["# Deadbranch Propagation Comparison", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.extend(["", "This is static analysis/reporting only.", ""])
    return "\n".join(lines)
