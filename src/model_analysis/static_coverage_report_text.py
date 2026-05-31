"""Markdown rendering for the cross-model static coverage study."""

from __future__ import annotations

from typing import Any


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def static_coverage_report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-Model Static Coverage Study",
        "",
        "## 1. Purpose",
        "",
        "This report measures how far the static pruning-analysis pipeline gets for each configured model. It distinguishes complete support from partial, skipped, or failed support and records where new model-specific semantics are needed.",
        "",
        "This is static analysis/reporting only. It does not choose pruning indices, modify models, execute pruning, download models, rewrite full ONNX models, train, or evaluate accuracy.",
        "",
        "## 2. Model status table",
        "",
        _table(
            report.get("model_status_table", []),
            ["model_name", "final_status", "completed_stages", "missing_artifacts", "safe_candidates", "plans", "valid_plans", "notes"],
        ),
        "",
        "## 3. Stage coverage table",
        "",
        _table(
            report.get("stage_coverage_table", []),
            [
                "model_name",
                "tensor_ir",
                "op_semantics",
                "region_tree",
                "dimension_ir",
                "region_semantics",
                "ranking",
                "plans",
                "validation",
                "deadbranch",
                "layer_packs",
                "full_report",
            ],
        ),
        "",
        "## 4. Opportunity coverage table",
        "",
        _table(
            report.get("opportunity_coverage_table", []),
            ["model_name", "safe", "constrained", "blocked", "auxiliary", "unknown", "mlp_safe", "generic_mlp_safe", "generic_mlp_constrained", "valid_plans", "deadbranch_pairs", "deadbranch_ffn_pairs", "deadbranch_value_pairs", "deadbranch_qk_blocked"],
        ),
        "",
        "## 5. Semantic coverage table",
        "",
        _table(
            report.get("semantic_coverage_table", []),
            [
                "model_name",
                "parameterized_matmul",
                "attention_score",
                "attention_context",
                "ffn_blocks",
                "generic_mlp_regions",
                "residuals",
                "layernorms",
                "unknown_ops",
            ],
        ),
        "",
        "## 6. Per-model summaries",
        "",
    ]
    for status in report.get("models", []):
        summary = status.get("summary", {})
        lines.extend(
            [
                f"### {status.get('model_name')}",
                "",
                f"- Final status: `{status.get('final_status')}`",
                f"- Completed/skipped/failed stages: `{summary.get('completed_stages', 0)}` / `{summary.get('skipped_stages', 0)}` / `{summary.get('failed_stages', 0)}`",
                f"- Missing artifacts: `{len(summary.get('missing_artifacts', []))}`",
                f"- Strongest recovered structure: `{strongest_structure(status)}`",
                f"- Next work: {next_work(status)}",
                "",
            ]
        )
    lines.extend(["## 7. Generalization conclusions", ""])
    for conclusion in report.get("conclusions", []):
        lines.append(f"- {conclusion}")
    lines.append("")
    return "\n".join(lines)


def strongest_structure(status: dict[str, Any]) -> str:
    artifacts = status.get("artifacts", {})
    report = artifacts.get("full_model_report", {})
    if report.get("valid_plans", 0):
        return "validated symbolic FFN plans"
    ranking = artifacts.get("ranking", {})
    if ranking.get("safe", 0):
        return "safe static ranking candidates"
    if ranking:
        return "ranking-level pruning classes"
    if artifacts.get("validation"):
        return "plan validation artifacts"
    if artifacts.get("plans"):
        return "symbolic plan artifacts"
    return "none detected"


def next_work(status: dict[str, Any]) -> str:
    if status.get("final_status") == "complete":
        return "use as a reference case and compare against other architectures."
    missing = status.get("summary", {}).get("missing_artifacts", [])
    if missing:
        return "build or provide missing static artifacts: " + ", ".join(missing[:3])
    return "inspect failed/skipped stages and add model-specific semantics where needed."
