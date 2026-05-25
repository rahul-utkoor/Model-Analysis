"""Cross-model summaries for full static analysis reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir, safe_model_name
from model_analysis.reporting import write_json, write_markdown


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_cross_model_analysis_report(root: Path, model_names: list[str], report_root: Path) -> dict[str, Any]:
    models = []
    missing = []
    for model in model_names:
        safe = safe_model_name(model)
        path = report_root / safe / "index.json"
        report = _load(path)
        if not report:
            missing.append({"model": model, "path": str(path)})
            models.append({"model_name": model, "status": "missing_report", "missing_artifacts": [str(path)]})
            continue
        summary = report.get("model_summary", {})
        ranking = summary.get("ranking", {})
        validations = summary.get("plan_validation", {})
        op_counts = summary.get("op_semantic_counts", {})
        region_counts = summary.get("region_semantic_counts", {})
        model_row = {
            "model_name": model,
            "status": "ok" if not report.get("missing_artifacts") else "partial",
            "missing_artifacts": report.get("missing_artifacts", []),
            "layers": summary.get("layers_generated", 0),
            "safe": ranking.get("safe", 0),
            "constrained": ranking.get("constrained", 0),
            "blocked": ranking.get("blocked", 0),
            "auxiliary": ranking.get("auxiliary", 0),
            "unknown": ranking.get("unknown", 0),
            "plans": summary.get("plans", {}).get("total_plans", 0),
            "valid_plans": validations.get("valid", 0),
            "ffn_safe_plans": validations.get("valid", 0),
            "attention_constrained": ranking.get("attention_constrained_candidates", 0),
            "residual_blocked": ranking.get("residual_blocked_candidates", 0),
            "layernorm_blocked": ranking.get("layernorm_blocked_candidates", 0),
            "unknown_candidates": ranking.get("unknown", 0),
            "parameterized_projections": op_counts.get("parameterized_linear_matmul", 0),
            "attention_contractions": op_counts.get("attention_score_matmul", 0) + op_counts.get("attention_context_matmul", 0),
            "residuals": region_counts.get("residual_merge", 0),
            "layernorms": region_counts.get("layer_norm", 0),
            "unknown_ops": op_counts.get("unknown", 0),
        }
        models.append(model_row)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": model_names,
        "model_summaries": models,
        "opportunity_comparison": [
            {
                "model": item.get("model_name"),
                "ffn_safe_plans": item.get("ffn_safe_plans", 0),
                "attention_constrained": item.get("attention_constrained", 0),
                "residual_blocked": item.get("residual_blocked", 0),
                "layernorm_blocked": item.get("layernorm_blocked", 0),
                "unknown_candidates": item.get("unknown_candidates", 0),
            }
            for item in models
        ],
        "semantic_coverage": [
            {
                "model": item.get("model_name"),
                "parameterized_projections": item.get("parameterized_projections", 0),
                "attention_contractions": item.get("attention_contractions", 0),
                "residuals": item.get("residuals", 0),
                "layernorms": item.get("layernorms", 0),
                "unknown_ops": item.get("unknown_ops", 0),
            }
            for item in models
        ],
        "missing_artifacts": missing,
        "conclusions": [
            "Cross-model report aggregates generated full-model reports only.",
            "BERT-like FFN safe plans generalize when model reports expose validated feed-forward plans.",
            "Missing reports or high unknown counts identify models needing additional structural rules.",
        ],
    }
    del root
    return report


def write_cross_model_analysis_report(report: dict[str, Any], report_root: Path, markdown_fn) -> None:
    out = report_root / "cross_model"
    ensure_dir(out)
    write_json(report, out / "index.json")
    write_markdown(markdown_fn(report), out / "index.md")
    write_markdown(markdown_fn(report, section="models"), out / "model_summary_table.md")
    write_json(report.get("opportunity_comparison", []), out / "opportunity_comparison.json")
    write_markdown(markdown_fn(report, section="opportunity"), out / "opportunity_comparison.md")
    write_json(report.get("semantic_coverage", []), out / "plan_validation_comparison.json")
    write_markdown(markdown_fn(report, section="validation"), out / "plan_validation_comparison.md")

