"""Cross-model static pipeline coverage study."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_status(model_status: dict[str, Any], stage_name: str) -> str:
    for stage in model_status.get("stages", []):
        if stage.get("stage_name") == stage_name:
            return stage.get("status", "missing")
    return "missing"


def _artifact_counts(status: dict[str, Any]) -> dict[str, Any]:
    artifacts = status.get("artifacts", {})
    ranking = artifacts.get("ranking", {})
    validation = artifacts.get("validation", {})
    report = artifacts.get("full_model_report", {})
    return {
        "safe": ranking.get("safe", 0),
        "constrained": ranking.get("constrained", 0),
        "blocked": ranking.get("blocked", 0),
        "auxiliary": ranking.get("auxiliary", 0),
        "unknown": ranking.get("unknown", 0),
        "mlp_safe": ranking.get("mlp_safe", 0),
        "generic_mlp_safe": ranking.get("generic_mlp_safe", 0),
        "generic_mlp_constrained": ranking.get("generic_mlp_constrained", 0),
        "plans": artifacts.get("plans", {}).get("plans", 0),
        "valid_plans": validation.get("valid_plans", 0),
        "layers": report.get("layers", 0),
        "subgraphs": report.get("subgraphs", 0),
    }


def _semantic_counts(status: dict[str, Any], root: Path) -> dict[str, int]:
    model = status.get("model_name", "")
    safe = model.replace("/", "__")
    op_path = root / "reports" / "op_semantics" / f"{safe}.json"
    region_path = root / "reports" / "region_pruning_semantics" / f"{safe}.json"
    op = _load(op_path) or {}
    region = _load(region_path) or {}
    op_counts = op.get("summary", {}).get("semantic_kind_counts", {})
    region_counts = region.get("summary", {}).get("semantic_category_counts", {})
    return {
        "parameterized_matmul": op_counts.get("parameterized_linear_matmul", 0),
        "attention_score": op_counts.get("attention_score_matmul", 0),
        "attention_context": op_counts.get("attention_context_matmul", 0),
        "ffn_blocks": region_counts.get("feed_forward_block", 0),
        "generic_mlp_regions": region.get("summary", {}).get("generic_mlp_regions", 0),
        "residuals": region_counts.get("residual_merge", 0),
        "layernorms": region_counts.get("layer_norm", 0),
        "unknown_ops": op_counts.get("unknown", 0),
    }


def build_static_coverage_report(root: Path, statuses: list[dict[str, Any]]) -> dict[str, Any]:
    model_rows = []
    stage_rows = []
    opportunity_rows = []
    semantic_rows = []
    for status in statuses:
        model = status.get("model_name", "")
        summary = status.get("summary", {})
        counts = _artifact_counts(status)
        model_rows.append(
            {
                "model_name": model,
                "final_status": status.get("final_status", "skipped"),
                "completed_stages": summary.get("completed_stages", 0),
                "missing_artifacts": len(summary.get("missing_artifacts", [])),
                "safe_candidates": counts["safe"],
                "plans": counts["plans"],
                "valid_plans": counts["valid_plans"],
                "notes": "; ".join(status.get("notes", [])),
            }
        )
        stage_rows.append(
            {
                "model_name": model,
                "tensor_ir": _stage_status(status, "tensor_ir"),
                "op_semantics": _stage_status(status, "op_semantics"),
                "region_tree": _stage_status(status, "structural_region_tree"),
                "dimension_ir": _stage_status(status, "region_dimension_ir"),
                "region_semantics": _stage_status(status, "region_pruning_semantics"),
                "ranking": _stage_status(status, "pruning_opportunity_ranking"),
                "plans": _stage_status(status, "pruning_plan_synthesis"),
                "validation": _stage_status(status, "pruning_plan_validation"),
                "layer_packs": _stage_status(status, "layer_subgraph_validation"),
                "full_report": _stage_status(status, "full_model_report"),
            }
        )
        opportunity_rows.append(
            {
                "model_name": model,
                "safe": counts["safe"],
                "constrained": counts["constrained"],
                "blocked": counts["blocked"],
                "auxiliary": counts["auxiliary"],
                "unknown": counts["unknown"],
                "mlp_safe": counts["mlp_safe"],
                "generic_mlp_safe": counts["generic_mlp_safe"],
                "generic_mlp_constrained": counts["generic_mlp_constrained"],
                "valid_plans": counts["valid_plans"],
            }
        )
        semantic_rows.append({"model_name": model, **_semantic_counts(status, root)})
    final_counts = Counter(row["final_status"] for row in model_rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": statuses,
        "model_status_table": model_rows,
        "stage_coverage_table": stage_rows,
        "opportunity_coverage_table": opportunity_rows,
        "semantic_coverage_table": semantic_rows,
        "summary": {
            "complete_models": final_counts.get("complete", 0),
            "partial_models": final_counts.get("partial", 0),
            "skipped_models": final_counts.get("skipped", 0),
            "failed_models": final_counts.get("failed", 0),
            "total_models": len(statuses),
        },
        "conclusions": coverage_conclusions(model_rows),
    }


def coverage_conclusions(model_rows: list[dict[str, Any]]) -> list[str]:
    conclusions = [
        "BERT is the current complete reference case when it reaches complete status with validated FFN plans.",
        "Partial or skipped models identify missing upstream artifacts or model-specific structure rules.",
        "Decoder-only and vision-transformer models should not be forced into BERT encoder-layer structure.",
    ]
    complete = [row["model_name"] for row in model_rows if row.get("final_status") == "complete"]
    if complete:
        conclusions.append("Complete models: " + ", ".join(complete) + ".")
    skipped = [row["model_name"] for row in model_rows if row.get("final_status") == "skipped"]
    if skipped:
        conclusions.append("Skipped models require base static artifacts before scientific comparison: " + ", ".join(skipped) + ".")
    return conclusions


def write_static_coverage_report(report: dict[str, Any], output_root: Path, markdown_fn) -> None:
    ensure_dir(output_root)
    (output_root / "index.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_root / "index.md").write_text(markdown_fn(report), encoding="utf-8")
