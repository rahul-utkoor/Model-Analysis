from __future__ import annotations

import json
from pathlib import Path

from model_analysis.cross_model_analysis_report import build_cross_model_analysis_report
from model_analysis.cross_model_analysis_report_text import cross_model_report_to_markdown


def test_cross_model_summary_aggregates_generated_reports(tmp_path: Path):
    report_root = tmp_path / "reports" / "model_analysis_reports"
    model_root = report_root / "bert-base-uncased"
    model_root.mkdir(parents=True)
    (model_root / "index.json").write_text(
        json.dumps(
            {
                "model_name": "bert-base-uncased",
                "missing_artifacts": [],
                "model_summary": {
                    "layers_generated": 12,
                    "ranking": {
                        "safe": 24,
                        "constrained": 60,
                        "blocked": 89,
                        "auxiliary": 652,
                        "unknown": 87,
                        "attention_constrained_candidates": 48,
                        "residual_blocked_candidates": 25,
                        "layernorm_blocked_candidates": 26,
                    },
                    "plans": {"total_plans": 12},
                    "plan_validation": {"valid": 12},
                    "op_semantic_counts": {
                        "parameterized_linear_matmul": 74,
                        "attention_score_matmul": 12,
                        "attention_context_matmul": 12,
                        "unknown": 33,
                    },
                    "region_semantic_counts": {"residual_merge": 25, "layer_norm": 26},
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_cross_model_analysis_report(
        tmp_path, ["bert-base-uncased", "distilbert-base-uncased"], report_root
    )
    assert report["model_summaries"][0]["valid_plans"] == 12
    assert report["model_summaries"][1]["status"] == "missing_report"
    markdown = cross_model_report_to_markdown(report)
    assert "Cross-Model Static Analysis Summary" in markdown
    assert "bert-base-uncased" in markdown
