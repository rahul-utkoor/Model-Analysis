from __future__ import annotations

from pathlib import Path

from model_analysis.static_coverage_report import build_static_coverage_report


def test_static_coverage_aggregates_status_counts(tmp_path: Path):
    statuses = [
        {
            "model_name": "bert-base-uncased",
            "final_status": "complete",
            "summary": {"completed_stages": 10, "missing_artifacts": []},
            "stages": [],
            "artifacts": {
                "ranking": {"safe": 24, "constrained": 60, "blocked": 89, "auxiliary": 652, "unknown": 87, "mlp_safe": 12},
                "plans": {"plans": 12},
                "validation": {"valid_plans": 12},
                "full_model_report": {"layers": 12, "subgraphs": 204},
            },
            "notes": ["reference"],
        },
        {
            "model_name": "gpt2",
            "final_status": "skipped",
            "summary": {"completed_stages": 0, "missing_artifacts": ["reports/tensor_ir/gpt2.json"]},
            "stages": [],
            "artifacts": {},
            "notes": [],
        },
    ]
    report = build_static_coverage_report(tmp_path, statuses)
    assert report["summary"]["complete_models"] == 1
    assert report["summary"]["skipped_models"] == 1
    assert report["opportunity_coverage_table"][0]["safe"] == 24
    assert report["model_status_table"][1]["missing_artifacts"] == 1
