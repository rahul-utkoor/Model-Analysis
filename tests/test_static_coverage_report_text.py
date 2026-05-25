from __future__ import annotations

from model_analysis.static_coverage_report_text import static_coverage_report_to_markdown


def test_coverage_markdown_contains_required_tables():
    report = {
        "model_status_table": [{"model_name": "bert", "final_status": "complete"}],
        "stage_coverage_table": [{"model_name": "bert", "tensor_ir": "present_existing"}],
        "opportunity_coverage_table": [{"model_name": "bert", "safe": 1}],
        "semantic_coverage_table": [{"model_name": "bert", "parameterized_matmul": 1}],
        "models": [{"model_name": "bert", "final_status": "complete", "summary": {"completed_stages": 1, "missing_artifacts": []}, "artifacts": {"full_model_report": {"valid_plans": 1}}}],
        "conclusions": ["synthetic conclusion"],
    }
    markdown = static_coverage_report_to_markdown(report)
    assert "## 2. Model status table" in markdown
    assert "## 3. Stage coverage table" in markdown
    assert "synthetic conclusion" in markdown
