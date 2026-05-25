from __future__ import annotations

from model_analysis.full_model_analysis_report_text import subgraph_explanation_to_markdown


def test_subgraph_explanation_contains_verdict_and_why_no_plan():
    markdown = subgraph_explanation_to_markdown(
        {
            "display_name": "Layer 0 Attention",
            "semantic_category": "attention_skeleton",
            "classification": {
                "pruning_class": "blocked",
                "plan_status": "no_plan_expected",
                "validation_status": "not_applicable",
            },
            "verdict": "blocked/constrained; attention pruning requires head-axis mapping proof.",
            "why_no_plan": "semantic blocker prevents pruning under conservative rules.",
            "primitive_ops": [],
            "local_op_semantics": [],
            "local_ranking": [],
            "local_plans": [],
            "local_validations": [],
        }
    )
    assert "head-axis mapping proof" in markdown
    assert "semantic blocker" in markdown
