from __future__ import annotations

from model_analysis.rule_gap_diagnosis import compare_rule_gap_diagnoses
from model_analysis.rule_gap_diagnosis_compare import compare_rule_gaps, compare_rule_gaps_to_markdown


def diagnoses() -> list[dict]:
    return [
        {
            "model_name": "bert-base-uncased",
            "detected_model_family": "bert_encoder",
            "gaps": [],
            "plan_summary": {"total_plans": 12},
            "validation_summary": {"valid_plans": 12},
            "conclusion": "No gaps.",
        },
        {
            "model_name": "facebook/opt-125m",
            "detected_model_family": "opt_decoder",
            "gaps": [{"gap_type": "missing_ffn_evidence_binding"}],
            "plan_summary": {"total_plans": 12},
            "validation_summary": {"valid_plans": 0},
            "conclusion": "Detected gaps.",
        },
    ]


def test_rule_gap_compare_aggregates_gap_counts() -> None:
    comparison = compare_rule_gap_diagnoses(diagnoses())

    assert comparison["family_counts"]["bert_encoder"] == 1
    assert comparison["gap_type_counts"]["missing_ffn_evidence_binding"] == 1


def test_rule_gap_compare_wrapper_renders_markdown() -> None:
    comparison = compare_rule_gaps(diagnoses())
    markdown = compare_rule_gaps_to_markdown(comparison)

    assert comparison["gap_type_counts"]["missing_ffn_evidence_binding"] == 1
    assert "facebook/opt-125m" in markdown
