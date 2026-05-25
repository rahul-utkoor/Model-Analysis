from __future__ import annotations

from model_analysis.rule_gap_diagnosis_text import rule_gap_compare_to_markdown, rule_gap_diagnosis_to_markdown


def test_rule_gap_markdown_contains_gap_table() -> None:
    text = rule_gap_diagnosis_to_markdown(
        {
            "model_name": "facebook/opt-125m",
            "final_status": "partial",
            "detected_model_family": "opt_decoder",
            "plan_summary": {"total_plans": 12, "ready_symbolic": 0, "incomplete": 12},
            "validation_summary": {"valid_plans": 0, "invalid_plans": 12},
            "gaps": [
                {
                    "gap_type": "missing_ffn_evidence_binding",
                    "severity": "blocker",
                    "affected_stage": "plan_synthesis",
                    "affected_count": 12,
                    "explanation": "Missing generic FFN evidence binding.",
                }
            ],
            "candidate_repairs": [],
            "evidence_summary": {},
            "conclusion": "Detected gaps.",
        }
    )

    assert "# Rule-Gap Diagnosis: facebook/opt-125m" in text
    assert "missing_ffn_evidence_binding" in text
    assert "static diagnosis/reporting only" in text


def test_rule_gap_compare_markdown_contains_model_rows() -> None:
    text = rule_gap_compare_to_markdown(
        {
            "diagnoses": [
                {
                    "model_name": "bert-base-uncased",
                    "detected_model_family": "bert_encoder",
                    "gaps": [],
                    "plan_summary": {"total_plans": 12},
                    "validation_summary": {"valid_plans": 12},
                    "conclusion": "No blocking rule gaps detected.",
                }
            ],
            "gap_type_counts": {},
        }
    )

    assert "# Rule-Gap Diagnosis Comparison" in text
    assert "bert-base-uncased" in text
