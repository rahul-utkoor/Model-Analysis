from experimental.formalization.report import (
    FormalizationInputs,
    render_bert_case_study,
    render_paper_methodology,
    render_static_notes,
    render_teaching_slides,
)


def _inputs() -> FormalizationInputs:
    layers = [
        {
            "layer_index": layer,
            "ffn_plan_status": "ready_symbolic",
            "ffn_validation_status": "valid",
            "attention_path_status": "seedable",
            "attention_mapping_status": "proven",
            "attention_evidence_tier": "native_mlir_dependence_evidence",
            "ffn_verdict": "proven",
            "attention_verdict": "proven",
        }
        for layer in range(12)
    ]
    return FormalizationInputs(
        {"summary": {"ffn_found": 12, "ffn_proven": 12, "attention_found": 12, "attention_proven": 12, "total_proven": 24, "expected_plans": 24, "final_verdict": "complete_24_plan_proof"}, "layers": layers},
        {"summary": {"native_proven": 24, "missing_cases": 0}},
        {"summary": {"seedable": 12}},
        {"summary": {"valid": 12}},
    )


def test_notes_cover_core_story() -> None:
    text = render_static_notes(_inputs())
    assert "Sparse-weight pruning" in text
    assert "Structural pruning" in text
    assert "Sparsity is not the same as deadness" in text
    assert "DFA" in text
    assert "MLIR dependence evidence" in text


def test_bert_case_study_reports_complete_proof() -> None:
    text = render_bert_case_study(_inputs())
    assert "12 x 2 = 24" in text
    assert "FFN intermediate" in text
    assert "Attention value path" in text
    assert "complete_24_plan_proof" in text


def test_teaching_outline_has_at_least_fifteen_slides() -> None:
    assert render_teaching_slides(_inputs()).count("## Slide ") >= 15


def test_paper_outline_contains_formal_terms() -> None:
    text = render_paper_methodology(_inputs())
    assert "fact lattice" in text
    assert "transfer function" in text
    assert "fixed-point" in text
    assert "Soundness-Style Statement" in text


def test_missing_inputs_render_partial_warnings() -> None:
    text = render_bert_case_study(FormalizationInputs(warnings=("BERT proof missing",)))
    assert "`partial`" in text
    assert "BERT proof missing" in text
