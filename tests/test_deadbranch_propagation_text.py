from model_analysis.deadbranch_propagation_text import deadbranch_report_to_markdown


def test_markdown_explains_sparse_weight_and_structural_deadness() -> None:
    text = deadbranch_report_to_markdown(
        {
            "model_name": "facebook/opt-125m",
            "summary": {
                "total_pairs": 24,
                "ffn_pairs": 12,
                "attention_value_pairs": 12,
                "query_key_blocked_pairs": 24,
                "expected_sparsegpt_pairs": 24,
                "sparsegpt_alignment_status": "matches_expected",
            },
            "pairs": [],
            "blocked_pairs": [],
        }
    )

    assert "SparseGPT 2:4 does not expose dead channels" in text
    assert "Channel pruning exposes deadness" in text
    assert "QK^T blocks Q/K simple propagation" in text
    assert "v_proj -> out_proj" in text
