from model_analysis.deadbranch_propagation_compare import compare_deadbranch_reports, deadbranch_compare_to_markdown


def test_compare_aggregates_deadbranch_counts() -> None:
    comparison = compare_deadbranch_reports(
        [
            {"model_name": "a", "summary": {"total_pairs": 2, "ffn_pairs": 1, "attention_value_pairs": 1, "query_key_blocked_pairs": 2}},
            {"model_name": "b", "summary": {"total_pairs": 1, "ffn_pairs": 1, "attention_value_pairs": 0, "query_key_blocked_pairs": 0}},
        ]
    )

    assert comparison["summary"] == {"total_pairs": 3, "ffn_pairs": 2, "attention_value_pairs": 1, "query_key_blocked_pairs": 2}
    assert "Deadbranch Propagation Comparison" in deadbranch_compare_to_markdown(comparison)
