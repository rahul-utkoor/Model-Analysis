from model_analysis.attention_value_path_subgraph_compare import compare_attention_value_path_reports


def test_compare_report_aggregates_statuses() -> None:
    data = compare_attention_value_path_reports(
        [
            {"model_name": "opt", "total_paths": 2, "exported": 1, "skipped": 1, "failed": 0, "seedable": 1, "partial": 1, "blocked": 0, "unknown": 0},
            {"model_name": "gpt2", "total_paths": 1, "exported": 0, "skipped": 1, "failed": 0, "seedable": 0, "partial": 0, "blocked": 1, "unknown": 0},
        ]
    )
    assert data["summary"]["total_paths"] == 3
    assert data["summary"]["seedable"] == 1
    assert data["summary"]["partial"] == 1
    assert data["summary"]["blocked"] == 1
