from __future__ import annotations

from model_analysis.subgraph_compare import compare_subgraph_reports


def make_report(model_name: str, join_count: int, residual_count: int, pattern: str) -> dict:
    return {
        "model_name": model_name,
        "pattern_summaries": [
            {"pattern": "Gemm", "count": 2},
            {"pattern": pattern, "count": 1},
        ],
        "summary": {
            "pruning_class_counts": {"directly_prunable": 2, "residual_like": residual_count},
            "risk_level_counts": {"low": 2, "high": residual_count},
            "evidence_type_counts": {"direct_prunable_op": 2, "residual_hidden_equality": residual_count},
            "num_path_subgraphs": 3,
            "num_join_subgraphs": join_count,
            "num_residual_like_join_subgraphs": residual_count,
            "bias_add_count": 1,
            "residual_add_count": residual_count,
            "unknown_add_count": 0,
            "residual_like_pattern_count": residual_count,
        },
    }


def test_compare_subgraph_reports_builds_pattern_and_join_matrices() -> None:
    comparison = compare_subgraph_reports(
        [
            make_report("bert", 1, 1, "Join(Add)[Gemm, GraphInput] -> LayerNormalization"),
            make_report("vit", 1, 0, "Join(Concat)[Conv, Conv] -> Conv"),
        ]
    )

    assert comparison["num_models"] == 2
    assert comparison["pattern_matrix"]["bert"]["Gemm"] == 2
    assert comparison["pruning_class_matrix"]["bert"]["directly_prunable"] == 2
    assert comparison["join_summary_matrix"]["bert"]["residual_add"] == 1
    assert comparison["residual_summary_matrix"]["vit"]["residual_like_join_subgraphs"] == 0
    assert comparison["common_patterns"] == ["Gemm"]

