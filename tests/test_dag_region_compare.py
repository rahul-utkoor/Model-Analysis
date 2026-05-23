from __future__ import annotations

from model_analysis.dag_region_compare import compare_dag_region_reports


def make_report(model: str, common_count: int, jfj_count: int) -> dict:
    return {
        "model_name": model,
        "pattern_summaries": [
            {"pattern": "Fork(MatMul -> [MatMul, MatMul])", "count": common_count},
            {"pattern": f"JoinForkJoin({model})", "count": jfj_count},
        ],
        "summary": {
            "num_regions": common_count + jfj_count,
            "num_join_fork_join_regions": jfj_count,
            "num_residual_like_regions": jfj_count,
            "region_kind_counts": {"fork": common_count, "join_fork_join": jfj_count},
            "pruning_class_counts": {"propagation_relevant": common_count, "residual_like": jfj_count},
            "risk_level_counts": {"medium": common_count, "high": jfj_count},
            "suggested_constraint_counts": {
                "fanout_same_indices": common_count,
                "residual_equal_shape": jfj_count,
            },
        },
    }


def test_compare_dag_region_reports_builds_matrices() -> None:
    comparison = compare_dag_region_reports([make_report("bert", 2, 1), make_report("vit", 1, 1)])

    assert comparison["num_models"] == 2
    assert comparison["region_kind_matrix"]["bert"]["join_fork_join"] == 1
    assert comparison["pruning_class_matrix"]["vit"]["residual_like"] == 1
    assert comparison["suggested_constraint_matrix"]["bert"]["fanout_same_indices"] == 2
    assert comparison["common_region_patterns"] == ["Fork(MatMul -> [MatMul, MatMul])"]
    assert comparison["summary"]["total_join_fork_join_regions"] == 2

