from __future__ import annotations

from model_analysis.tensor_ir_compare import compare_tensor_graphs, tensor_ir_comparison_to_markdown


def graph_report(model: str, linear: int, forks: int, joins: int) -> dict:
    return {
        "model_name": model,
        "summary": {
            "num_ops": linear + joins,
            "num_values": 10,
            "canonical_op_type_counts": {"linear": linear, "elementwise_join": joins},
            "semantic_role_counts": {"activation": 8, "parameter": 2},
            "region_hint_counts": {"LinearProjection": linear, "ResidualJoin": joins},
            "num_fork_ops": forks,
            "num_join_ops": joins,
        },
    }


def test_compare_tensor_graphs_builds_matrices() -> None:
    comparison = compare_tensor_graphs([graph_report("bert", 2, 1, 1), graph_report("vit", 3, 2, 1)])

    assert comparison["num_models"] == 2
    assert comparison["canonical_op_type_matrix"]["bert"]["linear"] == 2
    assert comparison["region_hint_matrix"]["vit"]["LinearProjection"] == 3
    assert comparison["fork_join_matrix"]["vit"]["fork_ops"] == 2
    assert comparison["summary"]["total_join_ops"] == 2
    assert "Tensor IR Comparison" in tensor_ir_comparison_to_markdown(comparison)
