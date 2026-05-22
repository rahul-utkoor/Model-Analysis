from __future__ import annotations

from model_analysis.dimension_ir_compare import compare_dimension_irs


def make_ir(name: str, role: str, constraint_type: str, relation: str) -> dict:
    return {
        "model_name": name,
        "dimension_variables": [{"semantic_role": role}],
        "constraint_equations": [{"constraint_type": constraint_type, "relation": relation}],
        "equivalence_classes": [{"class_type": "mlp_intermediate"}],
        "blocked_dimensions": ["d0"] if name == "bert" else [],
        "unresolved_constraints": ["c0"] if relation == "unknown" else [],
        "summary": {
            "semantic_role_counts": {role: 1},
            "constraint_type_counts": {constraint_type: 1},
            "relation_counts": {relation: 1},
            "num_dimension_variables": 1,
            "num_constraint_equations": 1,
            "num_blocked_dimensions": 1 if name == "bert" else 0,
            "num_unresolved_constraints": 1 if relation == "unknown" else 0,
        },
    }


def test_compare_dimension_irs_builds_matrices():
    comparison = compare_dimension_irs(
        [
            make_ir("bert", "coupled", "mlp_intermediate_consistency", "same_indices"),
            make_ir("gpt2", "producer", "unknown_mapping", "unknown"),
        ]
    )

    assert comparison["num_models"] == 2
    assert comparison["dimension_role_matrix"]["bert"]["coupled"] == 1
    assert comparison["constraint_type_matrix"]["gpt2"]["unknown_mapping"] == 1
    assert comparison["relation_matrix"]["bert"]["same_indices"] == 1
    assert comparison["blocked_dimension_matrix"]["bert"]["blocked_dimensions"] == 1
    assert comparison["unresolved_constraint_matrix"]["gpt2"]["unresolved_constraints"] == 1
