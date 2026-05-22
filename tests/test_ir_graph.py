from __future__ import annotations

from model_analysis.ir_graph import build_constraint_adjacency, extract_slice, get_equivalence_class_for_dimension


def synthetic_ir() -> dict:
    return {
        "model_name": "tiny",
        "dimension_variables": [
            {"var_id": "dim::a", "prunable": True, "size": 8},
            {"var_id": "dim::b", "prunable": True, "size": 8},
            {"var_id": "dim::c", "prunable": True, "size": 8},
        ],
        "constraint_equations": [
            {
                "constraint_id": "c1",
                "lhs": "dim::a",
                "rhs": "dim::b",
                "relation": "same_indices",
                "direction": "forward",
                "constraint_type": "mlp_intermediate_consistency",
                "blocking": False,
            },
            {
                "constraint_id": "c2",
                "lhs": "dim::b",
                "rhs": "dim::c",
                "relation": "unknown",
                "direction": "backward",
                "constraint_type": "unknown_mapping",
                "blocking": True,
            },
            {
                "constraint_id": "c3",
                "lhs": "dim::a",
                "rhs": "dim::c",
                "relation": "eq",
                "direction": "bidirectional",
                "constraint_type": "residual_hidden_equality",
                "blocking": True,
            },
        ],
        "equivalence_classes": [{"class_id": "e1", "members": ["dim::a", "dim::b"], "class_type": "mlp_intermediate"}],
    }


def test_constraint_adjacency_has_directions():
    adjacency = build_constraint_adjacency(synthetic_ir())

    assert adjacency["dim::a"]["outgoing"][0]["constraint_id"] == "c1"
    assert adjacency["dim::b"]["incoming"][0]["constraint_id"] == "c1"
    assert adjacency["dim::a"]["bidirectional"][0]["constraint_id"] == "c3"


def test_equivalence_class_lookup_works():
    eq_class = get_equivalence_class_for_dimension(synthetic_ir(), "dim::b")

    assert eq_class["class_id"] == "e1"


def test_forward_and_backward_slices_follow_expected_edges():
    forward = extract_slice(synthetic_ir(), "dim::a", "forward")
    backward = extract_slice(synthetic_ir(), "dim::c", "backward")

    assert "c1" in forward.constraints
    assert "c3" in forward.blocking_constraints
    assert "c2" in backward.unresolved_constraints
    assert "dim::b" in backward.dimensions
