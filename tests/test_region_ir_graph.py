from __future__ import annotations

from model_analysis.region_ir_graph import (
    build_region_constraint_adjacency,
    extract_region_slice,
    get_region_equivalence_class_for_dimension,
)


def synthetic_region_ir() -> dict:
    return {
        "model_name": "tiny-region-ir",
        "source_frontend": "synthetic",
        "root_region_id": "region::root",
        "dimension_variables": [
            {"var_id": "ffn_out", "region_id": "ffn", "region_type": "FeedForwardRegion", "region_name": "bert.encoder.layer.0.feedforward", "dim_name": "intermediate_dim", "axis_role": "intermediate", "size": 8, "prunable": True, "protected": False, "propagated": False, "blocked": False, "confidence": "high"},
            {"var_id": "ffn_in", "region_id": "ffn", "region_type": "FeedForwardRegion", "region_name": "bert.encoder.layer.0.feedforward", "dim_name": "intermediate_dim", "axis_role": "intermediate", "size": 8, "prunable": False, "protected": False, "propagated": True, "blocked": False, "confidence": "high"},
            {"var_id": "fanout_src", "region_id": "fork", "region_type": "ForkRegion", "region_name": "fork", "dim_name": "fanout_dim", "axis_role": "unknown", "size": 8, "prunable": True, "protected": False, "propagated": True, "blocked": False, "confidence": "medium"},
            {"var_id": "fanout_dst", "region_id": "fork", "region_type": "ForkRegion", "region_name": "fork", "dim_name": "fanout_dim", "axis_role": "unknown", "size": 8, "prunable": False, "protected": False, "propagated": True, "blocked": False, "confidence": "medium"},
            {"var_id": "residual_a", "region_id": "residual", "region_type": "ResidualMergeRegion", "region_name": "residual", "dim_name": "hidden_dim", "axis_role": "hidden", "size": 4, "prunable": True, "protected": True, "propagated": False, "blocked": True, "confidence": "medium"},
            {"var_id": "residual_b", "region_id": "residual", "region_type": "ResidualMergeRegion", "region_name": "residual", "dim_name": "hidden_dim", "axis_role": "hidden", "size": 4, "prunable": False, "protected": True, "propagated": False, "blocked": True, "confidence": "medium"},
            {"var_id": "norm_in", "region_id": "norm", "region_type": "LayerNormRegion", "region_name": "norm", "dim_name": "hidden_dim", "axis_role": "hidden", "size": 4, "prunable": True, "protected": True, "propagated": True, "blocked": False, "confidence": "high"},
            {"var_id": "norm_width", "region_id": "norm", "region_type": "LayerNormRegion", "region_name": "norm", "dim_name": "hidden_dim", "axis_role": "hidden", "size": 4, "prunable": False, "protected": True, "propagated": True, "blocked": False, "confidence": "high"},
            {"var_id": "axis_in", "region_id": "axis", "region_type": "AxisTransformRegion", "region_name": "axis", "dim_name": "symbolic_axis", "axis_role": "shape", "size": 8, "prunable": True, "protected": False, "propagated": True, "blocked": False, "confidence": "medium"},
            {"var_id": "axis_out", "region_id": "axis", "region_type": "AxisTransformRegion", "region_name": "axis", "dim_name": "symbolic_axis", "axis_role": "shape", "size": 8, "prunable": False, "protected": False, "propagated": True, "blocked": False, "confidence": "medium"},
            {"var_id": "local", "region_id": "linear", "region_type": "LinearProjectionRegion", "region_name": "linear", "dim_name": "out_features", "axis_role": "hidden", "size": 6, "prunable": True, "protected": False, "propagated": False, "blocked": False, "confidence": "high"},
        ],
        "constraint_equations": [
            {"constraint_id": "c_mlp", "region_id": "ffn", "region_type": "FeedForwardRegion", "lhs": "ffn_out", "rhs": "ffn_in", "relation": "same_indices", "constraint_type": "mlp_intermediate_same_indices", "blocking": False, "confidence": "high", "reason": "same intermediate"},
            {"constraint_id": "c_fanout", "region_id": "fork", "region_type": "ForkRegion", "lhs": "fanout_src", "rhs": "fanout_dst", "relation": "fanout", "constraint_type": "fork_fanout_propagation", "blocking": False, "confidence": "medium", "reason": "fanout"},
            {"constraint_id": "c_residual", "region_id": "residual", "region_type": "ResidualMergeRegion", "lhs": "residual_a", "rhs": "residual_b", "relation": "join_equal", "constraint_type": "residual_hidden_equality", "blocking": True, "confidence": "medium", "reason": "residual"},
            {"constraint_id": "c_norm", "region_id": "norm", "region_type": "LayerNormRegion", "lhs": "norm_in", "rhs": "norm_width", "relation": "eq", "constraint_type": "layernorm_hidden_equality", "blocking": True, "confidence": "high", "reason": "norm"},
            {"constraint_id": "c_axis", "region_id": "axis", "region_type": "AxisTransformRegion", "lhs": "axis_in", "rhs": "axis_out", "relation": "reshape_map", "constraint_type": "axis_transform_mapping", "blocking": True, "confidence": "medium", "reason": "axis"},
        ],
        "equivalence_classes": [
            {"class_id": "eq_mlp", "members": ["ffn_out", "ffn_in"], "representative": "ffn_in", "class_type": "mlp_intermediate"},
            {"class_id": "eq_residual", "members": ["residual_a", "residual_b"], "representative": "residual_a", "class_type": "residual_hidden"},
        ],
        "blocked_dimensions": ["residual_a", "residual_b"],
        "unresolved_constraints": ["c_axis"],
    }


def test_region_adjacency_infers_fanout_and_bidirectional_relations() -> None:
    adjacency = build_region_constraint_adjacency(synthetic_region_ir())

    assert adjacency["fanout_src"]["outgoing"][0]["constraint_id"] == "c_fanout"
    assert adjacency["fanout_dst"]["incoming"][0]["constraint_id"] == "c_fanout"
    assert adjacency["ffn_out"]["bidirectional"][0]["constraint_id"] == "c_mlp"


def test_equivalence_lookup_and_forward_backward_slices() -> None:
    ir = synthetic_region_ir()
    equivalent = get_region_equivalence_class_for_dimension(ir, "ffn_out")
    forward = extract_region_slice(ir, "fanout_src", "forward")
    backward = extract_region_slice(ir, "norm_width", "backward")

    assert equivalent["class_type"] == "mlp_intermediate"
    assert forward.dimensions == ["fanout_dst", "fanout_src"]
    assert forward.constraints == ["c_fanout"]
    assert "norm_in" in backward.dimensions
    assert backward.blocking_constraints == ["c_norm"]
