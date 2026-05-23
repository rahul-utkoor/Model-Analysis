from __future__ import annotations

from model_analysis.region_dimension_ir import (
    build_region_constraint_equations,
    build_region_dimension_equivalence_classes,
    build_region_dimension_ir,
    build_region_dimension_variables,
)


def synthetic_region_tree() -> dict:
    region_types = [
        ("region::linear", "LinearProjectionRegion", "high", [{"type": "bias_follows_output"}]),
        ("region::ffn", "FeedForwardRegion", "high", [{"type": "same_indices"}]),
        ("region::residual", "ResidualMergeRegion", "medium", [{"type": "branch_hidden_equality"}]),
        ("region::norm", "LayerNormRegion", "high", [{"type": "hidden_equals_normalization_parameter"}]),
        ("region::axis", "AxisTransformRegion", "medium", [{"type": "reshape_transpose_axis_mapping_required"}]),
        ("region::activation", "ActivationRegion", "high", [{"type": "activation_preserves_shape"}]),
        ("region::attention", "AttentionSkeletonRegion", "medium", [{"type": "head_axis_mapping_required"}]),
        ("region::fork", "ForkRegion", "medium", [{"type": "fanout_same_indices"}]),
        ("region::join", "JoinRegion", "medium", [{"type": "branch_compatibility"}]),
    ]
    return {
        "model_name": "tiny-regions",
        "source_frontend": "onnx",
        "root_region_id": "region::root",
        "regions": [
            {"region_id": "region::root", "region_type": "ModelRegion", "name": "root", "confidence": "high"},
            *[
                {"region_id": region_id, "region_type": region_type, "name": region_type, "confidence": confidence}
                for region_id, region_type, confidence, _ in region_types
            ],
        ],
        "interfaces": [
            {"region_id": "region::root", "region_type": "ModelRegion", "constraints": [], "pruning_role": "analysis_only"},
            *[
                {
                    "region_id": region_id,
                    "region_type": region_type,
                    "constraints": constraints,
                    "pruning_role": "analysis_only",
                }
                for region_id, region_type, _, constraints in region_types
            ],
        ],
        "summary": {},
    }


def test_region_types_create_symbolic_dimensions() -> None:
    dimensions = build_region_dimension_variables(synthetic_region_tree())

    assert any(item.region_type == "LinearProjectionRegion" and item.dim_name == "out_features" and item.prunable for item in dimensions)
    assert any(item.region_type == "LinearProjectionRegion" and item.dim_name == "in_features" and item.propagated for item in dimensions)
    assert any(item.region_type == "FeedForwardRegion" and item.dim_name == "intermediate_dim" and item.prunable for item in dimensions)
    assert any(item.region_type == "FeedForwardRegion" and item.dim_name == "hidden_dim" and item.protected for item in dimensions)
    assert any(item.region_type == "ResidualMergeRegion" and item.dim_name == "hidden_dim" and item.blocked for item in dimensions)
    assert any(item.region_type == "LayerNormRegion" and item.protected for item in dimensions)
    assert any(item.region_type == "AxisTransformRegion" and item.dim_name == "symbolic_axis" for item in dimensions)


def test_attention_dimensions_remain_protected_and_analysis_only() -> None:
    dimensions = build_region_dimension_variables(synthetic_region_tree())
    attention = [item for item in dimensions if item.region_type == "AttentionSkeletonRegion"]

    assert {item.dim_name for item in attention} == {"num_heads", "head_dim", "hidden_dim", "sequence_dim"}
    assert all(item.protected and item.propagated and not item.prunable for item in attention)


def test_region_constraints_include_mlp_blocking_and_unresolved_equations() -> None:
    dimensions = build_region_dimension_variables(synthetic_region_tree())
    constraints = build_region_constraint_equations(synthetic_region_tree(), dimensions)

    mlp = next(item for item in constraints if item.constraint_type == "mlp_intermediate_same_indices")
    residual = next(item for item in constraints if item.constraint_type == "residual_hidden_equality")
    axis = next(item for item in constraints if item.constraint_type == "axis_transform_mapping")
    attention = next(item for item in constraints if item.constraint_type == "attention_head_axis_mapping")
    assert mlp.relation == "same_indices" and mlp.blocking is False
    assert residual.relation == "join_equal" and residual.blocking is True
    assert axis.relation == "reshape_map"
    assert attention.relation == "unknown" and attention.blocking is True


def test_equivalence_classes_group_same_indices_and_blocked_dimensions() -> None:
    dimensions = build_region_dimension_variables(synthetic_region_tree())
    constraints = build_region_constraint_equations(synthetic_region_tree(), dimensions)
    classes = build_region_dimension_equivalence_classes(dimensions, constraints)
    ir = build_region_dimension_ir(synthetic_region_tree())

    mlp = next(item for item in classes if item.class_type == "mlp_intermediate")
    assert len(mlp.members) == 2
    assert any("residual" in item for item in ir.blocked_dimensions)
    assert len(ir.unresolved_constraints) == 2
    assert ir.summary["num_constraint_equations"] == len(constraints)
    assert ir.summary["prunable_dimension_count"] >= 2
