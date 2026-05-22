from __future__ import annotations

from model_analysis.pruning_opportunity import (
    build_model_pruning_map,
    extract_propagation_constraints,
    extract_pruning_dimensions,
)


def synthetic_graph() -> dict:
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "prunable_units": [
            {
                "unit_id": "torch:linear:linear",
                "name": "linear",
                "source": "torch",
                "unit_type": "linear",
                "module_or_node_name": "linear",
                "prunable_dims": ["out_features"],
                "parameter_count": 20,
                "shape": [4, 4],
                "confidence": "medium",
                "reason": "linear",
            },
            {
                "unit_id": "torch:linear:fc1",
                "name": "fc1",
                "source": "torch",
                "unit_type": "mlp_expansion",
                "module_or_node_name": "fc1",
                "prunable_dims": ["out_features", "intermediate_dim"],
                "parameter_count": 40,
                "shape": [8, 4],
                "confidence": "medium",
                "reason": "fc1",
            },
            {
                "unit_id": "torch:linear:fc2",
                "name": "fc2",
                "source": "torch",
                "unit_type": "mlp_projection",
                "module_or_node_name": "fc2",
                "prunable_dims": ["in_features", "intermediate_dim"],
                "parameter_count": 36,
                "shape": [4, 8],
                "confidence": "medium",
                "reason": "fc2",
            },
            {
                "unit_id": "torch:linear:q",
                "name": "q",
                "source": "torch",
                "unit_type": "linear",
                "module_or_node_name": "q",
                "prunable_dims": ["out_features"],
                "parameter_count": 72,
                "shape": [8, 8],
                "confidence": "medium",
                "reason": "q",
            },
            {
                "unit_id": "torch:linear:k",
                "name": "k",
                "source": "torch",
                "unit_type": "linear",
                "module_or_node_name": "k",
                "prunable_dims": ["out_features"],
                "parameter_count": 72,
                "shape": [8, 8],
                "confidence": "medium",
                "reason": "k",
            },
            {
                "unit_id": "torch:linear:residual",
                "name": "residual",
                "source": "torch",
                "unit_type": "linear",
                "module_or_node_name": "residual",
                "prunable_dims": ["out_features"],
                "parameter_count": 20,
                "shape": [4, 4],
                "confidence": "medium",
                "reason": "residual",
            },
        ],
        "dependency_edges": [
            {
                "src": "torch:linear:fc1",
                "dst": "torch:linear:fc2",
                "edge_type": "mlp_hidden_coupling",
                "affected_dims": ["intermediate_dim"],
                "direction": "bidirectional",
                "confidence": "medium",
                "reason": "mlp pair",
            },
            {
                "src": "torch:linear:q",
                "dst": "torch:linear:k",
                "edge_type": "qkv_coupling",
                "affected_dims": ["num_heads", "head_dim", "hidden_dim"],
                "direction": "bidirectional",
                "confidence": "medium",
                "reason": "qkv",
            },
            {
                "src": "torch:linear:residual",
                "dst": "torch:linear:linear",
                "edge_type": "residual_coupling",
                "affected_dims": ["hidden_dim"],
                "direction": "bidirectional",
                "confidence": "medium",
                "reason": "residual",
            },
        ],
    }


def test_linear_out_features_dimension_extracted():
    dimensions = extract_pruning_dimensions(synthetic_graph())

    linear_dims = [item for item in dimensions if item.unit_id == "torch:linear:linear"]
    assert linear_dims[0].dim_name == "out_features"
    assert linear_dims[0].size == 4


def test_mlp_and_qkv_constraints_are_typed():
    dimensions = extract_pruning_dimensions(synthetic_graph())
    constraints = extract_propagation_constraints(synthetic_graph(), dimensions)

    assert any(item.constraint_type == "mlp_same_intermediate_indices" for item in constraints)
    assert any(item.constraint_type == "qkv_same_heads" for item in constraints)


def test_model_pruning_map_classifies_opportunities_and_risks():
    model_map = build_model_pruning_map(synthetic_graph())

    opportunities = {item.opportunity_type: item for item in model_map.opportunities}
    assert "mlp_intermediate" in opportunities
    assert opportunities["mlp_intermediate"].executability == "pair_executable"
    assert any(item.opportunity_type == "attention_qkv_heads" and item.executability == "analysis_only" for item in model_map.opportunities)
    assert any(item.opportunity_type == "blocked_residual_hidden" for item in model_map.opportunities)
    assert any(risk["risk_type"] == "residual_shape_coupling" for risk in model_map.structural_risks)
    assert model_map.summary["num_pruning_dimensions"] == len(model_map.pruning_dimensions)
    assert model_map.summary["num_opportunities"] == len(model_map.opportunities)
