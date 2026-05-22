from __future__ import annotations

from model_analysis.dimension_ir import (
    build_constraint_equations_from_pruning_map,
    build_dimension_equivalence_classes,
    build_dimension_variables_from_pruning_map,
    build_index_variables,
    build_pruning_ir,
)


def synthetic_pruning_map() -> dict:
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "pruning_dimensions": [
            {
                "dim_id": "dim::torch:linear:fc1::intermediate_dim",
                "unit_id": "torch:linear:fc1",
                "unit_name": "fc1",
                "unit_type": "mlp_expansion",
                "dim_name": "intermediate_dim",
                "size": 8,
                "structural_role": "coupled",
                "confidence": "medium",
                "reason": "fc1 intermediate",
            },
            {
                "dim_id": "dim::torch:linear:fc2::intermediate_dim",
                "unit_id": "torch:linear:fc2",
                "unit_name": "fc2",
                "unit_type": "mlp_projection",
                "dim_name": "intermediate_dim",
                "size": 8,
                "structural_role": "coupled",
                "confidence": "medium",
                "reason": "fc2 intermediate",
            },
            {
                "dim_id": "dim::torch:linear:residual::hidden_dim",
                "unit_id": "torch:linear:residual",
                "unit_name": "residual",
                "unit_type": "linear",
                "dim_name": "hidden_dim",
                "size": 4,
                "structural_role": "coupled",
                "confidence": "low",
                "reason": "residual hidden",
            },
            {
                "dim_id": "dim::torch:linear:peer::hidden_dim",
                "unit_id": "torch:linear:peer",
                "unit_name": "peer",
                "unit_type": "linear",
                "dim_name": "hidden_dim",
                "size": 4,
                "structural_role": "coupled",
                "confidence": "low",
                "reason": "peer hidden",
            },
        ],
        "propagation_constraints": [
            {
                "constraint_id": "constraint_00001",
                "src_dim_id": "dim::torch:linear:fc1::intermediate_dim",
                "dst_dim_id": "dim::torch:linear:fc2::intermediate_dim",
                "constraint_type": "mlp_same_intermediate_indices",
                "direction": "bidirectional",
                "edge_type": "mlp_hidden_coupling",
                "confidence": "medium",
                "evidence": [],
                "reason": "MLP pair",
            },
            {
                "constraint_id": "constraint_00002",
                "src_dim_id": "dim::torch:linear:residual::hidden_dim",
                "dst_dim_id": "dim::torch:linear:peer::hidden_dim",
                "constraint_type": "residual_equal_shape",
                "direction": "bidirectional",
                "edge_type": "residual_coupling",
                "confidence": "medium",
                "evidence": [],
                "reason": "Residual path",
            },
            {
                "constraint_id": "constraint_00003",
                "src_dim_id": "dim::missing::x",
                "dst_dim_id": "dim::torch:linear:peer::hidden_dim",
                "constraint_type": "unknown_mapping",
                "direction": "forward",
                "edge_type": "shape_dependency",
                "confidence": "low",
                "evidence": [],
                "reason": "Unknown mapping",
            },
        ],
        "opportunities": [
            {
                "opportunity_id": "opp::blocked",
                "risk_level": "blocked",
                "executability": "blocked",
                "prunable_dimensions": ["dim::torch:linear:residual::hidden_dim"],
            }
        ],
        "blocked_opportunities": ["opp::blocked"],
    }


def test_pruning_dimension_becomes_dimension_variable():
    variables = build_dimension_variables_from_pruning_map(synthetic_pruning_map())

    assert variables[0].var_id.startswith("dim::")
    assert any(variable.owner_name == "fc1" and variable.dim_name == "intermediate_dim" for variable in variables)
    assert any(variable.semantic_role == "blocked" for variable in variables)


def test_index_variable_created_for_prunable_dimension():
    variables = build_dimension_variables_from_pruning_map(synthetic_pruning_map())
    indices = build_index_variables(variables)

    fc1_index = [item for item in indices if "fc1" in item.dimension_var_id][0]
    assert fc1_index.allowed_range == [0, 8]
    assert fc1_index.symbolic is True


def test_constraints_convert_to_symbolic_equations():
    variables = build_dimension_variables_from_pruning_map(synthetic_pruning_map())
    equations = build_constraint_equations_from_pruning_map(synthetic_pruning_map(), variables)

    mlp = [item for item in equations if item.constraint_type == "mlp_intermediate_consistency"][0]
    residual = [item for item in equations if item.constraint_type == "residual_hidden_equality"][0]
    unknown = [item for item in equations if item.constraint_type == "unknown_mapping"][0]
    assert mlp.relation == "same_indices"
    assert residual.relation == "eq"
    assert residual.blocking is True
    assert unknown.relation == "unknown"


def test_union_find_groups_mlp_dimensions():
    variables = build_dimension_variables_from_pruning_map(synthetic_pruning_map())
    equations = build_constraint_equations_from_pruning_map(synthetic_pruning_map(), variables)
    classes = build_dimension_equivalence_classes(variables, equations)

    mlp_classes = [item for item in classes if item.class_type == "mlp_intermediate"]
    assert len(mlp_classes) == 1
    assert len(mlp_classes[0].members) == 2


def test_full_pruning_ir_summary_counts():
    ir = build_pruning_ir(synthetic_pruning_map())

    assert ir.summary["num_dimension_variables"] == 4
    assert ir.summary["num_index_variables"] == 3
    assert ir.summary["num_constraint_equations"] == 3
    assert ir.summary["num_unresolved_constraints"] == 1
    assert ir.unresolved_constraints == ["constraint_00003"]
