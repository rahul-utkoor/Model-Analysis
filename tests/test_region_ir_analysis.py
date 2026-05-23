from __future__ import annotations

from model_analysis.region_ir_analysis import (
    check_region_pruning_legality,
    compute_region_minimal_repair_set,
    make_region_pruning_request,
)
from test_region_ir_graph import synthetic_region_ir


def test_local_region_dimension_without_constraints_is_legal() -> None:
    request = make_region_pruning_request("tiny", "local", indices=[0])
    result = check_region_pruning_legality(synthetic_region_ir(), request)

    assert result.status == "legal"


def test_mlp_intermediate_request_requires_same_index_repair() -> None:
    request = make_region_pruning_request("tiny", "ffn_out", count=2)
    result = check_region_pruning_legality(synthetic_region_ir(), request)

    assert result.status == "legal_with_repairs"
    assert result.required_propagations[0]["constraint_type"] == "mlp_intermediate_same_indices"
    assert result.minimal_repair_set[0].repair_type == "same_indices"


def test_residual_and_layernorm_hidden_requests_are_rejected() -> None:
    residual = check_region_pruning_legality(synthetic_region_ir(), make_region_pruning_request("tiny", "residual_a", indices=[0]))
    layernorm = check_region_pruning_legality(synthetic_region_ir(), make_region_pruning_request("tiny", "norm_in", indices=[0]))

    assert residual.status == "rejected"
    assert residual.blocking_reasons[0]["type"] == "root_blocked"
    assert layernorm.status == "rejected"
    assert layernorm.blocking_reasons[0]["constraint_type"] == "layernorm_hidden_equality"


def test_axis_mapping_is_ambiguous_and_produces_mapping_obligation() -> None:
    result = check_region_pruning_legality(synthetic_region_ir(), make_region_pruning_request("tiny", "axis_in"))

    assert result.status == "ambiguous"
    assert result.unresolved_items[0]["constraint_type"] == "axis_transform_mapping"
    assert result.minimal_repair_set[0].repair_type == "require_axis_mapping"


def test_invalid_and_all_indices_are_rejected() -> None:
    out_of_bounds = check_region_pruning_legality(synthetic_region_ir(), make_region_pruning_request("tiny", "local", indices=[9]))
    all_indices = check_region_pruning_legality(synthetic_region_ir(), make_region_pruning_request("tiny", "local", count=6))

    assert out_of_bounds.status == "rejected"
    assert all_indices.status == "rejected"


def test_repair_set_helper_contains_mlp_same_indices_obligation() -> None:
    result = check_region_pruning_legality(synthetic_region_ir(), make_region_pruning_request("tiny", "ffn_out", indices=[0]))
    repairs = compute_region_minimal_repair_set(synthetic_region_ir(), result.request, result.constraint_satisfaction)

    assert any(item.repair_type == "same_indices" for item in repairs)
