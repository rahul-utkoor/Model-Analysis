from __future__ import annotations

from model_analysis.ir_analysis import check_pruning_legality, compute_minimal_repair_set, make_symbolic_pruning_request


def base_dimension(var_id: str, size: int = 8, prunable: bool = True, owner_name: str | None = None) -> dict:
    return {
        "var_id": var_id,
        "owner_name": owner_name or var_id,
        "owner_type": "linear",
        "size": size,
        "prunable": prunable,
        "semantic_role": "producer",
        "confidence": "medium",
    }


def local_ir() -> dict:
    return {
        "model_name": "tiny",
        "dimension_variables": [base_dimension("dim::local")],
        "constraint_equations": [],
        "equivalence_classes": [{"class_id": "e1", "members": ["dim::local"], "class_type": "unknown"}],
        "blocked_dimensions": [],
        "unresolved_constraints": [],
    }


def same_indices_ir() -> dict:
    return {
        "model_name": "tiny",
        "dimension_variables": [
            base_dimension("dim::fc1", owner_name="bert.encoder.layer.0.intermediate.dense"),
            base_dimension("dim::fc2", owner_name="bert.encoder.layer.0.output.dense"),
        ],
        "constraint_equations": [
            {
                "constraint_id": "c1",
                "lhs": "dim::fc1",
                "rhs": "dim::fc2",
                "relation": "same_indices",
                "direction": "bidirectional",
                "constraint_type": "mlp_intermediate_consistency",
                "confidence": "medium",
                "blocking": False,
                "reason": "MLP pair",
            }
        ],
        "equivalence_classes": [{"class_id": "e1", "members": ["dim::fc1", "dim::fc2"], "class_type": "mlp_intermediate"}],
        "blocked_dimensions": [],
        "unresolved_constraints": [],
    }


def residual_ir() -> dict:
    data = same_indices_ir()
    data["dimension_variables"] = [base_dimension("dim::hidden"), base_dimension("dim::peer")]
    data["constraint_equations"] = [
        {
            "constraint_id": "c2",
            "lhs": "dim::hidden",
            "rhs": "dim::peer",
            "relation": "eq",
            "direction": "bidirectional",
            "constraint_type": "residual_hidden_equality",
            "confidence": "medium",
            "blocking": True,
            "reason": "Residual equality",
        }
    ]
    data["equivalence_classes"] = [{"class_id": "e1", "members": ["dim::hidden", "dim::peer"], "class_type": "residual_hidden"}]
    data["blocked_dimensions"] = ["dim::hidden", "dim::peer"]
    return data


def unknown_ir() -> dict:
    data = same_indices_ir()
    data["constraint_equations"][0]["relation"] = "unknown"
    data["constraint_equations"][0]["constraint_type"] = "unknown_mapping"
    data["constraint_equations"][0]["blocking"] = True
    return data


def test_legal_local_pruning_with_no_constraints():
    request = make_symbolic_pruning_request("tiny", "dim::local", count=2)

    result = check_pruning_legality(local_ir(), request)

    assert result.status == "legal"


def test_same_indices_returns_legal_with_repairs():
    request = make_symbolic_pruning_request("tiny", "dim::fc1", indices=[0, 1])

    result = check_pruning_legality(same_indices_ir(), request)

    assert result.status == "legal_with_repairs"
    assert result.minimal_repair_set[0].repair_type == "same_indices"
    assert result.minimal_repair_set[0].executable_backend == "experimental_bert_mlp"


def test_residual_hidden_equality_rejected_with_blocker():
    request = make_symbolic_pruning_request("tiny", "dim::hidden", count=1)

    result = check_pruning_legality(residual_ir(), request)

    assert result.status == "rejected"
    assert result.blocking_reasons


def test_unknown_mapping_is_ambiguous():
    request = make_symbolic_pruning_request("tiny", "dim::fc1", count=1)

    result = check_pruning_legality(unknown_ir(), request)

    assert result.status == "ambiguous"
    assert result.unresolved_items


def test_invalid_indices_and_pruning_all_reject():
    bad_index = make_symbolic_pruning_request("tiny", "dim::local", indices=[99])
    all_indices = make_symbolic_pruning_request("tiny", "dim::local", indices=list(range(8)))

    assert check_pruning_legality(local_ir(), bad_index).status == "rejected"
    assert check_pruning_legality(local_ir(), all_indices).status == "rejected"


def test_compute_minimal_repair_set_contains_same_indices():
    request = make_symbolic_pruning_request("tiny", "dim::fc1", count=1)
    result = check_pruning_legality(same_indices_ir(), request)
    repairs = compute_minimal_repair_set(same_indices_ir(), request, result.constraint_satisfaction)

    assert repairs[0].repair_type == "same_indices"
