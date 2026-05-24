from __future__ import annotations

from model_analysis.pruning_plan_synthesis import pruning_plan_set_to_dict, synthesize_pruning_plans


def candidate(kind: str = "feedforward_intermediate_pruning") -> dict:
    return {
        "candidate_id": "cand0",
        "region_id": "ffn0",
        "region_name": "Layer 0 Feed Forward",
        "candidate_region_id": "ffn0",
        "candidate_region_name": "Layer 0 Feed Forward",
        "candidate_kind": kind,
        "semantic_category": "feed_forward_block" if kind == "feedforward_intermediate_pruning" else "ffn_intermediate_projection",
        "pruning_class": "safe",
        "target_dimension": "intermediate_dim",
        "rank_score": 95,
        "confidence": "high",
        "required_repairs": [{"obligation_type": "same_indices_across_mlp"}, {"obligation_type": "prune_consumer_input"}],
        "blockers": [],
        "reason": "synthetic",
    }


def op(op_id: str, path: str, kind: str, op_type: str = "MatMul") -> dict:
    category = "elementwise_index_preserving" if kind.startswith("gelu") else "parameterized_projection"
    return {
        "op_id": op_id,
        "source_name": path,
        "op_type": op_type,
        "semantic_kind": kind,
        "semantic_category": category,
        "dimension_roles": {},
        "pruning_effect": {"direct_pruning": "allowed", "required_repairs": [], "blockers": [], "reason": ""},
    }


def full_ops() -> list[dict]:
    base = "/model/bert/encoder/layer.0"
    return [
        op("int_mm", f"{base}/intermediate/dense/MatMul", "parameterized_linear_matmul"),
        op("int_add", f"{base}/intermediate/dense/Add", "linear_bias_add", "Add"),
        op("gelu_erf", f"{base}/intermediate/intermediate_act_fn/Erf", "gelu_erf", "Erf"),
        op("gelu_mul", f"{base}/intermediate/intermediate_act_fn/Mul", "gelu_mul", "Mul"),
        op("out_mm", f"{base}/output/dense/MatMul", "parameterized_linear_matmul"),
        op("out_add", f"{base}/output/dense/Add", "linear_bias_add", "Add"),
        op("res", f"{base}/output/Add", "residual_add", "Add"),
        op("ln", f"{base}/output/LayerNorm/LayerNormalization", "layernorm", "LayerNormalization"),
    ]


def build_plan_set(candidates: list[dict] | None = None, ops: list[dict] | None = None) -> dict:
    return pruning_plan_set_to_dict(
        synthesize_pruning_plans(
            {"model_name": "synthetic", "candidates": candidates or [candidate()]},
            {"model_name": "synthetic", "regions": [{"region_id": "ffn0", "semantic_category": "feed_forward_block", "pruning_role": "directly_prunable"}]},
            {"model_name": "synthetic", "ops": ops if ops is not None else full_ops()},
        )
    )


def test_safe_feedforward_candidate_produces_ready_plan() -> None:
    data = build_plan_set()

    assert data["summary"]["total_plans"] == 1
    assert data["plans"][0]["plan_status"] == "ready_symbolic"


def test_plan_uses_stable_symbolic_index_set() -> None:
    plan = build_plan_set()["plans"][0]

    assert plan["symbolic_index_set"]["name"] == "I_layer_0_intermediate"


def test_plan_contains_required_prune_actions() -> None:
    actions = {action["action_type"]: action for action in build_plan_set()["plans"][0]["actions"]}

    assert actions["prune_producer_output"]["target_axis"] == "output_dim"
    assert actions["prune_producer_output"]["dimension"] == "intermediate_dim"
    assert actions["prune_bias"]["target_axis"] == "bias_dim"
    assert actions["prune_consumer_input"]["target_axis"] == "input_dim"


def test_plan_contains_gelu_propagation() -> None:
    propagation = build_plan_set()["plans"][0]["propagation"]

    assert any(step["semantic_kind"] == "gelu_erf" for step in propagation)
    assert all(step["index_mapping"] == "no_index_change" for step in propagation)


def test_plan_preserves_and_forbids_hidden_dim_pruning() -> None:
    plan = build_plan_set()["plans"][0]

    preserved_locations = " ".join(item["location"] for item in plan["preserved_dimensions"])
    forbidden_locations = " ".join(item["location"] for item in plan["forbidden_actions"])
    assert "output/dense/MatMul" in preserved_locations
    assert "output/dense/Add" in preserved_locations
    assert "output/Add" in forbidden_locations
    assert "LayerNorm" in forbidden_locations
    assert "output/dense/Add" in forbidden_locations


def test_plan_uses_ffn_residual_and_layernorm_not_attention_output() -> None:
    base = "/model/bert/encoder/layer.0"
    ops = [
        op("attn_res", f"{base}/attention/output/Add", "residual_add", "Add"),
        op("attn_ln", f"{base}/attention/output/LayerNorm/LayerNormalization", "layernorm", "LayerNormalization"),
        *full_ops(),
    ]
    plan = build_plan_set(ops=ops)["plans"][0]
    forbidden_locations = " ".join(item["location"] for item in plan["forbidden_actions"])

    assert "/attention/output/" not in forbidden_locations
    assert f"{base}/output/Add" in forbidden_locations
    assert f"{base}/output/LayerNorm/LayerNormalization" in forbidden_locations


def test_missing_required_evidence_marks_plan_incomplete() -> None:
    data = build_plan_set(ops=[op("int_mm", "/model/bert/encoder/layer.0/intermediate/dense/MatMul", "parameterized_linear_matmul")])
    plan = data["plans"][0]

    assert plan["plan_status"] == "incomplete"
    assert plan["warnings"]


def test_component_candidate_does_not_create_full_plan() -> None:
    data = build_plan_set(candidates=[candidate("projection_output_pruning")])

    assert data["summary"]["total_plans"] == 0
