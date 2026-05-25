from __future__ import annotations

from copy import deepcopy

from model_analysis.pruning_plan_synthesis import pruning_plan_set_to_dict, synthesize_pruning_plans
from model_analysis.pruning_plan_validation import pruning_plan_validation_set_to_dict, validate_pruning_plans


def candidate() -> dict:
    return {
        "candidate_id": "cand0",
        "region_id": "ffn0",
        "region_name": "Layer 0 Feed Forward",
        "candidate_region_id": "ffn0",
        "candidate_region_name": "Layer 0 Feed Forward",
        "candidate_kind": "feedforward_intermediate_pruning",
        "semantic_category": "feed_forward_block",
        "pruning_class": "safe",
        "target_dimension": "intermediate_dim",
        "rank_score": 95,
        "confidence": "high",
        "required_repairs": [{"obligation_type": "same_indices_across_mlp"}, {"obligation_type": "prune_consumer_input"}],
        "blockers": [],
    }


def region() -> dict:
    return {
        "region_id": "ffn0",
        "region_name": "Layer 0 Feed Forward",
        "semantic_category": "feed_forward_block",
        "pruning_role": "directly_prunable",
        "repair_obligations": [{"obligation_type": "same_indices_across_mlp", "required": True}, {"obligation_type": "prune_consumer_input", "required": True}],
        "propagation_rules": [{"rule_type": "same_indices_across_mlp"}],
        "blockers": [],
    }


def op(op_id: str, path: str, kind: str, op_type: str = "MatMul", roles: dict | None = None) -> dict:
    category = "elementwise_index_preserving" if kind.startswith("gelu") else "parameterized_projection"
    parameterized = kind in {"parameterized_linear_matmul", "linear_bias_add"}
    return {
        "op_id": op_id,
        "source_name": path,
        "op_type": op_type,
        "semantic_kind": kind,
        "semantic_category": category,
        "parameterized": parameterized,
        "dimension_roles": roles or {},
        "pruning_effect": {"direct_pruning": "allowed", "required_repairs": [], "blockers": [], "reason": ""},
    }


def full_ops() -> list[dict]:
    base = "/model/bert/encoder/layer.0"
    return [
        op("int_mm", f"{base}/intermediate/dense/MatMul", "parameterized_linear_matmul", roles={"input": "hidden_dim", "output": "intermediate_dim"}),
        op("int_add", f"{base}/intermediate/dense/Add", "linear_bias_add", "Add", {"output": "intermediate_dim"}),
        op("gelu_erf", f"{base}/intermediate/intermediate_act_fn/Erf", "gelu_erf", "Erf", {"input": "intermediate_dim", "output": "intermediate_dim"}),
        op("gelu_mul", f"{base}/intermediate/intermediate_act_fn/Mul", "gelu_mul", "Mul", {"input": "intermediate_dim", "output": "intermediate_dim"}),
        op("out_mm", f"{base}/output/dense/MatMul", "parameterized_linear_matmul", roles={"input": "intermediate_dim", "output": "hidden_dim"}),
        op("out_add", f"{base}/output/dense/Add", "linear_bias_add", "Add", {"output": "hidden_dim"}),
        op("res", f"{base}/output/Add", "residual_add", "Add", {"output": "hidden_dim"}),
        op("ln", f"{base}/output/LayerNorm/LayerNormalization", "layernorm", "LayerNormalization", {"output": "hidden_dim"}),
    ]


def fixtures() -> tuple[dict, dict, dict, dict]:
    ranking = {"model_name": "synthetic", "candidates": [candidate()]}
    regions = {"model_name": "synthetic", "regions": [region()]}
    ops = {"model_name": "synthetic", "ops": full_ops()}
    plan_set = pruning_plan_set_to_dict(synthesize_pruning_plans(ranking, regions, ops))
    return plan_set, ranking, regions, ops


def validated(plan_set: dict | None = None, ranking: dict | None = None, regions: dict | None = None, ops: dict | None = None) -> dict:
    default_plan, default_ranking, default_regions, default_ops = fixtures()
    return pruning_plan_validation_set_to_dict(
        validate_pruning_plans(
            plan_set or default_plan,
            ranking or default_ranking,
            regions or default_regions,
            ops or default_ops,
        )
    )


def failed_checks(data: dict) -> set[str]:
    return {check["check_type"] for check in data["validations"][0]["checks"] if check["status"] == "fail"}


def test_complete_ffn_plan_validates_as_valid() -> None:
    data = validated()

    assert data["summary"]["valid_plans"] == 1
    assert data["validations"][0]["validation_score"] == 100


def test_missing_prune_producer_output_action_invalid() -> None:
    plan_set, ranking, regions, ops = fixtures()
    plan_set["plans"][0]["actions"] = [a for a in plan_set["plans"][0]["actions"] if a["action_type"] != "prune_producer_output"]

    data = validated(plan_set, ranking, regions, ops)

    assert data["validations"][0]["validation_status"] == "invalid"
    assert "producer_output_pruned" in failed_checks(data)


def test_missing_prune_bias_action_invalid() -> None:
    plan_set, ranking, regions, ops = fixtures()
    plan_set["plans"][0]["actions"] = [a for a in plan_set["plans"][0]["actions"] if a["action_type"] != "prune_bias"]

    assert "bias_pruned" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_missing_prune_consumer_input_action_invalid() -> None:
    plan_set, ranking, regions, ops = fixtures()
    plan_set["plans"][0]["actions"] = [a for a in plan_set["plans"][0]["actions"] if a["action_type"] != "prune_consumer_input"]

    assert "consumer_input_pruned" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_missing_gelu_propagation_invalid() -> None:
    plan_set, ranking, regions, ops = fixtures()
    plan_set["plans"][0]["propagation"] = []

    assert "gelu_index_preserving" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_wrong_producer_op_semantics_fails() -> None:
    plan_set, ranking, regions, ops = fixtures()
    ops = deepcopy(ops)
    ops["ops"][0]["semantic_kind"] = "linear_bias_add"

    assert "op_semantics_agree" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_attention_contraction_as_producer_fails() -> None:
    plan_set, ranking, regions, ops = fixtures()
    ops = deepcopy(ops)
    ops["ops"][0]["semantic_kind"] = "attention_score_matmul"
    ops["ops"][0]["parameterized"] = False
    ops["ops"][0]["pruning_effect"]["direct_pruning"] = "blocked"

    assert "op_semantics_agree" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_missing_same_indices_repair_fails() -> None:
    plan_set, ranking, regions, ops = fixtures()
    plan_set["plans"][0]["required_repairs"] = [r for r in plan_set["plans"][0]["required_repairs"] if r["repair_type"] != "same_indices_across_mlp"]
    regions["regions"][0]["repair_obligations"] = []
    regions["regions"][0]["propagation_rules"] = []

    assert "same_indices_across_mlp" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_hidden_dim_pruning_action_invalid() -> None:
    plan_set, ranking, regions, ops = fixtures()
    bad = deepcopy(plan_set["plans"][0]["actions"][0])
    bad["action_id"] = "bad_hidden"
    bad["dimension"] = "hidden_dim"
    plan_set["plans"][0]["actions"].append(bad)

    assert "hidden_dim_preserved" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_missing_residual_forbidden_action_invalid() -> None:
    plan_set, ranking, regions, ops = fixtures()
    plan_set["plans"][0]["forbidden_actions"] = [a for a in plan_set["plans"][0]["forbidden_actions"] if "output/Add" not in a["location"]]

    assert "residual_hidden_not_pruned" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_missing_layernorm_forbidden_action_invalid() -> None:
    plan_set, ranking, regions, ops = fixtures()
    plan_set["plans"][0]["forbidden_actions"] = [a for a in plan_set["plans"][0]["forbidden_actions"] if "LayerNorm" not in a["location"]]

    assert "layernorm_hidden_not_pruned" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_output_dense_bias_pruning_action_invalid() -> None:
    plan_set, ranking, regions, ops = fixtures()
    bad = deepcopy(plan_set["plans"][0]["actions"][1])
    bad["action_id"] = "bad_output_bias"
    bad["target_source_name"] = "/model/bert/encoder/layer.0/output/dense/Add"
    bad["dimension"] = "hidden_dim"
    plan_set["plans"][0]["actions"].append(bad)

    assert "output_bias_not_pruned" in failed_checks(validated(plan_set, ranking, regions, ops))


def test_unknown_critical_op_semantics_invalid() -> None:
    plan_set, ranking, regions, ops = fixtures()
    ops = deepcopy(ops)
    ops["ops"][0]["semantic_kind"] = "unknown"

    assert "no_unknown_critical_ops" in failed_checks(validated(plan_set, ranking, regions, ops))
