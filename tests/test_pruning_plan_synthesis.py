from __future__ import annotations

from model_analysis.pruning_plan_synthesis import detect_ffn_evidence_for_candidate, pruning_plan_set_to_dict, synthesize_pruning_plans


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


def ffn_candidate_with_sources(model_name: str, sources: list[str]) -> dict:
    item = candidate()
    item["op_semantics_evidence"] = [
        {
            "source_name": source,
            "semantic_kind": "parameterized_linear_matmul",
            "semantic_category": "parameterized_projection",
        }
        for source in sources
    ]
    item["model_name"] = model_name
    return item


def generic_ops(expansion: str, activation: str | None, contraction: str, *, op_type: str = "MatMul") -> list[dict]:
    ops = [
        op("exp", expansion, "parameterized_linear_matmul", op_type),
        op("con", contraction, "parameterized_linear_matmul", op_type),
    ]
    if activation:
        ops.insert(1, op("act", activation, "gelu_elementwise", "Relu"))
    return ops


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
        op("attn_mm", f"{base}/attention/output/dense/MatMul", "parameterized_linear_matmul"),
        op("attn_bias", f"{base}/attention/output/dense/Add", "linear_bias_add", "Add"),
        op("attn_res", f"{base}/attention/output/Add", "residual_add", "Add"),
        op("attn_ln", f"{base}/attention/output/LayerNorm/LayerNormalization", "layernorm", "LayerNormalization"),
        *full_ops(),
    ]
    plan = build_plan_set(ops=ops)["plans"][0]
    action_locations = " ".join(item["target_source_name"] for item in plan["actions"])
    forbidden_locations = " ".join(item["location"] for item in plan["forbidden_actions"])

    assert "/attention/output/dense/" not in action_locations
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


def test_generic_ffn_matcher_finds_bert_pattern() -> None:
    base = "/model/bert/encoder/layer.0"
    ops = generic_ops(
        f"{base}/intermediate/dense/MatMul",
        f"{base}/intermediate/intermediate_act_fn/Erf",
        f"{base}/output/dense/MatMul",
    )
    ops.append(op("exp_bias", f"{base}/intermediate/dense/Add", "linear_bias_add", "Add"))
    ops.append(op("con_bias", f"{base}/output/dense/Add", "linear_bias_add", "Add"))
    evidence = detect_ffn_evidence_for_candidate(ffn_candidate_with_sources("bert-base-uncased", [op["source_name"] for op in ops]), {"ops": ops}, {}, "bert-base-uncased")

    assert evidence.family == "bert_encoder"
    assert evidence.evidence_status == "complete"


def test_generic_ffn_matcher_finds_distilbert_pattern() -> None:
    base = "/model/distilbert/transformer/layer.0"
    ops = generic_ops(f"{base}/ffn/lin1/MatMul", f"{base}/ffn/activation/Erf", f"{base}/ffn/lin2/MatMul")
    ops.append(op("exp_bias", f"{base}/ffn/lin1/Add", "linear_bias_add", "Add"))
    ops.append(op("con_bias", f"{base}/ffn/lin2/Add", "linear_bias_add", "Add"))
    evidence = detect_ffn_evidence_for_candidate(ffn_candidate_with_sources("distilbert-base-uncased", [op["source_name"] for op in ops]), {"ops": ops}, {}, "distilbert-base-uncased")

    assert evidence.family == "distilbert_encoder"
    assert evidence.evidence_status == "complete"


def test_generic_ffn_matcher_finds_opt_pattern() -> None:
    base = "/model/decoder/layers.0"
    ops = generic_ops(f"{base}/fc1/Gemm", f"{base}/activation_fn/Relu", f"{base}/fc2/Gemm", op_type="Gemm")
    evidence = detect_ffn_evidence_for_candidate(ffn_candidate_with_sources("facebook/opt-125m", [op["source_name"] for op in ops]), {"ops": ops}, {}, "facebook/opt-125m")

    assert evidence.family == "opt_decoder"
    assert evidence.evidence_status == "complete"


def test_generic_ffn_matcher_finds_vit_pattern() -> None:
    base = "/model/vit/layers.0"
    ops = generic_ops(f"{base}/mlp/fc1/MatMul", f"{base}/mlp/activation_fn/Erf", f"{base}/mlp/fc2/MatMul")
    ops.append(op("exp_bias", f"{base}/mlp/fc1/Add", "linear_bias_add", "Add"))
    ops.append(op("con_bias", f"{base}/mlp/fc2/Add", "linear_bias_add", "Add"))
    evidence = detect_ffn_evidence_for_candidate(ffn_candidate_with_sources("google/vit-base-patch16-224", [op["source_name"] for op in ops]), {"ops": ops}, {}, "google/vit-base-patch16-224")

    assert evidence.family == "vit_encoder"
    assert evidence.evidence_status == "complete"


def test_generic_ffn_matcher_finds_gpt2_pattern() -> None:
    base = "/model/transformer/h.0"
    ops = generic_ops(f"{base}/mlp/c_fc/Gemm", f"{base}/mlp/act/Tanh", f"{base}/mlp/c_proj/Gemm", op_type="Gemm")
    evidence = detect_ffn_evidence_for_candidate(ffn_candidate_with_sources("gpt2", [op["source_name"] for op in ops]), {"ops": ops}, {}, "gpt2")

    assert evidence.family == "gpt2_decoder"
    assert evidence.evidence_status == "complete"


def test_generic_ffn_matcher_reports_missing_activation() -> None:
    base = "/model/decoder/layers.0"
    ops = generic_ops(f"{base}/fc1/Gemm", None, f"{base}/fc2/Gemm", op_type="Gemm")
    evidence = detect_ffn_evidence_for_candidate(ffn_candidate_with_sources("facebook/opt-125m", [op["source_name"] for op in ops]), {"ops": ops}, {}, "facebook/opt-125m")

    assert evidence.evidence_status == "partial"
    assert "missing activation evidence" in evidence.missing_evidence


def test_plan_synthesis_uses_generic_opt_evidence() -> None:
    base = "/model/decoder/layers.0"
    ops = [
        op("fc1", f"{base}/fc1/Gemm", "parameterized_linear_matmul", "Gemm"),
        op("relu", f"{base}/activation_fn/Relu", "gelu_elementwise", "Relu"),
        op("fc2", f"{base}/fc2/Gemm", "parameterized_linear_matmul", "Gemm"),
    ]
    item = ffn_candidate_with_sources("facebook/opt-125m", [op["source_name"] for op in ops])
    data = build_plan_set(candidates=[item], ops=ops)
    plan = data["plans"][0]
    actions = {action["action_type"] for action in plan["actions"]}

    assert plan["plan_status"] == "ready_symbolic"
    assert {"prune_producer_output", "prune_bias", "prune_consumer_input"}.issubset(actions)
