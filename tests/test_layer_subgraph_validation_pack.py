from __future__ import annotations

from pathlib import Path

from model_analysis.layer_subgraph_validation_pack import (
    build_layer_subgraph_validation_pack,
    layer_subgraph_pack_to_dict,
    layer_subgraph_pack_to_markdown,
    layer_subgraph_record_to_markdown,
    select_expandable_layer_nodes,
)


def abstract_expansion() -> dict:
    return {
        "records": [
            {
                "region_id": "query0",
                "region_type": "LinearProjectionRegion",
                "name": "Layer 0 Query Projection",
                "section": "Encoder Layer 0",
                "op_range": "1-2",
                "recursive_primitive_leaves": [{"id": "op_q", "source_name": "/layer.0/attention/self/query/MatMul", "op_index": 1}],
            },
            {
                "region_id": "score0",
                "region_type": "LinearProjectionRegion",
                "name": "Layer 0 Attention Score MatMul",
                "section": "Encoder Layer 0",
                "op_range": "3-3",
                "recursive_primitive_leaves": [{"id": "op_score", "source_name": "/layer.0/attention/self/MatMul", "op_index": 3}],
            },
            {
                "region_id": "mask0",
                "region_type": "ResidualMergeRegion",
                "name": "Layer 0 Attention Mask Add",
                "section": "Encoder Layer 0",
                "op_range": "4-4",
                "recursive_primitive_leaves": [{"id": "op_mask", "source_name": "/layer.0/attention/self/Add", "op_index": 4}],
            },
            {
                "region_id": "ffn0",
                "region_type": "FeedForwardRegion",
                "name": "Layer 0 Feed Forward",
                "section": "Encoder Layer 0",
                "op_range": "10-20",
                "immediate_expansion": [{"id": "ffn_int0", "kind": "abstract"}],
                "recursive_primitive_leaves": [
                    {"id": "op_int", "source_name": "/layer.0/intermediate/dense/MatMul", "op_index": 10},
                    {"id": "op_gelu", "source_name": "/layer.0/intermediate/intermediate_act_fn/Erf", "op_index": 11},
                    {"id": "op_out", "source_name": "/layer.0/output/dense/MatMul", "op_index": 12},
                ],
            },
            {
                "region_id": "ffn_int0",
                "region_type": "LinearProjectionRegion",
                "name": "Layer 0 FFN Intermediate Projection",
                "section": "Encoder Layer 0",
                "op_range": "10-10",
                "recursive_primitive_leaves": [{"id": "op_int", "source_name": "/layer.0/intermediate/dense/MatMul", "op_index": 10}],
            },
            {
                "region_id": "dup_ffn_int0",
                "region_type": "LinearProjectionRegion",
                "name": "Layer 0 FFN Intermediate Projection",
                "section": "Encoder Layer 0",
                "op_range": "10-10",
                "recursive_primitive_leaves": [{"id": "op_int", "source_name": "/layer.0/intermediate/dense/MatMul", "op_index": 10}],
            },
        ]
    }


def tensor_ir() -> dict:
    ops = []
    for idx, (op_id, source, op_type) in enumerate(
        [
            ("op_q", "/layer.0/attention/self/query/MatMul", "MatMul"),
            ("op_score", "/layer.0/attention/self/MatMul", "MatMul"),
            ("op_mask", "/layer.0/attention/self/Add", "Add"),
            ("op_int", "/layer.0/intermediate/dense/MatMul", "MatMul"),
            ("op_gelu", "/layer.0/intermediate/intermediate_act_fn/Erf", "Erf"),
            ("op_out", "/layer.0/output/dense/MatMul", "MatMul"),
        ]
    ):
        ops.append({"op_id": op_id, "name": source, "source_node_name": source, "op_type": op_type, "inputs": [f"in{idx}"], "outputs": [f"out{idx}"], "predecessor_ops": [], "successor_ops": []})
    return {"ops": ops}


def op_semantics() -> dict:
    return {
        "ops": [
            {"op_id": "op_q", "source_name": "/layer.0/attention/self/query/MatMul", "semantic_kind": "parameterized_linear_matmul", "semantic_category": "parameterized_projection", "parameterized": True, "index_behavior": "creates_prunable_output_axis", "pruning_effect": {"direct_pruning": "allowed"}},
            {"op_id": "op_score", "source_name": "/layer.0/attention/self/MatMul", "semantic_kind": "attention_score_matmul", "semantic_category": "attention_contraction", "parameterized": False, "index_behavior": "axis_contraction", "pruning_effect": {"direct_pruning": "blocked"}},
            {"op_id": "op_mask", "source_name": "/layer.0/attention/self/Add", "semantic_kind": "attention_mask_add", "semantic_category": "attention_masking", "parameterized": False, "index_behavior": "broadcast_metadata", "pruning_effect": {"direct_pruning": "not_applicable"}},
            {"op_id": "op_int", "source_name": "/layer.0/intermediate/dense/MatMul", "semantic_kind": "parameterized_linear_matmul", "semantic_category": "parameterized_projection", "parameterized": True, "index_behavior": "creates_prunable_output_axis", "pruning_effect": {"direct_pruning": "allowed"}},
            {"op_id": "op_gelu", "source_name": "/layer.0/intermediate/intermediate_act_fn/Erf", "semantic_kind": "gelu_erf", "semantic_category": "elementwise_index_preserving", "parameterized": False, "index_behavior": "index_preserving", "pruning_effect": {"direct_pruning": "not_applicable"}},
            {"op_id": "op_out", "source_name": "/layer.0/output/dense/MatMul", "semantic_kind": "parameterized_linear_matmul", "semantic_category": "parameterized_projection", "parameterized": True, "index_behavior": "consumes_pruned_input_axis", "pruning_effect": {"direct_pruning": "allowed"}},
        ]
    }


def region_semantics() -> dict:
    return {
        "regions": [
            {"region_id": "query0", "region_name": "Layer 0 Query Projection", "source_region_type": "LinearProjectionRegion", "semantic_category": "query_projection", "pruning_role": "directly_prunable", "blockers": [{"blocker_type": "attention_head_mapping_unproven"}], "repair_obligations": []},
            {"region_id": "score0", "region_name": "Layer 0 Attention Score MatMul", "source_region_type": "LinearProjectionRegion", "semantic_category": "attention_score_matmul", "pruning_role": "constraint_carrier", "blockers": [{"blocker_type": "attention_head_mapping_unproven"}], "repair_obligations": []},
            {"region_id": "mask0", "region_name": "Layer 0 Attention Mask Add", "source_region_type": "ResidualMergeRegion", "semantic_category": "attention_mask_add", "pruning_role": "constraint_carrier", "blockers": [], "repair_obligations": []},
            {"region_id": "ffn0", "region_name": "Layer 0 Feed Forward", "source_region_type": "FeedForwardRegion", "semantic_category": "feed_forward_block", "pruning_role": "directly_prunable", "blockers": [], "repair_obligations": [{"obligation_type": "same_indices_across_mlp"}]},
            {"region_id": "ffn_int0", "region_name": "Layer 0 FFN Intermediate Projection", "source_region_type": "LinearProjectionRegion", "semantic_category": "ffn_intermediate_projection", "pruning_role": "directly_prunable", "blockers": [], "repair_obligations": []},
        ]
    }


def ranking() -> dict:
    return {
        "candidates": [
            {"candidate_id": "cand_query", "region_id": "query0", "region_name": "Layer 0 Query Projection", "semantic_category": "query_projection", "candidate_kind": "attention_projection_constrained_pruning", "pruning_class": "constrained", "rank_score": 55, "confidence": "medium", "target_dimension": "head_dim", "blockers": [{"blocker_type": "attention_head_mapping_unproven"}], "reason": "attention_head_mapping_unproven"},
            {"candidate_id": "cand_score", "region_id": "score0", "region_name": "Layer 0 Attention Score MatMul", "semantic_category": "attention_score_matmul", "candidate_kind": "attention_contraction_blocked", "pruning_class": "blocked", "rank_score": 10, "confidence": "high", "target_dimension": "head_dim", "blockers": [], "reason": "Q x K^T contraction, not learned projection"},
            {"candidate_id": "cand_mask", "region_id": "mask0", "region_name": "Layer 0 Attention Mask Add", "semantic_category": "attention_mask_add", "candidate_kind": "auxiliary_metadata_flow", "pruning_class": "auxiliary", "rank_score": 5, "confidence": "high", "target_dimension": "mask_dim", "blockers": [], "reason": "mask flow"},
            {"candidate_id": "cand_ffn", "region_id": "ffn0", "region_name": "Layer 0 Feed Forward", "semantic_category": "feed_forward_block", "candidate_kind": "feedforward_intermediate_pruning", "pruning_class": "safe", "rank_score": 95, "confidence": "high", "target_dimension": "intermediate_dim", "blockers": [], "reason": "safe FFN"},
            {"candidate_id": "cand_int", "region_id": "ffn_int0", "region_name": "Layer 0 FFN Intermediate Projection", "semantic_category": "ffn_intermediate_projection", "candidate_kind": "projection_output_pruning", "pruning_class": "safe", "rank_score": 85, "confidence": "high", "target_dimension": "intermediate_dim", "blockers": [], "reason": "component"},
        ]
    }


def plans() -> dict:
    return {"plans": [{"plan_id": "plan_ffn", "candidate_id": "cand_ffn", "candidate_region_id": "ffn0", "plan_kind": "feedforward_intermediate_dim_plan", "plan_status": "ready_symbolic", "target_dimension": "intermediate_dim", "actions": [{"action_type": "prune_producer_output"}], "symbolic_index_set": {"name": "I_layer_0_intermediate"}}]}


def validations() -> dict:
    return {"validations": [{"validation_id": "val_ffn", "plan_id": "plan_ffn", "candidate_region_name": "Layer 0 Feed Forward", "validation_status": "valid", "validation_score": 100, "checks": []}]}


def build_pack(tmp_path: Path | None = None, source_onnx_path: Path | None = None) -> dict:
    pack = build_layer_subgraph_validation_pack(
        model_name="bert-base-uncased",
        layer_index=0,
        tensor_ir=tensor_ir(),
        op_semantics=op_semantics(),
        structural_region_tree={},
        region_pruning_semantics=region_semantics(),
        ranking=ranking(),
        plans=plans(),
        validations=validations(),
        abstract_expansion=abstract_expansion(),
        report_root=tmp_path / "reports" if tmp_path else None,
        artifact_root=tmp_path / "artifacts" if tmp_path else None,
        source_onnx_path=source_onnx_path,
        export_onnx=source_onnx_path is not None,
    )
    return layer_subgraph_pack_to_dict(pack)


def by_name(pack: dict, name: str) -> dict:
    return next(item for item in pack["subgraphs"] if item["display_name"] == name)


def test_select_expandable_nodes_for_layer_and_deduplicates_by_region_id() -> None:
    nodes = select_expandable_layer_nodes(abstract_expansion(), region_semantics(), 0)
    ids = [item.get("region_id") for item in nodes]

    assert "ffn0" in ids
    assert len(ids) == len(set(ids))


def test_selected_nodes_preserve_topological_order() -> None:
    names = [item["name"] for item in select_expandable_layer_nodes(abstract_expansion(), region_semantics(), 0)]

    assert names.index("Layer 0 Query Projection") < names.index("Layer 0 Attention Score MatMul") < names.index("Layer 0 Feed Forward")


def test_feedforward_record_is_safe_with_valid_plan() -> None:
    item = by_name(build_pack(), "Layer 0 Feed Forward")

    assert item["classification"]["pruning_class"] == "safe"
    assert item["classification"]["plan_status"] == "valid_plan"
    assert item["classification"]["validation_status"] == "valid"


def test_attention_score_matmul_is_blocked_no_plan_expected() -> None:
    item = by_name(build_pack(), "Layer 0 Attention Score MatMul")

    assert item["classification"]["pruning_class"] == "blocked"
    assert item["classification"]["plan_status"] == "no_plan_expected"
    assert "not a learned parameter projection" in item["explanation"]


def test_query_projection_is_constrained_with_attention_blocker() -> None:
    item = by_name(build_pack(), "Layer 0 Query Projection")

    assert item["classification"]["pruning_class"] == "constrained"
    assert "attention_head_mapping_unproven" in str(item["local_ranking"])


def test_attention_mask_add_is_auxiliary() -> None:
    item = by_name(build_pack(), "Layer 0 Attention Mask Add")

    assert item["classification"]["pruning_class"] == "auxiliary"
    assert item["classification"]["plan_status"] == "no_plan_expected"


def test_local_slices_include_expected_analysis() -> None:
    item = by_name(build_pack(), "Layer 0 Query Projection")

    assert item["primitive_ops"]
    assert item["local_op_semantics"]
    assert item["local_region_semantics"]
    assert item["local_ranking"]


def test_feedforward_local_slices_include_plan_and_validation() -> None:
    item = by_name(build_pack(), "Layer 0 Feed Forward")

    assert item["local_plans"]
    assert item["local_validations"]


def test_onnx_export_failure_is_recorded_without_failing_pack(tmp_path: Path) -> None:
    bad = tmp_path / "bad.onnx"
    bad.write_text("not onnx", encoding="utf-8")
    pack = build_pack(tmp_path, bad)

    assert pack["summary"]["onnx_failed"] >= 1


def test_index_markdown_contains_subgraph_table() -> None:
    text = layer_subgraph_pack_to_markdown(build_pack())

    assert "## Subgraph Table" in text
    assert "Layer 0 Feed Forward" in text


def test_per_node_explanation_contains_verdict() -> None:
    text = layer_subgraph_record_to_markdown(by_name(build_pack(), "Layer 0 Feed Forward"))

    assert "## Verdict" in text
    assert "safe" in text


def test_generic_opt_layer_pack_falls_back_to_block_grouping() -> None:
    base = "/model/decoder/layers.0"
    ops = [
        {"op_id": "fc1", "name": f"{base}/fc1/MatMul", "source_node_name": f"{base}/fc1/MatMul", "op_type": "MatMul", "inputs": [], "outputs": [], "predecessor_ops": [], "successor_ops": []},
        {"op_id": "act", "name": f"{base}/activation_fn/Relu", "source_node_name": f"{base}/activation_fn/Relu", "op_type": "Relu", "inputs": [], "outputs": [], "predecessor_ops": [], "successor_ops": []},
        {"op_id": "fc2", "name": f"{base}/fc2/MatMul", "source_node_name": f"{base}/fc2/MatMul", "op_type": "MatMul", "inputs": [], "outputs": [], "predecessor_ops": [], "successor_ops": []},
    ]
    opsem = {
        "ops": [
            {"op_id": "fc1", "source_name": f"{base}/fc1/MatMul", "op_type": "MatMul", "topological_index": 1, "semantic_kind": "parameterized_linear_matmul", "semantic_category": "parameterized_projection", "parameterized": True, "dimension_roles": {"input": "hidden_dim", "output": "intermediate_dim"}, "pruning_effect": {"direct_pruning": "allowed"}},
            {"op_id": "act", "source_name": f"{base}/activation_fn/Relu", "op_type": "Relu", "topological_index": 2, "semantic_kind": "gelu_elementwise", "semantic_category": "elementwise_index_preserving", "parameterized": False, "dimension_roles": {"input": "intermediate_dim", "output": "intermediate_dim"}, "pruning_effect": {"direct_pruning": "not_applicable"}},
            {"op_id": "fc2", "source_name": f"{base}/fc2/MatMul", "op_type": "MatMul", "topological_index": 3, "semantic_kind": "parameterized_linear_matmul", "semantic_category": "parameterized_projection", "parameterized": True, "dimension_roles": {"input": "intermediate_dim", "output": "hidden_dim"}, "pruning_effect": {"direct_pruning": "allowed"}},
        ]
    }
    rank = {"candidates": [{"candidate_id": "cand", "candidate_kind": "feedforward_intermediate_pruning", "semantic_category": "feed_forward_block", "pruning_class": "safe", "op_semantics_evidence": [{"source_name": item["source_name"]} for item in opsem["ops"]]}]}
    plan = {"plans": [{"plan_id": "plan", "candidate_id": "cand", "plan_status": "ready_symbolic"}]}
    valid = {"validations": [{"validation_id": "val", "plan_id": "plan", "validation_status": "valid", "validation_score": 100, "checks": []}]}
    pack = build_layer_subgraph_validation_pack(
        model_name="facebook/opt-125m",
        layer_index=0,
        tensor_ir={"ops": ops},
        op_semantics=opsem,
        structural_region_tree={},
        region_pruning_semantics={"regions": []},
        ranking=rank,
        plans=plan,
        validations=valid,
        abstract_expansion=None,
        export_onnx=False,
    )
    data = layer_subgraph_pack_to_dict(pack)

    assert by_name(data, "OPT Decoder Block 0 MLP Block")["classification"]["plan_status"] == "valid_plan"
    assert any(item["display_name"].endswith("MLP Expansion Projection") for item in data["subgraphs"])
