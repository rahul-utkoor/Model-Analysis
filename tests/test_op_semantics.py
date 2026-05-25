from __future__ import annotations

from model_analysis.op_semantics import build_op_semantics_ir, op_semantics_ir_to_dict


def op(op_id: str, path: str, op_type: str) -> dict:
    return {
        "op_id": op_id,
        "name": path,
        "source_node_name": path,
        "op_type": op_type,
        "canonical_op_type": op_type.lower(),
        "source_location": {"node_index": int(op_id.split("_")[-1])},
    }


def build_records() -> dict[str, dict]:
    tensor_ir = {
        "model_name": "synthetic",
        "source_frontend": "onnx",
        "ops": [
            op("op_0", "/model/bert/encoder/layer.0/intermediate/dense/MatMul", "MatMul"),
            op("op_1", "/model/bert/encoder/layer.0/output/dense/MatMul", "MatMul"),
            op("op_2", "/model/bert/encoder/layer.0/attention/self/query/MatMul", "MatMul"),
            op("op_3", "/model/bert/encoder/layer.0/attention/self/MatMul", "MatMul"),
            op("op_4", "/model/bert/encoder/layer.0/attention/self/MatMul_1", "MatMul"),
            op("op_5", "/model/bert/encoder/layer.0/attention/self/Add", "Add"),
            op("op_6", "/model/bert/encoder/layer.0/attention/output/Add", "Add"),
            op("op_7", "/model/bert/encoder/layer.0/intermediate/intermediate_act_fn/Erf", "Erf"),
            op("op_8", "/model/bert/encoder/layer.0/intermediate/intermediate_act_fn/Mul", "Mul"),
            op("op_9", "/model/bert/encoder/layer.0/output/LayerNorm/LayerNormalization", "LayerNormalization"),
            op("op_10", "/model/bert/embeddings/word_embeddings/Gather", "Gather"),
            op("op_11", "/model/bert/encoder/layer.0/attention/self/Reshape", "Reshape"),
            op("op_12", "/model/bert/encoder/layer.0/attention/self/Constant", "Constant"),
            op("op_13", "/model/bert/unknown/Foo", "Foo"),
            op("op_14", "/model/decoder/layers.0/fc1/Gemm", "Gemm"),
            op("op_15", "/model/decoder/layers.0/fc2/Gemm", "Gemm"),
            op("op_16", "/model/distilbert/transformer/layer.0/ffn/lin1/MatMul", "MatMul"),
            op("op_17", "/model/vit/layers.0/mlp/fc2/MatMul", "MatMul"),
            op("op_18", "/model/transformer/h.0/mlp/c_fc/Gemm", "Gemm"),
            op("op_19", "/model/decoder/layers.0/activation_fn/Relu", "Relu"),
        ],
    }
    data = op_semantics_ir_to_dict(build_op_semantics_ir(tensor_ir))
    return {item["op_id"]: item for item in data["ops"]}


def test_ffn_intermediate_matmul_is_parameterized_projection() -> None:
    record = build_records()["op_0"]

    assert record["semantic_kind"] == "parameterized_linear_matmul"
    assert record["semantic_category"] == "parameterized_projection"
    assert record["parameterized"] is True
    assert record["pruning_effect"]["direct_pruning"] == "allowed"
    assert record["dimension_roles"]["output"] == "intermediate_dim"


def test_ffn_output_matmul_roles() -> None:
    record = build_records()["op_1"]

    assert record["semantic_kind"] == "parameterized_linear_matmul"
    assert record["dimension_roles"]["input"] == "intermediate_dim"
    assert record["dimension_roles"]["output"] == "hidden_dim"


def test_query_matmul_is_parameterized_projection() -> None:
    record = build_records()["op_2"]

    assert record["semantic_kind"] == "parameterized_linear_matmul"
    assert record["semantic_category"] == "parameterized_projection"


def test_attention_score_matmul_is_blocked_contraction() -> None:
    record = build_records()["op_3"]

    assert record["semantic_kind"] == "attention_score_matmul"
    assert record["parameterized"] is False
    assert record["pruning_effect"]["direct_pruning"] == "blocked"
    assert "attention_head_mapping_unproven" in record["pruning_effect"]["blockers"]


def test_attention_context_matmul_is_blocked_contraction() -> None:
    record = build_records()["op_4"]

    assert record["semantic_kind"] == "attention_context_matmul"
    assert record["parameterized"] is False
    assert record["pruning_effect"]["direct_pruning"] == "blocked"


def test_attention_mask_add_is_not_residual() -> None:
    record = build_records()["op_5"]

    assert record["semantic_kind"] == "attention_mask_add"
    assert record["semantic_category"] == "attention_masking"
    assert record["semantic_kind"] != "residual_add"


def test_true_residual_add_blocks_hidden_dim() -> None:
    record = build_records()["op_6"]

    assert record["semantic_kind"] == "residual_add"
    assert record["semantic_category"] == "branch_merge"
    assert "residual_hidden_dim" in record["pruning_effect"]["blockers"]


def test_gelu_decomposition_is_index_preserving() -> None:
    records = build_records()

    assert records["op_7"]["semantic_category"] == "elementwise_index_preserving"
    assert records["op_8"]["index_behavior"] == "index_preserving"


def test_layernorm_protects_hidden_dim() -> None:
    record = build_records()["op_9"]

    assert record["semantic_kind"] == "layernorm"
    assert record["semantic_category"] == "normalization"
    assert "layernorm_hidden_dim" in record["pruning_effect"]["blockers"]


def test_embedding_gather_is_protected_lookup() -> None:
    record = build_records()["op_10"]

    assert record["semantic_kind"] == "embedding_gather"
    assert record["semantic_category"] == "embedding_lookup"
    assert record["parameterized"] is True
    assert record["pruning_effect"]["direct_pruning"] == "blocked"


def test_shape_and_constant_are_not_pruning_targets() -> None:
    records = build_records()

    assert records["op_11"]["semantic_category"] == "axis_transform"
    assert records["op_11"]["pruning_effect"]["direct_pruning"] == "not_applicable"
    assert records["op_12"]["semantic_category"] == "metadata_flow"


def test_unknown_op_gets_warning_blocker() -> None:
    record = build_records()["op_13"]

    assert record["semantic_kind"] == "unknown"
    assert record["semantic_category"] == "unknown"
    assert "unsupported_or_unknown_op_semantics" in record["pruning_effect"]["blockers"]


def test_opt_fc1_fc2_are_parameterized_ffn_roles() -> None:
    records = build_records()

    assert records["op_14"]["semantic_kind"] == "parameterized_linear_matmul"
    assert records["op_14"]["dimension_roles"]["output"] == "intermediate_dim"
    assert records["op_15"]["semantic_kind"] == "parameterized_linear_matmul"
    assert records["op_15"]["dimension_roles"]["input"] == "intermediate_dim"
    assert records["op_15"]["dimension_roles"]["output"] == "hidden_dim"


def test_distilbert_vit_gpt2_ffn_aliases_are_parameterized() -> None:
    records = build_records()

    assert records["op_16"]["dimension_roles"]["output"] == "intermediate_dim"
    assert records["op_17"]["dimension_roles"]["input"] == "intermediate_dim"
    assert records["op_18"]["dimension_roles"]["output"] == "intermediate_dim"


def test_generic_activation_alias_is_index_preserving() -> None:
    record = build_records()["op_19"]

    assert record["semantic_category"] == "elementwise_index_preserving"
    assert record["index_behavior"] == "index_preserving"
