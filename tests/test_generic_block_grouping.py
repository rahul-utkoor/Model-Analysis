from __future__ import annotations

from model_analysis.generic_block_grouping import detect_generic_blocks, generic_layer_records


def op(op_id: str, source: str, kind: str, category: str = "parameterized_projection", roles: dict | None = None, op_type: str = "MatMul") -> dict:
    return {
        "op_id": op_id,
        "source_name": source,
        "op_type": op_type,
        "topological_index": int(op_id.split("_")[-1]),
        "semantic_kind": kind,
        "semantic_category": category,
        "parameterized": category == "parameterized_projection",
        "dimension_roles": roles or {},
        "pruning_effect": {"direct_pruning": "allowed"},
    }


def mlp_ops(prefix: str, family: str = "distilbert") -> list[dict]:
    if family == "distilbert":
        return [
            op("op_1", f"{prefix}/ffn/lin1/MatMul", "parameterized_linear_matmul", roles={"input": "hidden_dim", "output": "intermediate_dim"}),
            op("op_2", f"{prefix}/ffn/lin1/Add", "linear_bias_add", roles={"output": "intermediate_dim"}, op_type="Add"),
            op("op_3", f"{prefix}/ffn/activation/Erf", "gelu_erf", "elementwise_index_preserving", {"input": "intermediate_dim", "output": "intermediate_dim"}, "Erf"),
            op("op_4", f"{prefix}/ffn/lin2/MatMul", "parameterized_linear_matmul", roles={"input": "intermediate_dim", "output": "hidden_dim"}),
        ]
    if family == "opt":
        return [
            op("op_1", f"{prefix}/fc1/MatMul", "parameterized_linear_matmul", roles={"input": "hidden_dim", "output": "intermediate_dim"}),
            op("op_2", f"{prefix}/activation_fn/Relu", "gelu_elementwise", "elementwise_index_preserving", {"input": "intermediate_dim", "output": "intermediate_dim"}, "Relu"),
            op("op_3", f"{prefix}/fc2/MatMul", "parameterized_linear_matmul", roles={"input": "intermediate_dim", "output": "hidden_dim"}),
        ]
    if family == "vit":
        return [
            op("op_1", f"{prefix}/mlp/fc1/MatMul", "parameterized_linear_matmul", roles={"input": "hidden_dim", "output": "intermediate_dim"}),
            op("op_2", f"{prefix}/mlp/activation_fn/Erf", "gelu_erf", "elementwise_index_preserving", {"input": "intermediate_dim", "output": "intermediate_dim"}, "Erf"),
            op("op_3", f"{prefix}/mlp/fc2/MatMul", "parameterized_linear_matmul", roles={"input": "intermediate_dim", "output": "hidden_dim"}),
        ]
    return [
        op("op_1", f"{prefix}/attn/c_proj/MatMul", "parameterized_linear_matmul", roles={"input": "hidden_dim", "output": "hidden_dim"}),
        op("op_2", f"{prefix}/mlp/c_fc/Gemm", "parameterized_linear_matmul", roles={"input": "hidden_dim", "output": "intermediate_dim"}, op_type="Gemm"),
        op("op_3", f"{prefix}/mlp/act/Gelu", "gelu_elementwise", "elementwise_index_preserving", {"input": "intermediate_dim", "output": "intermediate_dim"}, "Gelu"),
        op("op_4", f"{prefix}/mlp/c_proj/Gemm", "parameterized_linear_matmul", roles={"input": "intermediate_dim", "output": "hidden_dim"}, op_type="Gemm"),
    ]


def support(prefix: str) -> list[dict]:
    return [
        op("op_9", f"{prefix}/output/Add", "residual_add", "branch_merge", {"output": "hidden_dim"}, "Add"),
        op("op_10", f"{prefix}/LayerNorm/LayerNormalization", "layernorm", "normalization", {"output": "hidden_dim"}, "LayerNormalization"),
    ]


def candidate_for(ops: list[dict]) -> dict:
    return {
        "candidate_id": "cand_mlp",
        "candidate_kind": "feedforward_intermediate_pruning",
        "semantic_category": "feed_forward_block",
        "pruning_class": "safe",
        "op_semantics_evidence": [{"source_name": item["source_name"]} for item in ops],
    }


def plan_and_validation() -> tuple[dict, dict]:
    return (
        {"plans": [{"plan_id": "plan_mlp", "candidate_id": "cand_mlp", "plan_status": "ready_symbolic"}]},
        {"validations": [{"validation_id": "val_mlp", "plan_id": "plan_mlp", "validation_status": "valid", "validation_score": 100}]},
    )


def test_detects_distilbert_transformer_blocks_with_valid_mlp_group() -> None:
    ops = mlp_ops("/model/distilbert/transformer/layer.0", "distilbert")
    plans, validations = plan_and_validation()
    blocks = detect_generic_blocks("distilbert-base-uncased", {"ops": ops}, ranking={"candidates": [candidate_for(ops)]}, plans=plans, validations=validations)

    assert blocks[0].family == "distilbert"
    assert any(group.group_kind == "mlp_block" and group.plan_status == "valid_plan" for group in blocks[0].grouped_subgraphs)


def test_detects_opt_decoder_blocks() -> None:
    blocks = detect_generic_blocks("facebook/opt-125m", {"ops": mlp_ops("/model/decoder/layers.0", "opt")})

    assert blocks[0].family == "opt"
    assert {group.group_kind for group in blocks[0].grouped_subgraphs} >= {"mlp_block", "mlp_expansion_projection", "mlp_activation", "mlp_contraction_projection"}


def test_detects_gpt2_blocks_without_confusing_attention_c_proj() -> None:
    blocks = detect_generic_blocks("gpt2", {"ops": mlp_ops("/model/transformer/h.0", "gpt2")})

    mlp_expansion = next(group for group in blocks[0].grouped_subgraphs if group.group_kind == "mlp_expansion_projection")
    mlp_contraction = next(group for group in blocks[0].grouped_subgraphs if group.group_kind == "mlp_contraction_projection")
    assert "attn/c_proj" not in str(mlp_expansion.source_ops)
    assert "mlp/c_proj" in str(mlp_contraction.source_ops)


def test_detects_vit_layers() -> None:
    blocks = detect_generic_blocks("google/vit-base-patch16-224", {"ops": mlp_ops("/model/vit/layers.0", "vit")})

    assert blocks[0].family == "vit"
    assert blocks[0].block_kind == "vit_encoder_layer"


def test_generic_layer_records_are_pack_compatible() -> None:
    ops = mlp_ops("/model/decoder/layers.0", "opt")
    records = generic_layer_records("facebook/opt-125m", 0, {"ops": ops})

    assert any(record["semantic_category"] == "feed_forward_block" for record in records)
    assert all("recursive_primitive_leaves" in record for record in records)
