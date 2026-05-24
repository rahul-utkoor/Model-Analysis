from __future__ import annotations

from model_analysis.region_pruning_semantics import build_region_pruning_semantics, region_pruning_semantics_to_dict


def op(op_id: str, path: str, op_type: str, inputs: list[str], outputs: list[str]) -> dict:
    return {
        "op_id": op_id,
        "source_node_name": path,
        "op_type": op_type,
        "canonical_op_type": op_type.lower(),
        "inputs": inputs,
        "outputs": outputs,
    }


def region(rid: str, rtype: str, ops: list[str], children: list[str] | None = None, parent: str = "model") -> dict:
    return {
        "region_id": rid,
        "region_type": rtype,
        "name": rid,
        "op_ids": ops,
        "children": children or [],
        "parent": parent,
        "confidence": "high",
    }


def synthetic_inputs() -> tuple[dict, dict, dict]:
    tensor_ir = {
        "model_name": "synthetic",
        "source_frontend": "onnx",
        "ops": [
            op("ffn_mm", "/model/bert/encoder/layer.0/intermediate/dense/MatMul", "MatMul", ["x"], ["a"]),
            op("ffn_add", "/model/bert/encoder/layer.0/intermediate/dense/Add", "Add", ["a"], ["b"]),
            op("gelu_erf", "/model/bert/encoder/layer.0/intermediate/intermediate_act_fn/Erf", "Erf", ["b"], ["c"]),
            op("out_mm", "/model/bert/encoder/layer.0/output/dense/MatMul", "MatMul", ["c"], ["d"]),
            op("out_add", "/model/bert/encoder/layer.0/output/dense/Add", "Add", ["d"], ["e"]),
            op("res_add", "/model/bert/encoder/layer.0/output/Add", "Add", ["e", "x"], ["f"]),
            op("ln", "/model/bert/encoder/layer.0/output/LayerNorm/LayerNormalization", "LayerNormalization", ["f"], ["g"]),
            op("q_mm", "/model/bert/encoder/layer.0/attention/self/query/MatMul", "MatMul", ["x"], ["q"]),
            op("q_add", "/model/bert/encoder/layer.0/attention/self/query/Add", "Add", ["q"], ["qb"]),
            op("score", "/model/bert/encoder/layer.0/attention/self/MatMul", "MatMul", ["q"], ["s"]),
            op("mask_add", "/model/bert/encoder/layer.0/attention/self/Add", "Add", ["s", "mask"], ["sm"]),
            op("softmax", "/model/bert/encoder/layer.0/attention/self/Softmax", "Softmax", ["s"], ["p"]),
            op("ctx", "/model/bert/encoder/layer.0/attention/self/MatMul_1", "MatMul", ["p"], ["ctx"]),
            op("shape", "/model/bert/encoder/layer.0/attention/self/Reshape", "Reshape", ["q"], ["qr"]),
        ],
    }
    tree = {
        "model_name": "synthetic",
        "source_frontend": "onnx",
        "root_region_id": "model",
        "regions": [
            region("model", "ModelRegion", [item["op_id"] for item in tensor_ir["ops"]], children=["ffn", "ffn_int", "gelu", "ffn_out", "res", "ln", "q", "score_region", "ctx_region", "mask_add_region", "attn", "shape"], parent=None),
            region("ffn", "FeedForwardRegion", ["ffn_mm", "ffn_add", "gelu_erf", "out_mm", "out_add"], children=["ffn_int", "gelu", "ffn_out"]),
            region("ffn_int", "LinearProjectionRegion", ["ffn_mm", "ffn_add"], parent="ffn"),
            region("gelu", "ActivationRegion", ["gelu_erf"], parent="ffn"),
            region("ffn_out", "LinearProjectionRegion", ["out_mm", "out_add"], parent="ffn"),
            region("res", "ResidualMergeRegion", ["res_add"]),
            region("ln", "LayerNormRegion", ["ln"]),
            region("q", "LinearProjectionRegion", ["q_mm", "q_add"]),
            region("score_region", "LinearProjectionRegion", ["score"]),
            region("ctx_region", "LinearProjectionRegion", ["ctx"]),
            region("mask_add_region", "ResidualMergeRegion", ["mask_add"]),
            region("attn", "AttentionSkeletonRegion", ["score", "softmax", "ctx"]),
            region("shape", "AxisTransformRegion", ["shape"]),
        ],
    }
    rdim = {
        "dimension_variables": [
            {
                "var_id": "rdim_ffn_intermediate",
                "region_id": "ffn",
                "dim_name": "intermediate_dim",
                "axis_role": "intermediate",
                "prunable": True,
                "protected": False,
                "propagated": False,
                "blocked": False,
                "reason": "synthetic",
            }
        ],
        "constraint_equations": [],
    }
    return tree, tensor_ir, rdim


def records_by_type() -> dict[str, list[dict]]:
    tree, tensor_ir, rdim = synthetic_inputs()
    semantics = region_pruning_semantics_to_dict(
        build_region_pruning_semantics(tree, tensor_ir, region_dimension_ir=rdim)
    )
    out: dict[str, list[dict]] = {}
    for record in semantics["regions"]:
        out.setdefault(record["region_type"], []).append(record)
    return out


def test_feedforward_is_directly_prunable_with_mlp_repairs_and_rule() -> None:
    ffn = records_by_type()["FeedForwardRegion"][0]

    assert ffn["pruning_role"] == "directly_prunable"
    assert any(dim["dim_name"] == "intermediate_dim" and dim["status"] == "prunable" for dim in ffn["dimensions"])
    assert any(repair["obligation_type"] == "same_indices_across_mlp" for repair in ffn["repair_obligations"])
    assert any(rule["rule_type"] == "same_indices_across_mlp" for rule in ffn["propagation_rules"])


def test_gelu_activation_is_propagation_only_no_index_change() -> None:
    gelu = records_by_type()["ActivationRegion"][0]

    assert gelu["pruning_role"] == "propagation_only"
    assert any(rule["index_mapping"] == "no_index_change" for rule in gelu["propagation_rules"])


def test_residual_and_layernorm_are_protected_or_blocked() -> None:
    records = records_by_type()
    residual = records["ResidualMergeRegion"][0]
    layernorm = records["LayerNormRegion"][0]

    assert residual["pruning_role"] == "blocked"
    assert any(blocker["blocker_type"] == "residual_hidden_dim" for blocker in residual["blockers"])
    assert any(dim["status"] == "protected" for dim in residual["dimensions"])
    assert layernorm["pruning_role"] == "protected"
    assert any(repair["obligation_type"] == "layernorm_parameter_repair" for repair in layernorm["repair_obligations"])


def test_attention_and_shape_semantics_are_conservative() -> None:
    records = records_by_type()
    attention = records["AttentionSkeletonRegion"][0]
    shape = records["AxisTransformRegion"][0]

    assert any(blocker["blocker_type"] == "attention_head_mapping_unproven" for blocker in attention["blockers"])
    assert any(repair["obligation_type"] == "attention_axis_mapping_required" for repair in attention["repair_obligations"])
    assert shape["pruning_role"] == "propagation_only"
    assert not any(dim["status"] == "prunable" for dim in shape["dimensions"])


def test_linear_projection_semantics_are_path_aware() -> None:
    projections = records_by_type()["LinearProjectionRegion"]
    ffn_intermediate = next(item for item in projections if item["region_name"] == "Layer 0 FFN Intermediate Projection")
    ffn_output = next(item for item in projections if item["region_name"] == "Layer 0 FFN Output Projection")

    assert ffn_intermediate["pruning_role"] == "directly_prunable"
    assert any(dim["symbolic_role"] == "intermediate_dim" and dim["status"] == "prunable" for dim in ffn_intermediate["dimensions"])
    assert any(dim["symbolic_role"] == "hidden_dim" and dim["status"] == "protected" for dim in ffn_output["dimensions"])


def test_attention_score_matmul_is_not_directly_prunable() -> None:
    projections = records_by_type()["LinearProjectionRegion"]
    score = next(item for item in projections if item["region_name"] == "Layer 0 Attention Score MatMul")

    assert score["pruning_role"] in {"constraint_carrier", "propagation_only"}
    assert not any(dim["status"] == "prunable" for dim in score["dimensions"])
    assert any(blocker["blocker_type"] == "attention_head_mapping_unproven" for blocker in score["blockers"])
    assert any("Q x K" in rule["explanation"] for rule in score["propagation_rules"])


def test_attention_context_matmul_is_not_directly_prunable() -> None:
    projections = records_by_type()["LinearProjectionRegion"]
    context = next(item for item in projections if item["region_name"] == "Layer 0 Attention Context MatMul")

    assert context["pruning_role"] in {"constraint_carrier", "propagation_only"}
    assert not any(dim["status"] == "prunable" for dim in context["dimensions"])
    assert any(blocker["blocker_type"] == "attention_head_mapping_unproven" for blocker in context["blockers"])


def test_attention_mask_add_is_not_residual_semantics() -> None:
    residuals = records_by_type()["ResidualMergeRegion"]
    mask_add = next(item for item in residuals if item["region_name"] == "Layer 0 Attention Mask Add")

    assert mask_add["pruning_role"] in {"constraint_carrier", "propagation_only"}
    assert not any(blocker["blocker_type"] == "residual_hidden_dim" for blocker in mask_add["blockers"])
    assert not any(repair["obligation_type"] == "residual_branch_repair" for repair in mask_add["repair_obligations"])
    assert any(repair["obligation_type"] == "shape_metadata_update" for repair in mask_add["repair_obligations"])


def test_qkv_projection_remains_directly_prunable_with_attention_warning() -> None:
    projections = records_by_type()["LinearProjectionRegion"]
    query = next(item for item in projections if item["region_name"] == "Layer 0 Query Projection")

    assert query["pruning_role"] == "directly_prunable"
    assert any(blocker["blocker_type"] == "attention_head_mapping_unproven" for blocker in query["blockers"])
    assert any(repair["obligation_type"] == "prune_bias" for repair in query["repair_obligations"])
