from __future__ import annotations

import importlib.util
from pathlib import Path


def load_exporter_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "export_abstract_node_expansion_report.py"
    spec = importlib.util.spec_from_file_location("export_abstract_node_expansion_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _op(op_id: str, source: str, op_type: str, inputs: list[str], outputs: list[str]) -> dict:
    return {
        "op_id": op_id,
        "source_node_name": source,
        "op_type": op_type,
        "canonical_op_type": op_type.lower(),
        "inputs": inputs,
        "outputs": outputs,
    }


def synthetic_inputs() -> tuple[dict, dict]:
    tensor_ir = {
        "ops": [
            _op("op_emb_gather", "/model/bert/embeddings/word_embeddings/Gather", "Gather", ["ids"], ["emb0"]),
            _op("op_emb_add", "/model/bert/embeddings/Add", "Add", ["emb0", "tok"], ["emb1"]),
            _op("op_emb_ln", "/model/bert/embeddings/LayerNorm/LayerNormalization", "LayerNormalization", ["emb1"], ["emb2"]),
            _op("op_q_mm", "/model/bert/encoder/layer.0/attention/self/query/MatMul", "MatMul", ["emb2"], ["q0"]),
            _op("op_q_add", "/model/bert/encoder/layer.0/attention/self/query/Add", "Add", ["q0"], ["q"]),
            _op("op_k_mm", "/model/bert/encoder/layer.0/attention/self/key/MatMul", "MatMul", ["emb2"], ["k0"]),
            _op("op_k_add", "/model/bert/encoder/layer.0/attention/self/key/Add", "Add", ["k0"], ["k"]),
            _op("op_v_mm", "/model/bert/encoder/layer.0/attention/self/value/MatMul", "MatMul", ["emb2"], ["v0"]),
            _op("op_v_add", "/model/bert/encoder/layer.0/attention/self/value/Add", "Add", ["v0"], ["v"]),
            _op("op_score", "/model/bert/encoder/layer.0/attention/self/MatMul", "MatMul", ["q", "k"], ["score"]),
            _op("op_mask_add", "/model/bert/encoder/layer.0/attention/self/Add", "Add", ["score", "mask"], ["masked"]),
            _op("op_softmax", "/model/bert/encoder/layer.0/attention/self/Softmax", "Softmax", ["masked"], ["prob"]),
            _op("op_context", "/model/bert/encoder/layer.0/attention/self/MatMul_1", "MatMul", ["prob", "v"], ["ctx"]),
            _op("op_attn_out_mm", "/model/bert/encoder/layer.0/attention/output/dense/MatMul", "MatMul", ["ctx"], ["ao0"]),
            _op("op_attn_out_add", "/model/bert/encoder/layer.0/attention/output/dense/Add", "Add", ["ao0"], ["ao"]),
            _op("op_attn_res", "/model/bert/encoder/layer.0/attention/output/Add", "Add", ["ao", "emb2"], ["ar"]),
            _op("op_attn_ln", "/model/bert/encoder/layer.0/attention/output/LayerNorm/LayerNormalization", "LayerNormalization", ["ar"], ["aln"]),
            _op("op_ffn_int_mm", "/model/bert/encoder/layer.0/intermediate/dense/MatMul", "MatMul", ["aln"], ["fi0"]),
            _op("op_ffn_int_add", "/model/bert/encoder/layer.0/intermediate/dense/Add", "Add", ["fi0"], ["fi"]),
            _op("op_gelu_erf", "/model/bert/encoder/layer.0/intermediate/intermediate_act_fn/Erf", "Erf", ["fi"], ["gelu"]),
            _op("op_ffn_out_mm", "/model/bert/encoder/layer.0/output/dense/MatMul", "MatMul", ["gelu"], ["fo0"]),
            _op("op_ffn_out_add", "/model/bert/encoder/layer.0/output/dense/Add", "Add", ["fo0"], ["fo"]),
            _op("op_ffn_res", "/model/bert/encoder/layer.0/output/Add", "Add", ["fo", "aln"], ["fr"]),
            _op("op_ffn_ln", "/model/bert/encoder/layer.0/output/LayerNorm/LayerNormalization", "LayerNormalization", ["fr"], ["fln"]),
            _op("op_cls_mm", "/model/bert/cls/predictions/transform/dense/MatMul", "MatMul", ["fln"], ["cls"]),
            _op("op_reshape", "/model/bert/encoder/layer.0/attention/self/Reshape", "Reshape", ["q"], ["qr"]),
            _op("op_ge", "/model/bert/attention_mask/GreaterOrEqual", "GreaterOrEqual", ["mask"], ["ge"]),
            _op("op_equal", "/model/bert/attention_mask/Equal", "Equal", ["ge"], ["eq"]),
            _op("op_and", "/model/bert/attention_mask/And", "And", ["eq"], ["and"]),
            _op("op_where", "/model/bert/encoder/layer.0/attention/self/Where", "Where", ["and", "score"], ["where"]),
            _op("op_isnan", "/model/bert/attention_mask/IsNaN", "IsNaN", ["mask"], ["isnan"]),
            _op("op_cos", "/model/bert/attention_mask/ConstantOfShape", "ConstantOfShape", ["mask"], ["cos"]),
            _op("op_shapeop", "/utility/shape_helper/Shape", "shape_op", ["fln"], ["shape"]),
        ]
    }

    def region(rid: str, rtype: str, ops: list[str], parent: str = "r_model", children: list[str] | None = None) -> dict:
        return {
            "region_id": rid,
            "region_type": rtype,
            "op_ids": ops,
            "children": children or [],
            "parent": parent,
            "confidence": "high",
            "reason": f"synthetic {rtype}",
        }

    regions = [
        region(
            "r_model",
            "ModelRegion",
            [op["op_id"] for op in tensor_ir["ops"]],
            parent=None,
            children=[
                "r_emb_add",
                "r_emb_ln",
                "r_q",
                "r_k",
                "r_v",
                "r_attention",
                "r_attn_out",
                "r_attn_res",
                "r_attn_ln",
                "r_ffn",
                "r_ffn_res",
                "r_ffn_ln",
                "r_cls",
                "r_shape",
                "r_predicate",
                "r_fork_shape",
            ],
        ),
        region("r_emb_add", "ResidualMergeRegion", ["op_emb_gather", "op_emb_add"]),
        region("r_emb_ln", "LayerNormRegion", ["op_emb_ln"]),
        region("r_q", "LinearProjectionRegion", ["op_q_mm", "op_q_add"]),
        region("r_k", "LinearProjectionRegion", ["op_k_mm", "op_k_add"]),
        region("r_v", "LinearProjectionRegion", ["op_v_mm", "op_v_add"]),
        region(
            "r_attention",
            "AttentionSkeletonRegion",
            ["op_score", "op_mask_add", "op_softmax", "op_context"],
            children=["p_score", "p_mask_add", "p_softmax", "p_context"],
        ),
        region("p_score", "PrimitiveRegion", ["op_score"], parent="r_attention"),
        region("p_mask_add", "PrimitiveRegion", ["op_mask_add"], parent="r_attention"),
        region("p_softmax", "PrimitiveRegion", ["op_softmax"], parent="r_attention"),
        region("p_context", "PrimitiveRegion", ["op_context"], parent="r_attention"),
        region("r_attn_out", "LinearProjectionRegion", ["op_attn_out_mm", "op_attn_out_add"]),
        region("r_attn_res", "ResidualMergeRegion", ["op_attn_res"]),
        region("r_attn_ln", "LayerNormRegion", ["op_attn_ln"]),
        region(
            "r_ffn",
            "FeedForwardRegion",
            ["op_ffn_int_mm", "op_ffn_int_add", "op_gelu_erf", "op_ffn_out_mm", "op_ffn_out_add"],
            children=["r_ffn_int", "r_gelu", "r_ffn_out"],
        ),
        region("r_ffn_int", "LinearProjectionRegion", ["op_ffn_int_mm", "op_ffn_int_add"], parent="r_ffn"),
        region("r_gelu", "ActivationRegion", ["op_gelu_erf"], parent="r_ffn"),
        region("r_ffn_out", "LinearProjectionRegion", ["op_ffn_out_mm", "op_ffn_out_add"], parent="r_ffn"),
        region("r_ffn_res", "ResidualMergeRegion", ["op_ffn_res"]),
        region("r_ffn_ln", "LayerNormRegion", ["op_ffn_ln"]),
        region("r_cls", "LinearProjectionRegion", ["op_cls_mm"]),
        region("r_shape", "AxisTransformRegion", ["op_reshape"]),
        region("r_predicate", "JoinRegion", ["op_ge"]),
        region("r_fork_shape", "ForkRegion", ["op_shapeop"]),
    ]
    tree = {
        "regions": regions,
        "interfaces": [
            {"region_id": "r_q", "pruning_role": "directly_prunable"},
            {"region_id": "r_ffn", "pruning_role": "directly_prunable"},
        ],
    }
    return tensor_ir, tree


def build_records(view: str, *, include_root_leaves: bool = False, include_single_op_shape_regions: bool = False) -> list[dict]:
    mod = load_exporter_module()
    tensor_ir, tree = synthetic_inputs()
    tm = mod.build_tensor_maps(tensor_ir)
    region_by_id, children_by_parent, interface_by_region = mod.build_region_maps(tree)
    recursive_leaf_ops = mod.compute_recursive_leaf_ops(region_by_id, children_by_parent)
    return mod.build_records(
        "bert-base-uncased",
        region_by_id,
        children_by_parent,
        interface_by_region,
        recursive_leaf_ops,
        tm,
        view=view,
        max_leaf_names=50,
        compress_single_op_wrappers=True,
        include_root_leaves=include_root_leaves,
        include_single_op_shape_regions=include_single_op_shape_regions,
    )


def record(records: list[dict], name: str) -> dict:
    return next(r for r in records if r["name"] == name)


def names(items: list[dict]) -> list[str]:
    return [item["name"] for item in items]


def test_model_immediate_expansion_contains_sections() -> None:
    records = build_records("main")
    expansion = names(record(records, "Model")["immediate_expansion"])

    assert "Embeddings" in expansion
    assert "Encoder Layer 0" in expansion
    assert "Prediction Head" in expansion


def test_encoder_layer_section_immediate_expansion_is_populated() -> None:
    records = build_records("main")
    expansion = names(record(records, "Encoder Layer 0")["immediate_expansion"])

    assert "Layer 0 Query Projection" in expansion
    assert "Layer 0 Feed Forward" in expansion
    assert expansion


def test_feedforward_expands_to_intermediate_gelu_output_projection() -> None:
    records = build_records("main")
    expansion = names(record(records, "Layer 0 Feed Forward")["immediate_expansion"])

    assert "Layer 0 FFN Intermediate Projection" in expansion
    assert "Layer 0 GELU" in expansion
    assert "Layer 0 FFN Output Projection" in expansion


def test_attention_expands_to_semantic_attention_primitive_names() -> None:
    records = build_records("main")
    expansion = names(record(records, "Layer 0 Attention")["immediate_expansion"])

    assert "Layer 0 Attention Score MatMul" in expansion
    assert "Layer 0 Attention Mask Add" in expansion
    assert "Layer 0 Attention Softmax" in expansion
    assert "Layer 0 Attention Context MatMul" in expansion


def test_main_view_excludes_auxiliary_only_shape_regions() -> None:
    records = build_records("main")
    region_ids = {r["region_id"] for r in records}

    assert "r_shape" not in region_ids
    assert "r_predicate" not in region_ids
    assert "r_fork_shape" not in region_ids


def test_main_view_has_no_fork_region_records_with_only_shape_op_leaves() -> None:
    records = build_records("main")
    fork_records = [r for r in records if r["region_type"] == "ForkRegion"]

    assert not fork_records


def test_shape_view_includes_motifs_and_suppresses_single_op_shape_regions_by_default() -> None:
    records = build_records("shape")
    region_types = {r["region_type"] for r in records}
    region_ids = {r["region_id"] for r in records}
    sections = {r["section"] for r in records}

    assert "ShapeMotifRegion" in region_types
    assert "r_shape" not in region_ids
    assert "Other Main Flow" not in sections

    debug_records = build_records("shape", include_single_op_shape_regions=True)
    assert "r_shape" in {r["region_id"] for r in debug_records}


def test_root_section_and_motif_leaves_are_hidden_by_default_and_included_by_flag() -> None:
    default_records = build_records("shape")
    debug_records = build_records("shape", include_root_leaves=True)

    assert record(default_records, "Model")["recursive_primitive_leaves"] == []
    assert record(default_records, "Encoder Layer 0")["recursive_primitive_leaves"] == []
    motif = next(r for r in default_records if r["region_type"] == "ShapeMotifRegion")
    assert motif["recursive_primitive_leaves"] == []

    assert record(debug_records, "Model")["recursive_primitive_leaves"]
    assert record(debug_records, "Encoder Layer 0")["recursive_primitive_leaves"]
    debug_motif = next(r for r in debug_records if r["region_type"] == "ShapeMotifRegion")
    assert debug_motif["recursive_primitive_leaves"]


def test_predicate_mask_ops_classify_as_auxiliary_shape_mask_flow() -> None:
    mod = load_exporter_module()
    tensor_ir, _tree = synthetic_inputs()
    tm = mod.build_tensor_maps(tensor_ir)

    for op_id in ["op_ge", "op_equal", "op_and", "op_where", "op_isnan", "op_cos", "op_shapeop"]:
        assert mod.op_section(op_id, tm) == "Auxiliary Shape / Mask Flow"
