from __future__ import annotations

import importlib.util
from pathlib import Path


def load_api_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "ordered_control_tree_api_server.py"
    spec = importlib.util.spec_from_file_location("ordered_control_tree_api_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def tensor_ir() -> dict:
    ops = [
        {"op_id": "op::0", "name": "/model/bert/embeddings/Add", "op_type": "Add", "canonical_op_type": "residual_add", "source_node_name": "/model/bert/embeddings/Add"},
        {"op_id": "op::1", "name": "/model/bert/encoder/layer.0/attention/self/query/MatMul", "op_type": "MatMul", "canonical_op_type": "linear", "source_node_name": "/model/bert/encoder.layer.0/attention.self.query/MatMul"},
        {"op_id": "op::2", "name": "/model/bert/encoder/layer.0/attention/Softmax", "op_type": "Softmax", "canonical_op_type": "softmax", "source_node_name": "/model/bert/encoder.layer.0/attention/Softmax"},
        {"op_id": "op::3", "name": "/model/bert/encoder/layer.0/output/Add", "op_type": "Add", "canonical_op_type": "residual_add", "source_node_name": "/model/bert/encoder.layer.0/output/Add"},
        {"op_id": "op::4", "name": "/model/bert/encoder/layer.0/intermediate/dense/MatMul", "op_type": "MatMul", "canonical_op_type": "linear", "source_node_name": "/model/bert/encoder.layer.0/intermediate.dense/MatMul"},
        {"op_id": "op::5", "name": "/model/bert/encoder/layer.0/intermediate/Gelu/Erf", "op_type": "Erf", "canonical_op_type": "activation", "source_node_name": "/model/bert/encoder.layer.0/intermediate/Gelu/Erf"},
        {"op_id": "op::6", "name": "/model/bert/encoder/layer.0/output/dense/MatMul", "op_type": "MatMul", "canonical_op_type": "linear", "source_node_name": "/model/bert/encoder.layer.0/output.dense/MatMul"},
    ]
    return {"model_name": "tiny", "source_frontend": "onnx", "ops": ops}


def structural_tree() -> dict:
    regions = [
        {"region_id": "r_model", "region_type": "ModelRegion", "name": "model", "parent": None, "children": ["r_ffn", "r_residual", "r_attention"], "op_ids": ["op::0", "op::1", "op::2", "op::3", "op::4", "op::5", "op::6"], "confidence": "high", "reason": "root", "metadata": {}},
        {"region_id": "r_ffn", "region_type": "FeedForwardRegion", "name": "ffn", "parent": "r_model", "children": ["r_linear1", "r_gelu", "r_linear2"], "op_ids": ["op::4", "op::5", "op::6"], "confidence": "high", "reason": "ffn", "metadata": {}},
        {"region_id": "r_residual", "region_type": "ResidualMergeRegion", "name": "residual", "parent": "r_model", "children": ["r_residual_leaf"], "op_ids": ["op::3"], "confidence": "medium", "reason": "residual", "metadata": {}},
        {"region_id": "r_attention", "region_type": "AttentionSkeletonRegion", "name": "attention", "parent": "r_model", "children": ["r_query", "r_softmax"], "op_ids": ["op::1", "op::2"], "confidence": "medium", "reason": "attention", "metadata": {}},
        {"region_id": "r_query", "region_type": "LinearProjectionRegion", "name": "query", "parent": "r_attention", "children": ["p_query"], "op_ids": ["op::1"], "confidence": "high", "reason": "query", "metadata": {}},
        {"region_id": "r_softmax", "region_type": "PrimitiveRegion", "name": "softmax", "parent": "r_attention", "children": [], "op_ids": ["op::2"], "confidence": "high", "reason": "softmax", "metadata": {}},
        {"region_id": "r_linear1", "region_type": "LinearProjectionRegion", "name": "linear1", "parent": "r_ffn", "children": ["p_linear1"], "op_ids": ["op::4"], "confidence": "high", "reason": "linear1", "metadata": {}},
        {"region_id": "r_gelu", "region_type": "ActivationRegion", "name": "gelu", "parent": "r_ffn", "children": ["p_gelu"], "op_ids": ["op::5"], "confidence": "high", "reason": "gelu", "metadata": {"activation_kind": "gelu"}},
        {"region_id": "r_linear2", "region_type": "LinearProjectionRegion", "name": "linear2", "parent": "r_ffn", "children": ["p_linear2"], "op_ids": ["op::6"], "confidence": "high", "reason": "linear2", "metadata": {}},
        {"region_id": "r_residual_leaf", "region_type": "PrimitiveRegion", "name": "add", "parent": "r_residual", "children": [], "op_ids": ["op::3"], "confidence": "high", "reason": "add", "metadata": {}},
        {"region_id": "p_query", "region_type": "PrimitiveRegion", "name": "query matmul", "parent": "r_query", "children": [], "op_ids": ["op::1"], "confidence": "high", "reason": "query primitive", "metadata": {}},
        {"region_id": "p_linear1", "region_type": "PrimitiveRegion", "name": "linear1 primitive", "parent": "r_linear1", "children": [], "op_ids": ["op::4"], "confidence": "high", "reason": "linear primitive", "metadata": {}},
        {"region_id": "p_gelu", "region_type": "PrimitiveRegion", "name": "gelu primitive", "parent": "r_gelu", "children": [], "op_ids": ["op::5"], "confidence": "high", "reason": "gelu primitive", "metadata": {}},
        {"region_id": "p_linear2", "region_type": "PrimitiveRegion", "name": "linear2 primitive", "parent": "r_linear2", "children": [], "op_ids": ["op::6"], "confidence": "high", "reason": "linear primitive", "metadata": {}},
    ]
    interfaces = [
        {"region_id": "r_ffn", "region_type": "FeedForwardRegion", "pruning_role": "directly_prunable"},
        {"region_id": "r_residual", "region_type": "ResidualMergeRegion", "pruning_role": "blocked"},
        {"region_id": "r_attention", "region_type": "AttentionSkeletonRegion", "pruning_role": "analysis_only"},
        {"region_id": "r_linear1", "region_type": "LinearProjectionRegion", "pruning_role": "directly_prunable"},
    ]
    return {"model_name": "tiny", "source_frontend": "onnx", "root_region_id": "r_model", "regions": regions, "interfaces": interfaces, "summary": {"region_type_counts": {"FeedForwardRegion": 1}}}


def dim_ir() -> dict:
    return {
        "model_name": "tiny",
        "dimension_variables": [
            {"var_id": "d_ffn_intermediate", "region_id": "r_ffn", "dim_name": "intermediate_dim", "axis_role": "intermediate", "prunable": True, "blocked": False, "protected": False},
            {"var_id": "d_residual_hidden", "region_id": "r_residual", "dim_name": "hidden_dim", "axis_role": "hidden", "prunable": False, "blocked": True, "protected": True},
        ],
    }


def store():
    api = load_api_module()
    return api.OrderedTreeStore("tiny", structural_tree(), tensor_ir=tensor_ir(), dim_ir=dim_ir())


def test_topological_ordering_uses_op_order_not_region_type() -> None:
    s = store()
    children = s.children_payload("r_model")["children"]

    assert [child["region_id"] for child in children] == ["r_attention", "r_residual", "r_ffn"]
    assert children[0]["region_type"] == "AttentionSkeletonRegion"
    assert children[-1]["region_type"] == "FeedForwardRegion"


def test_children_summaries_preserve_source_order() -> None:
    s = store()
    children = s.children_payload("r_ffn")["children"]

    assert [child["region_id"] for child in children] == ["r_linear1", "r_gelu", "r_linear2"]
    assert [child["source_op_range"] for child in children] == ["4", "5", "6"]


def test_humanized_labels_for_feedforward_and_residual() -> None:
    s = store()
    ffn = s.region_summary("r_ffn")
    residual = s.region_summary("r_residual")

    assert ffn["display_title"] == "Layer 0 Feed-Forward Block"
    assert "compound region" in ffn["compiler_analogy"].lower()
    assert residual["display_title"] == "Layer 0 Residual Merge"
    assert residual["pruning_role"] == "blocked"


def test_humanized_primitive_label_uses_tensor_ir_source_op() -> None:
    s = store()
    primitive = s.region_summary("p_query")

    assert primitive["display_title"] == "Primitive: MatMul"
    assert "attention.self.query" in primitive["display_subtitle"]


def test_path_to_root_works() -> None:
    s = store()
    path = s.path_to_root("p_linear2")

    assert [item["region_id"] for item in path] == ["r_model", "r_ffn", "r_linear2", "p_linear2"]


def test_leaf_ops_are_ordered_by_source_order() -> None:
    s = store()
    leaves = s.leaf_ops("r_ffn")

    assert [item["region_id"] for item in leaves] == ["p_linear1", "p_gelu", "p_linear2"]


def test_search_returns_matching_region_summaries() -> None:
    s = store()
    result = s.search("feed-forward")

    assert result["hits"]
    assert result["hits"][0]["region_id"] == "r_ffn"
    assert result["hits"][0]["path"][0]["region_id"] == "r_model"


def test_node_payload_includes_teaching_and_dimensions() -> None:
    s = store()
    payload = s.node_payload("r_ffn")

    assert payload["display"]["compiler_analogy"]
    assert payload["display"]["pruning_relevance"]
    assert payload["display"]["dimension_summary"]["prunable_count"] == 1
    assert payload["dimensions"][0]["dim_name"] == "intermediate_dim"
