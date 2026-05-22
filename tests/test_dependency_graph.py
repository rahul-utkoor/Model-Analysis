from model_analysis.dependency_graph import (
    augment_dependency_graph_with_onnx_summary,
    build_dependency_graph_from_torch_summary,
)


def fake_torch_summary():
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "linear_layers": [
            {"name": "block.attn.q_proj", "in_features": 8, "out_features": 8, "bias": True, "parameters": 72},
            {"name": "block.attn.k_proj", "in_features": 8, "out_features": 8, "bias": True, "parameters": 72},
            {"name": "block.attn.v_proj", "in_features": 8, "out_features": 8, "bias": True, "parameters": 72},
            {"name": "block.attn.out_proj", "in_features": 8, "out_features": 8, "bias": True, "parameters": 72},
            {"name": "block.mlp.fc1", "in_features": 8, "out_features": 16, "bias": True, "parameters": 144},
            {"name": "block.mlp.fc2", "in_features": 16, "out_features": 8, "bias": True, "parameters": 136},
            {"name": "block.extra_linear", "in_features": 8, "out_features": 8, "bias": True, "parameters": 72},
        ],
        "embedding_layers": [
            {"name": "embedding", "num_embeddings": 32, "embedding_dim": 8, "parameters": 256},
        ],
        "normalization_layers": [
            {"name": "block.norm", "type": "LayerNorm", "parameters": 16},
        ],
        "pruning_relevant_groups": [
            {
                "group_name": "block.attn:qkv_projections",
                "group_type": "attention_qkv",
                "members": ["block.attn.q_proj", "block.attn.k_proj", "block.attn.v_proj"],
                "reason": "Q/K/V projection layers share a common parent name.",
                "confidence": "high",
            },
            {
                "group_name": "block.attn.out_proj:attention_output_projection",
                "group_type": "attention_output_projection",
                "members": ["block.attn.out_proj"],
                "reason": "Layer name suggests an attention output projection.",
                "confidence": "medium",
            },
            {
                "group_name": "block.mlp:mlp_projections",
                "group_type": "mlp_projection_pair",
                "members": ["block.mlp.fc1", "block.mlp.fc2"],
                "reason": "MLP-like expansion/projection layers share a common parent.",
                "confidence": "medium",
            },
            {
                "group_name": "embedding:embedding_matrix",
                "group_type": "embedding_matrix",
                "members": ["embedding"],
                "reason": "Embedding matrix caveat.",
                "confidence": "low",
            },
        ],
    }


def fake_onnx_summary():
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "graph_summary": {
            "num_nodes": 5,
            "op_type_counts": {"MatMul": 1, "Add": 1, "Reshape": 1, "Transpose": 1, "Softmax": 1},
        },
        "initializer_shapes": {"matmul.weight": [8, 8]},
        "nodes": [
            {"name": "MatMul_0", "op_type": "MatMul", "inputs": ["x", "matmul.weight"], "outputs": ["y"]},
            {"name": "Add_1", "op_type": "Add", "inputs": ["y", "skip"], "outputs": ["z"]},
            {"name": "Reshape_2", "op_type": "Reshape", "inputs": ["z", "shape"], "outputs": ["r"]},
            {"name": "Transpose_3", "op_type": "Transpose", "inputs": ["r"], "outputs": ["t"]},
            {"name": "Softmax_4", "op_type": "Softmax", "inputs": ["t"], "outputs": ["o"]},
        ],
        "pruning_relevant_nodes": [
            {"name": "MatMul_0", "op_type": "MatMul", "reason": "projection", "confidence": "high"},
            {"name": "Add_1", "op_type": "Add", "reason": "residual", "confidence": "medium"},
            {"name": "Reshape_2", "op_type": "Reshape", "reason": "shape", "confidence": "medium"},
            {"name": "Transpose_3", "op_type": "Transpose", "reason": "shape", "confidence": "medium"},
            {"name": "Softmax_4", "op_type": "Softmax", "reason": "attention", "confidence": "low"},
        ],
    }


def test_build_dependency_graph_from_torch_summary():
    graph = build_dependency_graph_from_torch_summary(fake_torch_summary())

    assert graph.prunable_units
    assert any(unit.unit_type == "attention_qkv" for unit in graph.prunable_units)
    assert any(edge.edge_type == "qkv_coupling" for edge in graph.dependency_edges)
    assert any(edge.edge_type == "mlp_hidden_coupling" for edge in graph.dependency_edges)
    assert graph.ambiguous_units
    assert isinstance(graph.independent_units, list)


def test_augment_dependency_graph_with_onnx_summary():
    graph = build_dependency_graph_from_torch_summary(fake_torch_summary())
    graph = augment_dependency_graph_with_onnx_summary(graph, fake_onnx_summary())

    assert any(unit.source == "onnx" and unit.unit_type == "matmul" for unit in graph.prunable_units)
    assert graph.metadata["onnx_evidence"]["num_onnx_nodes"] == 5
    assert graph.metadata["onnx_evidence"]["propagation_relevant_node_count"] == 4
    assert any(edge.edge_type == "residual_coupling" for edge in graph.dependency_edges)
    assert any("Add" in item["reason"] for item in graph.ambiguous_units)
