from __future__ import annotations

from model_analysis.subgraph_analysis import (
    build_onnx_adjacency,
    build_subgraph_analysis_report,
    classify_add_node_kind,
    classify_subgraph_pattern,
    compute_subgraph_tensor_sets,
    enumerate_join_subgraphs,
    enumerate_node_path_subgraphs,
    generate_subgraph_pruning_evidence,
    summarize_subgraph_patterns,
)


def synthetic_onnx_summary() -> dict:
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "inputs": [{"name": "x", "shape": [1, 4]}],
        "outputs": [{"name": "norm", "shape": [1, 4]}, {"name": "biased", "shape": [1, 4]}],
        "initializers": [
            {"name": "w1", "dims": [4, 8]},
            {"name": "w2", "dims": [8, 4]},
            {"name": "w3", "dims": [4, 4]},
            {"name": "bias", "dims": [4]},
            {"name": "gamma", "dims": [4]},
            {"name": "beta", "dims": [4]},
        ],
        "tensor_shape_map": {
            "x": [1, 4],
            "transformed": [1, 4],
            "joined": [1, 4],
            "norm": [1, 4],
        },
        "nodes": [
            {"name": "gemm1", "op_type": "Gemm", "inputs": ["x", "w1"], "outputs": ["hidden"]},
            {"name": "gelu", "op_type": "Gelu", "inputs": ["hidden"], "outputs": ["activated"]},
            {"name": "gemm2", "op_type": "Gemm", "inputs": ["activated", "w2"], "outputs": ["transformed"]},
            {"name": "residual_add", "op_type": "Add", "inputs": ["transformed", "x"], "outputs": ["joined"]},
            {"name": "norm_node", "op_type": "LayerNormalization", "inputs": ["joined", "gamma", "beta"], "outputs": ["norm"]},
            {"name": "gemm3", "op_type": "Gemm", "inputs": ["norm", "w3"], "outputs": ["projected"]},
            {"name": "bias_add", "op_type": "Add", "inputs": ["projected", "bias"], "outputs": ["biased"]},
            {"name": "reshape", "op_type": "Reshape", "inputs": ["norm", "shape"], "outputs": ["reshaped"]},
            {"name": "transpose", "op_type": "Transpose", "inputs": ["reshaped"], "outputs": ["transposed"]},
        ],
    }


def test_adjacency_builds_tensor_and_node_maps() -> None:
    adjacency = build_onnx_adjacency(synthetic_onnx_summary())

    assert adjacency.producer_of_tensor["hidden"] == "gemm1"
    assert adjacency.consumers_of_tensor["hidden"] == ["gelu"]
    assert adjacency.successors["gemm1"] == ["gelu"]
    assert "gemm2" in adjacency.predecessors["residual_add"]


def test_path_enumeration_includes_sizes_and_boundaries() -> None:
    summary = synthetic_onnx_summary()
    adjacency = build_onnx_adjacency(summary)
    paths = enumerate_node_path_subgraphs(adjacency, summary, "tiny", max_nodes=3)

    assert len([path for path in paths if path.size == 1]) == len(summary["nodes"])
    assert any(path.op_types == ["Gemm", "Gelu"] for path in paths)
    mlp = next(path for path in paths if path.op_types == ["Gemm", "Gelu", "Gemm"])
    tensors = compute_subgraph_tensor_sets(mlp.node_names, adjacency, summary)
    assert tensors["internal_tensors"] == ["hidden", "activated"]
    assert "x" in tensors["boundary_input_tensors"]
    assert "transformed" in tensors["boundary_output_tensors"]
    assert set(tensors["initializer_tensors"]) == {"w1", "w2"}


def test_bias_add_and_residual_add_are_distinguished() -> None:
    summary = synthetic_onnx_summary()
    adjacency = build_onnx_adjacency(summary)

    assert classify_add_node_kind(adjacency.node_by_name["bias_add"], adjacency, summary)[0] == "bias_add"
    add_kind, confidence, _ = classify_add_node_kind(adjacency.node_by_name["residual_add"], adjacency, summary)
    assert add_kind == "residual_add"
    assert confidence == "high"


def test_join_subgraph_is_created_for_residual_add() -> None:
    summary = synthetic_onnx_summary()
    adjacency = build_onnx_adjacency(summary)
    joins = enumerate_join_subgraphs(adjacency, summary, "tiny")

    assert len(joins) == 1
    assert joins[0].join_node == "residual_add"
    assert joins[0].is_residual_like is True
    assert "Join(Add)" in joins[0].pattern


def test_pattern_classification_recognizes_local_classes() -> None:
    assert classify_subgraph_pattern(["Gemm"])[0] == "directly_prunable"
    assert classify_subgraph_pattern(["Gemm", "Gelu", "Gemm"])[0] == "mlp_like"
    assert classify_subgraph_pattern(["MatMul", "Softmax", "MatMul"])[0] == "attention_like"
    assert classify_subgraph_pattern(["Add", "LayerNormalization"])[0] in {"residual_like", "normalization_like"}
    assert classify_subgraph_pattern(["Reshape", "Transpose"])[0] == "shape_transform"


def test_evidence_and_pattern_summary_keep_join_semantics() -> None:
    summary = synthetic_onnx_summary()
    adjacency = build_onnx_adjacency(summary)
    paths = enumerate_node_path_subgraphs(adjacency, summary, "tiny", max_nodes=3)
    joins = enumerate_join_subgraphs(adjacency, summary, "tiny")
    evidence = generate_subgraph_pruning_evidence(paths, joins, summary)
    patterns = summarize_subgraph_patterns(paths, joins)

    assert any(item.evidence_type == "direct_prunable_op" for item in evidence)
    assert any(item.evidence_type == "residual_hidden_equality" for item in evidence)
    assert any(item.subgraph_kind == "path" for item in patterns)
    assert any(item.subgraph_kind == "join" for item in patterns)


def test_full_report_counts_bias_and_residual_adds() -> None:
    report = build_subgraph_analysis_report(
        synthetic_onnx_summary(),
        {"name": "tiny", "hf_id": "local/tiny", "task": "unit-test"},
        max_nodes=3,
    )

    assert report.summary["bias_add_count"] == 1
    assert report.summary["residual_add_count"] == 1
    assert report.summary["num_residual_like_join_subgraphs"] == 1

