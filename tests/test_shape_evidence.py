from model_analysis.shape_evidence import build_shape_evidence, shape_evidence_report_to_markdown


def fake_onnx_summary():
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "inputs": [{"name": "input_ids", "shape": ["batch", "seq"], "data_type": "INT64"}],
        "outputs": [{"name": "logits", "shape": ["batch", "seq", 10], "data_type": "FLOAT"}],
        "value_info_shapes": {"hidden": ["batch", "seq", 8]},
        "initializers": [{"name": "linear.weight", "dims": [10, 8], "data_type": "FLOAT"}],
        "nodes": [
            {"name": "linear/Gemm", "op_type": "Gemm", "inputs": ["hidden", "linear.weight"], "outputs": ["logits"]},
            {"name": "residual/Add", "op_type": "Add", "inputs": ["hidden", "hidden"], "outputs": ["hidden2"]},
        ],
    }


def test_build_shape_evidence_collects_tensors_and_node_shapes():
    report = build_shape_evidence(fake_onnx_summary())

    tensors = {item.tensor_name: item.shape for item in report.tensor_shapes}
    assert tensors["input_ids"] == ["batch", "seq"]
    assert tensors["linear.weight"] == [10, 8]
    assert any(node.node_name == "linear/Gemm" and node.shape_constraints for node in report.node_shapes)
    assert report.summary["num_tensor_shapes"] >= 4


def test_shape_evidence_markdown_has_sections():
    markdown = shape_evidence_report_to_markdown(build_shape_evidence(fake_onnx_summary()))

    assert "# Shape Evidence: tiny" in markdown
    assert "## Tensor Shapes" in markdown
    assert "## Node Shapes" in markdown
    assert "## Shape Constraints" in markdown
