from model_analysis.reporting import (
    onnx_summary_to_markdown,
    pruning_hints_to_markdown,
    structural_inventory_to_markdown,
)


def _torch_summary():
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "parameter_summary": {
            "total_parameters": 10,
            "trainable_parameters": 10,
            "non_trainable_parameters": 0,
        },
        "module_summary": {
            "total_modules": 2,
            "module_type_counts": {"Tiny": 1, "Linear": 1},
            "parameter_distribution_by_module_type": {"Tiny": 0, "Linear": 15},
        },
        "linear_layers": [{"name": "linear", "in_features": 4, "out_features": 3, "bias": True, "parameters": 15}],
        "embedding_layers": [],
        "normalization_layers": [],
        "attention_like_modules": [],
        "mlp_like_modules": [],
        "pruning_relevant_groups": [],
    }


def _onnx_summary():
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "onnx_path": "/tmp/tiny.onnx",
        "graph_summary": {
            "num_nodes": 1,
            "num_initializers": 2,
            "num_inputs": 1,
            "num_outputs": 1,
            "op_type_counts": {"Gemm": 1},
        },
        "inputs": [{"name": "inputs", "shape": [1, 4], "data_type": "FLOAT"}],
        "outputs": [{"name": "outputs", "shape": [1, 3], "data_type": "FLOAT"}],
        "initializers": [{"name": "linear.weight", "dims": [3, 4], "data_type": "FLOAT"}],
        "nodes": [{"name": "linear", "op_type": "Gemm", "inputs": ["inputs"], "outputs": ["outputs"]}],
        "pruning_relevant_nodes": [
            {"name": "linear", "op_type": "Gemm", "reason": "projection", "confidence": "high"}
        ],
    }


def test_structural_inventory_markdown_has_expected_sections():
    markdown = structural_inventory_to_markdown(_torch_summary())
    assert "# Structural Inventory: tiny" in markdown
    assert "## Linear Layers" in markdown
    assert "## Pruning-Relevant Groups" in markdown


def test_onnx_markdown_has_expected_sections():
    markdown = onnx_summary_to_markdown(_onnx_summary())
    assert "# ONNX Graph Summary: tiny" in markdown
    assert "## Graph Summary" in markdown
    assert "## Pruning-Relevant Nodes" in markdown


def test_pruning_hints_markdown_has_expected_sections():
    markdown = pruning_hints_to_markdown(_torch_summary(), _onnx_summary())
    assert "# Pruning Hints: tiny" in markdown
    assert "## What appears structurally prunable" in markdown
    assert "## Forward propagation considerations" in markdown
    assert "## Backward propagation considerations" in markdown
    assert "## Caveats" in markdown
