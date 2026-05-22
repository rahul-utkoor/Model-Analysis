from model_analysis.dependency_analyzer import analyze_dependency_graph, dependency_analysis_to_markdown
from model_analysis.dependency_graph import build_dependency_graph_from_torch_summary


def fake_torch_summary():
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "linear_layers": [
            {"name": "block.attn.q_proj", "in_features": 8, "out_features": 8, "bias": True, "parameters": 72},
            {"name": "block.attn.k_proj", "in_features": 8, "out_features": 8, "bias": True, "parameters": 72},
            {"name": "block.attn.v_proj", "in_features": 8, "out_features": 8, "bias": True, "parameters": 72},
            {"name": "block.mlp.fc1", "in_features": 8, "out_features": 16, "bias": True, "parameters": 144},
            {"name": "block.mlp.fc2", "in_features": 16, "out_features": 8, "bias": True, "parameters": 136},
        ],
        "embedding_layers": [
            {"name": "embedding", "num_embeddings": 32, "embedding_dim": 8, "parameters": 256},
        ],
        "normalization_layers": [],
        "pruning_relevant_groups": [
            {
                "group_name": "block.attn:qkv_projections",
                "group_type": "attention_qkv",
                "members": ["block.attn.q_proj", "block.attn.k_proj", "block.attn.v_proj"],
                "reason": "Q/K/V projection layers share a common parent name.",
                "confidence": "high",
            },
            {
                "group_name": "block.mlp:mlp_projections",
                "group_type": "mlp_projection_pair",
                "members": ["block.mlp.fc1", "block.mlp.fc2"],
                "reason": "MLP-like expansion/projection layers share a common parent.",
                "confidence": "medium",
            },
        ],
    }


def test_analyze_dependency_graph_reports_counts_and_targets():
    graph = build_dependency_graph_from_torch_summary(fake_torch_summary())
    summary = analyze_dependency_graph(graph)

    assert summary["num_prunable_units"] > 0
    assert summary["edge_type_counts"]["qkv_coupling"] > 0
    assert summary["edge_type_counts"]["mlp_hidden_coupling"] > 0
    assert any(target["unit_type"] == "attention_qkv" for target in summary["high_value_pruning_targets"])
    assert any(target["unit_type"] in {"mlp_expansion", "mlp_projection"} for target in summary["high_value_pruning_targets"])
    assert summary["manual_review_items"]


def test_dependency_analysis_markdown_has_expected_sections():
    graph = build_dependency_graph_from_torch_summary(fake_torch_summary())
    summary = analyze_dependency_graph(graph)
    markdown = dependency_analysis_to_markdown(summary)

    assert "# Dependency Summary: tiny" in markdown
    assert "## High-Value Pruning Targets" in markdown
    assert "## Forward Propagation Paths" in markdown
    assert "## Backward Propagation Constraints" in markdown
    assert "## Manual Review Items" in markdown
