from model_analysis.action_generation import generate_candidate_actions
from model_analysis.dependency_graph import DependencyGraph, PrunableUnit


def test_generate_candidate_actions_is_non_empty_and_small():
    graph = DependencyGraph(
        model_name="tiny",
        hf_id="local/tiny",
        task="unit-test",
        prunable_units=[
            PrunableUnit("torch:attention_qkv:block.attn", "block.attn", "torch", "attention_qkv", "block.attn", ["num_heads", "head_dim", "hidden_dim"], None, None, "medium", "qkv"),
            PrunableUnit("torch:linear:fc1", "fc1", "torch", "mlp_expansion", "fc1", ["out_features", "intermediate_dim"], 144, [16, 8], "medium", "fc1"),
            PrunableUnit("torch:linear:linear", "linear", "torch", "linear", "linear", ["out_features"], 32, [8, 4], "medium", "linear"),
        ],
    )

    actions = generate_candidate_actions(graph)

    assert actions
    assert any(action.target_unit_type == "attention_qkv" for action in actions)
    assert any(action.target_unit_type == "mlp_expansion" for action in actions)
    assert all(len(action.indices) <= 4 for action in actions)
