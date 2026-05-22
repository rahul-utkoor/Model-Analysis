import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from model_analysis.structural_inventory import summarize_torch_model


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 4)
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.out_proj = nn.Linear(4, 4)
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        hidden = self.out_proj(self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden))
        return self.norm(self.fc2(self.fc1(hidden)))


def test_summarize_tiny_torch_model_detects_structural_features():
    model = TinyBlock()
    summary = summarize_torch_model(
        model,
        "tiny",
        {"hf_id": "local/tiny", "task": "unit-test"},
    )

    assert summary["parameter_summary"]["total_parameters"] == sum(p.numel() for p in model.parameters())
    assert summary["parameter_summary"]["trainable_parameters"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert len(summary["linear_layers"]) == 6
    assert len(summary["embedding_layers"]) == 1
    assert any(entry["name"] == "q_proj" for entry in summary["attention_like_modules"])
    assert any(entry["name"] == "fc1" for entry in summary["mlp_like_modules"])
    assert summary["normalization_layers"][0]["name"] == "norm"
    assert summary["pruning_relevant_groups"]
