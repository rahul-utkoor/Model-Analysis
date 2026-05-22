import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from model_analysis.onnx_graph_analysis import summarize_onnx_graph


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, inputs):
        return self.linear(inputs)


def test_summarize_tiny_onnx_graph(tmp_path):
    model = TinyLinear().eval()
    onnx_path = tmp_path / "tiny.onnx"

    torch.onnx.export(
        model,
        (torch.randn(2, 4),),
        onnx_path,
        input_names=["inputs"],
        output_names=["outputs"],
        opset_version=17,
    )

    summary = summarize_onnx_graph(onnx_path, "tiny", {"hf_id": "local/tiny", "task": "unit-test"})

    assert summary["graph_summary"]["num_nodes"] > 0
    assert {"MatMul", "Gemm"} & set(summary["graph_summary"]["op_type_counts"])
    assert summary["pruning_relevant_nodes"]
