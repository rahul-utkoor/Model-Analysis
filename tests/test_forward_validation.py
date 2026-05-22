from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model_analysis.forward_validation import forward_smoke_result_to_markdown, run_forward_smoke_test, summarize_model_output


class TinyTensorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, inputs):
        return self.fc(inputs)


class FailingModel(torch.nn.Module):
    def forward(self, inputs):
        raise RuntimeError("intentional failure")


def test_tensor_forward_smoke_passes_and_records_shape():
    result = run_forward_smoke_test(TinyTensorModel(), {"name": "tiny", "task": "unit-test"}, input_kind="tensor")

    assert result.status == "passed"
    assert result.output_summary["shape"] == [1, 2]


def test_failing_model_returns_failed_result():
    result = run_forward_smoke_test(FailingModel(), {"name": "bad", "task": "unit-test"}, input_kind="tensor")

    assert result.status == "failed"
    assert result.error_message


def test_output_summary_and_markdown_have_expected_sections():
    summary = summarize_model_output(torch.zeros(1, 3))
    result = run_forward_smoke_test(TinyTensorModel(), {"name": "tiny", "task": "unit-test"}, input_kind="tensor")
    markdown = forward_smoke_result_to_markdown(result)

    assert summary["shape"] == [1, 3]
    assert "## Status" in markdown
    assert "## Output Summary" in markdown
    assert "## Caveats" in markdown
