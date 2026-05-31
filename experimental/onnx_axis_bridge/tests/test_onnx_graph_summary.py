from __future__ import annotations

import pytest

pytest.importorskip("onnx")

from experimental.onnx_axis_bridge.onnx_graph_summary import summarize_subgraph
from experimental.onnx_axis_bridge.onnx_loader import load_onnx_subgraph
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHintKind, infer_pattern_hints
from experimental.onnx_axis_bridge.tests.helpers import make_ffn, make_residual


def test_load_synthetic_ffn_onnx(tmp_path) -> None:
    subgraph = load_onnx_subgraph(make_ffn(tmp_path / "ffn.onnx"))
    summary = summarize_subgraph(subgraph)

    assert summary.num_nodes == 3
    assert summary.op_type_counts == {"MatMul": 2, "Relu": 1}
    assert summary.producer_by_tensor["Intermediate"] == "node_000"
    assert summary.consumers_by_tensor["Activated"] == ["node_002"]


def test_ffn_hint_detected(tmp_path) -> None:
    subgraph = load_onnx_subgraph(make_ffn(tmp_path / "ffn.onnx"))
    hints = infer_pattern_hints(subgraph, summarize_subgraph(subgraph))

    assert any(hint.kind == OnnxPatternHintKind.FFN_LIKE for hint in hints)


def test_residual_hint_detected(tmp_path) -> None:
    subgraph = load_onnx_subgraph(make_residual(tmp_path / "residual.onnx"))
    hints = infer_pattern_hints(subgraph, summarize_subgraph(subgraph))

    assert any(hint.kind == OnnxPatternHintKind.RESIDUAL_LIKE for hint in hints)
