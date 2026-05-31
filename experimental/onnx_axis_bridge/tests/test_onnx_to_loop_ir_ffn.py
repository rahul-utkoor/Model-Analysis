from __future__ import annotations

import pytest

pytest.importorskip("onnx")

from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.pattern_recognition import PatternKind, recognize_patterns
from experimental.onnx_axis_bridge.onnx_graph_summary import summarize_subgraph
from experimental.onnx_axis_bridge.onnx_loader import load_onnx_subgraph
from experimental.onnx_axis_bridge.onnx_to_loop_ir import lower_onnx_hint_to_region_spec
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHintKind, infer_pattern_hints
from experimental.onnx_axis_bridge.tests.helpers import make_ffn


def test_ffn_lowering_to_region_spec(tmp_path) -> None:
    subgraph = load_onnx_subgraph(make_ffn(tmp_path / "ffn.onnx"))
    hint = next(hint for hint in infer_pattern_hints(subgraph, summarize_subgraph(subgraph)) if hint.kind == OnnxPatternHintKind.FFN_LIKE)
    region = lower_onnx_hint_to_region_spec(subgraph, hint)
    summary = analyze_region(region)
    patterns = recognize_patterns(region, summary)

    assert region.attrs["source"] == "onnx_axis_bridge"
    assert region.attrs["hint_kind"] == "FFN_LIKE"
    assert any(pattern.pattern_kind == PatternKind.FFN_INTERMEDIATE_CHAIN for pattern in patterns)
