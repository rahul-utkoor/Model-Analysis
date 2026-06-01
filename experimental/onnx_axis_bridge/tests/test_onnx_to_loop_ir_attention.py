from __future__ import annotations

import pytest

pytest.importorskip("onnx")

from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.axis_relations import AxisRelationKind
from experimental.axis_transfer_analysis.pattern_recognition import PatternKind, recognize_patterns
from experimental.onnx_axis_bridge.onnx_graph_summary import summarize_subgraph
from experimental.onnx_axis_bridge.onnx_loader import load_onnx_subgraph
from experimental.onnx_axis_bridge.onnx_to_loop_ir import lower_onnx_hint_to_region_spec
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHintKind, infer_pattern_hints
from experimental.onnx_axis_bridge.tests.helpers import make_attention_value_path, make_attention_value_path_with_cache_layout, make_context, make_qk


def test_qk_score_hint_and_blocker_detected(tmp_path) -> None:
    subgraph = load_onnx_subgraph(make_qk(tmp_path / "qk.onnx"))
    hint = next(hint for hint in infer_pattern_hints(subgraph, summarize_subgraph(subgraph)) if hint.kind == OnnxPatternHintKind.QK_SCORE_LIKE)
    region = lower_onnx_hint_to_region_spec(subgraph, hint)
    patterns = recognize_patterns(region, analyze_region(region))

    assert any(pattern.pattern_kind == PatternKind.QK_SCORE_BLOCKER for pattern in patterns)


def test_attention_context_hint_and_preserved_value_axis_detected(tmp_path) -> None:
    subgraph = load_onnx_subgraph(make_context(tmp_path / "context.onnx"))
    hint = next(hint for hint in infer_pattern_hints(subgraph, summarize_subgraph(subgraph)) if hint.kind == OnnxPatternHintKind.ATTENTION_CONTEXT_LIKE)
    summary = analyze_region(lower_onnx_hint_to_region_spec(subgraph, hint))

    assert any(
        transfer.source_tensor == "V"
        and transfer.source_axis == "value_dim"
        and transfer.target_tensor == "Context"
        and transfer.target_axis == "value_context_dim"
        and transfer.relation == AxisRelationKind.PRESERVED
        for transfer in summary.op_summaries[0].transfers
    )


def test_attention_value_path_hint_and_pattern_detected(tmp_path) -> None:
    subgraph = load_onnx_subgraph(make_attention_value_path(tmp_path / "value_path.onnx"))
    hint = next(hint for hint in infer_pattern_hints(subgraph, summarize_subgraph(subgraph)) if hint.kind == OnnxPatternHintKind.ATTENTION_VALUE_PATH_LIKE)
    region = lower_onnx_hint_to_region_spec(subgraph, hint)
    patterns = recognize_patterns(region, analyze_region(region))

    assert any(pattern.pattern_kind == PatternKind.ATTENTION_VALUE_PATH for pattern in patterns)


def test_attention_value_path_hint_accepts_cache_concat_and_cast(tmp_path) -> None:
    subgraph = load_onnx_subgraph(make_attention_value_path_with_cache_layout(tmp_path / "value_path_cache.onnx"))
    hints = infer_pattern_hints(subgraph, summarize_subgraph(subgraph))

    assert any(hint.kind == OnnxPatternHintKind.ATTENTION_VALUE_PATH_LIKE for hint in hints)
    assert not any(hint.kind == OnnxPatternHintKind.FFN_LIKE for hint in hints)
