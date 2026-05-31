from __future__ import annotations

import pytest

pytest.importorskip("onnx")

from experimental.dfa_pruning_propagation.lattice import FactKind
from experimental.onnx_axis_bridge.bridge_runner import analyze_onnx_subgraph
from experimental.onnx_axis_bridge.tests.helpers import make_attention_value_path, make_ffn, make_qk


def _facts(result):
    return [
        fact
        for lowered in result.lowered_regions
        if lowered.bridge_result
        for fact in lowered.bridge_result.dfa_result.state.values()
    ]


def test_ffn_bridge_runner_reaches_dfa_dead_producer(tmp_path) -> None:
    result = analyze_onnx_subgraph(make_ffn(tmp_path / "ffn.onnx"))

    assert "FFN_INTERMEDIATE_CHAIN" in result.summary["axis_patterns"]
    assert result.summary["dfa_propagation_results"] == 1
    assert any(fact.kind == FactKind.DEAD and fact.axis.tensor == "producer_from_axis_summary.output" for fact in _facts(result))


def test_qk_bridge_runner_reaches_blocked_dfa_result(tmp_path) -> None:
    result = analyze_onnx_subgraph(make_qk(tmp_path / "qk.onnx"))

    assert "QK_SCORE_BLOCKER" in result.summary["axis_patterns"]
    assert result.summary["blocked_results"] == 1
    assert any(fact.kind == FactKind.BLOCKED for fact in _facts(result))


def test_attention_value_bridge_runner_reaches_dead_value_producer(tmp_path) -> None:
    result = analyze_onnx_subgraph(make_attention_value_path(tmp_path / "value_path.onnx"))

    assert "ATTENTION_VALUE_PATH" in result.summary["axis_patterns"]
    assert any(fact.kind == FactKind.DEAD and fact.axis.tensor == "value_producer_from_axis_summary.output" for fact in _facts(result))
