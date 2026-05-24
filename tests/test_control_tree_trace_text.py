from __future__ import annotations

from model_analysis.control_tree_trace import build_control_tree_trace
from model_analysis.control_tree_trace_text import control_tree_trace_to_text
from test_control_tree_trace import tiny_tensor_graph, tiny_tree


def test_control_tree_trace_text_contains_key_passes() -> None:
    trace = build_control_tree_trace(tiny_tensor_graph(), structural_region_tree=tiny_tree(), max_snapshot_nodes=20)
    text = control_tree_trace_to_text(trace)

    assert "control_tree.trace" in text
    assert "initialize_primitives" in text
    assert "collapse_feedforward" in text
