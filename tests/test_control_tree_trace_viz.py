from __future__ import annotations

import importlib.util
from pathlib import Path

from model_analysis.control_tree_trace import build_control_tree_trace, control_tree_trace_to_dict
from model_analysis.control_tree_trace_viz import control_tree_step_to_dot, write_control_tree_step_dot_files
from test_control_tree_trace import tiny_tensor_graph, tiny_tree


def test_control_tree_step_to_dot_contains_digraph_and_created_region() -> None:
    trace = build_control_tree_trace(tiny_tensor_graph(), structural_region_tree=tiny_tree(), max_snapshot_nodes=20)
    collapse_step = next(step for step in control_tree_trace_to_dict(trace)["steps"] if step["action"] == "collapse")
    dot = control_tree_step_to_dot(collapse_step)

    assert "digraph" in dot
    assert collapse_step["created_region_id"] in dot


def test_write_control_tree_step_dot_files(tmp_path: Path) -> None:
    trace = build_control_tree_trace(tiny_tensor_graph(), structural_region_tree=tiny_tree(), max_snapshot_nodes=20)
    paths = write_control_tree_step_dot_files(control_tree_trace_to_dict(trace), tmp_path, max_steps=2)

    assert len(paths) == 2
    assert paths[0].read_text(encoding="utf-8").startswith("digraph")


def test_mindnode_trace_outline_contains_ordered_steps() -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "export_control_tree_trace_mindnode.py"
    spec = importlib.util.spec_from_file_location("export_control_tree_trace_mindnode", tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    trace = control_tree_trace_to_dict(build_control_tree_trace(tiny_tensor_graph(), structural_region_tree=tiny_tree()))

    lines = module.make_outline_lines(trace)

    assert lines[0] == "Control Tree Trace: tiny"
    assert any("Step 000" in line for line in lines)
    assert any("collapse" in line for line in lines)
