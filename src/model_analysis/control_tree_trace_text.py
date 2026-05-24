"""Textual dump for stepwise control-tree construction traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.control_tree_trace import ControlTreeTrace, control_tree_trace_to_dict
from model_analysis.paths import ensure_dir


def _quote(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _fmt_ids(values: list[str], prefix: str) -> str:
    if not values:
        return f"{prefix}()"
    return f"{prefix}(" + ", ".join(f"%{item}" for item in values) + ")"


def control_tree_trace_to_text(trace: ControlTreeTrace | dict) -> str:
    data = control_tree_trace_to_dict(trace)
    lines = [f'control_tree.trace @{_quote(data.get("model_name", ""))} {{']
    for step in data.get("steps", []):
        pass_name = step.get("pass_name", "unknown")
        lines.append(f"  step {int(step.get('step_index', 0)):03d} {pass_name} {{")
        before = step.get("before_summary", {}).get("num_active_nodes", 0)
        after = step.get("after_summary", {}).get("num_active_nodes", 0)
        lines.append(f'    action "{_quote(step.get("action", ""))}"')
        if step.get("created_region_id"):
            lines.append(
                f'    create %{step.get("created_region_id")} : {step.get("created_region_type", "Region")}'
            )
        if step.get("collapsed_op_ids"):
            lines.append(f"    collapse {_fmt_ids(step.get('collapsed_op_ids', []), 'ops')}")
        if step.get("collapsed_region_ids"):
            lines.append(f"    collapse {_fmt_ids(step.get('collapsed_region_ids', []), 'regions')}")
        if step.get("collapsed_node_ids"):
            lines.append(f"    active_nodes {before} -> {after}")
        else:
            lines.append(f"    active_nodes_before = {before}")
            lines.append(f"    active_nodes_after = {after}")
        lines.append(f'    confidence "{_quote(step.get("confidence", ""))}"')
        lines.append(f'    reason "{_quote(step.get("reason", ""))}"')
        lines.append("  }")
        lines.append("")
    lines.append("}")
    return "\n".join(lines)


def write_control_tree_trace_text(trace: ControlTreeTrace | dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(control_tree_trace_to_text(trace), encoding="utf-8")
