"""DOT visualization helpers for control-tree construction trace steps."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


def _quote(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _node_color(node: dict[str, Any], created_region: str | None) -> str:
    if node.get("node_id") == created_region:
        return "#ffe08a"
    if node.get("node_kind") == "model_root":
        return "#b7e4c7"
    if node.get("node_kind") == "abstract_region":
        role = node.get("pruning_role")
        if role == "blocked":
            return "#ffadad"
        if role == "directly_prunable":
            return "#caffbf"
        if role == "analysis_only":
            return "#d0d1ff"
        return "#cde7ff"
    return "#f1f3f5"


def control_tree_step_to_dot(step: dict[str, Any]) -> str:
    snapshot = step.get("graph_snapshot", {})
    created_region = step.get("created_region_id")
    collapsed_nodes = set(step.get("collapsed_node_ids", []))
    lines = [
        "digraph control_tree_step {",
        "  rankdir=LR;",
        '  graph [fontname="Helvetica"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=9];',
        f'  label="step {step.get("step_index", 0)} {step.get("pass_name", "")}";',
    ]
    for node in snapshot.get("nodes", []):
        node_id = node.get("node_id")
        label = node.get("label") or node_id
        if node.get("region_type"):
            label = f"{label}\\n{node.get('region_type')}"
        elif node.get("canonical_op_type"):
            label = f"{label}\\n{node.get('canonical_op_type')}"
        attrs = [
            f'label="{_quote(label)}"',
            f'fillcolor="{_node_color(node, created_region)}"',
        ]
        if node_id in collapsed_nodes:
            attrs.append('style="rounded,filled,dashed"')
        lines.append(f'  "{_quote(node_id)}" [{", ".join(attrs)}];')
    for edge in snapshot.get("edges", []):
        label = edge.get("label") or edge.get("tensor_or_value_id") or ""
        attrs = f' [label="{_quote(label)}"]' if label else ""
        lines.append(f'  "{_quote(edge.get("src"))}" -> "{_quote(edge.get("dst"))}"{attrs};')
    if snapshot.get("truncated"):
        lines.append('  "__truncated__" [label="snapshot truncated", fillcolor="#eeeeee"];')
    lines.append("}")
    return "\n".join(lines)


def write_control_tree_step_dot_files(
    trace: dict[str, Any],
    output_dir: Path,
    max_steps: int | None = None,
    render_svg: bool = False,
) -> list[Path]:
    ensure_dir(output_dir)
    paths: list[Path] = []
    steps = trace.get("steps", [])
    selected_steps = steps[:max_steps] if max_steps is not None else steps
    dot_bin = shutil.which("dot") if render_svg else None
    for step in selected_steps:
        index = int(step.get("step_index", 0))
        dot_path = output_dir / f"step_{index:03d}.dot"
        dot_path.write_text(control_tree_step_to_dot(step), encoding="utf-8")
        paths.append(dot_path)
        if dot_bin:
            svg_path = output_dir / f"step_{index:03d}.svg"
            try:
                subprocess.run([dot_bin, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=False)
                if svg_path.exists():
                    paths.append(svg_path)
            except OSError:
                pass
    return paths
