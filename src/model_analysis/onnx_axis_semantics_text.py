"""Text, DOT, and SVG rendering for strict ONNX axis-semantics annotations."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from model_analysis.onnx_axis_semantics import AxisSemanticClass, NodeAxisSemantics


COLOR_BY_CLASS = {
    AxisSemanticClass.MLIR_DERIVED_INDEX_PRESERVING: "palegreen",
    AxisSemanticClass.MLIR_DERIVED_ELEMENTWISE_PRESERVE: "palegreen",
    AxisSemanticClass.MLIR_DERIVED_PROJECTION_EXPAND: "lightskyblue",
    AxisSemanticClass.MLIR_DERIVED_PROJECTION_CONTRACT: "plum",
    AxisSemanticClass.MLIR_DERIVED_MATMUL_ATTENTION_CONTEXT: "paleturquoise",
    AxisSemanticClass.MLIR_DERIVED_MATMUL_QK_SCORE: "lightcoral",
    AxisSemanticClass.MLIR_DERIVED_BLOCKER: "lightcoral",
    AxisSemanticClass.MLIR_DERIVED_REDUCTION: "orange",
    AxisSemanticClass.MLIR_DERIVED_BRANCH_MERGE: "khaki",
    AxisSemanticClass.MLIR_HIGH_LEVEL_INSUFFICIENT: "gray85",
    AxisSemanticClass.NO_ACCESS_EVIDENCE: "gray85",
    AxisSemanticClass.MLIR_LOWERING_FAILED: "firebrick1",
    AxisSemanticClass.UNKNOWN: "gray92",
}


def write_annotated_dot(model: Any, nodes: list[NodeAxisSemantics], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_name = {node.node_name: node for node in nodes}
    tensor_producers: dict[str, str] = {}
    for node in model.graph.node:
        node_name = node.name or f"{node.op_type}_{len(tensor_producers)}"
        for output in node.output:
            tensor_producers[output] = node_name

    lines = [
        "digraph axis_semantics {",
        "  rankdir=LR;",
        "  node [shape=box style=\"rounded,filled\" fontname=\"Helvetica\" fontsize=10];",
    ]
    for index, node in enumerate(model.graph.node):
        node_name = node.name or f"{node.op_type}_{index}"
        semantic = by_name.get(node_name)
        color = COLOR_BY_CLASS.get(semantic.semantic_class if semantic else AxisSemanticClass.UNKNOWN, "gray92")
        label = "\\n".join(
            [
                _escape_dot(node_name),
                _escape_dot(node.op_type),
                _escape_dot(semantic.semantic_class.value if semantic else AxisSemanticClass.UNKNOWN.value),
                _escape_dot(semantic.evidence_tier.value if semantic else "NONE"),
                _escape_dot(semantic.mlir_evidence.blocker_kind.value if semantic else "unknown"),
                _escape_dot(semantic.leader_candidate_kind if semantic else "unknown"),
            ]
        )
        lines.append(f'  "{_escape_dot(node_name)}" [label="{label}" fillcolor="{color}"];')

    for index, node in enumerate(model.graph.node):
        target = node.name or f"{node.op_type}_{index}"
        for input_name in node.input:
            producer = tensor_producers.get(input_name)
            if producer:
                lines.append(f'  "{_escape_dot(producer)}" -> "{_escape_dot(target)}" [label="{_escape_dot(input_name)}"];')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_svg_from_dot(dot_path: str | Path, svg_path: str | Path) -> tuple[Path | None, str | None]:
    dot_binary = shutil.which("dot")
    if not dot_binary:
        return None, "graphviz dot executable was not found; SVG was not rendered"
    output = Path(svg_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run([dot_binary, "-Tsvg", str(dot_path), "-o", str(output)], capture_output=True, text=True, check=False)
    if completed.returncode:
        return None, f"graphviz dot failed with exit code {completed.returncode}: {completed.stderr.strip()}"
    return output, None


def _escape_dot(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
