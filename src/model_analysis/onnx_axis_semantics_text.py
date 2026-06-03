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
    AxisSemanticClass.MLIR_DERIVED_MATMUL_GENERIC: "lightskyblue",
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


def short_semantic_class_name(cls: AxisSemanticClass | str) -> str:
    value = cls.value if isinstance(cls, AxisSemanticClass) else str(cls)
    return {
        AxisSemanticClass.MLIR_DERIVED_MATMUL_GENERIC.value: "MatMul",
        AxisSemanticClass.MLIR_DERIVED_ELEMENTWISE_PRESERVE.value: "Preserve",
        AxisSemanticClass.MLIR_DERIVED_INDEX_PRESERVING.value: "Preserve",
        AxisSemanticClass.MLIR_DERIVED_REDUCTION.value: "Reduce",
        AxisSemanticClass.MLIR_DERIVED_BLOCKER.value: "Blocker",
        AxisSemanticClass.MLIR_DERIVED_MATMUL_QK_SCORE.value: "QK blocker",
        AxisSemanticClass.MLIR_HIGH_LEVEL_INSUFFICIENT.value: "MLIR insufficient",
        AxisSemanticClass.MLIR_LOWERING_FAILED.value: "Lowering failed",
        AxisSemanticClass.NO_ACCESS_EVIDENCE.value: "No access evidence",
        AxisSemanticClass.UNKNOWN.value: "Unknown",
    }.get(value, value.removeprefix("MLIR_DERIVED_").replace("_", " ").title())


def short_evidence_tier_name(value: str) -> str:
    return {
        "NATIVE_MLIR_DEPENDENCE": "native",
        "PYTHON_MLIR_ACCESS": "python-access",
        "HIGH_LEVEL_MLIR_ONLY": "high-level",
        "MLIR_LOWERING_FAILED": "lowering-failed",
        "NONE": "none",
    }.get(value, value.lower().replace("_", "-"))


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
                _escape_dot(_short_node_name(node_name)),
                _escape_dot(node.op_type),
                _escape_dot(short_semantic_class_name(semantic.semantic_class if semantic else AxisSemanticClass.UNKNOWN)),
                _escape_dot(short_evidence_tier_name(semantic.evidence_tier.value if semantic else "NONE")),
                _escape_dot(f"leader={semantic.leader_candidate_kind if semantic else 'unknown'}"),
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


def _short_node_name(value: str, limit: int = 42) -> str:
    if len(value) <= limit:
        return value
    return "..." + value[-(limit - 3) :]
