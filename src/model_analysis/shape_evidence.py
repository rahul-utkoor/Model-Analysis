"""Static ONNX tensor and node shape evidence."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class TensorShapeEvidence:
    tensor_name: str
    shape: list[int | str]
    source: str
    confidence: str
    reason: str


@dataclass
class NodeShapeEvidence:
    node_name: str
    op_type: str
    input_shapes: dict[str, list[int | str]]
    output_shapes: dict[str, list[int | str]]
    shape_constraints: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "low"
    reason: str = ""


@dataclass
class ShapeEvidenceReport:
    model_name: str
    hf_id: str
    task: str
    tensor_shapes: list[TensorShapeEvidence] = field(default_factory=list)
    node_shapes: list[NodeShapeEvidence] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_shape(shape: Any) -> list[int | str]:
    if shape is None or shape == "unknown":
        return []
    return [dim if isinstance(dim, int) else str(dim) for dim in shape]


def _tensor_shape_map(onnx_summary: dict[str, Any]) -> dict[str, list[int | str]]:
    tensor_shapes: dict[str, list[int | str]] = {}
    for item in onnx_summary.get("inputs", []):
        tensor_shapes[item.get("name", "")] = _normalize_shape(item.get("shape"))
    for item in onnx_summary.get("outputs", []):
        tensor_shapes[item.get("name", "")] = _normalize_shape(item.get("shape"))
    for name, shape in onnx_summary.get("value_info_shapes", {}).items():
        tensor_shapes[name] = _normalize_shape(shape)
    for item in onnx_summary.get("initializers", []):
        tensor_shapes[item.get("name", "")] = _normalize_shape(item.get("dims"))
    for name, shape in onnx_summary.get("tensor_shape_map", {}).items():
        tensor_shapes.setdefault(name, _normalize_shape(shape))
    return {name: shape for name, shape in tensor_shapes.items() if name}


def _tensor_entries(onnx_summary: dict[str, Any]) -> list[TensorShapeEvidence]:
    entries: dict[str, TensorShapeEvidence] = {}
    for item in onnx_summary.get("inputs", []):
        name = item.get("name", "")
        entries[name] = TensorShapeEvidence(name, _normalize_shape(item.get("shape")), "onnx_input", "high", "Shape declared on ONNX graph input.")
    for item in onnx_summary.get("outputs", []):
        name = item.get("name", "")
        entries[name] = TensorShapeEvidence(name, _normalize_shape(item.get("shape")), "onnx_output", "high", "Shape declared on ONNX graph output.")
    for name, shape in onnx_summary.get("value_info_shapes", {}).items():
        entries.setdefault(name, TensorShapeEvidence(name, _normalize_shape(shape), "onnx_value_info", "medium", "Shape found in ONNX value_info."))
    for item in onnx_summary.get("initializers", []):
        name = item.get("name", "")
        entries[name] = TensorShapeEvidence(name, _normalize_shape(item.get("dims")), "onnx_initializer", "high", "Shape declared by ONNX initializer dimensions.")
    return [entry for entry in entries.values() if entry.tensor_name]


def _node_constraints(node: dict[str, Any], input_shapes: dict[str, list[int | str]], output_shapes: dict[str, list[int | str]]) -> list[dict[str, Any]]:
    op_type = node.get("op_type", "")
    constraints = []
    if op_type in {"Gemm", "MatMul"}:
        constraints.append({"constraint_type": "matrix_projection", "reason": "Matrix projection dimensions constrain input/output feature axes."})
    elif op_type == "Conv":
        constraints.append({"constraint_type": "channel_projection", "reason": "Convolution weight output channels constrain downstream channel dimension."})
    elif op_type in {"Add", "SkipLayerNormalization"}:
        constraints.append({"constraint_type": "residual_shape_match", "reason": "Inputs to residual-like Add operations must be shape-compatible."})
    elif op_type in {"Reshape", "Transpose", "Gather"}:
        constraints.append({"constraint_type": "shape_transform", "reason": f"{op_type} can remap pruning indices across tensor dimensions."})
    elif op_type in {"LayerNormalization", "Softmax"}:
        constraints.append({"constraint_type": "axis_dependency", "reason": f"{op_type} depends on a specific tensor axis."})
    known = sum(1 for shape in list(input_shapes.values()) + list(output_shapes.values()) if shape)
    if known:
        constraints.append({"constraint_type": "known_tensor_shapes", "known_shape_count": known, "reason": "At least one node tensor has static shape evidence."})
    return constraints


def build_shape_evidence(onnx_summary: dict) -> ShapeEvidenceReport:
    tensor_map = _tensor_shape_map(onnx_summary)
    tensor_entries = _tensor_entries(onnx_summary)
    node_entries = []
    for node in onnx_summary.get("nodes", []):
        input_shapes = {
            tensor: tensor_map[tensor]
            for tensor in node.get("inputs", [])
            if tensor in tensor_map
        }
        output_shapes = {
            tensor: tensor_map[tensor]
            for tensor in node.get("outputs", [])
            if tensor in tensor_map
        }
        constraints = _node_constraints(node, input_shapes, output_shapes)
        confidence = "medium" if input_shapes or output_shapes else "low"
        if input_shapes and output_shapes:
            confidence = "high"
        node_entries.append(
            NodeShapeEvidence(
                node_name=node.get("name", ""),
                op_type=node.get("op_type", ""),
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                shape_constraints=constraints,
                confidence=confidence,
                reason="Shape evidence collected from ONNX metadata." if constraints else "No static tensor shape evidence found for this node.",
            )
        )

    report = ShapeEvidenceReport(
        model_name=onnx_summary.get("model_name", ""),
        hf_id=onnx_summary.get("hf_id", ""),
        task=onnx_summary.get("task", ""),
        tensor_shapes=tensor_entries,
        node_shapes=node_entries,
        summary={
            "num_tensor_shapes": len(tensor_entries),
            "num_node_shapes": len(node_entries),
            "node_confidence_counts": dict(Counter(item.confidence for item in node_entries)),
            "tensor_source_counts": dict(Counter(item.source for item in tensor_entries)),
        },
        metadata={"source": "onnx_static_metadata"},
    )
    return report


def shape_evidence_report_to_dict(report: ShapeEvidenceReport) -> dict[str, Any]:
    return asdict(report)


def write_shape_evidence_json(report: ShapeEvidenceReport, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(shape_evidence_report_to_dict(report), indent=2), encoding="utf-8")


def load_shape_evidence_json(path: Path) -> ShapeEvidenceReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ShapeEvidenceReport(
        model_name=data["model_name"],
        hf_id=data.get("hf_id", ""),
        task=data.get("task", ""),
        tensor_shapes=[TensorShapeEvidence(**item) for item in data.get("tensor_shapes", [])],
        node_shapes=[NodeShapeEvidence(**item) for item in data.get("node_shapes", [])],
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if not rows:
        return "_None detected._"
    selected = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if limit and len(rows) > limit:
        lines.append("| ... | " + f"{len(rows) - limit} more rows omitted" + " |" * (len(columns) - 1))
    return "\n".join(lines)


def shape_evidence_report_to_markdown(report: ShapeEvidenceReport) -> str:
    data = shape_evidence_report_to_dict(report)
    tensor_rows = [
        {"tensor": item["tensor_name"], "shape": item["shape"], "source": item["source"], "confidence": item["confidence"], "reason": item["reason"]}
        for item in data["tensor_shapes"]
    ]
    node_rows = [
        {
            "node": item["node_name"],
            "op_type": item["op_type"],
            "input_shapes": item["input_shapes"],
            "output_shapes": item["output_shapes"],
            "confidence": item["confidence"],
            "reason": item["reason"],
        }
        for item in data["node_shapes"]
    ]
    constraint_rows = [
        {"node": item["node_name"], **constraint}
        for item in data["node_shapes"]
        for constraint in item.get("shape_constraints", [])
    ]
    lines = [
        f"# Shape Evidence: {report.model_name}",
        "",
        "## Summary",
        "",
        f"- Tensor shapes: `{data['summary'].get('num_tensor_shapes', 0)}`",
        f"- Node shape entries: `{data['summary'].get('num_node_shapes', 0)}`",
        f"- Node confidence counts: `{data['summary'].get('node_confidence_counts', {})}`",
        "",
        "## Tensor Shapes",
        "",
        _markdown_table(tensor_rows, ["tensor", "shape", "source", "confidence", "reason"], limit=300),
        "",
        "## Node Shapes",
        "",
        _markdown_table(node_rows, ["node", "op_type", "input_shapes", "output_shapes", "confidence", "reason"], limit=300),
        "",
        "## Shape Constraints",
        "",
        _markdown_table(constraint_rows, ["node", "constraint_type", "known_shape_count", "reason"], limit=300),
        "",
        "## Caveats",
        "",
        "Shape evidence is static ONNX metadata. It does not prove runtime correctness for a pruning transform.",
        "",
    ]
    return "\n".join(lines)
