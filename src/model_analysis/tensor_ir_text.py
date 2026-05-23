"""Readable textual dump for frontend-independent Tensor Graph IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir
from model_analysis.tensor_ir import TensorGraph, tensor_graph_to_dict


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _ref(identifier: str | None) -> str:
    return f"%{identifier}" if identifier else "none"


def _dtype(value: str | None) -> str:
    return {
        "FLOAT": "f32",
        "DOUBLE": "f64",
        "FLOAT16": "f16",
        "BFLOAT16": "bf16",
        "INT64": "i64",
        "INT32": "i32",
        "INT16": "i16",
        "INT8": "i8",
        "UINT8": "ui8",
        "BOOL": "i1",
    }.get(value or "", (value or "?").lower())


def _shape(value: dict[str, Any]) -> str:
    shape = value.get("shape")
    dtype = _dtype(value.get("dtype"))
    if shape is None:
        return f"tensor<*x{dtype}>"
    dimensions = "x".join(str(item) for item in shape) if shape else ""
    return f"tensor<{dimensions}x{dtype}>" if dimensions else f"tensor<{dtype}>"


def tensor_graph_to_text(graph: TensorGraph | dict) -> str:
    data = tensor_graph_to_dict(graph) if isinstance(graph, TensorGraph) else graph
    graph_inputs = set(data.get("graph_inputs", []))
    graph_outputs = set(data.get("graph_outputs", []))
    initializers = set(data.get("initializers", []))
    lines = [
        f'tensor.graph @{_escape(data.get("model_name", "model"))} frontend("{_escape(data.get("source_frontend", "unknown"))}") {{'
    ]
    for value in data.get("values", []):
        if value["value_id"] not in graph_inputs and value["value_id"] not in initializers:
            continue
        markers = []
        if value["value_id"] in graph_inputs:
            markers.append("graph_input")
        if value["value_id"] in initializers:
            markers.append("initializer")
        lines.append(
            f'  {_ref(value["value_id"])} : {_shape(value)} role("{_escape(value.get("semantic_role", "unknown"))}") {" ".join(markers)}'
        )
    if graph_inputs or initializers:
        lines.append("")
    for op in data.get("ops", []):
        outputs = ", ".join(_ref(item) for item in op.get("outputs", [])) or "_"
        inputs = ", ".join(_ref(item) for item in op.get("inputs", []))
        suffix = []
        if op.get("is_fork"):
            suffix.append("fork(true)")
        if op.get("is_join"):
            suffix.append("join(true)")
        if op.get("region_hint"):
            suffix.append(f'hint("{_escape(op["region_hint"])}")')
        source = op.get("source_node_name") or op.get("name", "")
        lines.append(
            f'  {outputs} = tensor.op "{_escape(op.get("canonical_op_type", "unknown"))}"({inputs})'
        )
        lines.append(
            f'        source("{_escape(op.get("source_frontend", "unknown"))}::{_escape(source)}") {" ".join(suffix)}'.rstrip()
        )
    if graph_outputs:
        lines.append("")
        lines.append("  tensor.return " + ", ".join(_ref(item) for item in data.get("graph_outputs", [])))
    summary = data.get("summary", {})
    lines.extend(
        [
            "",
            f'  // forks: {summary.get("num_fork_ops", 0)}, joins: {summary.get("num_join_ops", 0)}',
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def write_tensor_graph_text(graph: TensorGraph | dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(tensor_graph_to_text(graph), encoding="utf-8")
