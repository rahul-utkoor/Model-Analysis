"""Frontend-independent tensor dataflow intermediate representation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class TensorValue:
    value_id: str
    name: str
    producer: str | None
    consumers: list[str]
    shape: list[int | str] | None
    dtype: str | None
    is_initializer: bool
    is_graph_input: bool
    is_graph_output: bool
    semantic_role: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorOp:
    op_id: str
    name: str
    op_type: str
    canonical_op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any]
    predecessor_ops: list[str]
    successor_ops: list[str]
    is_fork: bool
    is_join: bool
    region_hint: str | None
    source_frontend: str
    source_node_name: str | None
    source_location: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorGraph:
    graph_id: str
    model_name: str
    source_frontend: str
    ops: list[TensorOp]
    values: list[TensorValue]
    graph_inputs: list[str]
    graph_outputs: list[str]
    initializers: list[str]
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def tensor_value_to_dict(value: TensorValue) -> dict[str, Any]:
    return asdict(value)


def tensor_op_to_dict(op: TensorOp) -> dict[str, Any]:
    return asdict(op)


def tensor_graph_to_dict(graph: TensorGraph) -> dict[str, Any]:
    return asdict(graph)


def write_tensor_graph_json(graph: TensorGraph, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(tensor_graph_to_dict(graph), indent=2), encoding="utf-8")


def load_tensor_graph_json(path: Path) -> TensorGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TensorGraph(
        graph_id=data["graph_id"],
        model_name=data["model_name"],
        source_frontend=data.get("source_frontend", "unknown"),
        ops=[TensorOp(**item) for item in data.get("ops", [])],
        values=[TensorValue(**item) for item in data.get("values", [])],
        graph_inputs=data.get("graph_inputs", []),
        graph_outputs=data.get("graph_outputs", []),
        initializers=data.get("initializers", []),
        summary=data.get("summary", {}),
        metadata=data.get("metadata", {}),
    )


def canonicalize_op_type(
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    initializer_names: set[str],
    attributes: dict | None = None,
) -> tuple[str, str | None, str]:
    """Map a frontend operation into a conservative tensor-IR operation type."""
    del outputs, attributes
    non_initializer_inputs = [name for name in inputs if name and name not in initializer_names]
    has_initializer = any(name in initializer_names for name in inputs)
    if op_type == "Gemm" or (op_type == "MatMul" and has_initializer):
        return "linear", "LinearProjection", "Parameterized matrix projection consumes an initializer."
    if op_type == "MatMul":
        return "matmul", "AttentionSkeleton", "Matrix multiplication has no initializer evidence in this frontend record."
    if op_type == "Conv":
        return "linear", "LinearProjection", "Convolution is a parameterized tensor projection."
    if op_type in {"Add", "Sum"} and has_initializer and len(non_initializer_inputs) <= 1:
        return "bias_add", "LinearProjection", "Add consumes an initializer and is treated as a projection bias."
    if op_type in {"Add", "Sum"} and len(non_initializer_inputs) >= 2:
        return "elementwise_join", "ResidualJoin", "Multiple activation tensors merge; residual semantics require later evidence."
    if op_type in {"LayerNormalization", "SkipLayerNormalization"}:
        return "layer_norm", "ResidualJoin", "Normalization commonly constrains a joined hidden dimension."
    if op_type == "Gather" and has_initializer:
        return "embedding_lookup", "Embedding", "Gather consumes a parameter initializer and an index-like tensor."
    if op_type in {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Concat", "Slice", "Gather", "Shape", "Range", "Cast", "Flatten", "Split"}:
        return "shape_op", "ShapeTransform", "Operation may transform or construct tensor-axis structure."
    if op_type == "Softmax":
        return "softmax", "AttentionSkeleton", "Softmax commonly marks attention probability structure."
    if op_type in {"Erf", "Gelu", "Relu", "QuickGelu", "Sigmoid", "Tanh"}:
        return "activation", "FeedForward", "Elementwise activation may lie in a feed-forward region."
    if op_type in {"Constant", "ConstantOfShape"}:
        return "constant", None, "Operation creates a constant value."
    if op_type == "Where":
        return "mask_or_select", "ShapeTransform", "Selection can encode masking or conditional tensor flow."
    return "unknown", None, "No conservative frontend-independent classification is available."


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 300) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(_escape_cell(row.get(column, "")) for column in columns) + " |")
    if len(rows) > limit:
        lines.append(f"| ... | {len(rows) - limit} more entries omitted |" + " |" * (len(columns) - 2))
    return "\n".join(lines)


def tensor_graph_to_markdown(graph: TensorGraph | dict) -> str:
    data = tensor_graph_to_dict(graph) if isinstance(graph, TensorGraph) else graph
    summary = data.get("summary", {})
    values = [
        {
            "value_id": item.get("value_id"),
            "name": item.get("name"),
            "shape": item.get("shape"),
            "role": item.get("semantic_role"),
            "producer": item.get("producer"),
            "consumers": item.get("consumers"),
        }
        for item in data.get("values", [])
    ]
    ops = [
        {
            "op_id": item.get("op_id"),
            "name": item.get("name"),
            "frontend_type": item.get("op_type"),
            "canonical_type": item.get("canonical_op_type"),
            "fork": item.get("is_fork"),
            "join": item.get("is_join"),
            "hint": item.get("region_hint"),
        }
        for item in data.get("ops", [])
    ]
    return "\n".join(
        [
            f"# Tensor Graph IR: {data.get('model_name', '')}",
            "",
            "## Summary",
            "",
            f"- Frontend: `{data.get('source_frontend', 'unknown')}`",
            f"- Operations: `{summary.get('num_ops', 0)}`",
            f"- Tensor values: `{summary.get('num_values', 0)}`",
            f"- Initializers: `{summary.get('num_initializers', 0)}`",
            f"- Fork operations: `{summary.get('num_fork_ops', 0)}`",
            f"- Join operations: `{summary.get('num_join_ops', 0)}`",
            "",
            "## Canonical Operation Counts",
            "",
            _table([{"type": key, "count": value} for key, value in sorted(summary.get("canonical_op_type_counts", {}).items())], ["type", "count"]),
            "",
            "## Operations",
            "",
            _table(ops, ["op_id", "name", "frontend_type", "canonical_type", "fork", "join", "hint"]),
            "",
            "## Tensor Values",
            "",
            _table(values, ["value_id", "name", "shape", "role", "producer", "consumers"]),
            "",
            "## Interpretation",
            "",
            "Tensor IR is a frontend-independent tensor-dataflow substrate. This instance was imported from ONNX, but later region-tree and propagation analyses should consume Tensor IR rather than depend directly on ONNX nodes.",
            "",
        ]
    )
