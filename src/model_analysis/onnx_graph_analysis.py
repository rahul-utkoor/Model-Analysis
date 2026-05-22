"""ONNX graph inventory helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


def load_onnx_model(path: Path):
    """Load an ONNX model from disk."""
    return onnx.load(path)


def _shape_from_value_info(value_info) -> list[Any]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []

    dims: list[Any] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def _data_type_name(data_type: int) -> str:
    try:
        return TensorProto.DataType.Name(data_type)
    except ValueError:
        return str(data_type)


def infer_initializer_shapes(model) -> dict[str, list[int]]:
    """Return initializer tensor shapes keyed by tensor name."""
    return {initializer.name: list(initializer.dims) for initializer in model.graph.initializer}


def infer_value_info_shapes(model) -> dict[str, list[Any]]:
    """Return available input/output/intermediate shapes keyed by tensor name."""
    shapes: dict[str, list[Any]] = {}
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        shapes[value_info.name] = _shape_from_value_info(value_info)
    return shapes


def node_summary(model) -> dict[str, Any]:
    """Return node counts by ONNX op type."""
    op_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "num_nodes": len(model.graph.node),
        "op_type_counts": dict(op_counts),
    }


def edge_summary(model) -> dict[str, Any]:
    """Return simple tensor edge statistics for the graph."""
    produced = [output for node in model.graph.node for output in node.output]
    consumed = [input_name for node in model.graph.node for input_name in node.input if input_name]
    return {
        "num_produced_tensors": len(produced),
        "num_consumed_tensors": len(consumed),
        "num_unique_tensors": len(set(produced) | set(consumed)),
    }


def find_pruning_relevant_onnx_nodes(model) -> list[dict[str, str]]:
    """Find ONNX nodes that are directly prunable or relevant to pruning propagation."""
    entries = []
    direct_prunable = {"MatMul", "Gemm", "Conv"}
    propagation_relevant = {
        "Add",
        "Mul",
        "Reshape",
        "Transpose",
        "Gather",
        "LayerNormalization",
        "Softmax",
    }

    for index, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{index}"
        if node.op_type in direct_prunable:
            reason = "high-interest parameterized projection/convolution node"
            if node.op_type == "Conv":
                reason = "high-interest convolution node, potentially patch projection in vision models"
            entries.append(
                {
                    "name": name,
                    "op_type": node.op_type,
                    "reason": reason,
                    "confidence": "high",
                }
            )
        elif node.op_type in propagation_relevant:
            entries.append(
                {
                    "name": name,
                    "op_type": node.op_type,
                    "reason": "propagation-relevant graph operation, not directly marked prunable",
                    "confidence": "medium" if node.op_type in {"Add", "Reshape", "Transpose", "LayerNormalization"} else "low",
                }
            )
    return entries


def summarize_onnx_graph(onnx_path: Path, model_name: str, model_config: dict) -> dict[str, Any]:
    """Build a structural summary for an ONNX graph."""
    model = load_onnx_model(onnx_path)
    op_counts = Counter(node.op_type for node in model.graph.node)

    inputs = [
        {
            "name": value_info.name,
            "shape": _shape_from_value_info(value_info),
            "data_type": _data_type_name(value_info.type.tensor_type.elem_type),
        }
        for value_info in model.graph.input
    ]
    outputs = [
        {
            "name": value_info.name,
            "shape": _shape_from_value_info(value_info),
            "data_type": _data_type_name(value_info.type.tensor_type.elem_type),
        }
        for value_info in model.graph.output
    ]
    initializers = [
        {
            "name": initializer.name,
            "dims": list(initializer.dims),
            "data_type": _data_type_name(initializer.data_type),
        }
        for initializer in model.graph.initializer
    ]
    nodes = [
        {
            "name": node.name or f"{node.op_type}_{index}",
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        for index, node in enumerate(model.graph.node)
    ]

    return {
        "model_name": model_name,
        "hf_id": model_config.get("hf_id"),
        "task": model_config.get("task"),
        "onnx_path": str(onnx_path),
        "graph_summary": {
            "num_nodes": len(model.graph.node),
            "num_initializers": len(model.graph.initializer),
            "num_inputs": len(model.graph.input),
            "num_outputs": len(model.graph.output),
            "op_type_counts": dict(op_counts),
        },
        "edge_summary": edge_summary(model),
        "initializer_shapes": infer_initializer_shapes(model),
        "inputs": inputs,
        "outputs": outputs,
        "value_info_shapes": infer_value_info_shapes(model),
        "initializers": initializers,
        "nodes": nodes,
        "pruning_relevant_nodes": find_pruning_relevant_onnx_nodes(model),
    }
