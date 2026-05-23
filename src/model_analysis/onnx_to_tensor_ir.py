"""ONNX frontend importer for the frontend-independent Tensor IR."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from model_analysis.tensor_ir import TensorGraph, TensorOp, TensorValue, canonicalize_op_type


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return token or "unnamed"


def _stable_ids(names: list[str], prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        if name in result:
            continue
        base = f"{prefix}::{_safe_token(name)}"
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        result[name] = candidate
    return result


def _entry_name(entry: Any) -> str:
    return entry["name"] if isinstance(entry, dict) else str(entry)


def _normalize_shape(shape: Any) -> list[int | str] | None:
    if shape is None:
        return None
    return ["?" if value is None else value for value in list(shape)]


def _tensor_names_in_order(onnx_summary: dict[str, Any]) -> list[str]:
    names: list[str] = []
    names.extend(_entry_name(item) for item in onnx_summary.get("inputs", []))
    names.extend(_entry_name(item) for item in onnx_summary.get("initializers", []))
    for node in onnx_summary.get("nodes", []):
        names.extend(name for name in node.get("inputs", []) if name)
        names.extend(name for name in node.get("outputs", []) if name)
    names.extend(_entry_name(item) for item in onnx_summary.get("outputs", []))
    return list(dict.fromkeys(names))


def _base_role(name: str, is_initializer: bool, is_graph_input: bool) -> str:
    lower = name.lower()
    if is_initializer:
        return "parameter"
    if "mask" in lower:
        return "mask"
    if is_graph_input and ("input_ids" in lower or "token_type_ids" in lower or lower.endswith("ids")):
        return "index"
    return "activation" if is_graph_input else "unknown"


def build_tensor_graph_from_onnx_summary(
    onnx_summary: dict,
    model_config: dict,
) -> TensorGraph:
    """Import an ONNX summary into tensor-dataflow IR without retaining ONNX as the abstraction."""
    nodes = list(onnx_summary.get("nodes", []))
    initializers = {_entry_name(item) for item in onnx_summary.get("initializers", [])}
    graph_input_names = [_entry_name(item) for item in onnx_summary.get("inputs", [])]
    graph_output_names = [_entry_name(item) for item in onnx_summary.get("outputs", [])]
    all_tensor_names = _tensor_names_in_order(onnx_summary)
    value_ids = _stable_ids(all_tensor_names, "value")

    op_names = [node.get("name") or f"{node.get('op_type', 'Op')}_{index}" for index, node in enumerate(nodes)]
    op_ids = [
        f"op::{index:06d}::{_safe_token(name)}"
        for index, name in enumerate(op_names)
    ]
    producer_by_tensor: dict[str, str] = {}
    consumers_by_tensor: dict[str, list[str]] = {name: [] for name in all_tensor_names}
    for node, op_id in zip(nodes, op_ids):
        for output in node.get("outputs", []):
            if output:
                producer_by_tensor[output] = op_id
        for input_name in node.get("inputs", []):
            if input_name:
                consumers_by_tensor.setdefault(input_name, []).append(op_id)

    shape_map = dict(onnx_summary.get("tensor_shape_map", {}))
    shape_map.update(onnx_summary.get("value_info_shapes", {}))
    dtype_map: dict[str, str] = {}
    for item in onnx_summary.get("inputs", []) + onnx_summary.get("outputs", []) + onnx_summary.get("initializers", []):
        if isinstance(item, dict) and "shape" in item:
            shape_map.setdefault(item["name"], item["shape"])
        if isinstance(item, dict) and "dims" in item:
            shape_map.setdefault(item["name"], item["dims"])
        if isinstance(item, dict) and item.get("data_type"):
            dtype_map[item["name"]] = item["data_type"]

    operation_records: list[TensorOp] = []
    operation_reason: dict[str, str] = {}
    canonical_by_op_id: dict[str, str] = {}
    source_type_by_op_id: dict[str, str] = {}
    for index, (node, name) in enumerate(zip(nodes, op_names)):
        inputs = [value_ids[value] for value in node.get("inputs", []) if value]
        outputs = [value_ids[value] for value in node.get("outputs", []) if value]
        canonical_type, region_hint, reason = canonicalize_op_type(
            node.get("op_type", "Unknown"),
            list(node.get("inputs", [])),
            list(node.get("outputs", [])),
            initializers,
            node.get("attributes", {}),
        )
        op_id = op_ids[index]
        canonical_by_op_id[op_id] = canonical_type
        source_type_by_op_id[op_id] = node.get("op_type", "Unknown")
        operation_reason[op_id] = reason
        predecessors = sorted({
            producer_by_tensor[input_name]
            for input_name in node.get("inputs", [])
            if input_name in producer_by_tensor
        })
        successors = sorted({
            consumer
            for output in node.get("outputs", [])
            for consumer in consumers_by_tensor.get(output, [])
            if consumer != op_id
        })
        non_initializer_inputs = [
            input_name for input_name in node.get("inputs", [])
            if input_name and input_name not in initializers
        ]
        is_join = node.get("op_type") in {"Add", "Sum", "Concat", "Where"} and len(non_initializer_inputs) >= 2
        is_fork = any(len(set(consumers_by_tensor.get(output, []))) > 1 for output in node.get("outputs", []))
        operation_records.append(
            TensorOp(
                op_id=op_id,
                name=name,
                op_type=node.get("op_type", "Unknown"),
                canonical_op_type=canonical_type,
                inputs=inputs,
                outputs=outputs,
                attributes=dict(node.get("attributes", {})),
                predecessor_ops=predecessors,
                successor_ops=successors,
                is_fork=is_fork,
                is_join=is_join,
                region_hint=region_hint,
                source_frontend="onnx",
                source_node_name=name,
                source_location={"node_index": index},
                metadata={"canonicalization_reason": reason},
            )
        )

    values: list[TensorValue] = []
    for name in all_tensor_names:
        producer = producer_by_tensor.get(name)
        role = _base_role(name, name in initializers, name in graph_input_names)
        if producer:
            producer_type = source_type_by_op_id.get(producer)
            canonical_type = canonical_by_op_id.get(producer)
            if producer_type in {"Shape", "Size", "Range", "ConstantOfShape"}:
                role = "shape_tensor"
            elif canonical_type == "constant":
                role = "constant"
            elif role == "unknown":
                role = "activation"
        values.append(
            TensorValue(
                value_id=value_ids[name],
                name=name,
                producer=producer,
                consumers=sorted(set(consumers_by_tensor.get(name, []))),
                shape=_normalize_shape(shape_map.get(name)),
                dtype=dtype_map.get(name),
                is_initializer=name in initializers,
                is_graph_input=name in graph_input_names,
                is_graph_output=name in graph_output_names,
                semantic_role=role,
                metadata={"source_tensor_name": name},
            )
        )

    canonical_counts = Counter(op.canonical_op_type for op in operation_records)
    role_counts = Counter(value.semantic_role for value in values)
    hint_counts = Counter(op.region_hint for op in operation_records if op.region_hint)
    graph_name = model_config.get("name", onnx_summary.get("model_name", "model"))
    return TensorGraph(
        graph_id=f"tensor_graph::{_safe_token(graph_name)}",
        model_name=graph_name,
        source_frontend="onnx",
        ops=operation_records,
        values=values,
        graph_inputs=[value_ids[name] for name in graph_input_names if name in value_ids],
        graph_outputs=[value_ids[name] for name in graph_output_names if name in value_ids],
        initializers=[value_ids[name] for name in all_tensor_names if name in initializers],
        summary={
            "num_ops": len(operation_records),
            "num_values": len(values),
            "num_initializers": len(initializers),
            "num_graph_inputs": len(graph_input_names),
            "num_graph_outputs": len(graph_output_names),
            "canonical_op_type_counts": dict(canonical_counts),
            "semantic_role_counts": dict(role_counts),
            "num_fork_ops": sum(op.is_fork for op in operation_records),
            "num_join_ops": sum(op.is_join for op in operation_records),
            "region_hint_counts": dict(hint_counts),
        },
        metadata={
            "hf_id": model_config.get("hf_id", onnx_summary.get("hf_id")),
            "task": model_config.get("task", onnx_summary.get("task")),
            "source_onnx_path": onnx_summary.get("onnx_path"),
            "frontend_note": "Imported from ONNX summary; Tensor IR is frontend-independent.",
        },
    )
