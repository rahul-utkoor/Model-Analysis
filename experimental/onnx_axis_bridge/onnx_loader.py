"""Read small ONNX subgraphs without executing or mutating them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OnnxTensorInfo:
    name: str
    shape: tuple[int | str | None, ...]
    is_initializer: bool


@dataclass(frozen=True)
class OnnxNodeInfo:
    node_id: str
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OnnxSubgraph:
    path: str
    graph_name: str
    nodes: tuple[OnnxNodeInfo, ...]
    tensors: dict[str, OnnxTensorInfo]
    graph_inputs: tuple[str, ...]
    graph_outputs: tuple[str, ...]
    initializers: tuple[str, ...]


def _require_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("The ONNX bridge requires the 'onnx' Python package. Use an environment with onnx installed.") from exc
    return onnx


def _shape_from_value_info(value_info: Any) -> tuple[int | str | None, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return ()
    shape: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return tuple(shape)


def _basic_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_basic_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _basic_value(value.tolist())
    return str(value)


def load_onnx_subgraph(path: str | Path) -> OnnxSubgraph:
    """Load ONNX graph metadata only. The model is never executed or rewritten."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX subgraph does not exist: {source}")
    onnx = _require_onnx()
    model = onnx.load(str(source), load_external_data=False)
    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    tensors: dict[str, OnnxTensorInfo] = {}
    for value_info in [*graph.input, *graph.value_info, *graph.output]:
        shape = _shape_from_value_info(value_info)
        existing = tensors.get(value_info.name)
        if existing is None or (not existing.shape and shape):
            tensors[value_info.name] = OnnxTensorInfo(value_info.name, shape, value_info.name in initializer_names)
    for initializer in graph.initializer:
        tensors[initializer.name] = OnnxTensorInfo(initializer.name, tuple(initializer.dims), True)
    nodes: list[OnnxNodeInfo] = []
    for index, node in enumerate(graph.node):
        attrs = {
            attribute.name: _basic_value(onnx.helper.get_attribute_value(attribute))
            for attribute in node.attribute
        }
        nodes.append(
            OnnxNodeInfo(
                node_id=f"node_{index:03d}",
                name=node.name or f"{node.op_type}_{index}",
                op_type=node.op_type,
                inputs=tuple(node.input),
                outputs=tuple(node.output),
                attrs=attrs,
            )
        )
        for tensor_name in [*node.input, *node.output]:
            if tensor_name and tensor_name not in tensors:
                tensors[tensor_name] = OnnxTensorInfo(tensor_name, (), tensor_name in initializer_names)
    return OnnxSubgraph(
        path=str(source),
        graph_name=graph.name or source.stem,
        nodes=tuple(nodes),
        tensors=tensors,
        graph_inputs=tuple(value.name for value in graph.input),
        graph_outputs=tuple(value.name for value in graph.output),
        initializers=tuple(initializer.name for initializer in graph.initializer),
    )
