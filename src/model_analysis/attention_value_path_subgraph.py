"""Build seedable attention value-path ONNX evidence artifacts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.onnx_subgraph_extractor import extract_onnx_subgraph_model, get_node_name
from model_analysis.paths import ensure_dir, safe_model_name


LAYOUT_OPS = {"reshape", "transpose", "flatten", "squeeze", "unsqueeze"}
VALUE_SLICE_OPS = {"split", "slice", "gather"}
METADATA_OPS = {"constant", "shape", "gather", "unsqueeze", "squeeze", "concat", "slice", "cast"}


@dataclass
class AttentionValuePathSubgraph:
    model_name: str
    layer_index: int
    path_id: str
    path_name: str
    value_projection_ops: list[dict[str, Any]] = field(default_factory=list)
    fused_qkv_projection_ops: list[dict[str, Any]] = field(default_factory=list)
    value_slice_ops: list[dict[str, Any]] = field(default_factory=list)
    value_slice_status: str = "not_fused"
    value_slice_evidence: list[str] = field(default_factory=list)
    qkv_layout: str = "separate_qkv"
    value_layout_ops: list[dict[str, Any]] = field(default_factory=list)
    attention_context_ops: list[dict[str, Any]] = field(default_factory=list)
    context_layout_ops: list[dict[str, Any]] = field(default_factory=list)
    output_projection_ops: list[dict[str, Any]] = field(default_factory=list)
    residual_or_output_ops: list[dict[str, Any]] = field(default_factory=list)
    source_ops: list[dict[str, Any]] = field(default_factory=list)
    boundary_inputs: list[str] = field(default_factory=list)
    boundary_outputs: list[str] = field(default_factory=list)
    axis_mapping: dict[str, Any] = field(default_factory=dict)
    export_status: str = "skipped"
    artifact_paths: dict[str, str] = field(default_factory=dict)
    analysis_status: str = "unknown"
    explanation: str = ""


@dataclass
class AttentionValuePathReport:
    model_name: str
    generated_at: str
    total_layers: int
    total_paths: int
    exported: int
    skipped: int
    failed: int
    seedable: int
    partial: int
    blocked: int
    unknown: int
    paths: list[AttentionValuePathSubgraph] = field(default_factory=list)


def attention_value_path_to_dict(path: AttentionValuePathSubgraph | dict[str, Any]) -> dict[str, Any]:
    return asdict(path) if isinstance(path, AttentionValuePathSubgraph) else path


def attention_value_path_report_to_dict(report: AttentionValuePathReport | dict[str, Any]) -> dict[str, Any]:
    return asdict(report) if isinstance(report, AttentionValuePathReport) else report


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _summary(node: Any, index: int) -> dict[str, Any]:
    return {"source_name": get_node_name(node, index), "op_type": node.op_type, "topological_index": index}


def _pair_path(pair: dict[str, Any]) -> AttentionValuePathSubgraph:
    layer = int(pair.get("layer_index", 0))
    family = str(pair.get("family", "unknown"))
    path_id = f"{family}_layer_{layer}_attention_value_path"
    mapping_status = str(pair.get("mapping_status", "unproven"))
    status = "seedable" if pair.get("status") == "propagatable" and mapping_status == "proven" else "partial"
    producer = {"source_name": pair.get("producer_op_name", ""), "op_type": pair.get("producer_op_type", "")}
    consumer = {"source_name": pair.get("consumer_op_name", ""), "op_type": pair.get("consumer_op_type", "")}
    evidence = list(pair.get("evidence_ops", []))
    fused = any(token in str(producer.get("source_name", "")).lower() for token in ("c_attn", "qkv"))
    if fused:
        status = "blocked"
    return AttentionValuePathSubgraph(
        model_name=str(pair.get("model_name", "")),
        layer_index=layer,
        path_id=path_id,
        path_name=f"{family.upper()} Layer {layer} Attention Value Path",
        value_projection_ops=[producer],
        fused_qkv_projection_ops=[producer] if fused else [],
        value_slice_status="unsupported" if fused else "not_fused",
        value_slice_evidence=(
            ["The semantic anchor identifies a fused QKV producer but does not independently prove its value slice."]
            if fused else []
        ),
        qkv_layout="fused_qkv_unknown" if fused else "separate_qkv",
        value_layout_ops=[op for op in evidence if str(op.get("op_type", "")).lower() in LAYOUT_OPS],
        attention_context_ops=[
            op for op in evidence
            if str(op.get("op_type", "")).lower() == "matmul"
            and op.get("source_name") not in {producer["source_name"], consumer["source_name"]}
        ],
        output_projection_ops=[consumer],
        source_ops=evidence,
        axis_mapping={
            "value_projection_output_axis": "value_dim",
            "context_value_axis": "value_context_dim",
            "output_projection_input_axis": "value_context_dim",
            "mapping_status": "unproven" if fused else mapping_status,
            "evidence": [
                "V.value_dim -> Context.value_context_dim is preserved through attention context.",
                "Context.value_context_dim -> out_proj input value/context channel is index-aligned.",
            ],
        },
        analysis_status=status,
        explanation=(
            "Fused QKV projection does not expose a separately proven value slice."
            if fused
            else "The selected local path makes output-projection input deadness seedable backward through attention context to the value-projection output."
        ),
    )


def detect_attention_value_paths(
    model_name: str,
    deadbranch_report: dict[str, Any],
    source_model: Any | None = None,
) -> list[AttentionValuePathSubgraph]:
    """Create value-path records from static deadbranch semantic anchors."""
    pairs = [
        pair for pair in deadbranch_report.get("pairs", [])
        if pair.get("pair_kind") == "attention_value_deadness"
    ]
    paths = [_pair_path({**pair, "model_name": pair.get("model_name") or model_name}) for pair in pairs]
    if source_model is None:
        return paths
    existing_layers = {path.layer_index for path in paths}
    return [
        *paths,
        *[
            path
            for path in discover_source_attention_value_paths(model_name, source_model)
            if path.layer_index not in existing_layers
        ],
    ]


def _graph_maps(source_model: Any) -> tuple[dict[str, Any], dict[str, str], dict[str, list[str]], dict[str, int]]:
    nodes: dict[str, Any] = {}
    producer: dict[str, str] = {}
    consumers: dict[str, list[str]] = {}
    indices: dict[str, int] = {}
    for index, node in enumerate(source_model.graph.node):
        name = get_node_name(node, index)
        nodes[name] = node
        indices[name] = index
        for tensor in node.output:
            producer[tensor] = name
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(name)
    return nodes, producer, consumers, indices


def _find_path(nodes: dict[str, Any], consumers: dict[str, list[str]], start: str, end: str) -> list[str]:
    pending: list[tuple[str, list[str]]] = [(start, [start])]
    visited: set[str] = set()
    while pending:
        current, path = pending.pop(0)
        if current == end:
            return path
        if current in visited:
            continue
        visited.add(current)
        for tensor in nodes[current].output:
            for successor in consumers.get(tensor, []):
                if successor not in visited:
                    pending.append((successor, [*path, successor]))
    return []


def _find_path_to_input(
    nodes: dict[str, Any],
    consumers: dict[str, list[str]],
    start: str,
    end: str,
    target_tensor: str,
) -> list[str]:
    pending: list[tuple[str, list[str]]] = [(start, [start])]
    visited: set[str] = set()
    while pending:
        current, path = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for tensor in nodes[current].output:
            for successor in consumers.get(tensor, []):
                if successor == end:
                    if tensor == target_tensor:
                        return [*path, successor]
                    continue
                if successor not in visited:
                    pending.append((successor, [*path, successor]))
    return []


def _reverse_path(
    nodes: dict[str, Any],
    producer: dict[str, str],
    start: str,
    predicate: Any,
) -> list[str]:
    pending: list[tuple[str, list[str]]] = [(start, [start])]
    visited: set[str] = set()
    while pending:
        current, reverse_path = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if predicate(current, nodes[current]):
            return list(reversed(reverse_path))
        for tensor in nodes[current].input:
            parent = producer.get(tensor)
            if parent and parent not in visited:
                pending.append((parent, [*reverse_path, parent]))
    return []


def _layer_index(name: str) -> int | None:
    match = re.search(r"(?:layers|layer|h)[./](\d+)", name)
    return int(match.group(1)) if match else None


def _output_projection_candidate(model_name: str, name: str, node: Any) -> bool:
    lower = name.lower()
    if node.op_type.lower() not in {"matmul", "gemm"} or "/mlp/" in lower:
        return False
    if "gpt2" in model_name:
        return "/attn/c_proj/" in lower
    if "vit" in model_name:
        return "/attention/o_proj/" in lower or "/attention/output/" in lower
    return False


def _value_projection_candidate(model_name: str, name: str, node: Any) -> bool:
    lower = name.lower()
    if node.op_type.lower() not in {"matmul", "gemm"}:
        return False
    if "gpt2" in model_name:
        return "/attn/c_attn/" in lower
    if "vit" in model_name:
        return "/attention/v_proj/" in lower or "/attention/value/" in lower or "/attention/qkv/" in lower
    return False


def _slice_layout(path_names: list[str], nodes: dict[str, Any]) -> tuple[str, str, list[str]]:
    slice_types = [nodes[name].op_type.lower() for name in path_names if nodes[name].op_type.lower() in VALUE_SLICE_OPS]
    if "split" in slice_types:
        return "fused_qkv_split", "recovered", ["The context value operand traces backward through an explicit Split output to the fused QKV projection."]
    if any(op_type in {"slice", "gather"} for op_type in slice_types):
        return "fused_qkv_concat", "recovered", ["The context value operand traces backward through an explicit Slice/Gather value branch to the fused QKV projection."]
    return "fused_qkv_unknown", "unsupported", ["A fused QKV projection was found, but no independently recoverable value-slice operation reaches the context value operand."]


def discover_source_attention_value_paths(model_name: str, source_model: Any) -> list[AttentionValuePathSubgraph]:
    """Recover source-graph value paths when semantic deadbranch anchors are absent."""
    nodes, producer, consumers, indices = _graph_maps(source_model)
    recovered: list[AttentionValuePathSubgraph] = []
    for output_name, output_node in nodes.items():
        if not _output_projection_candidate(model_name, output_name, output_node):
            continue
        layer = _layer_index(output_name)
        if layer is None:
            continue
        output_parent = producer.get(output_node.input[0], "")
        context_path = _reverse_path(
            nodes,
            producer,
            output_parent,
            lambda name, node: node.op_type.lower() == "matmul" and name != output_name,
        ) if output_parent else []
        if not context_path:
            continue
        context_name = context_path[0]
        context_node = nodes[context_name]
        if len(context_node.input) < 2:
            continue
        value_parent = producer.get(context_node.input[-1], "")
        value_path = _reverse_path(
            nodes,
            producer,
            value_parent,
            lambda name, node: _value_projection_candidate(model_name, name, node),
        ) if value_parent else []
        if not value_path:
            continue
        value_name = value_path[0]
        fused = any(token in value_name.lower() for token in ("c_attn", "qkv"))
        qkv_layout, slice_status, slice_evidence = (
            _slice_layout(value_path, nodes) if fused else ("separate_qkv", "not_fused", [])
        )
        mapping_status = "proven" if not fused or slice_status == "recovered" else "unproven"
        analysis_status = "seedable" if mapping_status == "proven" else "blocked"
        family = "gpt2" if "gpt2" in model_name else "vit"
        recovered.append(
            AttentionValuePathSubgraph(
                model_name=model_name,
                layer_index=layer,
                path_id=f"{family}_layer_{layer}_attention_value_path",
                path_name=f"{family.upper()} Layer {layer} Attention Value Path",
                value_projection_ops=[_summary(nodes[value_name], indices[value_name])],
                fused_qkv_projection_ops=[_summary(nodes[value_name], indices[value_name])] if fused else [],
                value_slice_ops=[
                    _summary(nodes[name], indices[name])
                    for name in value_path
                    if nodes[name].op_type.lower() in VALUE_SLICE_OPS
                ],
                value_slice_status=slice_status,
                value_slice_evidence=slice_evidence,
                qkv_layout=qkv_layout,
                attention_context_ops=[_summary(context_node, indices[context_name])],
                output_projection_ops=[_summary(output_node, indices[output_name])],
                axis_mapping={
                    "fused_qkv_output_value_slice_axis": "value_slice_dim" if fused else None,
                    "value_projection_output_axis": "value_dim",
                    "value_slice_axis": "value_dim",
                    "context_value_axis": "value_context_dim",
                    "output_projection_input_axis": "value_context_dim",
                    "mapping_status": mapping_status,
                    "evidence": [
                        "The attention context value operand traces backward to the recovered value branch.",
                        "V.value_dim -> Context.value_context_dim is preserved through attention context.",
                        "Context.value_context_dim -> output projection input value/context channel is index-aligned.",
                        *slice_evidence,
                    ],
                },
                analysis_status=analysis_status,
                explanation=(
                    "The source ONNX graph proves a seedable fused-QKV value-slice path from output-projection input through context to the recovered value branch."
                    if fused and analysis_status == "seedable"
                    else "The source ONNX graph proves a seedable value-projection -> context -> output-projection path."
                    if analysis_status == "seedable"
                    else "The fused QKV value branch could not be recovered conservatively."
                ),
            )
        )
    return sorted(recovered, key=lambda path: path.layer_index)


def _metadata_dependencies(selected: set[str], nodes: dict[str, Any], producer: dict[str, str]) -> set[str]:
    pending = list(selected)
    while pending:
        current = pending.pop()
        for tensor in nodes[current].input:
            parent = producer.get(tensor)
            if parent and parent not in selected and nodes[parent].op_type.lower() in METADATA_OPS:
                selected.add(parent)
                pending.append(parent)
    return selected


def _boundaries(
    selected: set[str],
    nodes: dict[str, Any],
    producer: dict[str, str],
    consumers: dict[str, list[str]],
    indices: dict[str, int],
) -> tuple[list[str], list[str]]:
    inputs = []
    outputs = []
    initializer_like = set()
    for name in selected:
        node = nodes[name]
        if node.op_type.lower() == "constant":
            initializer_like.update(node.output)
        for tensor in node.input:
            if tensor and producer.get(tensor) not in selected and tensor not in initializer_like:
                inputs.append(tensor)
        for tensor in node.output:
            if tensor and any(consumer not in selected for consumer in consumers.get(tensor, [])):
                outputs.append(tensor)
    if not outputs:
        terminal = max(selected, key=lambda name: indices.get(name, -1))
        outputs.extend(nodes[terminal].output)
    return list(dict.fromkeys(inputs)), list(dict.fromkeys(outputs))


def bind_path_to_onnx(path: AttentionValuePathSubgraph, source_model: Any) -> AttentionValuePathSubgraph:
    """Resolve semantic anchors to a connected local ONNX path."""
    nodes, producer, consumers, indices = _graph_maps(source_model)
    value = path.value_projection_ops[0].get("source_name", "") if path.value_projection_ops else ""
    output = path.output_projection_ops[0].get("source_name", "") if path.output_projection_ops else ""
    context_candidates = [
        str(op.get("source_name", "")) for op in path.attention_context_ops
        if str(op.get("source_name", "")) in nodes
    ]
    if value not in nodes or output not in nodes or not context_candidates:
        path.analysis_status = "partial"
        path.explanation = "Value projection, attention context, or output projection anchor was not found in the source ONNX graph."
        return path
    context = context_candidates[-1]
    context_value_tensor = nodes[context].input[-1] if nodes[context].input else ""
    value_path = _find_path_to_input(nodes, consumers, value, context, context_value_tensor)
    output_path = _find_path(nodes, consumers, context, output)
    if not value_path or not output_path:
        path.analysis_status = "partial"
        path.explanation = "The source ONNX graph did not expose a connected value projection -> context -> output projection chain."
        return path
    selected = _metadata_dependencies(set([*value_path, *output_path]), nodes, producer)
    ordered = sorted(selected, key=indices.get)
    summaries = [_summary(nodes[name], indices[name]) for name in ordered]
    context_index = ordered.index(context)
    output_index = ordered.index(output)
    path.source_ops = summaries
    path.value_projection_ops = [summary for summary in summaries if summary["source_name"] == value]
    fused = any(token in value.lower() for token in ("c_attn", "qkv"))
    path.fused_qkv_projection_ops = list(path.value_projection_ops) if fused else []
    path.value_slice_ops = [summary for summary in summaries[:context_index] if summary["op_type"].lower() in VALUE_SLICE_OPS]
    if fused:
        path.qkv_layout, path.value_slice_status, path.value_slice_evidence = _slice_layout(ordered[:context_index], nodes)
        path.axis_mapping["fused_qkv_output_value_slice_axis"] = "value_slice_dim"
        path.axis_mapping["value_slice_axis"] = "value_dim"
        path.axis_mapping["mapping_status"] = "proven" if path.value_slice_status == "recovered" else "unproven"
        path.axis_mapping["evidence"] = [*path.axis_mapping.get("evidence", []), *path.value_slice_evidence]
    else:
        path.qkv_layout = "separate_qkv"
        path.value_slice_status = "not_fused"
    path.value_layout_ops = [summary for summary in summaries[:context_index] if summary["op_type"].lower() in LAYOUT_OPS]
    path.attention_context_ops = [summary for summary in summaries if summary["source_name"] == context]
    path.context_layout_ops = [summary for summary in summaries[context_index + 1 : output_index] if summary["op_type"].lower() in LAYOUT_OPS]
    path.output_projection_ops = [summary for summary in summaries if summary["source_name"] == output]
    path.boundary_inputs, path.boundary_outputs = _boundaries(selected, nodes, producer, consumers, indices)
    if path.axis_mapping.get("mapping_status") == "proven":
        path.analysis_status = "seedable"
        if fused:
            path.explanation = "The recovered fused-QKV value slice makes output-projection input deadness seedable backward through context to the value branch."
    return path


def _record(path: AttentionValuePathSubgraph, source_path: Path) -> dict[str, Any]:
    return {
        "subgraph_id": path.path_id,
        "subgraph_kind": "attention_value_path",
        "node_names": [op["source_name"] for op in path.source_ops],
        "op_types": [op["op_type"] for op in path.source_ops],
        "pattern": "value projection -> attention context -> output projection",
        "boundary_input_tensors": path.boundary_inputs,
        "boundary_output_tensors": path.boundary_outputs,
        "source_onnx_path": str(source_path),
        "reason": "Seedable attention value-path evidence artifact.",
    }


def _write_dot(path: AttentionValuePathSubgraph, artifact_dir: Path, render_svg: bool) -> None:
    lines = ["digraph attention_value_path {", "  rankdir=LR;"]
    for index, op in enumerate(path.source_ops):
        name = op["source_name"]
        lines.append(f'  "{name}" [label="{name.split("/")[-1]}\\n{op["op_type"]}"];')
        if index:
            lines.append(f'  "{path.source_ops[index - 1]["source_name"]}" -> "{name}";')
    lines.append("}")
    dot = artifact_dir / "subgraph.dot"
    dot.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if render_svg and shutil.which("dot"):
        subprocess.run(["dot", "-Tsvg", str(dot), "-o", str(dot.with_suffix(".svg"))], check=False)


def export_attention_value_path(path: AttentionValuePathSubgraph, source_model: Any, source_path: Path, artifact_root: Path, render_svg: bool = False) -> AttentionValuePathSubgraph:
    artifact_dir = ensure_dir(artifact_root / safe_model_name(path.model_name) / "layers" / f"layer_{path.layer_index}" / _slug(path.path_id))
    onnx_path = artifact_dir / "subgraph.onnx"
    export = extract_onnx_subgraph_model(source_model, _record(path, source_path), onnx_path, path.model_name)
    path.export_status = "exported" if export.status == "success" else "failed"
    path.boundary_inputs = export.boundary_input_tensors or path.boundary_inputs
    path.boundary_outputs = export.boundary_output_tensors or path.boundary_outputs
    path.artifact_paths = {
        "onnx": str(onnx_path),
        "dot": str(artifact_dir / "subgraph.dot"),
        "svg": str(artifact_dir / "subgraph.svg"),
    }
    if export.status == "success":
        _write_dot(path, artifact_dir, render_svg)
    else:
        path.explanation = f"{path.explanation} ONNX export failed: {export.reason}".strip()
    return path


def make_attention_value_path_report(model_name: str, paths: list[AttentionValuePathSubgraph]) -> AttentionValuePathReport:
    statuses = [path.analysis_status for path in paths]
    exports = [path.export_status for path in paths]
    return AttentionValuePathReport(
        model_name,
        datetime.now(timezone.utc).isoformat(),
        len({path.layer_index for path in paths}),
        len(paths),
        exports.count("exported"),
        exports.count("skipped"),
        exports.count("failed"),
        statuses.count("seedable"),
        statuses.count("partial"),
        statuses.count("blocked"),
        statuses.count("unknown"),
        paths,
    )


def write_attention_value_path_report(report: AttentionValuePathReport | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(attention_value_path_report_to_dict(report), indent=2) + "\n", encoding="utf-8")
