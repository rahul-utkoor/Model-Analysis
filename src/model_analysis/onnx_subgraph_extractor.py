"""Extract report-selected ONNX subgraphs as Netron visualization artifacts."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper

from model_analysis.paths import ensure_dir, get_project_root


@dataclass
class ExtractedOnnxSubgraph:
    export_id: str
    model_name: str
    source_onnx_path: str
    output_onnx_path: str
    subgraph_id: str
    subgraph_kind: str
    node_names: list[str]
    op_types: list[str]
    pattern: str
    graph_inputs: list[str]
    graph_outputs: list[str]
    initializers: list[str]
    boundary_input_tensors: list[str]
    boundary_output_tensors: list[str]
    internal_tensors: list[str]
    status: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgraphExportReport:
    model_name: str
    source_onnx_path: str
    output_root: str
    exports: list[ExtractedOnnxSubgraph] = field(default_factory=list)
    failed_exports: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def extracted_onnx_subgraph_to_dict(export: ExtractedOnnxSubgraph) -> dict[str, Any]:
    return asdict(export)


def subgraph_export_report_to_dict(report: SubgraphExportReport) -> dict[str, Any]:
    return asdict(report)


def write_subgraph_export_report_json(report: SubgraphExportReport, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(subgraph_export_report_to_dict(report), indent=2), encoding="utf-8")


def get_node_name(node: onnx.NodeProto, fallback_index: int) -> str:
    """Return the stable node name used by ``onnx_graph_analysis``."""
    return node.name or f"{node.op_type}_{fallback_index}"


def build_value_info_lookup(source_model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    """Collect available tensor type/shape metadata keyed by tensor name."""
    values = (
        list(source_model.graph.input)
        + list(source_model.graph.output)
        + list(source_model.graph.value_info)
    )
    return {value.name: copy.deepcopy(value) for value in values}


def make_fallback_value_info(name: str) -> onnx.ValueInfoProto:
    """Create a Netron-friendly unknown-shaped float tensor boundary."""
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, ["unknown_dim"])


def collect_required_initializers(
    source_model: onnx.ModelProto,
    selected_nodes: list[onnx.NodeProto],
) -> list[onnx.TensorProto]:
    """Collect initializers consumed by any selected node."""
    consumed = {input_name for node in selected_nodes for input_name in node.input if input_name}
    return [
        copy.deepcopy(initializer)
        for initializer in source_model.graph.initializer
        if initializer.name in consumed
    ]


def copy_model_metadata(
    source_model: onnx.ModelProto,
    target_model: onnx.ModelProto,
    extra_metadata: dict[str, Any],
) -> None:
    """Copy existing metadata and add extraction provenance values."""
    metadata = {item.key: item.value for item in source_model.metadata_props}
    metadata.update({key: str(value) for key, value in extra_metadata.items()})
    del target_model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = target_model.metadata_props.add()
        entry.key = key
        entry.value = value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_record(record: dict[str, Any], kind: str) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {}))
    for key in ("region_kind", "is_residual_like", "residual_confidence", "suggested_constraints"):
        if key in record:
            metadata[key] = record[key]
    return {
        "subgraph_id": record.get("subgraph_id") or record.get("region_id", ""),
        "subgraph_kind": kind,
        "node_names": list(record.get("node_names", [])),
        "op_types": list(record.get("op_types", [])),
        "pattern": record.get("pattern", ""),
        "boundary_input_tensors": list(record.get("boundary_input_tensors", [])),
        "boundary_output_tensors": list(record.get("boundary_output_tensors", [])),
        "internal_tensors": list(record.get("internal_tensors", [])),
        "initializer_tensors": list(record.get("initializer_tensors", [])),
        "pruning_class": record.get("pruning_class") or metadata.get("pruning_class"),
        "risk_level": record.get("risk_level") or metadata.get("risk_level"),
        "reason": record.get("reason") or metadata.get("classification_reason", ""),
        "metadata": metadata,
    }


def load_subgraph_records(
    model_name: str,
    safe_name: str,
    kinds: list[str],
) -> list[dict[str, Any]]:
    """Load and normalize report records for export selection."""
    del model_name
    root = get_project_root()
    requested = set(kinds)
    records: list[dict[str, Any]] = []
    subgraphs = _read_json(root / "reports" / "subgraphs" / f"{safe_name}.json")
    if "path" in requested and subgraphs:
        records.extend(_normalize_record(item, "path") for item in subgraphs.get("path_subgraphs", []))
    if "join" in requested:
        joins = _read_json(root / "reports" / "join_subgraphs" / f"{safe_name}.json")
        source = joins.get("join_subgraphs", []) if joins else (subgraphs or {}).get("join_subgraphs", [])
        records.extend(_normalize_record(item, "join") for item in source)
    if "dag_region" in requested:
        dag = _read_json(root / "reports" / "dag_regions" / f"{safe_name}.json")
        if dag:
            records.extend(_normalize_record(item, "dag_region") for item in dag.get("regions", []))
    return records


def select_subgraphs_for_export(
    records: list[dict],
    subgraph_ids: list[str] | None = None,
    kinds: list[str] | None = None,
    pattern_contains: str | None = None,
    pruning_class: str | None = None,
    risk_level: str | None = None,
    max_exports: int | None = None,
) -> list[dict]:
    """Filter normalized records using stable, deterministic selection rules."""
    id_set = set(subgraph_ids or [])
    kind_set = set(kinds or [])
    if id_set:
        selected = [record for record in records if record.get("subgraph_id") in id_set]
    else:
        selected = list(records)
        if kind_set:
            selected = [record for record in selected if record.get("subgraph_kind") in kind_set]
        if pattern_contains:
            needle = pattern_contains.casefold()
            selected = [record for record in selected if needle in record.get("pattern", "").casefold()]
        if pruning_class:
            selected = [record for record in selected if record.get("pruning_class") == pruning_class]
        if risk_level:
            selected = [record for record in selected if record.get("risk_level") == risk_level]
    selected.sort(key=lambda record: (record.get("subgraph_kind", ""), record.get("subgraph_id", "")))
    return selected[:max_exports] if max_exports is not None else selected


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _derive_boundaries(
    source_model: onnx.ModelProto,
    selected_nodes: list[onnx.NodeProto],
    record: dict[str, Any],
    selected_node_names: set[str],
) -> tuple[list[str], list[str], list[str]]:
    selected_output_list = _unique(
        [tensor for node in selected_nodes for tensor in node.output if tensor]
    )
    selected_outputs = set(selected_output_list)
    selected_inputs = _unique(
        [tensor for node in selected_nodes for tensor in node.input if tensor]
    )
    initializer_names = {initializer.name for initializer in source_model.graph.initializer}
    actual_boundary_inputs = [
        tensor
        for tensor in selected_inputs
        if tensor not in selected_outputs and tensor not in initializer_names
    ]
    declared_inputs = [
        tensor
        for tensor in record.get("boundary_input_tensors", [])
        if tensor not in selected_outputs and tensor not in initializer_names
    ]
    graph_output_names = {output.name for output in source_model.graph.output}
    consumers: dict[str, set[int]] = {}
    for index, node in enumerate(source_model.graph.node):
        for tensor in node.input:
            consumers.setdefault(tensor, set()).add(index)
    selected_indices = {
        index
        for index, node in enumerate(source_model.graph.node)
        if get_node_name(node, index) in selected_node_names
    }
    derived_outputs = [
        tensor
        for tensor in selected_output_list
        if tensor in graph_output_names or any(index not in selected_indices for index in consumers.get(tensor, set()))
    ]
    declared_outputs = [
        tensor for tensor in record.get("boundary_output_tensors", []) if tensor in selected_outputs
    ]
    output_tensors = _unique([*declared_outputs, *derived_outputs])
    if not output_tensors:
        output_tensors = list(selected_nodes[-1].output) if selected_nodes else []
    internal = [
        tensor
        for tensor in selected_output_list
        if tensor not in output_tensors and tensor not in actual_boundary_inputs
    ]
    return (_unique([*declared_inputs, *actual_boundary_inputs]), output_tensors, internal)


def _value_info_for(name: str, lookup: dict[str, onnx.ValueInfoProto]) -> onnx.ValueInfoProto:
    return copy.deepcopy(lookup[name]) if name in lookup else make_fallback_value_info(name)


def _failed_export(
    record: dict[str, Any],
    model_name: str,
    source_path: str,
    output_path: Path,
    reason: str,
) -> ExtractedOnnxSubgraph:
    return ExtractedOnnxSubgraph(
        export_id=f"export::{record.get('subgraph_kind', 'manual')}::{record.get('subgraph_id', 'unknown')}",
        model_name=model_name,
        source_onnx_path=source_path,
        output_onnx_path=str(output_path),
        subgraph_id=record.get("subgraph_id", ""),
        subgraph_kind=record.get("subgraph_kind", "manual"),
        node_names=list(record.get("node_names", [])),
        op_types=list(record.get("op_types", [])),
        pattern=record.get("pattern", ""),
        graph_inputs=[],
        graph_outputs=[],
        initializers=[],
        boundary_input_tensors=list(record.get("boundary_input_tensors", [])),
        boundary_output_tensors=list(record.get("boundary_output_tensors", [])),
        internal_tensors=list(record.get("internal_tensors", [])),
        status="failed",
        reason=reason,
        metadata={},
    )


def extract_onnx_subgraph_model(
    source_model: onnx.ModelProto,
    record: dict,
    output_path: Path,
    model_name: str,
    check_model: bool = True,
) -> ExtractedOnnxSubgraph:
    """Build and save a standalone ONNX visualization fragment for one record."""
    source_path = str(record.get("source_onnx_path", ""))
    source_nodes = {
        get_node_name(node, index): node for index, node in enumerate(source_model.graph.node)
    }
    requested = list(record.get("node_names", []))
    missing = [name for name in requested if name not in source_nodes]
    if missing:
        return _failed_export(
            record,
            model_name,
            source_path,
            output_path,
            f"Selected nodes not found in source ONNX graph: {missing}",
        )
    selected_nodes = [
        copy.deepcopy(node)
        for index, node in enumerate(source_model.graph.node)
        if get_node_name(node, index) in set(requested)
    ]
    if not selected_nodes:
        return _failed_export(record, model_name, source_path, output_path, "No selected ONNX nodes were resolved.")

    initializer_by_name = {initializer.name: initializer for initializer in source_model.graph.initializer}
    required = collect_required_initializers(source_model, selected_nodes)
    required_names = {initializer.name for initializer in required}
    for name in record.get("initializer_tensors", []):
        if name in initializer_by_name and name not in required_names:
            required.append(copy.deepcopy(initializer_by_name[name]))
            required_names.add(name)

    graph_inputs, graph_outputs, internal_tensors = _derive_boundaries(
        source_model, selected_nodes, record, set(requested)
    )
    value_info = build_value_info_lookup(source_model)
    input_values = [_value_info_for(name, value_info) for name in graph_inputs]
    output_values = [_value_info_for(name, value_info) for name in graph_outputs]
    value_info_values = [
        _value_info_for(name, value_info)
        for name in _unique([*record.get("internal_tensors", []), *internal_tensors])
        if name not in graph_inputs and name not in graph_outputs
    ]
    graph = helper.make_graph(
        selected_nodes,
        f"{model_name}_{record.get('subgraph_id', 'subgraph')}",
        input_values,
        output_values,
        initializer=required,
        value_info=value_info_values,
    )
    target_model = helper.make_model(
        graph,
        opset_imports=[copy.deepcopy(opset) for opset in source_model.opset_import],
        producer_name="model_analysis.onnx_subgraph_extractor",
    )
    target_model.ir_version = source_model.ir_version
    if source_model.domain:
        target_model.domain = source_model.domain
    if source_model.model_version:
        target_model.model_version = source_model.model_version
    target_model.functions.extend(copy.deepcopy(function) for function in source_model.functions)
    metadata = {
        "source_model": source_path or model_name,
        "subgraph_id": record.get("subgraph_id", ""),
        "subgraph_kind": record.get("subgraph_kind", "manual"),
        "pattern": record.get("pattern", ""),
        "extraction_reason": record.get("reason") or "Selected structural-analysis subgraph for Netron visualization.",
        "node_count": len(selected_nodes),
        "generated_by": "model_analysis.onnx_subgraph_extractor",
    }
    checker_status = "not_requested"
    checker_error = None
    if check_model:
        try:
            onnx.checker.check_model(target_model)
            checker_status = "passed"
        except Exception as exc:  # Checker warnings should not prevent Netron visualization output.
            checker_status = "warning"
            checker_error = str(exc)
    metadata["checker_status"] = checker_status
    if checker_error:
        metadata["checker_error"] = checker_error
    copy_model_metadata(source_model, target_model, metadata)
    try:
        ensure_dir(output_path.parent)
        onnx.save(target_model, output_path)
    except Exception as exc:
        return _failed_export(record, model_name, source_path, output_path, f"Failed to write ONNX file: {exc}")
    reason = "Extracted ONNX visualization artifact."
    if checker_status == "warning":
        reason = "Extracted ONNX visualization artifact with checker warning; inspect metadata."
    return ExtractedOnnxSubgraph(
        export_id=f"export::{record.get('subgraph_kind', 'manual')}::{record.get('subgraph_id', 'unknown')}",
        model_name=model_name,
        source_onnx_path=source_path,
        output_onnx_path=str(output_path),
        subgraph_id=record.get("subgraph_id", ""),
        subgraph_kind=record.get("subgraph_kind", "manual"),
        node_names=[get_node_name(node, index) for index, node in enumerate(source_model.graph.node) if get_node_name(node, index) in set(requested)],
        op_types=[node.op_type for node in selected_nodes],
        pattern=record.get("pattern", ""),
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        initializers=[initializer.name for initializer in required],
        boundary_input_tensors=graph_inputs,
        boundary_output_tensors=graph_outputs,
        internal_tensors=_unique([*record.get("internal_tensors", []), *internal_tensors]),
        status="success",
        reason=reason,
        metadata={"checker_status": checker_status, "checker_error": checker_error},
    )


def make_subgraph_export_report(
    model_name: str,
    source_onnx_path: Path,
    output_root: Path,
    results: list[ExtractedOnnxSubgraph],
    metadata: dict[str, Any] | None = None,
) -> SubgraphExportReport:
    successes = [item for item in results if item.status == "success"]
    failures = [extracted_onnx_subgraph_to_dict(item) for item in results if item.status != "success"]
    return SubgraphExportReport(
        model_name=model_name,
        source_onnx_path=str(source_onnx_path),
        output_root=str(output_root),
        exports=successes,
        failed_exports=failures,
        summary={
            "num_selected": len(results),
            "num_successful_exports": len(successes),
            "num_failed_exports": len(failures),
            "kind_counts": {
                kind: sum(item.subgraph_kind == kind for item in successes)
                for kind in sorted({item.subgraph_kind for item in successes})
            },
        },
        metadata=metadata or {},
    )


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def subgraph_export_report_to_markdown(report: SubgraphExportReport | dict) -> str:
    data = subgraph_export_report_to_dict(report) if isinstance(report, SubgraphExportReport) else report
    exports = []
    for item in data.get("exports", []):
        exports.append(
            {
                "export_id": item.get("export_id"),
                "subgraph_id": item.get("subgraph_id"),
                "kind": item.get("subgraph_kind"),
                "pattern": item.get("pattern"),
                "node_count": len(item.get("node_names", [])),
                "onnx_path": item.get("output_onnx_path"),
                "netron_command": f"netron {item.get('output_onnx_path')}",
            }
        )
    return "\n".join(
        [
            f"# ONNX Subgraph Export Report: {data.get('model_name', '')}",
            "",
            "## Summary",
            "",
            f"- Source ONNX path: `{data.get('source_onnx_path', '')}`",
            f"- Output root: `{data.get('output_root', '')}`",
            f"- Successful exports: `{data.get('summary', {}).get('num_successful_exports', 0)}`",
            f"- Failed exports: `{data.get('summary', {}).get('num_failed_exports', 0)}`",
            "",
            "## Exported Subgraphs",
            "",
            _markdown_table(exports, ["export_id", "subgraph_id", "kind", "pattern", "node_count", "onnx_path", "netron_command"]),
            "",
            "## Failed Exports",
            "",
            _markdown_table(data.get("failed_exports", []), ["subgraph_id", "subgraph_kind", "reason"]),
            "",
            "## Notes",
            "",
            "- These ONNX files are extracted visualization artifacts.",
            "- They are not executable model fragments in the semantic sense.",
            "- Boundary inputs and outputs are artificial graph boundaries introduced for inspection.",
            "- They are intended for Netron visual inspection and structural reasoning.",
            "",
        ]
    )


def netron_index_to_markdown(report: SubgraphExportReport | dict) -> str:
    data = subgraph_export_report_to_dict(report) if isinstance(report, SubgraphExportReport) else report
    lines = [f"# Netron Subgraph Index: {data.get('model_name', '')}", ""]
    for item in data.get("exports", []):
        lines.extend(
            [
                f"## {item.get('subgraph_id', '')}",
                "",
                f"- Kind: `{item.get('subgraph_kind', '')}`",
                f"- Pattern: `{item.get('pattern', '')}`",
                f"- File: `{item.get('output_onnx_path', '')}`",
                "",
                "```bash",
                f"netron {item.get('output_onnx_path', '')}",
                "```",
                "",
            ]
        )
    if not data.get("exports"):
        lines.extend(["_No successful exports._", ""])
    return "\n".join(lines)


def safe_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "subgraph"
