"""Validate dependency graphs using correspondence and shape evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any

from model_analysis.correspondence import CorrespondenceReport
from model_analysis.shape_evidence import ShapeEvidenceReport


def _corr_by_unit(report: CorrespondenceReport) -> dict[str, Any]:
    return {
        corr.torch_unit_id: corr
        for corr in report.module_node_correspondences
        if corr.torch_unit_id
    }


def _corr_by_module(report: CorrespondenceReport) -> dict[str, Any]:
    return {corr.torch_module_name: corr for corr in report.module_node_correspondences}


def _node_shape_by_name(report: ShapeEvidenceReport) -> dict[str, Any]:
    return {item.node_name: item for item in report.node_shapes}


def _compatible_shapes(src_corr: Any, dst_corr: Any) -> bool:
    src_shapes = [shape for shape in src_corr.output_shapes.values() if isinstance(shape, list) and shape]
    dst_shapes = [shape for shape in dst_corr.input_shapes.values() if isinstance(shape, list) and shape]
    if not src_shapes or not dst_shapes:
        src_shapes = [evidence.onnx_shape for evidence in src_corr.parameter_evidence if evidence.onnx_shape]
        dst_shapes = [evidence.onnx_shape for evidence in dst_corr.parameter_evidence if evidence.onnx_shape]
    for src_shape in src_shapes:
        for dst_shape in dst_shapes:
            if src_shape and dst_shape and (src_shape[-1] == dst_shape[-1] or src_shape[0] == dst_shape[0]):
                return True
    return False


def _confidence_for_corr(corr: Any) -> str:
    return corr.confidence if corr else "low"


def validate_dependency_graph_with_evidence(
    dependency_graph: dict,
    correspondence_report: CorrespondenceReport,
    shape_report: ShapeEvidenceReport,
) -> dict:
    corr_by_unit = _corr_by_unit(correspondence_report)
    corr_by_module = _corr_by_module(correspondence_report)
    node_shapes = _node_shape_by_name(shape_report)
    validated_units = []
    unvalidated_units = []
    validated_edges = []
    unvalidated_edges = []
    shape_supported_edges = []
    correspondence_supported_edges = []
    manual_review_items = []
    confidence_counts: Counter[str] = Counter()

    for unit in dependency_graph.get("prunable_units", []):
        corr = corr_by_unit.get(unit.get("unit_id")) or corr_by_module.get(unit.get("module_or_node_name")) or corr_by_module.get(unit.get("name"))
        if corr and corr.confidence in {"high", "medium"} and (corr.onnx_node_names or corr.onnx_initializer_names):
            entry = {"unit_id": unit.get("unit_id"), "confidence": corr.confidence, "reason": "Unit has medium/high correspondence evidence."}
            validated_units.append(entry)
            confidence_counts[corr.confidence] += 1
        else:
            unvalidated_units.append({"unit_id": unit.get("unit_id"), "reason": "No medium/high ONNX correspondence found.", "confidence": "low"})
            confidence_counts["low"] += 1

    for edge in dependency_graph.get("dependency_edges", []):
        src_corr = corr_by_unit.get(edge.get("src"))
        dst_corr = corr_by_unit.get(edge.get("dst"))
        corr_supported = bool(src_corr and dst_corr and src_corr.confidence in {"high", "medium"} and dst_corr.confidence in {"high", "medium"})
        shape_supported = bool(src_corr and dst_corr and _compatible_shapes(src_corr, dst_corr))
        reason = "Correspondence and shape evidence support this edge." if corr_supported and shape_supported else "Edge remains partially or fully unvalidated."

        if edge.get("edge_type") == "qkv_coupling" and corr_supported:
            shape_supported = shape_supported or _similar_projection_dims(src_corr, dst_corr)
        if edge.get("edge_type") == "mlp_hidden_coupling" and src_corr and dst_corr:
            shape_supported = shape_supported or _compatible_shapes(src_corr, dst_corr)
        if edge.get("edge_type") == "residual_coupling":
            add_nodes = [shape for shape in node_shapes.values() if shape.op_type == "Add" and shape.confidence in {"high", "medium"}]
            shape_supported = bool(add_nodes)
            manual_review_items.append({"edge": f"{edge.get('src')} -> {edge.get('dst')}", "reason": "Residual coupling remains manual review unless Add branch mapping is explicit.", "confidence": "medium"})
        if edge.get("edge_type") == "normalization_dependency" and dst_corr:
            shape_supported = shape_supported or any("LayerNormalization" in op for op in dst_corr.onnx_op_types)

        edge_entry = {
            "src": edge.get("src"),
            "dst": edge.get("dst"),
            "edge_type": edge.get("edge_type"),
            "confidence": "medium" if corr_supported or shape_supported else "low",
            "reason": reason,
        }
        if corr_supported:
            correspondence_supported_edges.append(edge_entry)
        if shape_supported:
            shape_supported_edges.append(edge_entry)
        if corr_supported or shape_supported:
            validated_edges.append(edge_entry)
            confidence_counts[edge_entry["confidence"]] += 1
        else:
            unvalidated_edges.append(edge_entry)
            manual_review_items.append({"edge": f"{edge.get('src')} -> {edge.get('dst')}", "reason": "No correspondence or shape support for dependency edge.", "confidence": "low"})
            confidence_counts["low"] += 1

    return {
        "model_name": dependency_graph.get("model_name"),
        "validated_units": validated_units,
        "validated_edges": validated_edges,
        "unvalidated_units": unvalidated_units,
        "unvalidated_edges": unvalidated_edges,
        "shape_supported_edges": shape_supported_edges,
        "correspondence_supported_edges": correspondence_supported_edges,
        "manual_review_items": manual_review_items,
        "summary": {
            "num_units": len(dependency_graph.get("prunable_units", [])),
            "num_edges": len(dependency_graph.get("dependency_edges", [])),
            "num_validated_units": len(validated_units),
            "num_validated_edges": len(validated_edges),
            "num_shape_supported_edges": len(shape_supported_edges),
            "num_correspondence_supported_edges": len(correspondence_supported_edges),
            "num_manual_review_items": len(manual_review_items),
            "validation_confidence_counts": dict(confidence_counts),
        },
        "metadata": {"source": "correspondence_shape_validation"},
    }


def _similar_projection_dims(src_corr: Any, dst_corr: Any) -> bool:
    src_shapes = [evidence.onnx_shape for evidence in src_corr.parameter_evidence if evidence.onnx_shape]
    dst_shapes = [evidence.onnx_shape for evidence in dst_corr.parameter_evidence if evidence.onnx_shape]
    return any(src == dst for src in src_shapes for dst in dst_shapes)


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


def dependency_validation_to_markdown(validation: dict) -> str:
    summary = validation.get("summary", {})
    lines = [
        f"# Validated Dependency Graph: {validation.get('model_name')}",
        "",
        "## Summary",
        "",
        f"- Units: `{summary.get('num_units', 0)}`",
        f"- Edges: `{summary.get('num_edges', 0)}`",
        f"- Validated units: `{summary.get('num_validated_units', 0)}`",
        f"- Validated edges: `{summary.get('num_validated_edges', 0)}`",
        f"- Shape-supported edges: `{summary.get('num_shape_supported_edges', 0)}`",
        f"- Correspondence-supported edges: `{summary.get('num_correspondence_supported_edges', 0)}`",
        f"- Manual review items: `{summary.get('num_manual_review_items', 0)}`",
        "",
        "## Validated Units",
        "",
        _markdown_table(validation.get("validated_units", []), ["unit_id", "confidence", "reason"], limit=250),
        "",
        "## Validated Edges",
        "",
        _markdown_table(validation.get("validated_edges", []), ["src", "dst", "edge_type", "confidence", "reason"], limit=250),
        "",
        "## Shape-Supported Edges",
        "",
        _markdown_table(validation.get("shape_supported_edges", []), ["src", "dst", "edge_type", "confidence", "reason"], limit=250),
        "",
        "## Correspondence-Supported Edges",
        "",
        _markdown_table(validation.get("correspondence_supported_edges", []), ["src", "dst", "edge_type", "confidence", "reason"], limit=250),
        "",
        "## Unvalidated Units and Edges",
        "",
        _markdown_table(validation.get("unvalidated_units", []), ["unit_id", "confidence", "reason"], limit=150),
        "",
        _markdown_table(validation.get("unvalidated_edges", []), ["src", "dst", "edge_type", "confidence", "reason"], limit=150),
        "",
        "## Manual Review Items",
        "",
        _markdown_table(validation.get("manual_review_items", []), ["edge", "reason", "confidence"], limit=250),
        "",
        "## Interpretation",
        "",
        "Validated dependency evidence increases confidence in the dry-run graph, but it is still static evidence and not an executable pruning transform.",
        "",
    ]
    return "\n".join(lines)
