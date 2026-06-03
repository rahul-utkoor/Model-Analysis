"""Exporter for strict MLIR-derived ONNX axis-semantics annotations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from model_analysis.onnx_axis_semantics import (
    BlockerKind,
    EvidenceTier,
    MlirEvidence,
    NodeAxisSemantics,
    derive_semantics_from_mlir_evidence,
)
from model_analysis.onnx_axis_semantics_mlir import collect_mlir_evidence_for_unit
from model_analysis.onnx_axis_semantics_subgraphs import create_evidence_units
from model_analysis.onnx_axis_semantics_text import render_svg_from_dot, write_annotated_dot


METADATA_FIELDS = (
    "axis_semantics.class",
    "axis_semantics.confidence",
    "axis_semantics.evidence_tier",
    "axis_semantics.reason",
    "axis_semantics.leader_candidate_kind",
    "axis_semantics.relations_json",
    "axis_semantics.mlir_evidence_json",
    "axis_semantics.blocker_kind",
    "axis_semantics.blocker_explanation",
)


def annotate_onnx_axis_semantics(
    *,
    input_path: str | Path,
    output_path: str | Path,
    sidecar_json: str | Path,
    dot_path: str | Path | None = None,
    svg_path: str | Path | None = None,
    mlir_output_dir: str | Path | None = None,
    onnx_mlir_path: str | None = None,
    mlir_opt_path: str | None = None,
    native_pass_tool: str | None = None,
    run_native_pass: bool = False,
    allow_no_mlir: bool = False,
    annotation_mode: str = "doc_string",
    fallback_doc_string: bool = False,
    check_onnx: bool = False,
    model_name: str | None = None,
    max_nodes: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    del mlir_opt_path  # reserved for compatibility with the MLIR bridge CLI surface
    input_onnx = Path(input_path)
    output_onnx = Path(output_path)
    sidecar = Path(sidecar_json)
    if annotation_mode not in {"attributes", "doc_string", "both"}:
        raise ValueError(f"unsupported annotation mode: {annotation_mode}")
    if not input_onnx.is_file():
        raise FileNotFoundError(f"input ONNX does not exist: {input_onnx}")

    try:
        import onnx  # type: ignore
        from onnx import helper  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when onnx is absent
        raise RuntimeError("onnx is required for annotation export") from exc

    original_bytes = input_onnx.read_bytes()
    model = onnx.load(str(input_onnx))
    graph_signature = _graph_signature(model)
    case_root = Path(mlir_output_dir) if mlir_output_dir else sidecar.with_suffix("").with_name(sidecar.stem + "_mlir")
    units, extraction_warnings = create_evidence_units(input_onnx, case_root / "evidence_units", max_nodes=max_nodes)
    unit = units[0]
    mlir_evidence, axis_relations, mlir_warnings = collect_mlir_evidence_for_unit(
        unit.unit_path,
        case_root,
        onnx_mlir_path=onnx_mlir_path,
        native_pass_tool=native_pass_tool,
        run_native_pass=run_native_pass,
        allow_no_mlir=allow_no_mlir,
        verbose=verbose,
    )

    nodes: list[NodeAxisSemantics] = []
    for index, node in enumerate(model.graph.node):
        node_id = node.name or f"{node.op_type}_{index}"
        if max_nodes is not None and index >= max_nodes:
            evidence = MlirEvidence(
                available=False,
                lowering_succeeded=False,
                blocker_kind=BlockerKind.NO_AXIS_RELATION_RECOVERED,
                blocker_explanation=f"max_nodes={max_nodes} skipped MLIR evidence for this node.",
            )
            node_semantics = derive_semantics_from_mlir_evidence(
                node_name=node_id,
                op_type=node.op_type,
                topological_index=index,
                input_names=list(node.input),
                output_names=list(node.output),
                mlir_evidence=evidence,
                axis_relations=[],
            )
        else:
            node_semantics = derive_semantics_from_mlir_evidence(
                node_name=node_id,
                op_type=node.op_type,
                topological_index=index,
                input_names=list(node.input),
                output_names=list(node.output),
                mlir_evidence=mlir_evidence,
                axis_relations=axis_relations,
            )
        _annotate_node(node, node_semantics, annotation_mode, helper)
        nodes.append(node_semantics)

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_onnx))
    checker = {"requested": check_onnx, "passed": None, "error": None, "doc_string_fallback_used": False}
    if check_onnx:
        try:
            onnx.checker.check_model(str(output_onnx))
            checker["passed"] = True
        except Exception as exc:
            checker["passed"] = False
            checker["error"] = str(exc)
            if fallback_doc_string and annotation_mode in {"attributes", "both"}:
                model = onnx.load(str(input_onnx))
                for index, node in enumerate(model.graph.node):
                    _annotate_node(node, nodes[index], "doc_string", helper)
                onnx.save(model, str(output_onnx))
                onnx.checker.check_model(str(output_onnx))
                checker["passed"] = True
                checker["doc_string_fallback_used"] = True

    warnings = extraction_warnings + mlir_warnings
    dot_warning: str | None = None
    if dot_path:
        write_annotated_dot(model, nodes, dot_path)
    if svg_path and dot_path:
        _, dot_warning = render_svg_from_dot(dot_path, svg_path)
        if dot_warning:
            warnings.append(dot_warning)

    output_model = onnx.load(str(output_onnx))
    sidecar_payload = _sidecar_payload(
        input_onnx=input_onnx,
        output_onnx=output_onnx,
        annotation_mode=annotation_mode,
        model_name=model_name,
        mlir_output_dir=case_root,
        nodes=nodes,
        warnings=warnings,
        checker=checker,
        original_graph_unchanged=(original_bytes == input_onnx.read_bytes() and graph_signature == _graph_signature(output_model)),
        evidence_unit_index=case_root / "evidence_units" / "evidence_unit_index.json",
        dot_path=Path(dot_path) if dot_path else None,
        svg_path=Path(svg_path) if svg_path and Path(svg_path).is_file() else None,
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(sidecar_payload, indent=2) + "\n", encoding="utf-8")
    return sidecar_payload


def _annotate_node(node: Any, semantics: NodeAxisSemantics, annotation_mode: str, helper: Any) -> None:
    relations_json = json.dumps([relation.to_dict() for relation in semantics.axis_relations], sort_keys=True)
    mlir_json = json.dumps(semantics.mlir_evidence.to_dict(), sort_keys=True)
    payload = {
        "axis_semantics.class": semantics.semantic_class.value,
        "axis_semantics.confidence": semantics.confidence,
        "axis_semantics.evidence_tier": semantics.evidence_tier.value,
        "axis_semantics.reason": semantics.reason,
        "axis_semantics.leader_candidate_kind": semantics.leader_candidate_kind,
        "axis_semantics.relations_json": relations_json,
        "axis_semantics.mlir_evidence_json": mlir_json,
        "axis_semantics.blocker_kind": semantics.mlir_evidence.blocker_kind.value,
        "axis_semantics.blocker_explanation": semantics.mlir_evidence.blocker_explanation,
    }
    semantics.attributes_added = list(payload)
    if annotation_mode in {"doc_string", "both"}:
        node.doc_string = "axis_semantics=" + json.dumps(payload, sort_keys=True)
    if annotation_mode in {"attributes", "both"}:
        for key, value in payload.items():
            node.attribute.append(helper.make_attribute(key, str(value)))


def _sidecar_payload(
    *,
    input_onnx: Path,
    output_onnx: Path,
    annotation_mode: str,
    model_name: str | None,
    mlir_output_dir: Path,
    nodes: list[NodeAxisSemantics],
    warnings: list[str],
    checker: dict[str, Any],
    original_graph_unchanged: bool,
    evidence_unit_index: Path,
    dot_path: Path | None,
    svg_path: Path | None,
) -> dict[str, Any]:
    semantic_counts = Counter(node.semantic_class.value for node in nodes)
    leader_counts = Counter(node.leader_candidate_kind for node in nodes)
    evidence_counts = Counter(node.evidence_tier.value for node in nodes)
    blocker_counts = Counter(node.mlir_evidence.blocker_kind.value for node in nodes)
    return {
        "input_onnx": str(input_onnx),
        "output_onnx": str(output_onnx),
        "annotation_mode": annotation_mode,
        "strict_mlir_semantics": True,
        "model_name": model_name,
        "mlir_output_dir": str(mlir_output_dir),
        "dot": str(dot_path) if dot_path else None,
        "svg": str(svg_path) if svg_path else None,
        "evidence_unit_index": str(evidence_unit_index),
        "node_count": len(nodes),
        "semantic_counts": dict(sorted(semantic_counts.items())),
        "leader_candidate_counts": dict(sorted(leader_counts.items())),
        "evidence_tier_counts": dict(sorted(evidence_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "nodes": [node.to_dict() for node in nodes],
        "checker": checker,
        "original_graph_unchanged": original_graph_unchanged,
        "warnings": warnings,
    }


def _graph_signature(model: Any) -> dict[str, Any]:
    return {
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
        "initializers": [value.name for value in model.graph.initializer],
        "node_inputs": [list(node.input) for node in model.graph.node],
        "node_outputs": [list(node.output) for node in model.graph.node],
    }
