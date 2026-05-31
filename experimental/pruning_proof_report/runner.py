"""Run selected ONNX subgraphs through the experimental cross-evidence stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experimental.mlir_axis_bridge.bridge_runner import MlirAxisBridgeResult, analyze_onnx_with_mlir_bridge
from experimental.mlir_axis_bridge.toolchain import find_native_pass_tool
from experimental.onnx_axis_bridge.bridge_runner import OnnxAxisBridgeResult, analyze_onnx_subgraph
from experimental.pruning_proof_report.proof_case import AxisRelationRecord, ProofCase, ProofEvidence


@dataclass(frozen=True)
class ProofRunOptions:
    use_mlir: bool = True
    run_native_pass: bool = True
    native_pass_tool: str | None = None
    onnx_mlir: str | None = None
    mlir_opt: str | None = None
    output_root: str = "reports/pruning_proof_report/artifacts"
    verbose: bool = False


EVIDENCE_PRIORITY = {
    "native_mlir_dependence_evidence": 4,
    "actual_loop_access_evidence": 3,
    "high_level_mlir_dialect_evidence": 2,
    "onnx_hint_fallback": 1,
    "unavailable": 0,
}


def _best_source(sources: list[str]) -> str:
    return max(sources or ["unavailable"], key=lambda source: EVIDENCE_PRIORITY.get(source, 0))


def _axis_relations(mlir: MlirAxisBridgeResult) -> list[AxisRelationRecord]:
    records: list[AxisRelationRecord] = []
    seen: set[tuple[str, str, str, str]] = set()
    for region in mlir.region_results:
        if region.axis_build.axis_summary is None:
            continue
        for operation in region.axis_build.axis_summary.op_summaries:
            for transfer in operation.transfers:
                source = f"{transfer.source_tensor}.{transfer.source_axis}"
                target = f"{transfer.target_tensor}.{transfer.target_axis}" if transfer.target_tensor and transfer.target_axis else "-"
                key = source, target, transfer.relation.value, transfer.proof
                if key not in seen:
                    records.append(AxisRelationRecord(source, target, transfer.relation.value, transfer.confidence, transfer.proof))
                    seen.add(key)
    return records


def _onnx_axis_relations(onnx: OnnxAxisBridgeResult) -> list[AxisRelationRecord]:
    records: list[AxisRelationRecord] = []
    seen: set[tuple[str, str, str, str]] = set()
    for region in onnx.lowered_regions:
        for operation in region.axis_summary.op_summaries:
            for transfer in operation.transfers:
                source = f"{transfer.source_tensor}.{transfer.source_axis}"
                target = f"{transfer.target_tensor}.{transfer.target_axis}" if transfer.target_tensor and transfer.target_axis else "-"
                key = source, target, transfer.relation.value, transfer.proof
                if key not in seen:
                    records.append(AxisRelationRecord(source, target, transfer.relation.value, transfer.confidence, transfer.proof))
                    seen.add(key)
    return records


def _dfa_summary(mlir: MlirAxisBridgeResult) -> dict[str, object]:
    dead: list[str] = []
    protected: list[str] = []
    blocked: list[str] = []
    fixed_point = True
    interpretations: list[str] = []
    ran = False
    for region in mlir.region_results:
        result = region.bridge_result
        if result is None:
            continue
        ran = True
        dead.extend(result.summary["dfa_final_dead_axes"])
        protected.extend(result.summary["dfa_protected_axes"])
        blocked.extend(result.summary["dfa_blocked_axes"])
        fixed_point = fixed_point and bool(result.summary["reached_fixed_point"])
        interpretations.append(str(result.summary["interpretation"]))
    return {
        "ran": ran,
        "final_dead_axes": sorted(set(dead)),
        "protected_axes": sorted(set(protected)),
        "blocked_axes": sorted(set(blocked)),
        "reached_fixed_point": fixed_point if ran else False,
        "interpretation": " ".join(dict.fromkeys(interpretations)),
    }


def _onnx_dfa_summary(onnx: OnnxAxisBridgeResult) -> dict[str, object]:
    dead: list[str] = []
    protected: list[str] = []
    blocked: list[str] = []
    fixed_point = True
    interpretations: list[str] = []
    ran = False
    for region in onnx.lowered_regions:
        result = region.bridge_result
        if result is None:
            continue
        ran = True
        dead.extend(result.summary["dfa_final_dead_axes"])
        protected.extend(result.summary["dfa_protected_axes"])
        blocked.extend(result.summary["dfa_blocked_axes"])
        fixed_point = fixed_point and bool(result.summary["reached_fixed_point"])
        interpretations.append(str(result.summary["interpretation"]))
    return {
        "ran": ran,
        "final_dead_axes": sorted(set(dead)),
        "protected_axes": sorted(set(protected)),
        "blocked_axes": sorted(set(blocked)),
        "reached_fixed_point": fixed_point if ran else False,
        "interpretation": " ".join(dict.fromkeys(interpretations)),
    }


def _verdict(case: ProofCase, source: str, patterns: list[str], relations: list[AxisRelationRecord], dfa: dict[str, object]) -> str:
    if case.expected_pattern == "QK_SCORE_BLOCKER" and "QK_SCORE_BLOCKER" in patterns:
        return "blocked"
    if case.expected_pattern == "ATTENTION_CONTEXT_LIKE" and any(
        relation.relation == "PRESERVED" and "value_dim" in relation.source and "value_context_dim" in relation.target
        for relation in relations
    ):
        return "partial" if not dfa["ran"] else "proven"
    if case.expected_pattern in patterns:
        if source in {"native_mlir_dependence_evidence", "actual_loop_access_evidence"}:
            return "proven"
        return "fallback_proven"
    if relations:
        return "partial"
    return "unknown"


def _onnx_summary(result: OnnxAxisBridgeResult) -> dict[str, object]:
    return {
        "num_nodes": result.graph_summary.num_nodes,
        "op_type_counts": result.graph_summary.op_type_counts,
        "graph_inputs": list(result.subgraph.graph_inputs),
        "graph_outputs": list(result.subgraph.graph_outputs),
        "pattern_hints": [hint.kind.value for hint in result.pattern_hints],
    }


def _missing(case: ProofCase) -> ProofEvidence:
    return ProofEvidence(
        case.case_id,
        case.model_name,
        case.layer_index,
        case.subgraph_name,
        case.onnx_path,
        False,
        verdict="unknown",
        limitations=[f"ONNX subgraph is missing: {case.onnx_path}"],
    )


def _onnx_fallback(case: ProofCase, onnx: OnnxAxisBridgeResult, limitations: list[str]) -> ProofEvidence:
    patterns = sorted({*onnx.summary["recognized_hints"], *onnx.summary["axis_patterns"]})
    relations = _onnx_axis_relations(onnx)
    dfa = _onnx_dfa_summary(onnx)
    source = "onnx_hint_fallback"
    return ProofEvidence(
        case.case_id,
        case.model_name,
        case.layer_index,
        case.subgraph_name,
        case.onnx_path,
        True,
        _onnx_summary(onnx),
        {"toolchain_available": False, "lowering_succeeded": False},
        source,
        relations,
        patterns,
        dfa,
        _verdict(case, source, patterns, relations, dfa),
        list(dict.fromkeys([*limitations, *onnx.warnings])),
    )


def run_proof_case(case: ProofCase, options: ProofRunOptions) -> ProofEvidence:
    """Collect best-effort evidence for one selected local subgraph."""
    if not Path(case.onnx_path).is_file():
        return _missing(case)
    limitations: list[str] = []
    try:
        onnx = analyze_onnx_subgraph(case.onnx_path)
    except Exception as exc:
        return ProofEvidence(case.case_id, case.model_name, case.layer_index, case.subgraph_name, case.onnx_path, True, verdict="failed", limitations=[f"ONNX axis bridge failed: {exc}"])
    if not options.use_mlir:
        return _onnx_fallback(case, onnx, ["MLIR analysis disabled by option"])
    native_tool = options.native_pass_tool
    if options.run_native_pass and native_tool is None:
        try:
            native_tool = str(find_native_pass_tool())
        except FileNotFoundError as exc:
            limitations.append(str(exc))
    case_root = Path(options.output_root) / case.case_id
    try:
        mlir = analyze_onnx_with_mlir_bridge(
            case.onnx_path,
            case_root / "mlir_artifacts",
            options.onnx_mlir,
            options.mlir_opt,
            run_native_pass=options.run_native_pass,
            native_pass_tool=native_tool,
            native_output_dir=case_root / "native",
        )
    except Exception as exc:
        return _onnx_fallback(case, onnx, [*limitations, f"MLIR axis bridge failed; used ONNX fallback: {exc}"])
    relations = _axis_relations(mlir)
    patterns = sorted({*onnx.summary["recognized_hints"], *mlir.summary["axis_patterns"]})
    dfa = _dfa_summary(mlir)
    source = _best_source(mlir.evidence_source)
    limitations.extend(onnx.warnings)
    limitations.extend(mlir.warnings)
    mlir_summary = {
        "toolchain_available": mlir.toolchain_status.onnx_mlir_available,
        "lowering_succeeded": all(command.returncode == 0 for command in mlir.lowering_result.commands),
        "generated_artifacts": [artifact.path for artifact in mlir.artifacts],
        "dialect_hints": mlir.summary["dialect_hints"],
        "native_pass_available": mlir.native_pass_result is not None and mlir.native_pass_result.returncode == 0,
        "native_pass_returncode": mlir.native_pass_result.returncode if mlir.native_pass_result else None,
        "native_json_path": mlir.native_pass_result.json_path if mlir.native_pass_result else None,
    }
    return ProofEvidence(
        case.case_id,
        case.model_name,
        case.layer_index,
        case.subgraph_name,
        case.onnx_path,
        True,
        _onnx_summary(onnx),
        mlir_summary,
        source,
        relations,
        patterns,
        dfa,
        _verdict(case, source, patterns, relations, dfa),
        list(dict.fromkeys(limitations)),
    )
