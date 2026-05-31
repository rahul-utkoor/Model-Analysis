"""Orchestrate local ONNX-MLIR evidence extraction through axis and DFA analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experimental.mlir_axis_bridge.access_extractor import MlirAccessSummary, extract_mlir_access_summary
from experimental.mlir_axis_bridge.axis_summary_builder import (
    MlirAxisBuildResult,
    build_axis_transfer_from_mlir,
    build_axis_transfer_from_native_dependence,
)
from experimental.mlir_axis_bridge.mlir_artifacts import MlirArtifact, discover_mlir_artifacts
from experimental.mlir_axis_bridge.native_dependence import (
    NativeDependenceReport,
    load_native_dependence_report,
    write_native_dependence_report,
)
from experimental.mlir_axis_bridge.native_pass_runner import NativePassRunResult, run_native_dependence_tool
from experimental.mlir_axis_bridge.onnx_mlir_runner import MlirLoweringResult, lower_onnx_subgraph_to_mlir
from experimental.mlir_axis_bridge.toolchain import ToolchainStatus, check_toolchain, find_native_pass_tool
from experimental.onnx_axis_bridge.bridge_runner import REQUESTED_HINTS, _seed_policy
from experimental.onnx_axis_bridge.onnx_graph_summary import summarize_subgraph
from experimental.onnx_axis_bridge.onnx_loader import load_onnx_subgraph
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHint, OnnxPatternHintKind, infer_pattern_hints
from experimental.pruning_analysis_bridge.axis_to_dfa import run_bridge_analysis
from experimental.pruning_analysis_bridge.bridge_ir import BridgeResult


@dataclass
class MlirRegionResult:
    hint: OnnxPatternHint
    mlir_summary: MlirAccessSummary
    axis_build: MlirAxisBuildResult
    bridge_result: BridgeResult | None = None
    warning: str | None = None


@dataclass
class MlirAxisBridgeResult:
    onnx_path: str
    toolchain_status: ToolchainStatus
    lowering_result: MlirLoweringResult
    artifacts: list[MlirArtifact]
    mlir_access_summaries: list[MlirAccessSummary]
    evidence_source: list[str]
    region_results: list[MlirRegionResult]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    native_dependence_report: NativeDependenceReport | None = None
    emitted_python_dependence_json: str | None = None
    native_pass_result: NativePassRunResult | None = None


def _selected_hints(hints: list[OnnxPatternHint], requested_hint: str) -> list[OnnxPatternHint]:
    if requested_hint == "auto":
        return [hint for hint in hints if hint.kind != OnnxPatternHintKind.UNKNOWN]
    try:
        kind = REQUESTED_HINTS[requested_hint]
    except KeyError as exc:
        raise ValueError(f"unknown requested hint: {requested_hint}") from exc
    return [hint for hint in hints if hint.kind == kind]


def _empty_summary() -> MlirAccessSummary:
    return MlirAccessSummary("<no-mlir-artifact>", "unavailable", (), {}, (), [], (), ["no MLIR text artifact was available"])


def _source_score(source: str) -> int:
    return {
        "native_mlir_dependence_evidence": 4,
        "actual_loop_access_evidence": 3,
        "high_level_mlir_dialect_evidence": 2,
        "onnx_hint_fallback": 1,
        "unavailable": 0,
    }.get(source, 0)


def _best_axis_build(summaries: list[MlirAccessSummary], hint: OnnxPatternHint) -> tuple[MlirAccessSummary, MlirAxisBuildResult]:
    candidates = [(summary, build_axis_transfer_from_mlir(summary, hint)) for summary in summaries or [_empty_summary()]]
    return max(candidates, key=lambda candidate: _source_score(candidate[1].evidence_source))


def _native_summary(report: NativeDependenceReport) -> MlirAccessSummary:
    return MlirAccessSummary(report.mlir_file, "native_dependence", report.dialects_seen, {}, (), [], (), list(report.warnings), report)


def _emit_python_dependence(summaries: list[MlirAccessSummary], path: str | Path | None, warnings: list[str]) -> str | None:
    if path is None:
        return None
    candidates = [summary for summary in summaries if summary.dependence_report is not None]
    if not candidates:
        warnings.append("python dependence JSON was requested but no MLIR access summary was available")
        return None
    selected = max(candidates, key=lambda summary: (len(summary.access_records), len(summary.dependence_report.relations)))
    return str(write_native_dependence_report(selected.dependence_report, path))


def _run_native_pass(
    summaries: list[MlirAccessSummary],
    tool_path: str | Path | None,
    output_dir: str | Path | None,
    warnings: list[str],
) -> tuple[NativePassRunResult | None, NativeDependenceReport | None]:
    candidates = [summary for summary in summaries if summary.access_records]
    if not candidates:
        warnings.append("native pass requested but no lowered MLIR artifact with indexed accesses was available")
        return None, None
    selected = max(candidates, key=lambda summary: len(summary.access_records))
    try:
        resolved_tool = find_native_pass_tool(str(tool_path) if tool_path else None)
    except FileNotFoundError as exc:
        warnings.append(f"native pass unavailable; falling back to Python extraction: {exc}")
        return NativePassRunResult((str(tool_path or "<auto>"), selected.artifact_path), 127, "", "", None, str(exc)), None
    native_root = Path(output_dir) if output_dir else Path(selected.artifact_path).parent / "native_dependence"
    output_json = native_root / f"{Path(selected.artifact_path).stem}.native_dependence.json"
    result = run_native_dependence_tool(selected.artifact_path, resolved_tool, output_json)
    if result.warning:
        warnings.append(f"{result.warning}; falling back to Python extraction")
        return result, None
    return result, load_native_dependence_report(output_json)


def analyze_onnx_with_mlir_bridge(
    onnx_path: str | Path,
    output_root: str | Path,
    onnx_mlir_path: str | None = None,
    mlir_opt_path: str | None = None,
    hint: str = "auto",
    native_dependence_json: str | Path | None = None,
    prefer_native_dependence: bool = False,
    emit_python_dependence_json: str | Path | None = None,
    run_native_pass: bool = False,
    native_pass_tool: str | Path | None = None,
    native_output_dir: str | Path | None = None,
) -> MlirAxisBridgeResult:
    """Use ONNX-MLIR as a read-only local evidence generator for one subgraph."""
    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX subgraph does not exist: {source}")
    toolchain = check_toolchain(onnx_mlir_path, mlir_opt_path)
    if not toolchain.onnx_mlir_available or not toolchain.onnx_mlir_path:
        raise FileNotFoundError("onnx-mlir is required for this bridge; " + "; ".join(toolchain.warnings))
    subgraph = load_onnx_subgraph(source)
    onnx_hints = infer_pattern_hints(subgraph, summarize_subgraph(subgraph))
    selected = _selected_hints(onnx_hints, hint)
    warnings = list(toolchain.warnings)
    if hint != "auto" and not selected:
        warnings.append(f"requested hint '{hint}' was not proven by local ONNX topology and shape evidence")
    lowering = lower_onnx_subgraph_to_mlir(source, Path(output_root), Path(toolchain.onnx_mlir_path))
    warnings.extend(lowering.warnings)
    artifacts = discover_mlir_artifacts(lowering)
    access_summaries = [extract_mlir_access_summary(artifact) for artifact in artifacts]
    native_report = load_native_dependence_report(native_dependence_json) if native_dependence_json else None
    emitted_python_json = _emit_python_dependence(access_summaries, emit_python_dependence_json, warnings)
    native_pass_result = None
    if run_native_pass and native_report is None:
        native_pass_result, native_report = _run_native_pass(access_summaries, native_pass_tool, native_output_dir, warnings)
    if prefer_native_dependence and native_report is None:
        warnings.append("--prefer-native-dependence was requested without --native-dependence-json; using Python extraction")
    region_results: list[MlirRegionResult] = []
    for selected_hint in selected:
        mlir_summary, axis_build = _best_axis_build(access_summaries, selected_hint)
        if native_report is not None:
            native_build = build_axis_transfer_from_native_dependence(native_report, selected_hint)
            if _source_score(native_build.evidence_source) >= _source_score(axis_build.evidence_source):
                mlir_summary, axis_build = _native_summary(native_report), native_build
            else:
                warnings.extend(native_build.warnings)
        warnings.extend(axis_build.warnings)
        bridge_result = None
        region_warning = None
        if axis_build.region_spec and axis_build.pattern_matches:
            policy = _seed_policy(axis_build.pattern_matches)
            if policy:
                bridge_result = run_bridge_analysis(
                    axis_build.region_spec,
                    policy,
                    example_name=f"mlir::{selected_hint.kind.value.lower()}",
                    interpretation="Selected ONNX-MLIR evidence was lowered into axis-transfer relations and DFA propagation.",
                )
            else:
                region_warning = f"{selected_hint.kind.value} produced axis evidence but has no standalone DFA seed policy"
                warnings.append(region_warning)
        else:
            region_warning = f"{selected_hint.kind.value} did not produce a complete axis-transfer pattern"
            warnings.append(region_warning)
        region_results.append(MlirRegionResult(selected_hint, mlir_summary, axis_build, bridge_result, region_warning))
    evidence_sources = sorted({item.axis_build.evidence_source for item in region_results})
    patterns = sorted({pattern.pattern_kind.value for item in region_results for pattern in item.axis_build.pattern_matches})
    dfa_results = [item.bridge_result for item in region_results if item.bridge_result]
    return MlirAxisBridgeResult(
        str(source),
        toolchain,
        lowering,
        artifacts,
        access_summaries,
        evidence_sources,
        region_results,
        {
            "num_artifacts": len(artifacts),
            "dialect_hints": sorted({dialect for artifact in artifacts for dialect in artifact.dialect_hints}),
            "evidence_source": evidence_sources,
            "recognized_hints": [item.kind.value for item in selected],
            "axis_patterns": patterns,
            "dfa_propagation_results": len(dfa_results),
            "blocked_results": sum(bool(result and result.summary["dfa_blocked_axes"]) for result in dfa_results),
            "warnings": len(warnings),
        },
        warnings,
        native_report,
        emitted_python_json,
        native_pass_result,
    )
