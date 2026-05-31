"""Orchestrate local ONNX-MLIR evidence extraction through axis and DFA analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experimental.mlir_axis_bridge.access_extractor import MlirAccessSummary, extract_mlir_access_summary
from experimental.mlir_axis_bridge.axis_summary_builder import MlirAxisBuildResult, build_axis_transfer_from_mlir
from experimental.mlir_axis_bridge.mlir_artifacts import MlirArtifact, discover_mlir_artifacts
from experimental.mlir_axis_bridge.onnx_mlir_runner import MlirLoweringResult, lower_onnx_subgraph_to_mlir
from experimental.mlir_axis_bridge.toolchain import ToolchainStatus, check_toolchain
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
        "actual_loop_access_evidence": 3,
        "high_level_mlir_dialect_evidence": 2,
        "onnx_hint_fallback": 1,
        "unavailable": 0,
    }.get(source, 0)


def _best_axis_build(summaries: list[MlirAccessSummary], hint: OnnxPatternHint) -> tuple[MlirAccessSummary, MlirAxisBuildResult]:
    candidates = [(summary, build_axis_transfer_from_mlir(summary, hint)) for summary in summaries or [_empty_summary()]]
    return max(candidates, key=lambda candidate: _source_score(candidate[1].evidence_source))


def analyze_onnx_with_mlir_bridge(
    onnx_path: str | Path,
    output_root: str | Path,
    onnx_mlir_path: str | None = None,
    mlir_opt_path: str | None = None,
    hint: str = "auto",
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
    region_results: list[MlirRegionResult] = []
    for selected_hint in selected:
        mlir_summary, axis_build = _best_axis_build(access_summaries, selected_hint)
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
    )
