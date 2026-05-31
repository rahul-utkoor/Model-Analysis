"""Run local ONNX evidence through loop/access and DFA bridge prototypes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.axis_relations import RegionAxisSummary
from experimental.axis_transfer_analysis.loop_ir import RegionSpec
from experimental.axis_transfer_analysis.pattern_recognition import PatternKind, PatternMatch, recognize_patterns
from experimental.onnx_axis_bridge.onnx_graph_summary import OnnxGraphSummary, summarize_subgraph
from experimental.onnx_axis_bridge.onnx_loader import OnnxSubgraph, load_onnx_subgraph
from experimental.onnx_axis_bridge.onnx_to_loop_ir import lower_onnx_hint_to_region_spec
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHint, OnnxPatternHintKind, infer_pattern_hints
from experimental.pruning_analysis_bridge.axis_to_dfa import run_bridge_analysis
from experimental.pruning_analysis_bridge.bridge_ir import BridgeResult, BridgeSeedPolicy


@dataclass
class LoweredRegionResult:
    hint: OnnxPatternHint
    region_spec: RegionSpec
    axis_summary: RegionAxisSummary
    pattern_matches: list[PatternMatch]
    bridge_result: BridgeResult | None = None
    warning: str | None = None


@dataclass
class OnnxAxisBridgeResult:
    onnx_path: str
    subgraph: OnnxSubgraph
    graph_summary: OnnxGraphSummary
    pattern_hints: list[OnnxPatternHint]
    lowered_regions: list[LoweredRegionResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


REQUESTED_HINTS = {
    "ffn": OnnxPatternHintKind.FFN_LIKE,
    "attention-context": OnnxPatternHintKind.ATTENTION_CONTEXT_LIKE,
    "qk-score": OnnxPatternHintKind.QK_SCORE_LIKE,
    "attention-value-path": OnnxPatternHintKind.ATTENTION_VALUE_PATH_LIKE,
    "residual": OnnxPatternHintKind.RESIDUAL_LIKE,
    "layernorm": OnnxPatternHintKind.LAYERNORM_LIKE,
}


def _seed_policy(patterns: list[PatternMatch]) -> BridgeSeedPolicy | None:
    kinds = {pattern.pattern_kind for pattern in patterns}
    if PatternKind.FFN_INTERMEDIATE_CHAIN in kinds:
        return BridgeSeedPolicy("ffn_consumer_input_dead", "consumer intermediate input", "seed: ONNX-lowered consumer intermediate input channel j is exactly dead")
    if PatternKind.ATTENTION_VALUE_PATH in kinds:
        return BridgeSeedPolicy("attention_output_input_dead", "attention output-projection value-context input", "seed: ONNX-lowered output-projection value-context input channel j is exactly dead")
    if PatternKind.QK_SCORE_BLOCKER in kinds:
        return BridgeSeedPolicy("qk_query_output_pruned", "query projection output head_dim", "seed: ONNX-lowered query output channel j pruning attempt")
    if PatternKind.RESIDUAL_HIDDEN_PROTECTED in kinds:
        return BridgeSeedPolicy("residual_hidden_pruned", "residual hidden input", "seed: ONNX-lowered residual hidden channel j pruning attempt")
    if PatternKind.LAYERNORM_HIDDEN_PROTECTED in kinds:
        return BridgeSeedPolicy("layernorm_hidden_pruned", "normalized hidden input", "seed: ONNX-lowered normalized hidden channel j pruning attempt")
    return None


def _supported_hints(hints: list[OnnxPatternHint], requested_hint: str | None) -> list[OnnxPatternHint]:
    if requested_hint in {None, "auto"}:
        return [hint for hint in hints if hint.kind != OnnxPatternHintKind.UNKNOWN]
    try:
        requested = REQUESTED_HINTS[requested_hint]
    except KeyError as exc:
        raise ValueError(f"unknown requested hint: {requested_hint}") from exc
    return [hint for hint in hints if hint.kind == requested]


def analyze_onnx_subgraph(path: str | Path, requested_hint: str | None = None) -> OnnxAxisBridgeResult:
    """Analyze one local ONNX artifact without executing or mutating it."""
    subgraph = load_onnx_subgraph(path)
    graph_summary = summarize_subgraph(subgraph)
    hints = infer_pattern_hints(subgraph, graph_summary)
    warnings: list[str] = []
    selected_hints = _supported_hints(hints, requested_hint)
    if requested_hint not in {None, "auto"} and not selected_hints:
        warnings.append(f"requested hint '{requested_hint}' was not proven by local ONNX topology and shape evidence")
    lowered: list[LoweredRegionResult] = []
    for hint in selected_hints:
        try:
            region = lower_onnx_hint_to_region_spec(subgraph, hint)
            axis_summary = analyze_region(region)
            patterns = recognize_patterns(region, axis_summary)
            policy = _seed_policy(patterns)
            bridge_result = None
            warning = None
            if policy:
                bridge_result = run_bridge_analysis(
                    region,
                    policy,
                    example_name=f"onnx::{hint.kind.value.lower()}",
                    interpretation="Local ONNX evidence was lowered into loop/access relations and then into DFA propagation.",
                )
            else:
                warning = f"{hint.kind.value} lowered to axis-transfer evidence but has no standalone DFA propagation seed policy"
                warnings.append(warning)
            lowered.append(LoweredRegionResult(hint, region, axis_summary, patterns, bridge_result, warning))
        except Exception as exc:  # keep best-effort local analysis reportable
            warnings.append(f"{hint.kind.value} lowering failed: {exc}")
    axis_patterns = sorted({pattern.pattern_kind.value for item in lowered for pattern in item.pattern_matches})
    bridge_results = [item.bridge_result for item in lowered if item.bridge_result is not None]
    blocked_results = [
        result
        for result in bridge_results
        if result and result.summary["dfa_blocked_axes"]
    ]
    return OnnxAxisBridgeResult(
        onnx_path=subgraph.path,
        subgraph=subgraph,
        graph_summary=graph_summary,
        pattern_hints=hints,
        lowered_regions=lowered,
        warnings=warnings,
        summary={
            "num_nodes": graph_summary.num_nodes,
            "recognized_hints": [hint.kind.value for hint in hints if hint.kind != OnnxPatternHintKind.UNKNOWN],
            "lowered_regions": len(lowered),
            "axis_patterns": axis_patterns,
            "dfa_propagation_results": len(bridge_results),
            "blocked_results": len(blocked_results),
            "warnings": len(warnings),
        },
    )
