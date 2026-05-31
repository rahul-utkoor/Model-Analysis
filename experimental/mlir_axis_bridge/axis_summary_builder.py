"""Translate extracted MLIR evidence into the existing loop/access RegionSpec."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.axis_relations import RegionAxisSummary
from experimental.axis_transfer_analysis.examples import (
    activation_example, attention_context_example, attention_value_path_example, ffn_example,
    layernorm_example, qk_score_example, residual_example,
)
from experimental.axis_transfer_analysis.loop_ir import RegionSpec
from experimental.axis_transfer_analysis.pattern_recognition import PatternMatch, recognize_patterns
from experimental.mlir_axis_bridge.access_extractor import MlirAccessSummary
from experimental.onnx_axis_bridge.pattern_hints import OnnxPatternHint, OnnxPatternHintKind


@dataclass
class MlirAxisBuildResult:
    region_spec: RegionSpec | None
    axis_summary: RegionAxisSummary | None
    pattern_matches: list[PatternMatch]
    evidence_source: str
    warnings: list[str] = field(default_factory=list)


TEMPLATES = {
    OnnxPatternHintKind.FFN_LIKE: ffn_example,
    OnnxPatternHintKind.ATTENTION_CONTEXT_LIKE: attention_context_example,
    OnnxPatternHintKind.QK_SCORE_LIKE: qk_score_example,
    OnnxPatternHintKind.ATTENTION_VALUE_PATH_LIKE: attention_value_path_example,
    OnnxPatternHintKind.RESIDUAL_LIKE: residual_example,
    OnnxPatternHintKind.LAYERNORM_LIKE: layernorm_example,
}


def _decorate(region: RegionSpec, source: str, summary: MlirAccessSummary, hint: OnnxPatternHint | None) -> RegionSpec:
    region = deepcopy(region)
    region.attrs.update(
        {
            "source": "mlir_axis_bridge",
            "evidence_source": source,
            "mlir_artifact": summary.artifact_path,
            "dialect_hints": list(summary.dialect_hints),
            "onnx_hint": hint.kind.value if hint else None,
        }
    )
    return region


def _actual_template(summary: MlirAccessSummary, hint: OnnxPatternHint | None) -> RegionSpec | None:
    accesses = summary.access_records
    reads = [record for record in accesses if record.access_kind == "read"]
    writes = [record for record in accesses if record.access_kind == "write"]
    if hint and hint.kind == OnnxPatternHintKind.QK_SCORE_LIKE and len(reads) >= 2 and writes:
        return qk_score_example().region
    if hint and hint.kind == OnnxPatternHintKind.ATTENTION_CONTEXT_LIKE and len(reads) >= 2 and writes:
        return attention_context_example().region
    for write in writes:
        candidates = [read for read in reads if len(read.indices) == len(write.indices) == 4]
        if len(candidates) < 2:
            continue
        common_reduced = set(candidates[0].indices) & set(candidates[1].indices) - set(write.indices)
        if "d" in common_reduced:
            return qk_score_example().region
        if write.indices[-1:] == ("d",) and "k" in common_reduced:
            return attention_context_example().region
    if len(reads) == 1 and len(writes) == 1 and reads[0].indices == writes[0].indices:
        return activation_example().region
    return None


def _result(region: RegionSpec, source: str, summary: MlirAccessSummary, hint: OnnxPatternHint | None, warnings: list[str]) -> MlirAxisBuildResult:
    decorated = _decorate(region, source, summary, hint)
    axis_summary = analyze_region(decorated)
    patterns = recognize_patterns(decorated, axis_summary)
    return MlirAxisBuildResult(decorated, axis_summary, patterns, source, warnings)


def build_axis_transfer_from_mlir(mlir_summary: MlirAccessSummary, onnx_hint: OnnxPatternHint | None = None) -> MlirAxisBuildResult:
    """Prefer actual accesses, then use explicit and honestly labeled fallback evidence."""
    warnings = list(mlir_summary.warnings)
    actual = _actual_template(mlir_summary, onnx_hint)
    if actual is not None:
        return _result(actual, "actual_loop_access_evidence", mlir_summary, onnx_hint, warnings)
    if onnx_hint and onnx_hint.kind in TEMPLATES:
        source = "high_level_mlir_dialect_evidence" if mlir_summary.recognized_high_level_ops else "onnx_hint_fallback"
        warnings.append(f"using {source}; full affine/scf access reconstruction was not proven")
        return _result(TEMPLATES[onnx_hint.kind]().region, source, mlir_summary, onnx_hint, warnings)
    warnings.append("no supported axis-transfer region could be constructed conservatively")
    return MlirAxisBuildResult(None, None, [], "unavailable", warnings)
