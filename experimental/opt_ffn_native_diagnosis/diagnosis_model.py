"""Data model for the OPT FFN native-evidence diagnosis report."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class BlockerKind(str, Enum):
    NONE = "none"
    NO_ONNX_ARTIFACT = "no_onnx_artifact"
    ONNX_MLIR_LOWERING_FAILED = "onnx_mlir_lowering_failed"
    NO_AFFINE_LOOPS = "no_affine_loops"
    NO_LOAD_STORE_ACCESSES = "no_load_store_accesses"
    ACTIVATION_DECOMPOSITION_UNLINKED = "activation_decomposition_unlinked"
    MATMUL_LOWERING_NOT_CANONICAL = "matmul_lowering_not_canonical"
    SHAPE_LAYOUT_NOISE = "shape_layout_noise"
    NATIVE_PASS_RELATION_GAP = "native_pass_relation_gap"
    AXIS_SUMMARY_BUILDER_GAP = "axis_summary_builder_gap"
    UNKNOWN = "unknown"


@dataclass
class OptFfnNativeDiagnosis:
    model_name: str
    layer_index: int
    onnx_path: str
    core_onnx_path: str | None = None
    lowering_succeeded: bool = False
    mlir_artifacts: list[str] = field(default_factory=list)
    dialect_hints: list[str] = field(default_factory=list)
    native_pass_ran: bool = False
    native_pass_returncode: int | None = None
    native_relations_count: int = 0
    preserved_relations: int = 0
    reduced_relations: int = 0
    mixed_relations: int = 0
    ffn_pattern_detected_by_native: bool = False
    ffn_pattern_detected_by_fallback: bool = False
    blocker_kind: BlockerKind = BlockerKind.UNKNOWN
    blocker_explanation: str = ""
    suggested_fix: str = ""
    fix_applied: bool = False
    report_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class OptFfnNativeDiagnosisReport:
    total_layers: int
    native_proven: int
    fallback_only: int
    failed: int
    blockers_by_kind: dict[str, int]
    layers: list[OptFfnNativeDiagnosis]
    final_recommendation: str
    generated_at: str

    @classmethod
    def create(cls, layers: list[OptFfnNativeDiagnosis]) -> "OptFfnNativeDiagnosisReport":
        blockers: dict[str, int] = {}
        for layer in layers:
            blockers[layer.blocker_kind.value] = blockers.get(layer.blocker_kind.value, 0) + 1
        native = sum(item.ffn_pattern_detected_by_native for item in layers)
        fallback = sum(item.ffn_pattern_detected_by_fallback and not item.ffn_pattern_detected_by_native for item in layers)
        failed = sum(not item.ffn_pattern_detected_by_native and not item.ffn_pattern_detected_by_fallback for item in layers)
        if native == len(layers):
            recommendation = (
                "Use the generated OPT FFN-core evidence artifacts for MLIR coverage. "
                "They remove unrelated LayerNorm/residual boundary noise while preserving fc1 -> activation -> fc2."
            )
        else:
            recommendation = "Retain fallback evidence for unresolved layers and inspect each recorded lowering or relation blocker."
        return cls(
            len(layers),
            native,
            fallback,
            failed,
            blockers,
            layers,
            recommendation,
            datetime.now(timezone.utc).isoformat(),
        )
