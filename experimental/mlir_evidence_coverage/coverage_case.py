"""Data records for the MLIR evidence coverage study."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from experimental.pruning_proof_report.proof_case import AxisRelationRecord


class CoveragePatternKind(str, Enum):
    FFN_MLP_INTERMEDIATE = "FFN_MLP_INTERMEDIATE"
    ATTENTION_QK_SCORE = "ATTENTION_QK_SCORE"
    ATTENTION_CONTEXT_VALUE_AXIS = "ATTENTION_CONTEXT_VALUE_AXIS"
    ATTENTION_VALUE_PATH = "ATTENTION_VALUE_PATH"
    RESIDUAL_HIDDEN_PROTECTED = "RESIDUAL_HIDDEN_PROTECTED"
    LAYERNORM_HIDDEN_PROTECTED = "LAYERNORM_HIDDEN_PROTECTED"
    UNKNOWN = "UNKNOWN"


class CoverageEvidenceTier(str, Enum):
    NATIVE_MLIR_DEPENDENCE = "native_mlir_dependence_evidence"
    PYTHON_AFFINE_ACCESS = "actual_loop_access_evidence"
    HIGH_LEVEL_MLIR_DIALECT = "high_level_mlir_dialect_evidence"
    ONNX_HINT_FALLBACK = "onnx_hint_fallback"
    UNAVAILABLE = "unavailable"


class CoverageVerdict(str, Enum):
    NATIVE_PROVEN = "native_proven"
    ACCESS_PROVEN = "access_proven"
    FALLBACK_PROVEN = "fallback_proven"
    BLOCKED_AS_EXPECTED = "blocked_as_expected"
    PARTIAL = "partial"
    MISSING = "missing"
    UNKNOWN = "unknown"
    FAILED = "failed"


@dataclass(frozen=True)
class CoverageCase:
    case_id: str
    model_name: str
    layer_index: int
    pattern_kind: CoveragePatternKind
    subgraph_name: str
    onnx_path: str
    expected_pattern: str
    expected_result: str
    required_for_model: bool
    notes: str = ""


@dataclass
class CoverageResult:
    case: CoverageCase
    found: bool = False
    onnx_lowered: bool = False
    mlir_artifacts_count: int = 0
    dialect_hints: list[str] = field(default_factory=list)
    native_tool_available: bool = False
    native_pass_ran: bool = False
    native_pass_returncode: int | None = None
    evidence_tier: CoverageEvidenceTier = CoverageEvidenceTier.UNAVAILABLE
    axis_relations: list[AxisRelationRecord] = field(default_factory=list)
    recognized_patterns: list[str] = field(default_factory=list)
    dfa_ran: bool = False
    dfa_final_dead_axes: list[str] = field(default_factory=list)
    dfa_blocked_axes: list[str] = field(default_factory=list)
    dfa_protected_axes: list[str] = field(default_factory=list)
    verdict: CoverageVerdict = CoverageVerdict.UNKNOWN
    warnings: list[str] = field(default_factory=list)
    report_path: str | None = None
