"""Run one MLIR evidence coverage case through the existing proof stack."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from experimental.mlir_axis_bridge.toolchain import find_native_pass_tool
from experimental.mlir_evidence_coverage.coverage_case import (
    CoverageCase,
    CoverageEvidenceTier,
    CoveragePatternKind,
    CoverageResult,
    CoverageVerdict,
)
from experimental.pruning_proof_report.proof_case import ProofCase, ProofEvidence
from experimental.pruning_proof_report.runner import ProofRunOptions, run_proof_case


@dataclass(frozen=True)
class CoverageRunOptions:
    output_root: str = "reports/mlir_evidence_coverage/artifacts"
    run_native_pass: bool = True
    native_pass_tool: str | None = None
    onnx_mlir: str | None = None
    mlir_opt: str | None = None
    verbose: bool = False
    keep_artifacts: bool = True


TIER_BY_SOURCE = {
    "native_mlir_dependence_evidence": CoverageEvidenceTier.NATIVE_MLIR_DEPENDENCE,
    "actual_loop_access_evidence": CoverageEvidenceTier.PYTHON_AFFINE_ACCESS,
    "high_level_mlir_dialect_evidence": CoverageEvidenceTier.HIGH_LEVEL_MLIR_DIALECT,
    "onnx_hint_fallback": CoverageEvidenceTier.ONNX_HINT_FALLBACK,
    "unavailable": CoverageEvidenceTier.UNAVAILABLE,
}


def _native_tool_available(options: CoverageRunOptions) -> bool:
    if not options.run_native_pass:
        return False
    try:
        find_native_pass_tool(options.native_pass_tool)
    except FileNotFoundError:
        return False
    return True


def _proof_case(case: CoverageCase) -> ProofCase:
    return ProofCase(
        case.case_id,
        case.model_name,
        case.layer_index,
        case.subgraph_name,
        case.onnx_path,
        case.expected_pattern,
        case.expected_result,
        case.notes,
    )


def _relation_proven(case: CoverageCase, proof: ProofEvidence) -> bool:
    if case.pattern_kind == CoveragePatternKind.ATTENTION_CONTEXT_VALUE_AXIS:
        return any(
            relation.relation == "PRESERVED"
            and "value_dim" in relation.source
            and "value_context_dim" in relation.target
            for relation in proof.axis_relations
        )
    return case.expected_pattern in proof.recognized_patterns


def _verdict(case: CoverageCase, proof: ProofEvidence, tier: CoverageEvidenceTier) -> CoverageVerdict:
    if not proof.found:
        return CoverageVerdict.MISSING
    if proof.verdict == "failed":
        return CoverageVerdict.FAILED
    if case.pattern_kind == CoveragePatternKind.ATTENTION_QK_SCORE and case.expected_pattern in proof.recognized_patterns:
        return CoverageVerdict.BLOCKED_AS_EXPECTED
    if case.pattern_kind == CoveragePatternKind.ATTENTION_CONTEXT_VALUE_AXIS and _relation_proven(case, proof):
        return CoverageVerdict.PARTIAL if not proof.dfa_summary.get("ran") else (
            CoverageVerdict.NATIVE_PROVEN if tier == CoverageEvidenceTier.NATIVE_MLIR_DEPENDENCE else CoverageVerdict.ACCESS_PROVEN
        )
    if not _relation_proven(case, proof):
        return CoverageVerdict.PARTIAL if proof.axis_relations else CoverageVerdict.UNKNOWN
    if tier == CoverageEvidenceTier.NATIVE_MLIR_DEPENDENCE:
        return CoverageVerdict.NATIVE_PROVEN
    if tier == CoverageEvidenceTier.PYTHON_AFFINE_ACCESS:
        return CoverageVerdict.ACCESS_PROVEN
    if tier in {CoverageEvidenceTier.HIGH_LEVEL_MLIR_DIALECT, CoverageEvidenceTier.ONNX_HINT_FALLBACK}:
        return CoverageVerdict.FALLBACK_PROVEN
    return CoverageVerdict.UNKNOWN


def run_coverage_case(case: CoverageCase, options: CoverageRunOptions) -> CoverageResult:
    """Evaluate one local ONNX artifact without executing or mutating the model."""
    native_available = _native_tool_available(options)
    if not Path(case.onnx_path).is_file():
        return CoverageResult(
            case,
            found=False,
            native_tool_available=native_available,
            verdict=CoverageVerdict.MISSING,
            warnings=[f"ONNX subgraph is missing: {case.onnx_path}"],
        )
    case_root = Path(options.output_root) / case.case_id
    if case_root.is_dir():
        shutil.rmtree(case_root)
    proof = run_proof_case(
        _proof_case(case),
        ProofRunOptions(
            use_mlir=True,
            run_native_pass=options.run_native_pass,
            native_pass_tool=options.native_pass_tool,
            onnx_mlir=options.onnx_mlir,
            mlir_opt=options.mlir_opt,
            output_root=options.output_root,
            verbose=options.verbose,
        ),
    )
    mlir = proof.mlir_summary
    dfa = proof.dfa_summary
    tier = TIER_BY_SOURCE.get(proof.evidence_source, CoverageEvidenceTier.UNAVAILABLE)
    result = CoverageResult(
        case,
        found=proof.found,
        onnx_lowered=bool(mlir.get("lowering_succeeded", False)),
        mlir_artifacts_count=len(mlir.get("generated_artifacts", [])),
        dialect_hints=list(mlir.get("dialect_hints", [])),
        native_tool_available=native_available,
        native_pass_ran=mlir.get("native_pass_returncode") is not None,
        native_pass_returncode=mlir.get("native_pass_returncode"),
        evidence_tier=tier,
        axis_relations=proof.axis_relations,
        recognized_patterns=proof.recognized_patterns,
        dfa_ran=bool(dfa.get("ran", False)),
        dfa_final_dead_axes=list(dfa.get("final_dead_axes", [])),
        dfa_blocked_axes=list(dfa.get("blocked_axes", [])),
        dfa_protected_axes=list(dfa.get("protected_axes", [])),
        verdict=_verdict(case, proof, tier),
        warnings=proof.limitations,
    )
    if not options.keep_artifacts and case_root.is_dir():
        shutil.rmtree(case_root)
    return result
