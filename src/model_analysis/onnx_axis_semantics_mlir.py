"""MLIR evidence collection for strict ONNX axis-semantics annotation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimental.mlir_axis_bridge.access_extractor import MlirAccessSummary, extract_mlir_access_summary
from experimental.mlir_axis_bridge.mlir_artifacts import MlirArtifact, artifact_from_path, discover_mlir_artifacts
from experimental.mlir_axis_bridge.native_dependence import native_dependence_report_to_dict, load_native_dependence_report
from experimental.mlir_axis_bridge.native_pass_runner import run_native_dependence_tool
from experimental.mlir_axis_bridge.onnx_mlir_runner import lower_onnx_subgraph_to_mlir
from experimental.mlir_axis_bridge.toolchain import find_native_pass_tool, find_onnx_mlir

from model_analysis.onnx_axis_semantics import (
    AxisRelation,
    BlockerKind,
    EvidenceTier,
    MlirEvidence,
    relations_from_native_report_dict,
    summarize_relations,
)


def collect_mlir_evidence_for_unit(
    onnx_path: str | Path,
    output_dir: str | Path,
    *,
    onnx_mlir_path: str | None = None,
    native_pass_tool: str | None = None,
    run_native_pass: bool = False,
    allow_no_mlir: bool = False,
    verbose: bool = False,
) -> tuple[MlirEvidence, list[AxisRelation], list[str]]:
    """Lower an evidence unit and recover axis relations from MLIR only."""
    warnings: list[str] = []
    source = Path(onnx_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    try:
        onnx_mlir = find_onnx_mlir(onnx_mlir_path)
    except FileNotFoundError as exc:
        warnings.append(str(exc))
        blocker = BlockerKind.MLIR_TOOLCHAIN_MISSING
        explanation = "ONNX-MLIR was not found; strict semantics are not inferred without MLIR."
        if allow_no_mlir:
            explanation += " --allow-no-mlir permits artifact emission with UNKNOWN semantics only."
        return MlirEvidence(False, False, [], None, None, [], {}, {}, blocker, explanation), [], warnings

    try:
        lowering = lower_onnx_subgraph_to_mlir(source, output, onnx_mlir, preserve_mlir=True)
    except Exception as exc:
        warnings.append(str(exc))
        return (
            MlirEvidence(
                available=False,
                lowering_succeeded=False,
                blocker_kind=BlockerKind.ONNX_MLIR_LOWERING_FAILED,
                blocker_explanation=f"ONNX-MLIR lowering raised {type(exc).__name__}: {exc}",
            ),
            [],
            warnings,
        )

    warnings.extend(lowering.warnings)
    failed_commands = [command for command in lowering.commands if command.returncode]
    artifacts = discover_mlir_artifacts(lowering)
    if not artifacts:
        blocker = BlockerKind.ONNX_MLIR_LOWERING_FAILED if failed_commands else BlockerKind.NO_MLIR_ARTIFACT
        return (
            MlirEvidence(
                available=False,
                lowering_succeeded=False,
                mlir_files=list(lowering.generated_files),
                blocker_kind=blocker,
                blocker_explanation="ONNX-MLIR did not emit an MLIR text artifact with usable evidence.",
            ),
            [],
            warnings,
        )

    summaries = [extract_mlir_access_summary(artifact) for artifact in artifacts]
    best_summary = _best_access_summary(summaries)
    python_report_dict = native_dependence_report_to_dict(best_summary.dependence_report) if best_summary.dependence_report else {}
    python_json = output / "python_dependence.json"
    python_json.write_text(json.dumps(python_report_dict, indent=2) + "\n", encoding="utf-8")

    native_json_path: Path | None = None
    native_report_dict: dict[str, Any] = {}
    if run_native_pass:
        try:
            native_tool = find_native_pass_tool(native_pass_tool)
            native_input = Path(best_summary.artifact_path)
            native_json_path = output / "native_dependence.json"
            native_run = run_native_dependence_tool(native_input, native_tool, native_json_path)
            if native_run.returncode:
                warnings.append(f"native dependence pass failed with exit code {native_run.returncode}: {native_run.stderr.strip()}")
                native_json_path = None
            elif native_json_path.is_file():
                native_report_dict = native_dependence_report_to_dict(load_native_dependence_report(native_json_path))
        except Exception as exc:
            warnings.append(f"native dependence pass unavailable: {exc}")

    relation_tier = EvidenceTier.NATIVE_MLIR_DEPENDENCE if native_report_dict.get("relations") else EvidenceTier.PYTHON_MLIR_ACCESS
    report_for_relations = native_report_dict if native_report_dict.get("relations") else python_report_dict
    relations = relations_from_native_report_dict(report_for_relations, relation_tier) if report_for_relations else []
    dialect_hints = sorted({hint for artifact in artifacts for hint in artifact.dialect_hints})
    relation_summary = summarize_relations(relations)
    access_summary = _access_summary_payload(summaries, best_summary)
    blocker = BlockerKind.NONE
    explanation = ""
    if not relations:
        if _has_high_level_mlir(summaries, dialect_hints):
            blocker = BlockerKind.HIGH_LEVEL_MLIR_ONLY
            explanation = "MLIR artifacts exist, but only high-level dialect evidence was recovered; no access/dependence relation is available."
        elif not access_summary.get("access_record_count", 0):
            blocker = BlockerKind.NO_AFFINE_OR_LOOP_ACCESS
            explanation = "No affine/scf/memref load-store access evidence was recovered from emitted MLIR."
        else:
            blocker = BlockerKind.NO_AXIS_RELATION_RECOVERED
            explanation = "MLIR access records exist, but no axis relation was recovered."

    evidence = MlirEvidence(
        available=True,
        lowering_succeeded=not bool(failed_commands),
        mlir_files=[artifact.path for artifact in artifacts],
        native_dependence_json=str(native_json_path) if native_json_path else None,
        python_dependence_json=str(python_json),
        dialect_hints=dialect_hints,
        access_summary=access_summary,
        relation_summary=relation_summary,
        blocker_kind=blocker,
        blocker_explanation=explanation,
    )
    if verbose:
        warnings.append(f"MLIR evidence: {relation_summary.get('total', 0)} relation(s), {access_summary.get('access_record_count', 0)} access record(s)")
    return evidence, relations, warnings


def evidence_from_mlir_text(path: str | Path) -> tuple[MlirEvidence, list[AxisRelation]]:
    """Test helper: derive strict evidence from a handwritten MLIR text file."""
    artifact = artifact_from_path(path, "synthetic")
    summary = extract_mlir_access_summary(artifact)
    report = native_dependence_report_to_dict(summary.dependence_report) if summary.dependence_report else {}
    relations = relations_from_native_report_dict(report, EvidenceTier.PYTHON_MLIR_ACCESS)
    evidence = MlirEvidence(
        available=True,
        lowering_succeeded=True,
        mlir_files=[str(path)],
        dialect_hints=list(artifact.dialect_hints),
        access_summary=_access_summary_payload([summary], summary),
        relation_summary=summarize_relations(relations),
        blocker_kind=BlockerKind.NONE if relations else BlockerKind.NO_AXIS_RELATION_RECOVERED,
        blocker_explanation="" if relations else "No axis relation was recovered from handwritten MLIR.",
    )
    return evidence, relations


def _best_access_summary(summaries: list[MlirAccessSummary]) -> MlirAccessSummary:
    return max(
        summaries,
        key=lambda item: (
            len(item.access_records),
            sum(item.operation_counts.get(op, 0) for op in ("affine.for", "scf.for", "affine.load", "affine.store", "memref.load", "memref.store")),
            -item.artifact_path.count("onnx"),
        ),
    )


def _access_summary_payload(summaries: list[MlirAccessSummary], best: MlirAccessSummary) -> dict[str, Any]:
    op_counts: dict[str, int] = {}
    for summary in summaries:
        for op_name, count in summary.operation_counts.items():
            op_counts[op_name] = op_counts.get(op_name, 0) + count
    return {
        "artifact_count": len(summaries),
        "best_artifact": best.artifact_path,
        "best_stage": best.stage,
        "access_record_count": len(best.access_records),
        "loop_kinds": list(best.loop_kinds),
        "recognized_high_level_ops": list(best.recognized_high_level_ops),
        "operation_counts": dict(sorted(op_counts.items())),
    }


def _has_high_level_mlir(summaries: list[MlirAccessSummary], dialect_hints: list[str]) -> bool:
    if any(summary.recognized_high_level_ops for summary in summaries):
        return True
    return any(hint in {"onnx.", "krnl.", "linalg."} for hint in dialect_hints)
