"""Build compact access summaries from emitted MLIR artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from experimental.mlir_axis_bridge.mlir_artifacts import MlirArtifact
from experimental.mlir_axis_bridge.mlir_parser import MlirAccessRecord, parse_mlir_artifact
from experimental.mlir_axis_bridge.native_dependence import NativeDependenceReport, build_python_dependence_report


@dataclass
class MlirAccessSummary:
    artifact_path: str
    stage: str
    dialect_hints: tuple[str, ...]
    operation_counts: dict[str, int]
    loop_kinds: tuple[str, ...]
    access_records: list[MlirAccessRecord]
    recognized_high_level_ops: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)
    dependence_report: NativeDependenceReport | None = None


def extract_mlir_access_summary(artifact: MlirArtifact) -> MlirAccessSummary:
    parsed = parse_mlir_artifact(artifact)
    op_counts = dict(sorted(Counter(op.op_name for op in parsed.operations).items()))
    loops = tuple(sorted(op for op in op_counts if op in {"affine.for", "scf.for", "krnl.iterate"}))
    high_level = tuple(sorted(op for op in op_counts if op.startswith(("onnx.", "linalg.", "krnl."))))
    warnings: list[str] = []
    if not parsed.accesses:
        warnings.append("no affine/memref load-store accesses were found; high-level dialect evidence or ONNX hints may be required")
    dependence_report = build_python_dependence_report(artifact.path, artifact.dialect_hints, parsed.accesses)
    return MlirAccessSummary(artifact.path, artifact.stage, artifact.dialect_hints, op_counts, loops, parsed.accesses, high_level, warnings, dependence_report)
