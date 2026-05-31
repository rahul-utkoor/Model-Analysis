"""Dependence-style evidence shared by the Python extractor and optional native pass."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from experimental.mlir_axis_bridge.mlir_parser import MlirAccessRecord


@dataclass(frozen=True)
class NativeDependenceRelation:
    relation_id: str
    source_tensor: str
    source_indices: tuple[str, ...]
    target_tensor: str | None
    target_indices: tuple[str, ...]
    loop_ivs: tuple[str, ...]
    relation_kind: str
    dependence_kind: str
    affine_evidence: tuple[str, ...]
    proof: str
    confidence: str


@dataclass
class NativeDependenceReport:
    mlir_file: str
    analysis_tool: str
    dialects_seen: tuple[str, ...]
    relations: list[NativeDependenceRelation]
    reductions: list[str] = field(default_factory=list)
    preserved_axes: list[str] = field(default_factory=list)
    blocked_axes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _relation_from_dict(payload: dict[str, Any]) -> NativeDependenceRelation:
    return NativeDependenceRelation(
        relation_id=str(payload["relation_id"]),
        source_tensor=str(payload["source_tensor"]),
        source_indices=tuple(payload.get("source_indices", ())),
        target_tensor=payload.get("target_tensor"),
        target_indices=tuple(payload.get("target_indices", ())),
        loop_ivs=tuple(payload.get("loop_ivs", ())),
        relation_kind=str(payload.get("relation_kind", "unknown")),
        dependence_kind=str(payload.get("dependence_kind", "unknown")),
        affine_evidence=tuple(payload.get("affine_evidence", ())),
        proof=str(payload.get("proof", "")),
        confidence=str(payload.get("confidence", "low")),
    )


def native_dependence_report_from_dict(payload: dict[str, Any]) -> NativeDependenceReport:
    return NativeDependenceReport(
        mlir_file=str(payload.get("mlir_file", "")),
        analysis_tool=str(payload.get("analysis_tool", "imported_json")),
        dialects_seen=tuple(payload.get("dialects_seen", ())),
        relations=[_relation_from_dict(item) for item in payload.get("relations", ())],
        reductions=list(payload.get("reductions", ())),
        preserved_axes=list(payload.get("preserved_axes", ())),
        blocked_axes=list(payload.get("blocked_axes", ())),
        warnings=list(payload.get("warnings", ())),
    )


def load_native_dependence_report(path: str | Path) -> NativeDependenceReport:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"native MLIR dependence report does not exist: {source}")
    return native_dependence_report_from_dict(json.loads(source.read_text(encoding="utf-8")))


def native_dependence_report_to_dict(report: NativeDependenceReport) -> dict[str, Any]:
    return asdict(report)


def write_native_dependence_report(report: NativeDependenceReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(native_dependence_report_to_dict(report), indent=2) + "\n", encoding="utf-8")
    return output


def _relation(
    ordinal: int,
    read: MlirAccessRecord,
    write: MlirAccessRecord,
    relation_kind: str,
    dependence_kind: str,
    proof: str,
    *,
    confidence: str = "high",
    source_indices: tuple[str, ...] | None = None,
    target_indices: tuple[str, ...] | None = None,
) -> NativeDependenceRelation:
    return NativeDependenceRelation(
        f"relation_{ordinal:04d}",
        read.tensor,
        source_indices if source_indices is not None else read.indices,
        write.tensor,
        target_indices if target_indices is not None else write.indices,
        tuple(dict.fromkeys((*read.loop_ivs, *write.loop_ivs))),
        relation_kind,
        dependence_kind,
        (read.raw_line, write.raw_line),
        proof,
        confidence,
    )


def _relations_for_accesses(accesses: Iterable[MlirAccessRecord]) -> list[NativeDependenceRelation]:
    records = list(accesses)
    reads = [record for record in records if record.access_kind == "read"]
    writes = [record for record in records if record.access_kind == "write"]
    relations: list[NativeDependenceRelation] = []
    ordinal = 1
    for write in writes:
        nearby_reads = [read for read in reads if read.line_no < write.line_no and (not write.loop_ivs or set(read.loop_ivs) <= set(write.loop_ivs))]
        if not nearby_reads:
            nearby_reads = reads
        for read in nearby_reads:
            common = [iv for iv in read.indices if iv in write.indices]
            reduced = [iv for iv in read.indices if iv not in write.indices]
            if common:
                proof = f"free IVs {', '.join(common)} are shared by {read.tensor} and {write.tensor}"
                relations.append(
                    _relation(
                        ordinal,
                        read,
                        write,
                        "preserved",
                        "access_equivalence",
                        proof,
                        source_indices=tuple(common),
                        target_indices=tuple(common),
                    )
                )
                ordinal += 1
            if reduced:
                proof = f"{', '.join(reduced)} index {read.tensor} but not {write.tensor}; the axis is reduced by the enclosing contraction"
                relations.append(
                    _relation(
                        ordinal,
                        read,
                        write,
                        "reduced",
                        "reduction",
                        proof,
                        source_indices=tuple(reduced),
                        target_indices=(),
                    )
                )
                ordinal += 1
            if not common and not reduced:
                proof = "indexed-access relation is not proven by the conservative extractor"
                relations.append(_relation(ordinal, read, write, "unknown", "unknown", proof))
                ordinal += 1
        if len(nearby_reads) >= 2:
            reduced_sets = [set(read.indices) - set(write.indices) for read in nearby_reads]
            mixed = set.intersection(*reduced_sets) if reduced_sets else set()
            for iv in sorted(mixed):
                for read in nearby_reads:
                    if iv not in read.indices:
                        continue
                    proof = f"{iv} participates in multiple reads and disappears from {write.tensor}; contraction mixes channels"
                    relations.append(_relation(ordinal, read, write, "mixed", "reduction", proof, source_indices=(iv,), target_indices=()))
                    ordinal += 1
    return relations


def build_python_dependence_report(
    mlir_file: str,
    dialects_seen: Iterable[str],
    accesses: Iterable[MlirAccessRecord],
) -> NativeDependenceReport:
    relations = _relations_for_accesses(accesses)
    reductions = sorted({iv for relation in relations if relation.relation_kind in {"reduced", "mixed"} for iv in relation.source_indices if iv not in relation.target_indices})
    preserved = sorted({iv for relation in relations if relation.relation_kind == "preserved" for iv in relation.source_indices if iv in relation.target_indices})
    blocked = sorted({relation.proof for relation in relations if relation.relation_kind == "blocked"})
    warnings = [] if relations else ["no dependence relations were derived from affine/memref accesses"]
    return NativeDependenceReport(
        mlir_file,
        "python_affine_extractor",
        tuple(dialects_seen),
        relations,
        reductions,
        preserved,
        blocked,
        warnings,
    )
