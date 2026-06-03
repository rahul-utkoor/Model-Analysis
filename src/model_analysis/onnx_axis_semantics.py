"""Strict MLIR-derived ONNX axis-semantics data model.

The classes in this module deliberately separate display metadata from
semantic claims. ONNX node names and op types may be recorded, but semantic
classes are only produced from MLIR access/dependence evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AxisSemanticClass(str, Enum):
    UNKNOWN = "UNKNOWN"
    MLIR_DERIVED_INDEX_PRESERVING = "MLIR_DERIVED_INDEX_PRESERVING"
    MLIR_DERIVED_ELEMENTWISE_PRESERVE = "MLIR_DERIVED_ELEMENTWISE_PRESERVE"
    MLIR_DERIVED_AXIS_PERMUTE = "MLIR_DERIVED_AXIS_PERMUTE"
    MLIR_DERIVED_AXIS_RESHAPE = "MLIR_DERIVED_AXIS_RESHAPE"
    MLIR_DERIVED_AXIS_BROADCAST = "MLIR_DERIVED_AXIS_BROADCAST"
    MLIR_DERIVED_AXIS_SPLIT = "MLIR_DERIVED_AXIS_SPLIT"
    MLIR_DERIVED_AXIS_GATHER = "MLIR_DERIVED_AXIS_GATHER"
    MLIR_DERIVED_PROJECTION = "MLIR_DERIVED_PROJECTION"
    MLIR_DERIVED_PROJECTION_EXPAND = "MLIR_DERIVED_PROJECTION_EXPAND"
    MLIR_DERIVED_PROJECTION_CONTRACT = "MLIR_DERIVED_PROJECTION_CONTRACT"
    MLIR_DERIVED_MATMUL_GENERIC = "MLIR_DERIVED_MATMUL_GENERIC"
    MLIR_DERIVED_MATMUL_ATTENTION_CONTEXT = "MLIR_DERIVED_MATMUL_ATTENTION_CONTEXT"
    MLIR_DERIVED_MATMUL_QK_SCORE = "MLIR_DERIVED_MATMUL_QK_SCORE"
    MLIR_DERIVED_REDUCTION = "MLIR_DERIVED_REDUCTION"
    MLIR_DERIVED_BRANCH_MERGE = "MLIR_DERIVED_BRANCH_MERGE"
    MLIR_DERIVED_NORMALIZATION = "MLIR_DERIVED_NORMALIZATION"
    MLIR_DERIVED_PROTECTED_INTERFACE = "MLIR_DERIVED_PROTECTED_INTERFACE"
    MLIR_DERIVED_BLOCKER = "MLIR_DERIVED_BLOCKER"
    MLIR_HIGH_LEVEL_INSUFFICIENT = "MLIR_HIGH_LEVEL_INSUFFICIENT"
    MLIR_LOWERING_FAILED = "MLIR_LOWERING_FAILED"
    NO_ACCESS_EVIDENCE = "NO_ACCESS_EVIDENCE"


class AxisRelationKind(str, Enum):
    UNKNOWN = "UNKNOWN"
    PRESERVED = "PRESERVED"
    PERMUTED = "PERMUTED"
    RESHAPED = "RESHAPED"
    BROADCAST = "BROADCAST"
    SPLIT = "SPLIT"
    GATHERED = "GATHERED"
    PRODUCED = "PRODUCED"
    CONSUMED = "CONSUMED"
    REDUCED = "REDUCED"
    MIXED = "MIXED"
    PROTECTED = "PROTECTED"
    BLOCKED = "BLOCKED"


class EvidenceTier(str, Enum):
    NONE = "NONE"
    MLIR_LOWERING_FAILED = "MLIR_LOWERING_FAILED"
    HIGH_LEVEL_MLIR_ONLY = "HIGH_LEVEL_MLIR_ONLY"
    PYTHON_MLIR_ACCESS = "PYTHON_MLIR_ACCESS"
    NATIVE_MLIR_DEPENDENCE = "NATIVE_MLIR_DEPENDENCE"


class BlockerKind(str, Enum):
    NONE = "none"
    MLIR_TOOLCHAIN_MISSING = "mlir_toolchain_missing"
    EVIDENCE_UNIT_EXPORT_FAILED = "evidence_unit_export_failed"
    ONNX_MLIR_LOWERING_FAILED = "onnx_mlir_lowering_failed"
    NO_MLIR_ARTIFACT = "no_mlir_artifact"
    HIGH_LEVEL_MLIR_ONLY = "high_level_mlir_only"
    NO_AFFINE_OR_LOOP_ACCESS = "no_affine_or_loop_access"
    NO_NATIVE_DEPENDENCE = "no_native_dependence"
    NO_AXIS_RELATION_RECOVERED = "no_axis_relation_recovered"
    UNSUPPORTED_OP_LOWERING = "unsupported_op_lowering"
    UNKNOWN = "unknown"


@dataclass
class AxisRelation:
    source_value: str
    source_axis: str
    target_value: str | None
    target_axis: str | None
    relation: AxisRelationKind
    evidence_tier: EvidenceTier
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return _enum_to_value(asdict(self))


@dataclass
class MlirEvidence:
    available: bool = False
    lowering_succeeded: bool = False
    mlir_files: list[str] = field(default_factory=list)
    native_dependence_json: str | None = None
    python_dependence_json: str | None = None
    dialect_hints: list[str] = field(default_factory=list)
    access_summary: dict[str, Any] = field(default_factory=dict)
    relation_summary: dict[str, Any] = field(default_factory=dict)
    blocker_kind: BlockerKind = BlockerKind.UNKNOWN
    blocker_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _enum_to_value(asdict(self))


@dataclass
class NodeAxisSemantics:
    node_name: str
    op_type: str
    topological_index: int
    semantic_class: AxisSemanticClass
    confidence: str
    evidence_tier: EvidenceTier
    reason: str
    input_names: list[str]
    output_names: list[str]
    axis_relations: list[AxisRelation] = field(default_factory=list)
    leader_candidate_kind: str = "unknown"
    mlir_evidence: MlirEvidence = field(default_factory=MlirEvidence)
    attributes_added: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["axis_relations"] = [relation.to_dict() for relation in self.axis_relations]
        payload["mlir_evidence"] = self.mlir_evidence.to_dict()
        return _enum_to_value(payload)


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_to_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_enum_to_value(item) for item in value]
    if isinstance(value, tuple):
        return [_enum_to_value(item) for item in value]
    return value


def relation_kind_from_mlir(kind: str) -> AxisRelationKind:
    normalized = kind.lower()
    if normalized == "preserved":
        return AxisRelationKind.PRESERVED
    if normalized == "reduced":
        return AxisRelationKind.REDUCED
    if normalized == "mixed":
        return AxisRelationKind.MIXED
    if normalized == "blocked":
        return AxisRelationKind.BLOCKED
    return AxisRelationKind.UNKNOWN


def relations_from_native_report_dict(report: dict[str, Any], tier: EvidenceTier) -> list[AxisRelation]:
    relations: list[AxisRelation] = []
    for item in report.get("relations", []) or []:
        source_indices = item.get("source_indices") or []
        target_indices = item.get("target_indices") or []
        kind = relation_kind_from_mlir(str(item.get("relation_kind", "unknown")))
        relations.append(
            AxisRelation(
                source_value=str(item.get("source_tensor", "")),
                source_axis=",".join(source_indices) if source_indices else "",
                target_value=item.get("target_tensor"),
                target_axis=",".join(target_indices) if target_indices else None,
                relation=kind,
                evidence_tier=tier,
                reason=str(item.get("proof", "")),
            )
        )
    return relations


def summarize_relations(relations: list[AxisRelation]) -> dict[str, Any]:
    counts = Counter(relation.relation.value for relation in relations)
    return {
        "total": len(relations),
        "counts": dict(sorted(counts.items())),
        "preserved": counts.get(AxisRelationKind.PRESERVED.value, 0),
        "reduced": counts.get(AxisRelationKind.REDUCED.value, 0),
        "mixed": counts.get(AxisRelationKind.MIXED.value, 0),
        "blocked": counts.get(AxisRelationKind.BLOCKED.value, 0),
    }


def derive_semantics_from_mlir_evidence(
    *,
    node_name: str,
    op_type: str,
    topological_index: int,
    input_names: list[str],
    output_names: list[str],
    mlir_evidence: MlirEvidence,
    axis_relations: list[AxisRelation],
) -> NodeAxisSemantics:
    """Classify a node only from MLIR-derived evidence and blockers."""
    if mlir_evidence.blocker_kind == BlockerKind.MLIR_TOOLCHAIN_MISSING:
        return _blocked_node(
            node_name,
            op_type,
            topological_index,
            input_names,
            output_names,
            AxisSemanticClass.UNKNOWN,
            EvidenceTier.NONE,
            mlir_evidence,
            "MLIR toolchain was unavailable; no semantic claim was made.",
        )
    if not mlir_evidence.lowering_succeeded and mlir_evidence.blocker_kind == BlockerKind.ONNX_MLIR_LOWERING_FAILED:
        return _blocked_node(
            node_name,
            op_type,
            topological_index,
            input_names,
            output_names,
            AxisSemanticClass.MLIR_LOWERING_FAILED,
            EvidenceTier.MLIR_LOWERING_FAILED,
            mlir_evidence,
            "ONNX-MLIR lowering failed for this evidence unit.",
        )
    if mlir_evidence.blocker_kind == BlockerKind.NO_MLIR_ARTIFACT:
        return _blocked_node(
            node_name,
            op_type,
            topological_index,
            input_names,
            output_names,
            AxisSemanticClass.MLIR_LOWERING_FAILED,
            EvidenceTier.MLIR_LOWERING_FAILED,
            mlir_evidence,
            "ONNX-MLIR did not emit an MLIR artifact.",
        )
    if mlir_evidence.blocker_kind in {BlockerKind.HIGH_LEVEL_MLIR_ONLY, BlockerKind.NO_AFFINE_OR_LOOP_ACCESS}:
        return _blocked_node(
            node_name,
            op_type,
            topological_index,
            input_names,
            output_names,
            AxisSemanticClass.MLIR_HIGH_LEVEL_INSUFFICIENT,
            EvidenceTier.HIGH_LEVEL_MLIR_ONLY,
            mlir_evidence,
            "MLIR exists, but no access/dependence relation was recovered.",
        )
    if not axis_relations:
        return _blocked_node(
            node_name,
            op_type,
            topological_index,
            input_names,
            output_names,
            AxisSemanticClass.NO_ACCESS_EVIDENCE,
            mlir_evidence_relation_tier(mlir_evidence),
            mlir_evidence,
            "No axis relation was recovered from MLIR access/dependence evidence.",
        )

    kinds = {relation.relation for relation in axis_relations}
    tier = mlir_evidence_relation_tier(mlir_evidence)
    if AxisRelationKind.MIXED in kinds or AxisRelationKind.BLOCKED in kinds:
        semantic_class = AxisSemanticClass.MLIR_DERIVED_MATMUL_QK_SCORE
        leader = "blocker"
        reason = "MLIR dependence evidence shows a reduced/mixed axis that is not one-to-one propagatable."
    elif AxisRelationKind.REDUCED in kinds and AxisRelationKind.PRESERVED in kinds:
        semantic_class = AxisSemanticClass.MLIR_DERIVED_MATMUL_GENERIC
        leader = "positive"
        reason = "MLIR access evidence shows preserved free axes plus a reduced/consumed axis."
    elif AxisRelationKind.REDUCED in kinds:
        semantic_class = AxisSemanticClass.MLIR_DERIVED_REDUCTION
        leader = "none"
        reason = "MLIR access evidence shows a read index absent from the write access."
    elif kinds <= {AxisRelationKind.PRESERVED}:
        semantic_class = AxisSemanticClass.MLIR_DERIVED_INDEX_PRESERVING
        leader = "positive"
        reason = "MLIR access evidence shows shared read/write index variables."
    else:
        semantic_class = AxisSemanticClass.NO_ACCESS_EVIDENCE
        leader = "unknown"
        reason = "MLIR evidence was present, but relation kinds were not enough to classify semantics."

    return NodeAxisSemantics(
        node_name=node_name,
        op_type=op_type,
        topological_index=topological_index,
        semantic_class=semantic_class,
        confidence="high" if tier == EvidenceTier.NATIVE_MLIR_DEPENDENCE else "medium",
        evidence_tier=tier,
        reason=reason,
        input_names=input_names,
        output_names=output_names,
        axis_relations=axis_relations,
        leader_candidate_kind=leader,
        mlir_evidence=mlir_evidence,
    )


def mlir_evidence_relation_tier(mlir_evidence: MlirEvidence) -> EvidenceTier:
    if mlir_evidence.native_dependence_json and mlir_evidence.relation_summary.get("total", 0):
        return EvidenceTier.NATIVE_MLIR_DEPENDENCE
    if mlir_evidence.access_summary.get("access_record_count", 0) or mlir_evidence.relation_summary.get("total", 0):
        return EvidenceTier.PYTHON_MLIR_ACCESS
    if mlir_evidence.available:
        return EvidenceTier.HIGH_LEVEL_MLIR_ONLY
    return EvidenceTier.NONE


def _blocked_node(
    node_name: str,
    op_type: str,
    topological_index: int,
    input_names: list[str],
    output_names: list[str],
    semantic_class: AxisSemanticClass,
    evidence_tier: EvidenceTier,
    mlir_evidence: MlirEvidence,
    reason: str,
) -> NodeAxisSemantics:
    return NodeAxisSemantics(
        node_name=node_name,
        op_type=op_type,
        topological_index=topological_index,
        semantic_class=semantic_class,
        confidence="none",
        evidence_tier=evidence_tier,
        reason=reason,
        input_names=input_names,
        output_names=output_names,
        leader_candidate_kind="unknown",
        mlir_evidence=mlir_evidence,
        warnings=[mlir_evidence.blocker_explanation] if mlir_evidence.blocker_explanation else [],
    )
