"""MLIR-derived leader-candidate discovery for axis semantics.

This module does not synthesize pruning plans. It only identifies nodes whose
strict MLIR-derived relation summaries look useful, blocked, or insufficient.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from model_analysis.onnx_axis_semantics import (
    AxisRelationKind,
    AxisSemanticClass,
    BlockerKind,
    EvidenceTier,
    NodeAxisSemantics,
)


class LeaderCandidateKind(str, Enum):
    POSITIVE = "positive"
    BLOCKER = "blocker"
    PROTECTED = "protected"
    UNKNOWN = "unknown"
    NONE = "none"


@dataclass
class LeaderReason:
    relation_summary: dict[str, Any]
    evidence_tier: str
    blocker_kind: str
    explanation: str


@dataclass
class LeaderCandidate:
    node_name: str
    op_type: str
    topological_index: int
    candidate_kind: LeaderCandidateKind
    evidence_tier: EvidenceTier
    semantic_class: AxisSemanticClass
    relation_counts: dict[str, int]
    root_axes: list[str]
    affected_relation_kinds: list[str]
    reason: LeaderReason
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _enum_to_value(asdict(self))


def discover_leader_candidates(nodes: list[NodeAxisSemantics]) -> list[LeaderCandidate]:
    candidates = [leader_candidate_for_node(node) for node in nodes]
    for node, candidate in zip(nodes, candidates):
        node.leader_candidate_kind = candidate.candidate_kind.value
    return candidates


def leader_candidate_for_node(node: NodeAxisSemantics) -> LeaderCandidate:
    relation_counts = count_relations(node)
    relation_kinds = {relation.relation for relation in node.axis_relations}
    blocker = node.mlir_evidence.blocker_kind
    evidence_tier = node.evidence_tier
    semantic_class = node.semantic_class
    warnings = list(node.warnings)

    if semantic_class == AxisSemanticClass.MLIR_DERIVED_PROTECTED_INTERFACE:
        kind = LeaderCandidateKind.PROTECTED
        explanation = "Explicit protected-interface metadata is present."
    elif AxisRelationKind.BLOCKED in relation_kinds or AxisRelationKind.MIXED in relation_kinds or semantic_class in {
        AxisSemanticClass.MLIR_DERIVED_BLOCKER,
        AxisSemanticClass.MLIR_DERIVED_MATMUL_QK_SCORE,
    }:
        kind = LeaderCandidateKind.BLOCKER
        explanation = "MLIR-derived relation summary contains blocked or mixed axis evidence."
    elif semantic_class in {
        AxisSemanticClass.MLIR_LOWERING_FAILED,
        AxisSemanticClass.MLIR_HIGH_LEVEL_INSUFFICIENT,
        AxisSemanticClass.NO_ACCESS_EVIDENCE,
        AxisSemanticClass.UNKNOWN,
    } or blocker in {
        BlockerKind.MLIR_TOOLCHAIN_MISSING,
        BlockerKind.ONNX_MLIR_LOWERING_FAILED,
        BlockerKind.NO_MLIR_ARTIFACT,
        BlockerKind.HIGH_LEVEL_MLIR_ONLY,
        BlockerKind.NO_AFFINE_OR_LOOP_ACCESS,
        BlockerKind.NO_AXIS_RELATION_RECOVERED,
        BlockerKind.UNSUPPORTED_OP_LOWERING,
        BlockerKind.UNKNOWN,
    }:
        kind = LeaderCandidateKind.UNKNOWN
        explanation = "MLIR evidence is missing, failed, high-level only, or insufficient for relation recovery."
    elif _is_positive_candidate(node, relation_counts):
        kind = LeaderCandidateKind.POSITIVE
        explanation = "MLIR-derived access/dependence evidence contains nontrivial preserved or reduced axis relations."
    else:
        kind = LeaderCandidateKind.NONE
        explanation = "No leader-candidate rule matched the MLIR-derived relation summary."

    return LeaderCandidate(
        node_name=node.node_name,
        op_type=node.op_type,
        topological_index=node.topological_index,
        candidate_kind=kind,
        evidence_tier=evidence_tier,
        semantic_class=semantic_class,
        relation_counts=relation_counts,
        root_axes=_root_axes(node),
        affected_relation_kinds=sorted(kind.value for kind in relation_kinds),
        reason=LeaderReason(
            relation_summary=node.mlir_evidence.relation_summary,
            evidence_tier=evidence_tier.value,
            blocker_kind=blocker.value,
            explanation=explanation,
        ),
        warnings=warnings,
    )


def count_relations(node: NodeAxisSemantics) -> dict[str, int]:
    counts = Counter(relation.relation.value for relation in node.axis_relations)
    return {
        "PRESERVED": counts.get(AxisRelationKind.PRESERVED.value, 0),
        "REDUCED": counts.get(AxisRelationKind.REDUCED.value, 0),
        "MIXED": counts.get(AxisRelationKind.MIXED.value, 0),
        "BLOCKED": counts.get(AxisRelationKind.BLOCKED.value, 0),
        "UNKNOWN": counts.get(AxisRelationKind.UNKNOWN.value, 0),
    }


def write_leader_report(candidates: list[LeaderCandidate], output_path: str) -> None:
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(candidate.candidate_kind.value for candidate in candidates)
    lines = [
        "# MLIR-Derived Leader Candidates",
        "",
        "## Summary",
        "",
        f"- Nodes analyzed: {len(candidates)}",
        f"- Positive candidates: {counts.get('positive', 0)}",
        f"- Blocker candidates: {counts.get('blocker', 0)}",
        f"- Unknown candidates: {counts.get('unknown', 0)}",
        f"- Protected candidates: {counts.get('protected', 0)}",
        "",
        "## Candidate Table",
        "",
        "Node | Op | Semantic | Evidence | Candidate | Relations | Reason",
        "--- | --- | --- | --- | --- | --- | ---",
    ]
    for candidate in candidates:
        relations = _relations_cell(candidate.relation_counts)
        lines.append(
            " | ".join(
                [
                    _md(candidate.node_name),
                    _md(candidate.op_type),
                    _md(candidate.semantic_class.value),
                    _md(candidate.evidence_tier.value),
                    _md(candidate.candidate_kind.value),
                    _md(relations),
                    _md(candidate.reason.explanation),
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is leader-candidate discovery only.",
            "- It does not synthesize final pruning propagation plans.",
            "- It does not classify MatMul into projection/context/QK roles unless MLIR evidence and later propagation prove it.",
            "- Full evidence remains in the sidecar JSON.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_positive_candidate(node: NodeAxisSemantics, relation_counts: dict[str, int]) -> bool:
    return (
        node.evidence_tier in {EvidenceTier.NATIVE_MLIR_DEPENDENCE, EvidenceTier.PYTHON_MLIR_ACCESS}
        and node.mlir_evidence.blocker_kind == BlockerKind.NONE
        and node.semantic_class.value.startswith("MLIR_DERIVED_")
        and bool(node.input_names)
        and bool(node.output_names)
        and relation_counts.get("UNKNOWN", 0) < sum(relation_counts.values())
        and (relation_counts.get("PRESERVED", 0) > 0 or relation_counts.get("REDUCED", 0) > 0)
    )


def _root_axes(node: NodeAxisSemantics) -> list[str]:
    axes = []
    for relation in node.axis_relations:
        if relation.source_axis:
            axes.append(relation.source_axis)
        if relation.target_axis:
            axes.append(relation.target_axis)
    return sorted(set(axes))


def _relations_cell(counts: dict[str, int]) -> str:
    return f"P={counts.get('PRESERVED', 0)} R={counts.get('REDUCED', 0)} M={counts.get('MIXED', 0)} B={counts.get('BLOCKED', 0)}"


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_to_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_enum_to_value(item) for item in value]
    return value


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
