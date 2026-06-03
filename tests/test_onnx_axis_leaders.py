from __future__ import annotations

from model_analysis.onnx_axis_leaders import leader_candidate_for_node
from model_analysis.onnx_axis_semantics import (
    AxisRelation,
    AxisRelationKind,
    AxisSemanticClass,
    BlockerKind,
    EvidenceTier,
    MlirEvidence,
    NodeAxisSemantics,
)


def _node(
    *,
    semantic_class: AxisSemanticClass,
    evidence_tier: EvidenceTier,
    blocker_kind: BlockerKind,
    relations: list[AxisRelation],
) -> NodeAxisSemantics:
    return NodeAxisSemantics(
        node_name="node",
        op_type="DisplayOnly",
        topological_index=0,
        semantic_class=semantic_class,
        confidence="high",
        evidence_tier=evidence_tier,
        reason="test",
        input_names=["X"],
        output_names=["Y"],
        axis_relations=relations,
        mlir_evidence=MlirEvidence(
            available=True,
            lowering_succeeded=True,
            relation_summary={"total": len(relations)},
            blocker_kind=blocker_kind,
        ),
    )


def test_leader_discovery_positive() -> None:
    node = _node(
        semantic_class=AxisSemanticClass.MLIR_DERIVED_MATMUL_GENERIC,
        evidence_tier=EvidenceTier.NATIVE_MLIR_DEPENDENCE,
        blocker_kind=BlockerKind.NONE,
        relations=[
            AxisRelation("A", "j", "B", "j", AxisRelationKind.PRESERVED, EvidenceTier.NATIVE_MLIR_DEPENDENCE, "preserved"),
            AxisRelation("A", "k", "B", None, AxisRelationKind.REDUCED, EvidenceTier.NATIVE_MLIR_DEPENDENCE, "reduced"),
        ],
    )

    candidate = leader_candidate_for_node(node)

    assert candidate.candidate_kind.value == "positive"
    assert candidate.relation_counts["PRESERVED"] == 1
    assert candidate.relation_counts["REDUCED"] == 1


def test_leader_discovery_unknown_for_insufficient_mlir() -> None:
    node = _node(
        semantic_class=AxisSemanticClass.MLIR_LOWERING_FAILED,
        evidence_tier=EvidenceTier.MLIR_LOWERING_FAILED,
        blocker_kind=BlockerKind.ONNX_MLIR_LOWERING_FAILED,
        relations=[],
    )

    candidate = leader_candidate_for_node(node)

    assert candidate.candidate_kind.value == "unknown"


def test_leader_discovery_blocker_for_mixed_relation() -> None:
    node = _node(
        semantic_class=AxisSemanticClass.MLIR_DERIVED_MATMUL_QK_SCORE,
        evidence_tier=EvidenceTier.NATIVE_MLIR_DEPENDENCE,
        blocker_kind=BlockerKind.NONE,
        relations=[
            AxisRelation("Q", "d", "S", None, AxisRelationKind.MIXED, EvidenceTier.NATIVE_MLIR_DEPENDENCE, "mixed"),
        ],
    )

    candidate = leader_candidate_for_node(node)

    assert candidate.candidate_kind.value == "blocker"
    assert candidate.relation_counts["MIXED"] == 1
