"""Recognize pruning patterns from axis-transfer evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from experimental.axis_transfer_analysis.axis_relations import AxisRelationKind, OperationAxisSummary, RegionAxisSummary
from experimental.axis_transfer_analysis.loop_ir import OperationSpec, RegionSpec


class PatternKind(str, Enum):
    INDEX_PRESERVING_UNARY = "INDEX_PRESERVING_UNARY"
    FFN_INTERMEDIATE_CHAIN = "FFN_INTERMEDIATE_CHAIN"
    ATTENTION_VALUE_PATH = "ATTENTION_VALUE_PATH"
    QK_SCORE_BLOCKER = "QK_SCORE_BLOCKER"
    RESIDUAL_HIDDEN_PROTECTED = "RESIDUAL_HIDDEN_PROTECTED"
    LAYERNORM_HIDDEN_PROTECTED = "LAYERNORM_HIDDEN_PROTECTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PatternMatch:
    pattern_kind: PatternKind
    ops: tuple[str, ...]
    status: str
    required_relations: tuple[str, ...] = field(default_factory=tuple)
    evidence: tuple[str, ...] = field(default_factory=tuple)
    explanation: str = ""


def _summary_by_op(summary: RegionAxisSummary) -> dict[str, OperationAxisSummary]:
    return {op_summary.op_id: op_summary for op_summary in summary.op_summaries}


def _writers(region: RegionSpec) -> dict[str, OperationSpec]:
    return {access.tensor: op for op in region.ops for access in op.writes}


def _read_tensors(op: OperationSpec) -> set[str]:
    return {access.tensor for access in op.reads}


def _writes_axis(region: RegionSpec, op: OperationSpec, axis: str) -> bool:
    return any(axis in region.tensor(access.tensor).axes for access in op.writes)


def _reads_axis(region: RegionSpec, op: OperationSpec, axis: str) -> bool:
    return any(axis in region.tensor(access.tensor).axes for access in op.reads)


def _has_relation(
    summary: OperationAxisSummary,
    relation: AxisRelationKind,
    *,
    source_axis: str | None = None,
    target_axis: str | None = None,
    proof: str | None = None,
) -> bool:
    return any(
        transfer.relation == relation
        and (source_axis is None or transfer.source_axis == source_axis)
        and (target_axis is None or transfer.target_axis == target_axis)
        and (proof is None or proof in transfer.proof)
        for transfer in summary.transfers
    )


def _recognize_unary(region: RegionSpec, summary: RegionAxisSummary) -> list[PatternMatch]:
    summaries = _summary_by_op(summary)
    matches: list[PatternMatch] = []
    for op in region.ops:
        op_summary = summaries[op.op_id]
        if op.op_kind == "unary_elementwise" and _has_relation(op_summary, AxisRelationKind.PRESERVED):
            matches.append(
                PatternMatch(
                    PatternKind.INDEX_PRESERVING_UNARY,
                    (op.op_id,),
                    "propagation_amenable",
                    ("input axis -> output axis PRESERVED",),
                    tuple(transfer.proof for transfer in op_summary.transfers if transfer.relation == AxisRelationKind.PRESERVED),
                    "The unary operation preserves indexed axes without reducing or mixing them.",
                )
            )
    return matches


def _recognize_ffn(region: RegionSpec, summary: RegionAxisSummary) -> list[PatternMatch]:
    summaries = _summary_by_op(summary)
    writers = _writers(region)
    matches: list[PatternMatch] = []
    for unary in region.ops:
        if unary.op_kind != "unary_elementwise" or not _writes_axis(region, unary, "intermediate_dim"):
            continue
        if not _has_relation(summaries[unary.op_id], AxisRelationKind.PRESERVED, source_axis="intermediate_dim", target_axis="intermediate_dim"):
            continue
        expansion = next((writers.get(tensor) for tensor in _read_tensors(unary) if writers.get(tensor)), None)
        if expansion is None or not _writes_axis(region, expansion, "intermediate_dim"):
            continue
        unary_outputs = {access.tensor for access in unary.writes}
        contraction = next(
            (
                op
                for op in region.ops
                if op.op_id != unary.op_id
                and unary_outputs & _read_tensors(op)
                and _reads_axis(region, op, "intermediate_dim")
                and _writes_axis(region, op, "hidden_dim")
                and _has_relation(summaries[op.op_id], AxisRelationKind.REDUCED, source_axis="intermediate_dim")
            ),
            None,
        )
        if contraction is None:
            continue
        matches.append(
            PatternMatch(
                PatternKind.FFN_INTERMEDIATE_CHAIN,
                (expansion.op_id, unary.op_id, contraction.op_id),
                "pruning_amenable",
                (
                    "expansion produces intermediate_dim",
                    "unary preserves intermediate_dim",
                    "contraction consumes intermediate_dim as input-feature reduction",
                    "contraction produces hidden_dim",
                ),
                ("intermediate axis is produced, preserved, then consumed",),
                "The intermediate axis is produced by the first projection, preserved by the activation, and consumed by the second projection. This is a pruning-amenable FFN intermediate chain.",
            )
        )
    return matches


def _recognize_attention_value(region: RegionSpec, summary: RegionAxisSummary) -> list[PatternMatch]:
    summaries = _summary_by_op(summary)
    writers = _writers(region)
    matches: list[PatternMatch] = []
    for context in region.ops:
        context_summary = summaries[context.op_id]
        value_transfer = next(
            (
                transfer
                for transfer in context_summary.transfers
                if transfer.relation == AxisRelationKind.PRESERVED
                and transfer.source_axis == "value_dim"
                and transfer.target_axis == "value_context_dim"
            ),
            None,
        )
        if value_transfer is None:
            continue
        value_projection = writers.get(value_transfer.source_tensor)
        if value_projection is None:
            continue
        context_outputs = {access.tensor for access in context.writes}
        output_projection = next(
            (
                op
                for op in region.ops
                if context_outputs & _read_tensors(op)
                and _reads_axis(region, op, "value_context_dim")
                and _writes_axis(region, op, "hidden_dim")
                and _has_relation(summaries[op.op_id], AxisRelationKind.REDUCED, source_axis="value_context_dim")
            ),
            None,
        )
        if output_projection is None:
            continue
        matches.append(
            PatternMatch(
                PatternKind.ATTENTION_VALUE_PATH,
                (value_projection.op_id, context.op_id, output_projection.op_id),
                "propagation_amenable",
                (
                    "V.value_dim -> Context.value_context_dim PRESERVED",
                    "output projection consumes value_context_dim",
                    "output projection produces hidden_dim",
                ),
                (value_transfer.proof,),
                "The value axis is preserved through attention context and consumed by the output projection. Dead output-projection input channels can propagate backward to V output channels.",
            )
        )
    return matches


def _recognize_qk(summary: RegionAxisSummary) -> list[PatternMatch]:
    matches: list[PatternMatch] = []
    for op_summary in summary.op_summaries:
        if _has_relation(op_summary, AxisRelationKind.BLOCKED, proof="qk_score_contraction_mixes_channels"):
            matches.append(
                PatternMatch(
                    PatternKind.QK_SCORE_BLOCKER,
                    (op_summary.op_id,),
                    "blocked",
                    ("Q/K feature axis is REDUCED", "simple one-to-one propagation is BLOCKED"),
                    ("qk_score_contraction_mixes_channels",),
                    "The feature axis is reduced inside QK^T, so it is not an output-preserved axis. Simple one-to-one deadness propagation through Q/K is blocked.",
                )
            )
    return matches


def _recognize_protected(summary: RegionAxisSummary) -> list[PatternMatch]:
    matches: list[PatternMatch] = []
    for op_summary in summary.op_summaries:
        if not op_summary.protected_axes:
            continue
        if op_summary.op_kind == "residual_add":
            matches.append(
                PatternMatch(
                    PatternKind.RESIDUAL_HIDDEN_PROTECTED,
                    (op_summary.op_id,),
                    "protected",
                    ("residual hidden axes aligned",),
                    tuple(transfer.proof for transfer in op_summary.transfers if transfer.relation == AxisRelationKind.PROTECTED),
                    "Residual hidden-axis pruning requires coordinated repair across both branches.",
                )
            )
        elif op_summary.op_kind == "layernorm":
            matches.append(
                PatternMatch(
                    PatternKind.LAYERNORM_HIDDEN_PROTECTED,
                    (op_summary.op_id,),
                    "protected",
                    ("normalized hidden axis protected",),
                    tuple(transfer.proof for transfer in op_summary.transfers if transfer.relation == AxisRelationKind.PROTECTED),
                    "LayerNorm hidden-axis pruning is protected because normalization statistics couple that axis.",
                )
            )
    return matches


def recognize_patterns(region: RegionSpec, summary: RegionAxisSummary | None = None) -> list[PatternMatch]:
    """Recognize pruning-relevant patterns from access-derived relations."""
    if summary is None:
        from experimental.axis_transfer_analysis.access_analysis import analyze_region

        summary = analyze_region(region)
    matches = [
        *_recognize_unary(region, summary),
        *_recognize_ffn(region, summary),
        *_recognize_attention_value(region, summary),
        *_recognize_qk(summary),
        *_recognize_protected(summary),
    ]
    summary.pattern_candidates = matches
    return matches
