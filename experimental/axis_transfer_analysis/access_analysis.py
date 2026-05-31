"""Infer pruning-relevant axis relations from loop/access descriptions."""

from __future__ import annotations

from experimental.axis_transfer_analysis.axis_relations import (
    AxisRelationKind,
    AxisTransfer,
    OperationAxisSummary,
    RegionAxisSummary,
)
from experimental.axis_transfer_analysis.loop_ir import OperationSpec, RegionSpec, Tensor, TensorAccess


def _axes_by_iv(tensor: Tensor, access: TensorAccess) -> dict[str, str]:
    if len(access.indices) != tensor.rank:
        raise ValueError(f"access rank mismatch for {tensor.name}: {access.indices}")
    return dict(zip(access.indices, tensor.axes))


def _append_transfer(summary: OperationAxisSummary, transfer: AxisTransfer) -> None:
    if transfer not in summary.transfers:
        summary.transfers.append(transfer)
    axis = f"{transfer.source_tensor}.{transfer.source_axis}"
    if transfer.relation == AxisRelationKind.REDUCED and axis not in summary.reduced_axes:
        summary.reduced_axes.append(axis)
    if transfer.relation in {AxisRelationKind.PRESERVED, AxisRelationKind.PERMUTED} and axis not in summary.preserved_axes:
        summary.preserved_axes.append(axis)
    if transfer.relation == AxisRelationKind.PROTECTED and axis not in summary.protected_axes:
        summary.protected_axes.append(axis)
    if transfer.relation == AxisRelationKind.BLOCKED and axis not in summary.blocked_axes:
        summary.blocked_axes.append(axis)


def _transfer(
    source_tensor: str,
    source_axis: str,
    target_tensor: str | None,
    target_axis: str | None,
    relation: AxisRelationKind,
    proof: str,
    *,
    confidence: str = "high",
) -> AxisTransfer:
    return AxisTransfer(source_tensor, source_axis, target_tensor, target_axis, relation, confidence, proof)


def _free_iv_relation(source_axis: str, target_axis: str) -> AxisRelationKind:
    if source_axis == target_axis or (source_axis, target_axis) == ("value_dim", "value_context_dim"):
        return AxisRelationKind.PRESERVED
    return AxisRelationKind.PERMUTED


def _first_write(op: OperationSpec) -> TensorAccess:
    if not op.writes:
        raise ValueError(f"operation {op.op_id} has no writes")
    return op.writes[0]


def _analyze_elementwise(region: RegionSpec, op: OperationSpec) -> OperationAxisSummary:
    summary = OperationAxisSummary(op.op_id, op.op_kind, explanation="Elementwise accesses preserve matching loop IVs.")
    write = _first_write(op)
    output_axes = _axes_by_iv(region.tensor(write.tensor), write)
    for read in op.reads:
        input_axes = _axes_by_iv(region.tensor(read.tensor), read)
        for iv, source_axis in input_axes.items():
            target_axis = output_axes.get(iv)
            if target_axis is None:
                relation = AxisRelationKind.REDUCED
                proof = f"{iv} appears in {read.tensor} but not {write.tensor}"
            elif source_axis == target_axis:
                relation = AxisRelationKind.PRESERVED
                proof = f"{iv} indexes {read.tensor}.{source_axis} and {write.tensor}.{target_axis}"
            else:
                relation = _free_iv_relation(source_axis, target_axis)
                proof = f"{iv} is retained while axis role changes from {source_axis} to {target_axis}"
            _append_transfer(summary, _transfer(read.tensor, source_axis, write.tensor, target_axis, relation, proof))
        for iv, target_axis in output_axes.items():
            if iv not in input_axes:
                _append_transfer(
                    summary,
                    _transfer(read.tensor, "*", write.tensor, target_axis, AxisRelationKind.BROADCAST, f"{iv} is generated or broadcast into {write.tensor}.{target_axis}"),
                )
    return summary


def _looks_like_qk_score(region: RegionSpec, op: OperationSpec) -> bool:
    if len(op.reads) != 2 or not op.writes:
        return False
    write_axes = set(region.tensor(op.writes[0].tensor).axes)
    read_axes = [set(region.tensor(read.tensor).axes) for read in op.reads]
    return (
        {"query_position", "key_position"} <= write_axes
        and all("head_dim" in axes for axes in read_axes)
        and any("query_position" in axes for axes in read_axes)
        and any("key_position" in axes for axes in read_axes)
    )


def _analyze_contraction(region: RegionSpec, op: OperationSpec) -> OperationAxisSummary:
    summary = OperationAxisSummary(op.op_id, op.op_kind)
    write = _first_write(op)
    output_axes = _axes_by_iv(region.tensor(write.tensor), write)
    qk_score = _looks_like_qk_score(region, op)
    for read in op.reads:
        input_axes = _axes_by_iv(region.tensor(read.tensor), read)
        for iv, source_axis in input_axes.items():
            target_axis = output_axes.get(iv)
            if target_axis is not None:
                relation = _free_iv_relation(source_axis, target_axis)
                proof = f"{iv} remains free in {write.tensor}.{target_axis}"
            else:
                relation = AxisRelationKind.REDUCED
                proof = f"{iv} is consumed inside the contraction and absent from {write.tensor}"
            _append_transfer(summary, _transfer(read.tensor, source_axis, write.tensor, target_axis, relation, proof))
            if qk_score and source_axis == "head_dim" and target_axis is None:
                _append_transfer(
                    summary,
                    _transfer(
                        read.tensor,
                        source_axis,
                        write.tensor,
                        None,
                        AxisRelationKind.MIXED,
                        "projected feature channels participate jointly in QK^T score contraction",
                    ),
                )
                _append_transfer(
                    summary,
                    _transfer(
                        read.tensor,
                        source_axis,
                        write.tensor,
                        None,
                        AxisRelationKind.BLOCKED,
                        "qk_score_contraction_mixes_channels",
                    ),
                )
    if qk_score:
        summary.explanation = "QK score contraction reduces the projected feature axis; simple one-to-one Q/K propagation is blocked."
    else:
        summary.explanation = "Contraction keeps free output IVs and reduces feature IVs that do not appear in the write."
    return summary


def _analyze_residual(region: RegionSpec, op: OperationSpec) -> OperationAxisSummary:
    summary = OperationAxisSummary(op.op_id, op.op_kind, explanation="Residual branches must keep the hidden axis aligned.")
    write = _first_write(op)
    output_axes = _axes_by_iv(region.tensor(write.tensor), write)
    for read in op.reads:
        input_axes = _axes_by_iv(region.tensor(read.tensor), read)
        for iv, source_axis in input_axes.items():
            target_axis = output_axes.get(iv)
            relation = AxisRelationKind.PROTECTED if source_axis == "hidden_dim" else AxisRelationKind.PRESERVED
            proof = "residual hidden axis requires coordinated branch pruning" if relation == AxisRelationKind.PROTECTED else f"{iv} is aligned across residual add"
            _append_transfer(summary, _transfer(read.tensor, source_axis, write.tensor, target_axis, relation, proof))
    return summary


def _analyze_layernorm(region: RegionSpec, op: OperationSpec) -> OperationAxisSummary:
    summary = OperationAxisSummary(op.op_id, op.op_kind, explanation="LayerNorm protects its normalized hidden axis.")
    write = _first_write(op)
    output_axes = _axes_by_iv(region.tensor(write.tensor), write)
    for read in op.reads:
        input_axes = _axes_by_iv(region.tensor(read.tensor), read)
        for iv, source_axis in input_axes.items():
            target_axis = output_axes.get(iv)
            relation = AxisRelationKind.PROTECTED if source_axis == "hidden_dim" else AxisRelationKind.PRESERVED
            proof = "normalization statistics couple the hidden axis" if relation == AxisRelationKind.PROTECTED else f"{iv} is preserved through normalization"
            _append_transfer(summary, _transfer(read.tensor, source_axis, write.tensor, target_axis, relation, proof))
    return summary


def _analyze_unknown(region: RegionSpec, op: OperationSpec) -> OperationAxisSummary:
    summary = OperationAxisSummary(op.op_id, op.op_kind, explanation="Access behavior is unsupported by this teaching prototype.")
    for read in op.reads:
        tensor = region.tensor(read.tensor)
        for source_axis in tensor.axes:
            _append_transfer(summary, _transfer(read.tensor, source_axis, None, None, AxisRelationKind.UNKNOWN, "unsupported operation kind", confidence="low"))
    return summary


def analyze_operation(region: RegionSpec, op: OperationSpec) -> OperationAxisSummary:
    if op.op_kind in {"unary_elementwise", "reshape", "transpose"}:
        return _analyze_elementwise(region, op)
    if op.op_kind in {"matmul", "contraction", "reduction"}:
        return _analyze_contraction(region, op)
    if op.op_kind == "residual_add":
        return _analyze_residual(region, op)
    if op.op_kind == "layernorm":
        return _analyze_layernorm(region, op)
    return _analyze_unknown(region, op)


def analyze_region(region: RegionSpec) -> RegionAxisSummary:
    """Summarize axis behavior for every operation in one region."""
    summaries = [analyze_operation(region, op) for op in region.ops]
    return RegionAxisSummary(
        region_id=region.region_id,
        op_summaries=summaries,
        explanation="Axis transfers are inferred from loop IV reuse and indexed tensor accesses.",
    )
