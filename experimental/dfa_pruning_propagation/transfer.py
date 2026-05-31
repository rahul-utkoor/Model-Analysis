"""Operation transfer functions for the DFA pruning propagation prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from experimental.dfa_pruning_propagation.ir import Axis, Node
from experimental.dfa_pruning_propagation.lattice import Fact, FactKind


STRUCTURAL_KINDS = {FactKind.DEAD, FactKind.PRUNED}


@dataclass(frozen=True)
class TransferEmission:
    fact: Fact
    action: str
    explanation: str


def _fact(axis: Axis, kind: FactKind, reason: str, node: Node, *evidence: str) -> Fact:
    return Fact(axis=axis, kind=kind, reason=reason, source_node=node.node_id, evidence=tuple(evidence))


def _emit(axis: Axis, kind: FactKind, reason: str, node: Node, action: str, explanation: str) -> TransferEmission:
    return TransferEmission(_fact(axis, kind, reason, node, node.op_kind), action, explanation)


def _other_axes(node: Node, changed_axis: Axis) -> list[Axis]:
    return [axis for axis in [*node.inputs, *node.outputs] if axis != changed_axis]


def transfer(node: Node, state: Mapping[Axis, Fact], changed_axis: Axis) -> list[TransferEmission]:
    """Apply one local transfer function after an axis fact changes."""
    changed = state.get(changed_axis)
    if changed is None or changed.kind in {FactKind.UNKNOWN, FactKind.BLOCKED}:
        return []
    if node.op_kind == "activation":
        if changed.kind in STRUCTURAL_KINDS:
            return [
                _emit(axis, changed.kind, "index-preserving activation", node, "propagated", "Activation preserves the intermediate-dimension index.")
                for axis in _other_axes(node, changed_axis)
                if axis.role == "intermediate_dim"
            ]
        return []
    if node.op_kind == "linear_contract":
        if changed_axis in node.inputs and changed.kind in STRUCTURAL_KINDS:
            return [
                _emit(axis, FactKind.PROTECTED, "linear contraction output hidden_dim is preserved", node, "protected", "Consumer-input pruning preserves the output hidden width.")
                for axis in node.outputs
                if axis.role == "hidden_dim"
            ]
        if changed_axis in node.outputs and changed.kind in STRUCTURAL_KINDS:
            return [_emit(changed_axis, FactKind.BLOCKED, "linear contraction output hidden_dim is protected", node, "blocked", "The structural FFN plan prunes the contraction input, not its hidden output.")]
        return []
    if node.op_kind == "attention_context":
        if changed.kind not in STRUCTURAL_KINDS:
            return []
        axes = [axis for axis in _other_axes(node, changed_axis) if axis.role in {"value_dim", "value_context_dim"}]
        if node.attrs.get("value_axis_mapping") == "proven":
            return [
                _emit(axis, changed.kind, "proven attention value-axis mapping", node, "propagated", "Attention context preserves the mapped V/context channel.")
                for axis in axes
            ]
        return [
            _emit(axis, FactKind.BLOCKED, "value_axis_mapping_unproven", node, "blocked", "Value-path deadness cannot cross attention context without a proven axis mapping.")
            for axis in axes
        ]
    if node.op_kind == "attention_output_projection":
        if changed_axis in node.inputs and changed.kind in STRUCTURAL_KINDS:
            return [
                _emit(axis, FactKind.PROTECTED, "attention output hidden_dim is preserved", node, "protected", "Value-context input pruning preserves the attention output hidden width.")
                for axis in node.outputs
                if axis.role == "hidden_dim"
            ]
        if changed_axis in node.outputs and changed.kind in STRUCTURAL_KINDS:
            return [_emit(changed_axis, FactKind.BLOCKED, "attention output hidden_dim is protected", node, "blocked", "The value-path rule prunes the projection input, not its hidden output.")]
        return []
    if node.op_kind in {"residual_add", "layernorm"}:
        if changed.kind in STRUCTURAL_KINDS:
            return [
                _emit(changed_axis, FactKind.BLOCKED, f"{node.op_kind}_hidden_dim_protected", node, "blocked", f"{node.name} protects hidden_dim from local structural pruning."),
                *[
                    _emit(axis, FactKind.PROTECTED, f"{node.op_kind} output hidden_dim protected", node, "protected", f"{node.name} preserves the hidden width.")
                    for axis in node.outputs
                    if axis.role == "hidden_dim"
                ],
            ]
        if changed.kind == FactKind.PROTECTED:
            return [
                _emit(axis, FactKind.PROTECTED, f"{node.op_kind} output hidden_dim protected", node, "protected", f"{node.name} preserves the hidden width.")
                for axis in node.outputs
                if axis.role == "hidden_dim"
            ]
        return []
    if node.op_kind == "score_matmul" and changed_axis in node.inputs and changed.kind in STRUCTURAL_KINDS:
        return [
            _emit(changed_axis, FactKind.BLOCKED, "qk_score_contraction_mixes_channels", node, "blocked", "QK^T mixes Q/K dimensions, so simple one-to-one deadness propagation is invalid.")
        ]
    return []
