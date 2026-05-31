"""Conservative dataflow lattice for pruning/deadness facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from experimental.dfa_pruning_propagation.ir import Axis


class FactKind(str, Enum):
    UNKNOWN = "UNKNOWN"
    LIVE = "LIVE"
    DEAD = "DEAD"
    PRUNED = "PRUNED"
    PROTECTED = "PROTECTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class Fact:
    axis: Axis
    kind: FactKind
    reason: str
    source_node: str | None = None
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        return f"{self.axis.label()} = {self.kind.value}: {self.reason}"


def unknown(axis: Axis) -> Fact:
    return Fact(axis=axis, kind=FactKind.UNKNOWN, reason="no_fact")


def _merged_evidence(a: Fact, b: Fact) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*a.evidence, *b.evidence]))


def _result(a: Fact, b: Fact, kind: FactKind, reason: str | None = None) -> Fact:
    return Fact(
        axis=a.axis,
        kind=kind,
        reason=reason or (b.reason if a.kind == FactKind.UNKNOWN else a.reason),
        source_node=b.source_node if a.kind == FactKind.UNKNOWN else a.source_node or b.source_node,
        evidence=_merged_evidence(a, b),
    )


def join(a: Fact, b: Fact) -> Fact:
    """Join two facts for one axis conservatively."""
    if a.axis != b.axis:
        raise ValueError("cannot join facts for different axes")
    if a.kind == b.kind:
        return _result(a, b, a.kind)
    if a.kind == FactKind.UNKNOWN:
        return _result(a, b, b.kind, b.reason)
    if b.kind == FactKind.UNKNOWN:
        return _result(a, b, a.kind, a.reason)
    if FactKind.BLOCKED in {a.kind, b.kind}:
        return _result(a, b, FactKind.BLOCKED, "blocked fact dominates join")
    if {a.kind, b.kind} == {FactKind.DEAD, FactKind.PRUNED}:
        return _result(a, b, FactKind.PRUNED, "legal pruning subsumes deadness")
    if FactKind.PROTECTED in {a.kind, b.kind} and ({a.kind, b.kind} & {FactKind.DEAD, FactKind.PRUNED}):
        return _result(a, b, FactKind.BLOCKED, "protected axis conflicts with structural deadness/pruning")
    if {a.kind, b.kind} == {FactKind.LIVE, FactKind.DEAD}:
        return _result(a, b, FactKind.BLOCKED, "live axis conflicts with deadness")
    if {a.kind, b.kind} == {FactKind.LIVE, FactKind.PRUNED}:
        return _result(a, b, FactKind.BLOCKED, "live axis conflicts with pruning")
    if FactKind.PROTECTED in {a.kind, b.kind}:
        return _result(a, b, FactKind.PROTECTED, "axis is protected")
    return _result(a, b, FactKind.BLOCKED, f"incompatible facts: {a.kind.value} and {b.kind.value}")
