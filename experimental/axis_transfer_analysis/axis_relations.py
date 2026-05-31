"""Axis-transfer records inferred from indexed tensor accesses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AxisRelationKind(str, Enum):
    UNKNOWN = "UNKNOWN"
    PRESERVED = "PRESERVED"
    PERMUTED = "PERMUTED"
    RESHAPED_SPLIT = "RESHAPED_SPLIT"
    RESHAPED_MERGED = "RESHAPED_MERGED"
    REDUCED = "REDUCED"
    BROADCAST = "BROADCAST"
    MIXED = "MIXED"
    PROTECTED = "PROTECTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class AxisTransfer:
    source_tensor: str
    source_axis: str
    target_tensor: str | None
    target_axis: str | None
    relation: AxisRelationKind
    confidence: str
    proof: str


@dataclass
class OperationAxisSummary:
    op_id: str
    op_kind: str
    transfers: list[AxisTransfer] = field(default_factory=list)
    reduced_axes: list[str] = field(default_factory=list)
    preserved_axes: list[str] = field(default_factory=list)
    protected_axes: list[str] = field(default_factory=list)
    blocked_axes: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class RegionAxisSummary:
    region_id: str
    op_summaries: list[OperationAxisSummary]
    pattern_candidates: list[object] = field(default_factory=list)
    explanation: str = ""
