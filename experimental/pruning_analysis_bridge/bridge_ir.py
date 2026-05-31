"""Records for lowering axis-transfer evidence into DFA propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experimental.axis_transfer_analysis.axis_relations import RegionAxisSummary
from experimental.axis_transfer_analysis.loop_ir import RegionSpec
from experimental.axis_transfer_analysis.pattern_recognition import PatternMatch
from experimental.dfa_pruning_propagation.ir import Graph
from experimental.dfa_pruning_propagation.lattice import Fact
from experimental.dfa_pruning_propagation.worklist import AnalysisResult


@dataclass(frozen=True)
class BridgeSeedPolicy:
    kind: str
    target_axis_hint: str
    explanation: str


@dataclass(frozen=True)
class BridgeInput:
    example_name: str
    region_spec: RegionSpec
    seed_policy: BridgeSeedPolicy
    interpretation: str


@dataclass(frozen=True)
class BridgeTraceEvent:
    stage: str
    message: str
    evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class BridgeResult:
    example_name: str
    region_spec: RegionSpec
    axis_summary: RegionAxisSummary
    pattern_matches: list[PatternMatch]
    dfa_graph: Graph
    seed_facts: list[Fact]
    dfa_result: AnalysisResult
    bridge_trace: list[BridgeTraceEvent]
    summary: dict[str, Any]
