"""Data records for selected-subgraph cross-evidence pruning proofs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProofCase:
    case_id: str
    model_name: str
    layer_index: int | None
    subgraph_name: str
    onnx_path: str
    expected_pattern: str | None
    expected_dfa_result: str | None
    notes: str = ""


@dataclass(frozen=True)
class AxisRelationRecord:
    source: str
    target: str
    relation: str
    confidence: str
    proof: str


@dataclass
class ProofEvidence:
    case_id: str
    model_name: str
    layer_index: int | None
    subgraph_name: str
    onnx_path: str
    found: bool
    onnx_summary: dict[str, Any] = field(default_factory=dict)
    mlir_summary: dict[str, Any] = field(default_factory=dict)
    evidence_source: str = "unavailable"
    axis_relations: list[AxisRelationRecord] = field(default_factory=list)
    recognized_patterns: list[str] = field(default_factory=list)
    dfa_summary: dict[str, Any] = field(default_factory=dict)
    verdict: str = "unknown"
    limitations: list[str] = field(default_factory=list)
