"""Aggregate BERT FFN and attention value-path evidence into a 24-plan proof."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"(?:Layer |layer[._])(\d+)")
PROVEN_COVERAGE_VERDICTS = {"native_proven", "access_proven", "fallback_proven"}


@dataclass(frozen=True)
class BertLayerPlanProof:
    layer_index: int
    ffn_plan_status: str = "missing"
    ffn_validation_status: str = "missing"
    ffn_evidence_tier: str = "static_symbolic_plan"
    ffn_verdict: str = "missing"
    attention_path_status: str = "missing"
    attention_mapping_status: str = "unproven"
    attention_evidence_tier: str = "unavailable"
    attention_verdict: str = "missing"
    attention_dfa_ran: bool = False
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Bert24PlanSummary:
    layers_total: int
    expected_plans: int
    ffn_expected: int
    ffn_found: int
    ffn_proven: int
    attention_expected: int
    attention_found: int
    attention_seedable: int
    attention_proven: int
    total_proven: int
    missing: int
    partial: int
    failed: int
    final_verdict: str


@dataclass(frozen=True)
class Bert24PlanProof:
    model_name: str
    summary: Bert24PlanSummary
    layers: tuple[BertLayerPlanProof, ...]
    source_paths: dict[str, str] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required report is missing: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _layer_index(value: str) -> int | None:
    match = LAYER_RE.search(value)
    return int(match.group(1)) if match else None


def _ffn_records(plans: dict[str, Any], validations: dict[str, Any]) -> dict[int, tuple[str, str]]:
    validation_by_plan = {
        str(item.get("plan_id")): str(item.get("validation_status", "unknown"))
        for item in validations.get("validations", [])
    }
    records: dict[int, tuple[str, str]] = {}
    for plan in plans.get("plans", []):
        if plan.get("plan_kind") != "feedforward_intermediate_dim_plan":
            continue
        text = " ".join(
            [
                str(plan.get("candidate_region_name", "")),
                *[str(action.get("target_source_name", "")) for action in plan.get("actions", [])],
            ]
        )
        layer = _layer_index(text)
        if layer is not None:
            records[layer] = (
                str(plan.get("plan_status", "unknown")),
                validation_by_plan.get(str(plan.get("plan_id")), "missing"),
            )
    return records


def _attention_paths(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(path.get("layer_index", 0)): path for path in report.get("paths", [])}


def _coverage_records(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    records = {}
    for item in report.get("cases", []):
        case = item.get("case", {})
        records[(int(case.get("layer_index", 0)), str(case.get("pattern_kind", "")))] = item
    return records


def build_bert_24_plan_proof(
    plans: dict[str, Any],
    validations: dict[str, Any],
    value_paths: dict[str, Any],
    coverage: dict[str, Any],
    *,
    layers_total: int = 12,
    source_paths: dict[str, str] | None = None,
) -> Bert24PlanProof:
    """Join static symbolic plans with MLIR-backed attention value-path coverage."""
    ffn = _ffn_records(plans, validations)
    attention = _attention_paths(value_paths)
    coverage_records = _coverage_records(coverage)
    layers: list[BertLayerPlanProof] = []
    for layer in range(layers_total):
        ffn_status, validation_status = ffn.get(layer, ("missing", "missing"))
        ffn_verdict = "proven" if ffn_status == "ready_symbolic" and validation_status == "valid" else "missing"
        path = attention.get(layer, {})
        coverage_item = coverage_records.get((layer, "ATTENTION_VALUE_PATH"), {})
        coverage_verdict = str(coverage_item.get("verdict", "missing"))
        attention_verdict = "proven" if coverage_verdict in PROVEN_COVERAGE_VERDICTS else coverage_verdict
        layers.append(
            BertLayerPlanProof(
                layer,
                ffn_status,
                validation_status,
                "static_symbolic_plan",
                ffn_verdict,
                str(path.get("analysis_status", "missing")),
                str(path.get("axis_mapping", {}).get("mapping_status", "unproven")),
                str(coverage_item.get("evidence_tier", "unavailable")),
                attention_verdict,
                bool(coverage_item.get("dfa_ran", False)),
                tuple(coverage_item.get("warnings", [])),
            )
        )
    ffn_found = sum(layer.ffn_plan_status != "missing" for layer in layers)
    ffn_proven = sum(layer.ffn_verdict == "proven" for layer in layers)
    attention_found = sum(layer.attention_path_status != "missing" for layer in layers)
    attention_seedable = sum(layer.attention_path_status == "seedable" for layer in layers)
    attention_proven = sum(layer.attention_verdict == "proven" and layer.attention_dfa_ran for layer in layers)
    total_proven = ffn_proven + attention_proven
    missing = sum(layer.ffn_verdict == "missing" for layer in layers) + sum(layer.attention_verdict == "missing" for layer in layers)
    partial = sum(layer.attention_verdict in {"partial", "unknown"} for layer in layers)
    failed = sum(layer.attention_verdict == "failed" for layer in layers)
    complete = total_proven == layers_total * 2 and not missing and not partial and not failed
    return Bert24PlanProof(
        "bert-base-uncased",
        Bert24PlanSummary(
            layers_total,
            layers_total * 2,
            layers_total,
            ffn_found,
            ffn_proven,
            layers_total,
            attention_found,
            attention_seedable,
            attention_proven,
            total_proven,
            missing,
            partial,
            failed,
            "complete_24_plan_proof" if complete else "failed" if failed else "partial",
        ),
        tuple(layers),
        source_paths or {},
        tuple(dict.fromkeys(limitation for layer in layers for limitation in layer.limitations)),
    )


def run_bert_24_plan_proof(
    *,
    plans_path: str | Path = "reports/pruning_plans/bert-base-uncased.json",
    validations_path: str | Path = "reports/pruning_plan_validation/bert-base-uncased.json",
    value_paths_path: str | Path = "reports/attention_value_path_subgraphs/bert-base-uncased/summary.json",
    coverage_path: str | Path = "reports/mlir_evidence_coverage_bert_24_plan/index.json",
) -> Bert24PlanProof:
    paths = {
        "pruning_plans": str(plans_path),
        "pruning_plan_validation": str(validations_path),
        "attention_value_paths": str(value_paths_path),
        "mlir_evidence_coverage": str(coverage_path),
    }
    return build_bert_24_plan_proof(
        _load_json(plans_path),
        _load_json(validations_path),
        _load_json(value_paths_path),
        _load_json(coverage_path),
        source_paths=paths,
    )


def proof_to_dict(proof: Bert24PlanProof) -> dict[str, Any]:
    return asdict(proof)
