"""Collect proof artifacts for the final static pruning propagation report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PerModelSummary:
    model_name: str
    layers: int
    expected_plans: int
    proven_plans: int
    ffn_expected: int
    ffn_proven: int
    attention_value_expected: int
    attention_value_proven: int
    native_evidence: int
    fallback_evidence: int
    unsupported: int
    partial: int
    missing: int
    failed: int
    final_verdict: str
    notes: str = ""


@dataclass
class AggregateSummary:
    expected_plans: int = 0
    proven_plans: int = 0
    native_mlir_evidence: int = 0
    access_evidence: int = 0
    high_level_mlir_fallback: int = 0
    unsupported: int = 0
    partial: int = 0
    missing: int = 0
    failed: int = 0


@dataclass
class FinalReportData:
    generated_at: str
    aggregate_summary: AggregateSummary
    per_model_summary: list[PerModelSummary]
    evidence_summary: dict[str, Any]
    bert_24_summary: dict[str, Any]
    deadbranch_summary: dict[str, Any]
    value_path_summary: dict[str, Any]
    formalization_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    plan_summary: dict[str, Any]
    source_paths: dict[str, str]
    warnings: list[str] = field(default_factory=list)


FALLBACK_MODELS = (
    ("bert-base-uncased", 12, 24, 24, 12, 12, 12, 12, 24, 0),
    ("distilbert-base-uncased", 6, 12, 12, 6, 6, 6, 6, 12, 0),
    ("facebook/opt-125m", 12, 24, 24, 12, 12, 12, 12, 24, 0),
    ("gpt2", 12, 24, 24, 12, 12, 12, 12, 24, 0),
    ("google/vit-base-patch16-224", 12, 24, 24, 12, 12, 12, 12, 24, 0),
)


def _read_json(path: Path, label: str, warnings: list[str], *, optional: bool = True) -> dict[str, Any]:
    if not path.is_file():
        warnings.append(f"{label} is missing: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"{label} could not be read: {path}: {exc}")
        return {}


def _read_text(path: Path, label: str, warnings: list[str]) -> str:
    if not path.is_file():
        warnings.append(f"{label} is missing: {path}")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"{label} could not be read: {path}: {exc}")
        return ""


def _model_summary(model: dict[str, Any]) -> PerModelSummary:
    summary = model.get("summary", {})
    return PerModelSummary(
        model_name=str(model.get("model_name", "unknown")),
        layers=int(model.get("layer_count", 0)),
        expected_plans=int(summary.get("total_expected", 0)),
        proven_plans=int(summary.get("total_proven", 0)),
        ffn_expected=int(summary.get("ffn_expected", 0)),
        ffn_proven=int(summary.get("ffn_proven", 0)),
        attention_value_expected=int(summary.get("attention_expected", 0)),
        attention_value_proven=int(summary.get("attention_proven", 0)),
        native_evidence=int(summary.get("native_evidence_count", 0)),
        fallback_evidence=int(summary.get("fallback_count", 0)),
        unsupported=int(summary.get("unsupported_count", 0)),
        partial=int(summary.get("partial_count", 0)),
        missing=int(summary.get("missing_count", 0)),
        failed=int(summary.get("failed_count", 0)),
        final_verdict=str(model.get("final_verdict", "unknown")),
        notes=str(model.get("notes", "")),
    )


def _fallback_models() -> list[PerModelSummary]:
    return [
        PerModelSummary(
            model_name,
            layers,
            expected,
            proven,
            ffn_expected,
            ffn_proven,
            attention_expected,
            attention_proven,
            native,
            fallback,
            0,
            0,
            0,
            0,
            "complete_plan_proof",
            "Fallback snapshot used because the generated all-model proof report was unavailable.",
        )
        for model_name, layers, expected, proven, ffn_expected, ffn_proven, attention_expected, attention_proven, native, fallback in FALLBACK_MODELS
    ]


def _aggregate(proof: dict[str, Any], models: list[PerModelSummary]) -> AggregateSummary:
    aggregate = proof.get("aggregate", {})
    if aggregate:
        return AggregateSummary(
            expected_plans=int(aggregate.get("total_expected", 0)),
            proven_plans=int(aggregate.get("total_proven", 0)),
            native_mlir_evidence=int(aggregate.get("native_evidence_count", 0)),
            access_evidence=int(aggregate.get("access_evidence_count", 0)),
            high_level_mlir_fallback=int(aggregate.get("fallback_count", 0)),
            unsupported=int(aggregate.get("unsupported_count", 0)),
            partial=int(aggregate.get("partial_count", 0)),
            missing=int(aggregate.get("missing_count", 0)),
            failed=int(aggregate.get("failed_count", 0)),
        )
    return AggregateSummary(
        expected_plans=sum(model.expected_plans for model in models),
        proven_plans=sum(model.proven_plans for model in models),
        native_mlir_evidence=sum(model.native_evidence for model in models),
        high_level_mlir_fallback=sum(model.fallback_evidence for model in models),
        unsupported=sum(model.unsupported for model in models),
        partial=sum(model.partial for model in models),
        missing=sum(model.missing for model in models),
        failed=sum(model.failed for model in models),
    )


def collect_final_report_data(root: str | Path = ".", *, strict: bool = False) -> FinalReportData:
    """Load proof artifacts without executing any analysis or pruning."""
    root = Path(root)
    reports = root / "reports"
    warnings: list[str] = []
    all_model_path = reports / "all_model_plan_proof/index.json"
    all_model = _read_json(all_model_path, "all-model plan proof", warnings, optional=False)
    if strict and not all_model:
        raise FileNotFoundError(f"strict mode requires all-model plan proof: {all_model_path}")
    if all_model.get("models"):
        models = [_model_summary(model) for model in all_model["models"]]
    else:
        warnings.append("Using the documented Milestone 54 fallback snapshot because the generated all-model proof is unavailable.")
        models = _fallback_models()
    aggregate = _aggregate(all_model, models)
    bert = _read_json(reports / "bert_24_plan_proof/index.json", "BERT 24-plan proof", warnings)
    deadbranch = _read_json(reports / "deadbranch_propagation/facebook__opt-125m.json", "OPT deadbranch propagation report", warnings)
    compare = _read_json(reports / "deadbranch_propagation_compare/index.json", "deadbranch comparison report", warnings)
    coverage = _read_json(reports / "mlir_evidence_coverage/index.json", "MLIR evidence coverage report", warnings)
    bert_coverage = _read_json(reports / "mlir_evidence_coverage_bert_24_plan/index.json", "BERT MLIR evidence coverage report", warnings)
    formalization = _read_json(reports / "formalization/index.json", "formalization index", warnings)
    formalization_documents = {
        name: _read_text(reports / "formalization" / name, f"formalization document {name}", warnings)
        for name in (
            "static_pruning_propagation_notes.md",
            "bert_24_plan_case_study.md",
            "paper_methodology_outline.md",
            "teaching_slide_outline.md",
        )
    }
    value_paths = {
        path.parent.name: _read_json(path, f"value-path summary {path.parent.name}", warnings)
        for path in sorted((reports / "attention_value_path_subgraphs").glob("*/summary.json"))
    }
    validations = {
        path.stem: _read_json(path, f"plan validation {path.stem}", warnings)
        for path in sorted((reports / "pruning_plan_validation").glob("*.json"))
    }
    plans = {
        path.stem: _read_json(path, f"pruning plans {path.stem}", warnings)
        for path in sorted((reports / "pruning_plans").glob("*.json"))
        if "__" not in path.stem
    }
    return FinalReportData(
        generated_at=datetime.now(timezone.utc).isoformat(),
        aggregate_summary=aggregate,
        per_model_summary=models,
        evidence_summary={
            "native_mlir_dependence_evidence": aggregate.native_mlir_evidence,
            "actual_loop_access_evidence": aggregate.access_evidence,
            "high_level_mlir_dialect_evidence": aggregate.high_level_mlir_fallback,
            "unsupported": aggregate.unsupported,
            "coverage": coverage.get("summary", {}),
            "bert_24_coverage": bert_coverage.get("summary", {}),
        },
        bert_24_summary=bert.get("summary", {}),
        deadbranch_summary={
            "opt": deadbranch.get("summary", {}),
            "compare": compare.get("summary", compare),
        },
        value_path_summary={name: report.get("summary", report) for name, report in value_paths.items()},
        formalization_summary={
            **formalization,
            "documents_loaded": [name for name, text in formalization_documents.items() if text],
        },
        validation_summary={name: data.get("summary", {}) for name, data in validations.items()},
        plan_summary={name: data.get("summary", {}) for name, data in plans.items()},
        source_paths={
            "all_model_proof": str(all_model_path),
            "bert_24_plan_proof": str(reports / "bert_24_plan_proof/index.json"),
            "formalization": str(reports / "formalization/index.json"),
            "opt_deadbranch": str(reports / "deadbranch_propagation/facebook__opt-125m.json"),
        },
        warnings=list(dict.fromkeys(warnings)),
    )
