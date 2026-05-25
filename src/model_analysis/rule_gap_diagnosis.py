"""Diagnose cross-model rule gaps in the static pruning-analysis pipeline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir, safe_model_name


@dataclass
class RuleGap:
    gap_id: str
    gap_type: str
    severity: str
    affected_stage: str
    affected_count: int
    evidence: dict[str, Any]
    explanation: str


@dataclass
class RuleRepairSuggestion:
    repair_id: str
    repair_type: str
    target_models: list[str]
    expected_effect: str
    priority: str


@dataclass
class RuleGapDiagnosis:
    model_name: str
    generated_at: str
    final_status: str
    plan_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    detected_model_family: str
    gaps: list[RuleGap] = field(default_factory=list)
    candidate_repairs: list[RuleRepairSuggestion] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    conclusion: str = ""


def diagnosis_to_dict(value: RuleGapDiagnosis | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    data = asdict(value)
    data["gaps"] = [asdict(gap) for gap in value.gaps]
    data["candidate_repairs"] = [asdict(item) for item in value.candidate_repairs]
    return data


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sources(op_semantics: dict[str, Any]) -> list[str]:
    return [str(op.get("source_name", "")).lower().replace(".", "/") for op in op_semantics.get("ops", [])]


def detect_model_family(model_name: str, op_semantics: dict[str, Any]) -> str:
    text = " ".join([model_name.lower().replace(".", "/"), *_sources(op_semantics)])
    if "distilbert" in text or "/ffn/lin1/" in text:
        return "distilbert_encoder"
    if "opt" in text or "/decoder/layers/" in text:
        return "opt_decoder"
    if "gpt2" in text or "/mlp/c_fc/" in text:
        return "gpt2_decoder"
    if "vit" in text or "/mlp/fc1/" in text:
        return "vit_encoder"
    if "/encoder/layer/" in text or "/intermediate/dense/" in text:
        return "bert_encoder"
    return "unknown"


def _ffn_like_counts(op_semantics: dict[str, Any]) -> dict[str, int]:
    sources = _sources(op_semantics)
    return {
        "bert_dense": sum("/intermediate/dense/" in s or "/output/dense/" in s for s in sources),
        "distilbert_lin": sum("/ffn/lin1/" in s or "/ffn/lin2/" in s for s in sources),
        "opt_fc": sum("/fc1/" in s or "/fc2/" in s for s in sources),
        "vit_mlp": sum("/mlp/fc1/" in s or "/mlp/fc2/" in s for s in sources),
        "gpt2_mlp": sum("/mlp/c_fc/" in s or "/mlp/c_proj/" in s for s in sources),
    }


def _gap(index: int, gap_type: str, severity: str, stage: str, count: int, evidence: dict[str, Any], explanation: str) -> RuleGap:
    return RuleGap(f"rule_gap::{index:03d}", gap_type, severity, stage, count, evidence, explanation)


def _suggest(index: int, repair_type: str, model: str, expected: str, priority: str) -> RuleRepairSuggestion:
    return RuleRepairSuggestion(f"rule_repair::{index:03d}", repair_type, [model], expected, priority)


def diagnose_rule_gaps_for_model(root: Path, model_name: str) -> RuleGapDiagnosis:
    safe = safe_model_name(model_name)
    status = _load(root / "reports" / "static_pipeline_status" / f"{safe}.json")
    op_sem = _load(root / "reports" / "op_semantics" / f"{safe}.json")
    ranking = _load(root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json")
    plans = _load(root / "reports" / "pruning_plans" / f"{safe}.json")
    validations = _load(root / "reports" / "pruning_plan_validation" / f"{safe}.json")
    family = detect_model_family(model_name, op_sem)
    plan_summary = plans.get("summary", {})
    validation_summary = validations.get("summary", {})
    ranking_summary = ranking.get("summary", {})
    ffn_counts = _ffn_like_counts(op_sem)
    gaps: list[RuleGap] = []
    repairs: list[RuleRepairSuggestion] = []
    idx = 0
    total_plans = int(plan_summary.get("total_plans", 0) or 0)
    incomplete = int(plan_summary.get("incomplete", 0) or 0)
    invalid = int(validation_summary.get("invalid_plans", 0) or 0)
    safe_candidates = int(ranking_summary.get("safe_candidates", 0) or 0)
    if total_plans and incomplete == total_plans:
        gaps.append(_gap(idx, "missing_ffn_evidence_binding", "blocker", "plan_synthesis", total_plans, {"missing_evidence_counts": plan_summary.get("missing_evidence_counts", {})}, "Plans exist but all are incomplete, indicating candidate-to-primitive FFN evidence binding is missing or too family-specific."))
        repairs.append(_suggest(idx, "add_generic_ffn_evidence_matcher", model_name, "Bind expansion, activation, and contraction evidence across model-family naming schemes.", "high"))
        idx += 1
    if total_plans and invalid == total_plans:
        gaps.append(_gap(idx, "validation_policy_too_bert_specific", "blocker", "plan_validation", total_plans, {"failed_checks_by_type": validation_summary.get("failed_checks_by_type", {})}, "Plans exist but validation rejects all of them; validation may be too BERT-path-specific or plans lack required actions."))
        repairs.append(_suggest(idx, "add_family_specific_ffn_patterns", model_name, "Allow generic FFN action/validation roles rather than BERT-only paths.", "high"))
        idx += 1
    if safe_candidates == 0 and any(ffn_counts.values()):
        gaps.append(_gap(idx, "missing_feedforward_fusion", "warning", "ranking", sum(ffn_counts.values()), {"ffn_like_ops": ffn_counts}, "FFN-like primitive ops exist but no safe candidates were ranked. Region fusion or region semantics likely needs model-family rules."))
        repairs.append(_suggest(idx, "add_family_specific_ffn_patterns", model_name, "Recover feed-forward regions for this model family.", "medium"))
        idx += 1
    if status and any(stage.get("stage_name") == "layer_subgraph_validation" and stage.get("status") == "skipped" for stage in status.get("stages", [])):
        gaps.append(_gap(idx, "missing_layer_grouping", "warning", "layer_subgraph_validation", 1, {"status": "skipped"}, "Base artifacts exist but layer/subgraph report support did not recover layer structure."))
        repairs.append(_suggest(idx, "add_decoder_layer_grouping" if family in {"opt_decoder", "gpt2_decoder"} else "add_vit_mlp_grouping", model_name, "Build learner-facing layer/section packs for this family.", "medium"))
        idx += 1
    unknown_ops = int(op_sem.get("summary", {}).get("unknown_ops", 0) or 0)
    if unknown_ops > 0:
        gaps.append(_gap(idx, "missing_op_semantics", "info", "op_semantics", unknown_ops, {"unknown_ops": unknown_ops}, "Some primitive ops remain unknown and may need future local semantics rules."))
    conclusion = "No blocking rule gaps detected." if not gaps else f"Detected {len(gaps)} rule-gap categories for {model_name}."
    return RuleGapDiagnosis(
        model_name=model_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        final_status=status.get("final_status", "unknown"),
        plan_summary=plan_summary,
        validation_summary=validation_summary,
        detected_model_family=family,
        gaps=gaps,
        candidate_repairs=repairs,
        evidence_summary={
            "ranking_summary": ranking_summary,
            "ffn_like_op_counts": ffn_counts,
            "unknown_ops": unknown_ops,
        },
        conclusion=conclusion,
    )


def write_rule_gap_diagnosis(value: RuleGapDiagnosis | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(diagnosis_to_dict(value), indent=2), encoding="utf-8")


def compare_rule_gap_diagnoses(diagnoses: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(item.get("detected_model_family", "unknown") for item in diagnoses)
    gap_counts = Counter(gap.get("gap_type", "unknown") for item in diagnoses for gap in item.get("gaps", []))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": [item.get("model_name") for item in diagnoses],
        "family_counts": dict(sorted(family_counts.items())),
        "gap_type_counts": dict(sorted(gap_counts.items())),
        "diagnoses": diagnoses,
    }

