"""Rank static pruning opportunities from region and op semantics."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class PruningOpportunityCandidate:
    candidate_id: str
    model_name: str
    region_id: str
    region_name: str
    source_region_type: str
    semantic_category: str
    section: str
    op_range: str
    candidate_kind: str
    target_dimension: str
    target_axis: str
    pruning_class: str
    rank_score: int
    confidence: str
    reason: str
    required_repairs: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    propagation_rules: list[dict[str, Any]] = field(default_factory=list)
    op_semantics_evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PruningOpportunityRanking:
    model_name: str
    generated_at: str
    source_region_pruning_semantics_path: str
    source_op_semantics_path: str | None
    candidates: list[PruningOpportunityCandidate]
    summary: dict[str, Any]


def pruning_opportunity_candidate_to_dict(value: PruningOpportunityCandidate) -> dict[str, Any]:
    return asdict(value)


def pruning_opportunity_ranking_to_dict(value: PruningOpportunityRanking) -> dict[str, Any]:
    return asdict(value)


def write_pruning_opportunity_ranking_json(value: PruningOpportunityRanking | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    data = pruning_opportunity_ranking_to_dict(value) if isinstance(value, PruningOpportunityRanking) else value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_pruning_opportunity_ranking_json(path: Path) -> PruningOpportunityRanking:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PruningOpportunityRanking(
        model_name=data.get("model_name", "model"),
        generated_at=data.get("generated_at", ""),
        source_region_pruning_semantics_path=data.get("source_region_pruning_semantics_path", ""),
        source_op_semantics_path=data.get("source_op_semantics_path"),
        candidates=[PruningOpportunityCandidate(**item) for item in data.get("candidates", [])],
        summary=data.get("summary", {}),
    )


def _compact_id(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "candidate"


def _repair_summaries(region: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "obligation_type": item.get("obligation_type"),
            "required": item.get("required"),
            "affected_dimensions": item.get("affected_dimensions", []),
            "explanation": item.get("explanation", ""),
        }
        for item in region.get("repair_obligations", [])
    ]


def _blocker_summaries(region: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "blocker_type": item.get("blocker_type"),
            "severity": item.get("severity"),
            "explanation": item.get("explanation", ""),
        }
        for item in region.get("blockers", [])
    ]


def _rule_summaries(region: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rule_type": item.get("rule_type"),
            "source_dimension": item.get("source_dimension"),
            "target_dimensions": item.get("target_dimensions", []),
            "index_mapping": item.get("index_mapping"),
            "explanation": item.get("explanation", ""),
        }
        for item in region.get("propagation_rules", [])
    ]


def _has_dim(region: dict[str, Any], dim_name: str, status: str | None = None) -> bool:
    for dim in region.get("dimensions", []):
        if dim.get("dim_name") == dim_name or dim.get("symbolic_role") == dim_name:
            if status is None or dim.get("status") == status:
                return True
    return False


def _has_repair(region: dict[str, Any], repair_type: str) -> bool:
    return any(item.get("obligation_type") == repair_type for item in region.get("repair_obligations", []))


def _has_blocker(region: dict[str, Any], blocker_type: str) -> bool:
    return any(item.get("blocker_type") == blocker_type for item in region.get("blockers", []))


def _op_evidence_map(op_semantics: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not op_semantics:
        return {}
    return {item.get("op_id", ""): item for item in op_semantics.get("ops", [])}


def _op_evidence(region: dict[str, Any], by_op: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for op_id in region.get("evidence", {}).get("source_ops", []):
        op = by_op.get(op_id)
        if not op:
            continue
        effect = op.get("pruning_effect", {})
        out.append(
            {
                "source_name": op.get("source_name"),
                "op_type": op.get("op_type"),
                "semantic_kind": op.get("semantic_kind"),
                "semantic_category": op.get("semantic_category"),
                "parameterized": op.get("parameterized"),
                "direct_pruning": effect.get("direct_pruning"),
            }
        )
    return out


def _confidence_with_op_evidence(base: str, evidence: list[dict[str, Any]], warnings: list[str]) -> str:
    if not evidence:
        warnings.append("missing_op_semantics_evidence")
        return "medium" if base == "high" else "low"
    if any(item.get("semantic_kind") == "unknown" for item in evidence):
        warnings.append("unknown_op_semantics_present")
        return "medium" if base == "high" else base
    return base


def _disagreement(region: dict[str, Any], evidence: list[dict[str, Any]], warnings: list[str]) -> bool:
    if region.get("pruning_role") != "directly_prunable":
        return False
    if any(item.get("direct_pruning") == "blocked" for item in evidence):
        warnings.append("op_region_semantics_disagreement")
        return True
    return False


def _base_candidate(
    model_name: str,
    region: dict[str, Any],
    index: int,
    *,
    candidate_kind: str,
    target_dimension: str,
    target_axis: str,
    pruning_class: str,
    rank_score: int,
    confidence: str,
    reason: str,
    op_semantics_evidence: list[dict[str, Any]],
    warnings: list[str],
) -> PruningOpportunityCandidate:
    return PruningOpportunityCandidate(
        candidate_id=f"prune_candidate::{index:06d}::{_compact_id(region.get('region_id', region.get('region_name', 'region')))}",
        model_name=model_name,
        region_id=region.get("region_id", ""),
        region_name=region.get("region_name", region.get("region_id", "")),
        source_region_type=region.get("source_region_type", region.get("region_type", "UnknownRegion")),
        semantic_category=region.get("semantic_category", "unknown"),
        section=region.get("section", ""),
        op_range=region.get("op_range", ""),
        candidate_kind=candidate_kind,
        target_dimension=target_dimension,
        target_axis=target_axis,
        pruning_class=pruning_class,
        rank_score=rank_score,
        confidence=confidence,
        reason=reason,
        required_repairs=_repair_summaries(region),
        blockers=_blocker_summaries(region),
        propagation_rules=_rule_summaries(region),
        op_semantics_evidence=op_semantics_evidence,
        warnings=warnings,
    )


def _classify_region(region: dict[str, Any], evidence: list[dict[str, Any]]) -> tuple[str, str, str, str, int, str, str, list[str]]:
    category = region.get("semantic_category", "unknown")
    role = region.get("pruning_role", "unknown")
    warnings: list[str] = []

    if category == "feed_forward_block" and role == "directly_prunable" and _has_dim(region, "intermediate_dim", "prunable") and _has_repair(region, "same_indices_across_mlp") and _has_repair(region, "prune_consumer_input") and not region.get("blockers"):
        confidence = _confidence_with_op_evidence("high", evidence, warnings)
        kinds = {item.get("semantic_kind") for item in evidence}
        if evidence and not ({"parameterized_linear_matmul", "gelu_erf"} & kinds and "parameterized_linear_matmul" in kinds):
            warnings.append("incomplete_feedforward_op_semantics_evidence")
            confidence = "medium"
        return (
            "feedforward_intermediate_pruning",
            "intermediate_dim",
            "producer_output",
            "safe",
            95,
            confidence,
            "FFN intermediate_dim pruning is structurally local: intermediate projection output, GELU, and FFN output input must use the same indices.",
            warnings,
        )
    if category == "ffn_intermediate_projection" and role == "directly_prunable":
        confidence = _confidence_with_op_evidence("high", evidence, warnings)
        pruning_class = "safe"
        score = 85
        if _disagreement(region, evidence, warnings):
            pruning_class = "constrained"
            score = 55
        return (
            "projection_output_pruning",
            "intermediate_dim",
            "producer_output",
            pruning_class,
            score,
            confidence,
            "Learned FFN intermediate projection output axis is prunable; the enclosing FeedForwardRegion should coordinate full repair.",
            warnings,
        )
    if category == "ffn_output_projection":
        return (
            "projection_input_repair",
            "intermediate_dim",
            "consumer_input",
            "constrained",
            50,
            _confidence_with_op_evidence("medium", evidence, warnings),
            "FFN output projection input columns are repaired as a consequence of pruning FFN intermediate_dim; output hidden_dim remains protected.",
            warnings,
        )
    if category in {"query_projection", "key_projection", "value_projection"}:
        return (
            "attention_projection_constrained_pruning",
            "head_dim",
            "producer_output",
            "constrained",
            55,
            _confidence_with_op_evidence("medium", evidence, warnings),
            "Learned Q/K/V projection exposes output pruning axes, but pruning must respect num_heads/head_dim reshape-transpose mapping.",
            warnings,
        )
    if category == "attention_output_projection":
        return (
            "attention_projection_constrained_pruning",
            "hidden_dim",
            "consumer_input",
            "constrained",
            45,
            _confidence_with_op_evidence("medium", evidence, warnings),
            "Attention output projection input depends on context/head-axis mapping and its output hidden_dim feeds a residual path.",
            warnings,
        )
    if category in {"attention_score_matmul", "attention_context_matmul"}:
        return (
            "attention_contraction_blocked",
            "head_dim",
            "unknown",
            "blocked",
            10,
            "high",
            "Attention score/context MatMul is a dataflow contraction, not a learned parameter projection; it has no independent pruning axis.",
            warnings,
        )
    if category in {"attention_mask_add", "attention_mask_axis_transform", "attention_mask_join", "attention_mask_fork", "auxiliary_attention_mask_flow", "shape_axis_transform", "shape_motif"}:
        return (
            "auxiliary_metadata_flow",
            "symbolic_axis" if category not in {"attention_mask_add"} else "mask_dim",
            "metadata_axis",
            "auxiliary",
            10 if category == "attention_mask_add" else 5,
            "medium",
            "Mask, shape, or axis flow carries metadata and may require shape updates, but it is not directly prunable.",
            warnings,
        )
    if category == "residual_merge":
        return (
            "residual_hidden_blocked",
            "hidden_dim",
            "branch_hidden",
            "blocked",
            5,
            "high",
            "Residual branches require hidden_dim agreement; hidden pruning is blocked unless all branches are jointly repaired.",
            warnings,
        )
    if category == "layer_norm":
        return (
            "layernorm_hidden_blocked",
            "hidden_dim",
            "branch_hidden",
            "blocked",
            5,
            "high",
            "LayerNorm hidden dimension is protected; gamma/beta repair would be required under hidden-width pruning.",
            warnings,
        )
    if category in {"embedding_lookup", "embedding_add"}:
        pruning_class = "blocked" if category == "embedding_lookup" else "auxiliary"
        return (
            "residual_hidden_blocked" if category == "embedding_add" else "unknown",
            "vocab_dim" if category == "embedding_lookup" else "hidden_dim",
            "unknown",
            pruning_class,
            10,
            "medium",
            "Embedding vocabulary, position, token-type, and hidden dimensions are protected by default.",
            warnings,
        )
    if category == "unknown" or role == "unknown":
        return (
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            20,
            "low",
            "Insufficient semantic evidence to rank this region confidently.",
            warnings,
        )
    if region.get("blockers"):
        return (
            "unknown",
            "unknown",
            "unknown",
            "blocked",
            20,
            "medium",
            "Region carries blockers but has no specialized ranking rule.",
            warnings,
        )
    return (
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        25,
        "low",
        "No specialized ranking rule matched this region.",
        warnings,
    )


def build_pruning_opportunity_ranking(
    region_pruning_semantics: dict[str, Any],
    *,
    op_semantics: dict[str, Any] | None = None,
    source_region_pruning_semantics_path: str = "",
    source_op_semantics_path: str | None = None,
) -> PruningOpportunityRanking:
    model_name = region_pruning_semantics.get("model_name", "model")
    by_op = _op_evidence_map(op_semantics)
    candidates: list[PruningOpportunityCandidate] = []
    for index, region in enumerate(region_pruning_semantics.get("regions", [])):
        evidence = _op_evidence(region, by_op)
        candidate_kind, target_dimension, target_axis, pruning_class, score, confidence, reason, warnings = _classify_region(region, evidence)
        candidates.append(
            _base_candidate(
                model_name,
                region,
                index,
                candidate_kind=candidate_kind,
                target_dimension=target_dimension,
                target_axis=target_axis,
                pruning_class=pruning_class,
                rank_score=score,
                confidence=confidence,
                reason=reason,
                op_semantics_evidence=evidence,
                warnings=warnings,
            )
        )
    candidates.sort(key=lambda item: (-item.rank_score, item.pruning_class, item.region_name, item.candidate_id))
    return PruningOpportunityRanking(
        model_name=model_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_region_pruning_semantics_path=source_region_pruning_semantics_path,
        source_op_semantics_path=source_op_semantics_path,
        candidates=candidates,
        summary=_summary(candidates),
    )


def _summary(candidates: list[PruningOpportunityCandidate]) -> dict[str, Any]:
    class_counts = Counter(item.pruning_class for item in candidates)
    kind_counts = Counter(item.candidate_kind for item in candidates)
    category_counts = Counter(item.semantic_category for item in candidates)
    return {
        "total_candidates": len(candidates),
        "safe_candidates": class_counts.get("safe", 0),
        "constrained_candidates": class_counts.get("constrained", 0),
        "blocked_candidates": class_counts.get("blocked", 0),
        "auxiliary_candidates": class_counts.get("auxiliary", 0),
        "unknown_candidates": class_counts.get("unknown", 0),
        "mlp_safe_candidates": sum(1 for item in candidates if item.candidate_kind == "feedforward_intermediate_pruning" and item.pruning_class == "safe"),
        "attention_constrained_candidates": sum(1 for item in candidates if item.candidate_kind == "attention_projection_constrained_pruning"),
        "residual_blocked_candidates": sum(1 for item in candidates if item.candidate_kind == "residual_hidden_blocked" and item.pruning_class == "blocked"),
        "layernorm_blocked_candidates": sum(1 for item in candidates if item.candidate_kind == "layernorm_hidden_blocked"),
        "unknown_op_semantics_candidates": sum(1 for item in candidates if "unknown_op_semantics_present" in item.warnings or "missing_op_semantics_evidence" in item.warnings),
        "candidate_kind_counts": dict(sorted(kind_counts.items())),
        "semantic_category_counts": dict(sorted(category_counts.items())),
        "pruning_class_counts": dict(sorted(class_counts.items())),
    }


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 80) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    if len(rows) > limit:
        lines.append("| ... | " + f"{len(rows) - limit} more rows omitted" + " |" * (len(columns) - 2))
    return "\n".join(lines)


def _row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": item.get("region_name"),
        "class": item.get("pruning_class"),
        "score": item.get("rank_score"),
        "confidence": item.get("confidence"),
        "kind": item.get("candidate_kind"),
        "target": item.get("target_dimension"),
        "repairs": ", ".join(r.get("obligation_type", "") for r in item.get("required_repairs", [])),
        "blockers": ", ".join(b.get("blocker_type", "") for b in item.get("blockers", [])),
        "reason": item.get("reason", ""),
    }


def _count_rows(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [{"item": key, "count": value} for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def pruning_opportunity_ranking_to_markdown(value: PruningOpportunityRanking | dict[str, Any], *, include_auxiliary_details: bool = False) -> str:
    data = pruning_opportunity_ranking_to_dict(value) if isinstance(value, PruningOpportunityRanking) else value
    summary = data.get("summary", {})
    candidates = data.get("candidates", [])
    safe = [item for item in candidates if item.get("pruning_class") == "safe"]
    constrained = [item for item in candidates if item.get("pruning_class") == "constrained"]
    blocked = [item for item in candidates if item.get("pruning_class") == "blocked"]
    auxiliary = [item for item in candidates if item.get("pruning_class") == "auxiliary"]
    unknown = [item for item in candidates if item.get("pruning_class") == "unknown"]
    lines = [
        f"# Pruning Opportunity Ranking: {data.get('model_name', '')}",
        "",
        "## Summary",
        "",
        f"- Total candidates: `{summary.get('total_candidates', 0)}`",
        f"- Safe: `{summary.get('safe_candidates', 0)}`",
        f"- Constrained: `{summary.get('constrained_candidates', 0)}`",
        f"- Blocked: `{summary.get('blocked_candidates', 0)}`",
        f"- Auxiliary: `{summary.get('auxiliary_candidates', 0)}`",
        f"- Unknown: `{summary.get('unknown_candidates', 0)}`",
        f"- Safe MLP opportunities: `{summary.get('mlp_safe_candidates', 0)}`",
        "",
        "## Candidate Kinds",
        "",
        _table(_count_rows(summary.get("candidate_kind_counts", {})), ["item", "count"], limit=80),
        "",
        "## Top Safe Candidates",
        "",
        _table([_row(item) for item in safe], ["candidate", "class", "score", "confidence", "target", "repairs", "reason"], limit=60),
        "",
        "## Constrained Candidates",
        "",
        _table([_row(item) for item in constrained], ["candidate", "score", "confidence", "kind", "target", "blockers", "reason"], limit=80),
        "",
        "## Blocked Candidates",
        "",
        _table([_row(item) for item in blocked], ["candidate", "score", "kind", "target", "blockers", "reason"], limit=80),
        "",
        "## Auxiliary Metadata Flow",
        "",
        _table(_count_rows(Counter(item.get("semantic_category", "unknown") for item in auxiliary)), ["item", "count"], limit=40),
        "",
    ]
    if include_auxiliary_details:
        lines.extend(["### Auxiliary Details", "", _table([_row(item) for item in auxiliary], ["candidate", "score", "kind", "target", "reason"], limit=300), ""])
    lines.extend(
        [
            "## Unknown / Needs Review",
            "",
            _table([_row(item) for item in unknown], ["candidate", "score", "confidence", "kind", "reason"], limit=80),
            "",
            "## Candidate Details",
            "",
        ]
    )
    for item in candidates[:200]:
        lines.extend(
            [
                f"### {item.get('region_name')}",
                "",
                f"- Class: `{item.get('pruning_class')}`",
                f"- Score: `{item.get('rank_score')}`",
                f"- Kind: `{item.get('candidate_kind')}`",
                f"- Semantic category: `{item.get('semantic_category')}`",
                f"- Target dimension: `{item.get('target_dimension')}`",
                f"- Reason: {item.get('reason')}",
                f"- Warnings: `{', '.join(item.get('warnings', [])) or 'none'}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "This ranking prioritizes static pruning candidates from Region Pruning Semantics and Op Semantics. It does not execute pruning or modify models.",
            "",
        ]
    )
    return "\n".join(lines)

