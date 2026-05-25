#!/usr/bin/env python3
"""Export a compact uploadable snapshot of the static pruning analysis.

This script summarizes existing generated artifacts only. It does not parse
ONNX, regenerate upstream reports, execute pruning, or mutate model files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_OP_KINDS = [
    "parameterized_linear_matmul",
    "attention_score_matmul",
    "attention_context_matmul",
    "attention_mask_add",
    "residual_add",
    "layernorm",
    "unknown",
]

KEY_REGION_CATEGORIES = [
    "feed_forward_block",
    "ffn_intermediate_projection",
    "query_projection",
    "key_projection",
    "value_projection",
    "attention_score_matmul",
    "attention_context_matmul",
    "residual_merge",
    "layer_norm",
]

RANKING_SUMMARY_KEYS = [
    "total_candidates",
    "safe_candidates",
    "constrained_candidates",
    "blocked_candidates",
    "auxiliary_candidates",
    "unknown_candidates",
    "mlp_safe_candidates",
]

PLAN_SUMMARY_KEYS = [
    "total_plans",
    "ready_symbolic",
    "incomplete",
    "blocked",
    "unknown",
]

LAYER_PACK_SUMMARY_KEYS = [
    "total_subgraphs",
    "onnx_exported",
    "onnx_skipped",
    "onnx_failed",
    "safe_subgraphs",
    "constrained_subgraphs",
    "blocked_subgraphs",
    "auxiliary_subgraphs",
    "unknown_subgraphs",
    "valid_plan_subgraphs",
]


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def repo_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_load_json(
    path: Path,
    missing_artifacts: list[dict[str, str]],
    label: str,
    *,
    required: bool = False,
) -> Any | None:
    if not path.exists():
        item = {"label": label, "path": str(path)}
        missing_artifacts.append(item)
        if required:
            raise FileNotFoundError(f"Required artifact missing: {label}: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        item = {"label": label, "path": str(path), "error": str(exc)}
        missing_artifacts.append(item)
        if required:
            raise
        return None


def trim_dict(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: data.get(key, 0) for key in keys}


def count_from_summary(data: dict[str, Any] | None, field: str, keys: list[str]) -> dict[str, Any]:
    if not data:
        return {}
    counts = data.get("summary", {}).get(field, {})
    if not isinstance(counts, dict):
        return {}
    return {key: counts.get(key, 0) for key in keys}


def normalize_validation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    status_counts = summary.get("validation_status_counts", {})
    if not isinstance(status_counts, dict):
        status_counts = {}
    return {
        "total_validations": summary.get("total_validations", summary.get("total_plans", 0)),
        "valid": summary.get("valid", summary.get("valid_plans", status_counts.get("valid", 0))),
        "warning": summary.get(
            "warning", summary.get("warning_plans", status_counts.get("warning", 0))
        ),
        "invalid": summary.get(
            "invalid", summary.get("invalid_plans", status_counts.get("invalid", 0))
        ),
        "unknown": summary.get(
            "unknown", summary.get("unknown_plans", status_counts.get("unknown", 0))
        ),
    }


def stage_presence(path: Path) -> dict[str, Any]:
    return {"present": path.exists(), "path": str(path)}


def artifact_file_map(root: Path, model: str, layer: int) -> dict[str, Path]:
    safe_model = safe_model_name(model)
    layer_dir = root / "reports" / "layer_subgraph_validation" / safe_model / f"layer_{layer}"
    return {
        "tensor_ir": root / "reports" / "tensor_ir" / f"{safe_model}.json",
        "op_semantics": root / "reports" / "op_semantics" / f"{safe_model}.json",
        "region_pruning_semantics": root
        / "reports"
        / "region_pruning_semantics"
        / f"{safe_model}.json",
        "ranking": root / "reports" / "pruning_opportunity_rankings" / f"{safe_model}.json",
        "plans": root / "reports" / "pruning_plans" / f"{safe_model}.json",
        "plan_validation": root / "reports" / "pruning_plan_validation" / f"{safe_model}.json",
        "layer_pack_index": layer_dir / "index.json",
        "abstract_expansion": root
        / "reports"
        / "abstract_node_expansions"
        / safe_model
        / "abstract_node_expansions_main.json",
        "structural_region_tree": root / "reports" / "structural_region_trees" / f"{safe_model}.json",
        "region_dimension_ir": root / "reports" / "region_dimension_ir" / f"{safe_model}.json",
    }


def summarize_pipeline(
    root: Path,
    paths: dict[str, Path],
    missing_artifacts: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    op_semantics = safe_load_json(paths["op_semantics"], missing_artifacts, "op_semantics")
    region_semantics = safe_load_json(
        paths["region_pruning_semantics"], missing_artifacts, "region_pruning_semantics"
    )
    ranking = safe_load_json(paths["ranking"], missing_artifacts, "pruning_opportunity_ranking")
    plans = safe_load_json(paths["plans"], missing_artifacts, "pruning_plans")
    validation = safe_load_json(paths["plan_validation"], missing_artifacts, "pruning_plan_validation")
    layer_pack = safe_load_json(
        paths["layer_pack_index"], missing_artifacts, "layer_subgraph_pack", required=True
    )

    optional_labels = [
        ("abstract_expansion", "abstract_node_expansion"),
        ("structural_region_tree", "structural_region_tree"),
        ("region_dimension_ir", "region_dimension_ir"),
    ]
    for key, label in optional_labels:
        safe_load_json(paths[key], missing_artifacts, label)

    summary = {
        "tensor_ir": {
            **stage_presence(paths["tensor_ir"]),
            "path": repo_path(root, paths["tensor_ir"]),
        },
        "op_semantics": {
            **stage_presence(paths["op_semantics"]),
            "path": repo_path(root, paths["op_semantics"]),
            "key_counts": count_from_summary(op_semantics, "semantic_kind_counts", KEY_OP_KINDS),
        },
        "region_pruning_semantics": {
            **stage_presence(paths["region_pruning_semantics"]),
            "path": repo_path(root, paths["region_pruning_semantics"]),
            "key_counts": count_from_summary(
                region_semantics, "semantic_category_counts", KEY_REGION_CATEGORIES
            ),
        },
        "ranking": {
            **stage_presence(paths["ranking"]),
            "path": repo_path(root, paths["ranking"]),
            "summary": trim_dict(ranking.get("summary", {}) if ranking else {}, RANKING_SUMMARY_KEYS),
        },
        "plans": {
            **stage_presence(paths["plans"]),
            "path": repo_path(root, paths["plans"]),
            "summary": trim_dict(plans.get("summary", {}) if plans else {}, PLAN_SUMMARY_KEYS),
        },
        "plan_validation": {
            **stage_presence(paths["plan_validation"]),
            "path": repo_path(root, paths["plan_validation"]),
            "summary": normalize_validation_summary(validation.get("summary", {}) if validation else {}),
        },
        "layer_subgraph_pack": {
            **stage_presence(paths["layer_pack_index"]),
            "path": repo_path(root, paths["layer_pack_index"]),
            "summary": trim_dict(
                layer_pack.get("summary", {}) if layer_pack else {}, LAYER_PACK_SUMMARY_KEYS
            ),
        },
    }
    return summary, layer_pack


def short_source_name(value: str, max_len: int = 96) -> str:
    if len(value) <= max_len:
        return value
    return "..." + value[-(max_len - 3) :]


def summarize_primitive_ops(ops: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    for op in ops[:limit]:
        result.append(
            {
                "op_id": op.get("op_id", ""),
                "source_name": op.get("source_name", ""),
                "op_type": op.get("op_type", ""),
                "topological_index": op.get("topological_index"),
            }
        )
    return result


def summarize_op_semantics(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    for item in items[:limit]:
        result.append(
            {
                "source_name": item.get("source_name") or item.get("op_id", ""),
                "semantic_kind": item.get("semantic_kind", ""),
                "semantic_category": item.get("semantic_category", ""),
                "parameterized": item.get("parameterized", "unknown"),
                "direct_pruning": item.get("direct_pruning", ""),
            }
        )
    return result


def summarize_ranking(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    for item in items[:limit]:
        result.append(
            {
                "candidate_id": item.get("candidate_id", ""),
                "candidate_kind": item.get("candidate_kind", ""),
                "pruning_class": item.get("pruning_class", ""),
                "rank_score": item.get("rank_score"),
                "confidence": item.get("confidence", ""),
                "target_dimension": item.get("target_dimension", ""),
                "blockers": item.get("blockers", []),
                "reason": item.get("reason", ""),
            }
        )
    return result


def summarize_plans(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    for item in items[:limit]:
        actions = item.get("actions", [])
        result.append(
            {
                "plan_id": item.get("plan_id", ""),
                "plan_kind": item.get("plan_kind", ""),
                "plan_status": item.get("plan_status", ""),
                "target_dimension": item.get("target_dimension", ""),
                "symbolic_index_set": item.get("symbolic_index_set", {}),
                "actions": [
                    {
                        "action_type": action.get("action_type", ""),
                        "target_source_name": action.get("target_source_name", ""),
                        "target_axis": action.get("target_axis", ""),
                        "dimension": action.get("dimension", ""),
                    }
                    for action in actions[:limit]
                ],
                "action_count": len(actions),
            }
        )
    return result


def summarize_validations(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    for item in items[:limit]:
        result.append(
            {
                "validation_id": item.get("validation_id", ""),
                "validation_status": item.get("validation_status", ""),
                "validation_score": item.get("validation_score"),
                "failed_checks": item.get("failed_checks", []),
                "warning_checks": item.get("warning_checks", []),
            }
        )
    return result


def verdict_for_subgraph(record: dict[str, Any]) -> str:
    category = record.get("semantic_category", "")
    name = record.get("display_name", "")
    classification = record.get("classification", {})
    validation_status = classification.get("validation_status", "")
    if category == "feed_forward_block":
        if validation_status == "valid":
            return "safe; symbolic plan exists and validates."
        return "safe candidate; symbolic plan evidence should be reviewed."
    if category == "ffn_intermediate_projection":
        return "safe component; full plan belongs to enclosing FeedForwardRegion."
    if category == "ffn_output_projection":
        return "repair component; consumer input is pruned as part of FFN plan."
    if category == "gelu_activation":
        return "index-preserving propagation; participates in FFN plan."
    if category in {"query_projection", "key_projection", "value_projection"}:
        return "constrained; learned projection but attention head-axis mapping is unproven."
    if category == "attention_score_matmul":
        return "blocked; Q x K^T contraction, not learned parameter projection."
    if category == "attention_context_matmul":
        return "blocked; Softmax(scores) x V contraction, not learned parameter projection."
    if category == "attention_mask_add":
        return "auxiliary/constraint carrier; mask broadcast over attention scores, not residual merge."
    if category == "residual_merge":
        return "blocked/protected; hidden_dim branch agreement required."
    if category == "layer_norm":
        return "blocked/protected; hidden_dim and gamma/beta semantics protected."
    if "shape" in category or "axis" in category or "mask" in category:
        return "auxiliary metadata flow."
    if "Attention Score MatMul" in name:
        return "blocked; Q x K^T contraction, not learned parameter projection."
    return "needs review."


def node_folder_for_record(layer_dir: Path, record: dict[str, Any]) -> Path | None:
    slug = record.get("node_slug")
    if slug:
        folder = layer_dir / slug
        if folder.exists():
            return folder
    ordinal = record.get("ordinal")
    if ordinal is not None:
        matches = sorted(layer_dir.glob(f"{int(ordinal):02d}_*"))
        if matches:
            return matches[0]
    return None


def load_subgraph_analysis(
    root: Path,
    layer_dir: Path,
    index_record: dict[str, Any],
    max_ops_per_subgraph: int,
    max_evidence_per_section: int,
    include_explanation_excerpts: bool,
) -> dict[str, Any]:
    folder = node_folder_for_record(layer_dir, index_record)
    analysis = index_record
    analysis_path = None
    explanation_path = None
    if folder:
        candidate = folder / "analysis.json"
        if candidate.exists():
            analysis_path = candidate
            try:
                analysis = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                analysis = index_record
        explanation_candidate = folder / "explanation.md"
        if explanation_candidate.exists():
            explanation_path = explanation_candidate

    primitive_ops = analysis.get("primitive_ops", [])
    local_op_semantics = analysis.get("local_op_semantics", [])
    local_ranking = analysis.get("local_ranking", [])
    local_plans = analysis.get("local_plans", [])
    local_validations = analysis.get("local_validations", [])
    classification = analysis.get("classification", {})
    onnx_export = analysis.get("onnx_export", {})

    summarized = {
        "ordinal": analysis.get("ordinal"),
        "display_name": analysis.get("display_name") or analysis.get("region_name", ""),
        "node_slug": analysis.get("node_slug", ""),
        "semantic_category": analysis.get("semantic_category", ""),
        "source_region_type": analysis.get("source_region_type", ""),
        "region_id": analysis.get("region_id", ""),
        "op_range": analysis.get("op_range"),
        "pruning_class": classification.get("pruning_class", ""),
        "plan_status": classification.get("plan_status", ""),
        "validation_status": classification.get("validation_status", ""),
        "onnx_export": {
            "attempted": onnx_export.get("attempted", False),
            "status": onnx_export.get("status", ""),
            "output_path": onnx_export.get("output_path", ""),
            "checker_status": onnx_export.get("checker_status", ""),
            "error": onnx_export.get("error", ""),
        },
        "primitive_ops_count": len(primitive_ops),
        "primitive_ops_sample": summarize_primitive_ops(primitive_ops, max_ops_per_subgraph),
        "op_semantics_count": len(local_op_semantics),
        "op_semantics_sample": summarize_op_semantics(
            local_op_semantics, max_evidence_per_section
        ),
        "ranking_count": len(local_ranking),
        "ranking_summary": summarize_ranking(local_ranking, max_evidence_per_section),
        "plan_count": len(local_plans),
        "plan_summary": summarize_plans(local_plans, max_evidence_per_section),
        "validation_count": len(local_validations),
        "validation_summary": summarize_validations(local_validations, max_evidence_per_section),
        "verdict": verdict_for_subgraph(analysis),
        "source_files": {
            "analysis": repo_path(root, analysis_path) if analysis_path else "",
            "explanation": repo_path(root, explanation_path) if explanation_path else "",
        },
    }

    if include_explanation_excerpts and explanation_path:
        text = explanation_path.read_text(encoding="utf-8", errors="replace")
        summarized["explanation_excerpt"] = compact_text(text, 700)

    return summarized


def compact_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_subgraph_summaries(
    root: Path,
    model: str,
    layer: int,
    layer_pack: dict[str, Any],
    max_ops_per_subgraph: int,
    max_evidence_per_section: int,
    include_explanation_excerpts: bool,
) -> list[dict[str, Any]]:
    safe_model = safe_model_name(model)
    layer_dir = root / "reports" / "layer_subgraph_validation" / safe_model / f"layer_{layer}"
    records = layer_pack.get("subgraphs", [])
    summaries = [
        load_subgraph_analysis(
            root,
            layer_dir,
            record,
            max_ops_per_subgraph,
            max_evidence_per_section,
            include_explanation_excerpts,
        )
        for record in records
    ]
    return sorted(summaries, key=lambda item: (item.get("ordinal") is None, item.get("ordinal") or 0))


def research_conclusions() -> list[str]:
    return [
        "FeedForwardRegion intermediate_dim pruning is the clean safe opportunity.",
        "Layer 0 Feed Forward has a valid symbolic plan when validation status is valid.",
        "Q/K/V projections are learned but constrained by attention_head_mapping_unproven.",
        "Attention Score MatMul and Attention Context MatMul are blocked because they are contractions, not learned projections.",
        "Attention Mask Add is auxiliary/constraint carrier, not residual merge.",
        "Residual and LayerNorm hidden_dim are protected.",
        "Shape/mask/axis flow is auxiliary metadata.",
    ]


def build_snapshot(
    root: Path,
    model: str,
    layer: int,
    max_ops_per_subgraph: int,
    max_evidence_per_section: int,
    include_explanation_excerpts: bool = False,
) -> dict[str, Any]:
    paths = artifact_file_map(root, model, layer)
    missing_artifacts: list[dict[str, str]] = []
    pipeline_summary, layer_pack = summarize_pipeline(root, paths, missing_artifacts)
    assert layer_pack is not None
    subgraphs = build_subgraph_summaries(
        root,
        model,
        layer,
        layer_pack,
        max_ops_per_subgraph,
        max_evidence_per_section,
        include_explanation_excerpts,
    )
    return {
        "model_name": model,
        "layer": layer,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "missing_artifacts": missing_artifacts,
        "pipeline_summary": pipeline_summary,
        "subgraphs": subgraphs,
        "research_conclusions": research_conclusions(),
    }


def presence_label(stage: dict[str, Any]) -> str:
    return "present" if stage.get("present") else "missing"


def format_count_lines(counts: dict[str, Any], indent: str = "- ") -> list[str]:
    if not counts:
        return [f"{indent}no summary counts available"]
    return [f"{indent}`{key}`: {value}" for key, value in counts.items()]


def markdown_pipeline_section(snapshot: dict[str, Any]) -> list[str]:
    pipeline = snapshot["pipeline_summary"]
    lines = ["## 2. Full pipeline state", ""]
    lines.append(f"- Tensor IR: **{presence_label(pipeline['tensor_ir'])}**")
    lines.append(f"- Op Semantics: **{presence_label(pipeline['op_semantics'])}**")
    lines.extend(format_count_lines(pipeline["op_semantics"].get("key_counts", {}), "  - "))
    lines.append(f"- Region Pruning Semantics: **{presence_label(pipeline['region_pruning_semantics'])}**")
    lines.extend(
        format_count_lines(pipeline["region_pruning_semantics"].get("key_counts", {}), "  - ")
    )
    lines.append(f"- Pruning Opportunity Ranking: **{presence_label(pipeline['ranking'])}**")
    lines.extend(format_count_lines(pipeline["ranking"].get("summary", {}), "  - "))
    lines.append(f"- Pruning Plans: **{presence_label(pipeline['plans'])}**")
    lines.extend(format_count_lines(pipeline["plans"].get("summary", {}), "  - "))
    lines.append(f"- Plan Validation: **{presence_label(pipeline['plan_validation'])}**")
    lines.extend(format_count_lines(pipeline["plan_validation"].get("summary", {}), "  - "))
    lines.append(f"- Layer Subgraph Pack: **{presence_label(pipeline['layer_subgraph_pack'])}**")
    lines.extend(format_count_lines(pipeline["layer_subgraph_pack"].get("summary", {}), "  - "))
    if snapshot["missing_artifacts"]:
        lines.append("")
        lines.append("Missing artifacts recorded, but the snapshot continued with available evidence:")
        for item in snapshot["missing_artifacts"]:
            lines.append(f"- `{item['label']}`: `{item['path']}`")
    lines.append("")
    return lines


def markdown_subgraph_table(subgraphs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## 3. Layer 0 subgraph table",
        "",
        "| # | Subgraph | Semantic category | Class | Primitive ops | Plan | Validation | ONNX |",
        "|---:|---|---|---|---:|---|---|---|",
    ]
    for item in subgraphs:
        ordinal = item.get("ordinal", "")
        lines.append(
            "| {ordinal} | {name} | `{category}` | `{cls}` | {ops} | `{plan}` | `{validation}` | `{onnx}` |".format(
                ordinal=ordinal,
                name=item.get("display_name", ""),
                category=item.get("semantic_category", ""),
                cls=item.get("pruning_class", ""),
                ops=item.get("primitive_ops_count", 0),
                plan=item.get("plan_status", ""),
                validation=item.get("validation_status", ""),
                onnx=item.get("onnx_export", {}).get("status", ""),
            )
        )
    lines.append("")
    return lines


def names_for_categories(subgraphs: list[dict[str, Any]], categories: set[str]) -> list[str]:
    return [
        item.get("display_name", "")
        for item in subgraphs
        if item.get("semantic_category") in categories
    ]


def markdown_stage_explanation(subgraphs: list[dict[str, Any]]) -> list[str]:
    groups = [
        (
            "A. Q/K/V projections",
            {"query_projection", "key_projection", "value_projection"},
            "These are learned attention input projections. They expose parameter axes, but pruning is constrained until head-axis mapping is proven.",
        ),
        (
            "B. Attention internals",
            {
                "attention_skeleton",
                "attention_score_matmul",
                "attention_mask_add",
                "attention_softmax",
                "attention_context_matmul",
            },
            "Attention internals include score/context contractions and mask application. Score/context MatMuls are blocked as non-parameterized contractions; mask flow is auxiliary.",
        ),
        (
            "C. Attention output/residual",
            {"attention_output_projection", "residual_merge"},
            "The attention output projection is constrained by context/head mapping and residual hidden-dim preservation. Residual merges protect hidden_dim branch agreement.",
        ),
        (
            "D. Feed-forward block",
            {"feed_forward_block"},
            "The enclosing feed-forward block is the clean safe pruning unit for intermediate_dim because the same index set propagates through intermediate projection, GELU, and output input columns.",
        ),
        (
            "E. FFN components",
            {"ffn_intermediate_projection", "gelu_activation", "ffn_output_projection"},
            "FFN components are evidence for the enclosing plan: intermediate projection output and bias are pruned, GELU preserves indices, and output projection input columns are repaired.",
        ),
        (
            "F. Residual/LayerNorm/protected path",
            {"layer_norm"},
            "LayerNorm and residual hidden paths remain protected by default. They are validation constraints, not independent safe pruning targets.",
        ),
    ]
    lines = ["## 4. Stage-by-stage explanation of Layer 0", ""]
    for title, categories, text in groups:
        present = names_for_categories(subgraphs, categories)
        if not present:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append("Subgraphs: " + ", ".join(present) + ".")
        lines.append("")
    return lines


def format_source_list(items: list[dict[str, Any]], max_lines: int, formatter) -> list[str]:
    if not items:
        return ["  - none"]
    lines = []
    for item in items[:max_lines]:
        lines.append("  - " + formatter(item))
    return lines


def markdown_subgraph_evidence(
    subgraphs: list[dict[str, Any]],
    max_evidence_per_section: int,
) -> list[str]:
    lines = ["## 5. Per-subgraph compact evidence", ""]
    for item in subgraphs:
        lines.append(f"### {item.get('ordinal')}. {item.get('display_name')}")
        lines.append("")
        lines.append(f"- semantic_category: `{item.get('semantic_category', '')}`")
        lines.append(f"- source_region_type: `{item.get('source_region_type', '')}`")
        lines.append(f"- pruning_class: `{item.get('pruning_class', '')}`")
        lines.append(f"- plan_status: `{item.get('plan_status', '')}`")
        lines.append(f"- validation_status: `{item.get('validation_status', '')}`")
        onnx = item.get("onnx_export", {})
        onnx_path = onnx.get("output_path", "")
        lines.append(f"- ONNX export: `{onnx.get('status', '')}` `{onnx_path}`")
        lines.append(
            f"- primitive ops sample ({len(item.get('primitive_ops_sample', []))}/{item.get('primitive_ops_count', 0)}):"
        )
        lines.extend(
            format_source_list(
                item.get("primitive_ops_sample", []),
                max_evidence_per_section,
                lambda op: f"`{short_source_name(op.get('source_name', ''))}` `{op.get('op_type', '')}` idx={op.get('topological_index')}",
            )
        )
        lines.append(
            f"- op semantics sample ({len(item.get('op_semantics_sample', []))}/{item.get('op_semantics_count', 0)}):"
        )
        lines.extend(
            format_source_list(
                item.get("op_semantics_sample", []),
                max_evidence_per_section,
                lambda op: "`{}` `{}` `{}` parameterized={} direct_pruning={}".format(
                    short_source_name(op.get("source_name", "")),
                    op.get("semantic_kind", ""),
                    op.get("semantic_category", ""),
                    op.get("parameterized", ""),
                    op.get("direct_pruning", ""),
                ),
            )
        )
        lines.append("- ranking summary:")
        lines.extend(
            format_source_list(
                item.get("ranking_summary", []),
                max_evidence_per_section,
                lambda r: "`{}` class=`{}` score={} confidence={} blockers={}".format(
                    r.get("candidate_kind", ""),
                    r.get("pruning_class", ""),
                    r.get("rank_score", ""),
                    r.get("confidence", ""),
                    r.get("blockers", []),
                ),
            )
        )
        if item.get("plan_summary"):
            lines.append("- plan summary:")
            for plan in item["plan_summary"][:max_evidence_per_section]:
                index_set = plan.get("symbolic_index_set", {}).get("name", "")
                lines.append(
                    f"  - `{plan.get('plan_kind', '')}` status=`{plan.get('plan_status', '')}` index_set=`{index_set}`"
                )
                for action in plan.get("actions", [])[:max_evidence_per_section]:
                    lines.append(
                        f"    - `{action.get('action_type', '')}` `{short_source_name(action.get('target_source_name', ''))}` axis=`{action.get('target_axis', '')}` dim=`{action.get('dimension', '')}`"
                    )
        if item.get("validation_summary"):
            lines.append("- validation summary:")
            for validation in item["validation_summary"][:max_evidence_per_section]:
                lines.append(
                    f"  - status=`{validation.get('validation_status', '')}` score={validation.get('validation_score')} failed={validation.get('failed_checks', [])} warnings={validation.get('warning_checks', [])}"
                )
        lines.append(f"- verdict: {item.get('verdict', '')}")
        if item.get("explanation_excerpt"):
            lines.append(f"- explanation excerpt: {item['explanation_excerpt']}")
        lines.append("")
    return lines


def markdown_conclusions(snapshot: dict[str, Any]) -> list[str]:
    lines = ["## 6. Key research conclusions", ""]
    for conclusion in snapshot["research_conclusions"]:
        lines.append(f"- {conclusion}")
    lines.append("")
    return lines


def markdown_upload_section(snapshot: dict[str, Any]) -> list[str]:
    lines = ["## 7. What to upload / inspect next", ""]
    lines.append(
        "This compact Markdown file and its JSON companion are intended for external upload/review."
    )
    lines.append("")
    lines.append("Original large files summarized:")
    for stage in snapshot["pipeline_summary"].values():
        path = stage.get("path")
        if path:
            lines.append(f"- `{path}`")
    for item in snapshot["subgraphs"]:
        source_files = item.get("source_files", {})
        for path in source_files.values():
            if path:
                lines.append(f"- `{path}`")
    lines.append("")
    return lines


def render_markdown(snapshot: dict[str, Any], max_evidence_per_section: int) -> str:
    model = snapshot["model_name"]
    layer = snapshot["layer"]
    lines = [
        f"# Compact Analysis Snapshot: {model} / Layer {layer}",
        "",
        "## 1. Purpose",
        "",
        (
            f"This snapshot summarizes the compiler-style static analysis pipeline for "
            f"{model} Layer {layer} expandable dataflow subgraphs."
        ),
        "",
        (
            "ONNX subgraphs are visualization/evidence artifacts. The analysis is inherited "
            "from the full-model TensorIR, op-semantics, region-semantics, ranking, plan, "
            "and validation pipeline."
        ),
        "",
        "This snapshot is static analysis only: it does not choose indices, prune weights, rewrite ONNX, train, evaluate, or download models.",
        "",
    ]
    lines.extend(markdown_pipeline_section(snapshot))
    lines.extend(markdown_subgraph_table(snapshot["subgraphs"]))
    lines.extend(markdown_stage_explanation(snapshot["subgraphs"]))
    lines.extend(markdown_subgraph_evidence(snapshot["subgraphs"], max_evidence_per_section))
    lines.extend(markdown_conclusions(snapshot))
    lines.extend(markdown_upload_section(snapshot))
    return "\n".join(lines).rstrip() + "\n"


def output_paths(output_dir: Path, model: str, layer: int) -> tuple[Path, Path]:
    safe_model = safe_model_name(model)
    stem = f"{safe_model}__layer_{layer}__snapshot"
    return output_dir / f"{stem}.md", output_dir / f"{stem}.json"


def write_outputs(
    snapshot: dict[str, Any],
    output_dir: Path,
    *,
    write_markdown: bool,
    write_json: bool,
    max_evidence_per_section: int,
) -> tuple[Path | None, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path, json_path = output_paths(output_dir, snapshot["model_name"], snapshot["layer"])
    written_md = None
    written_json = None
    if write_markdown:
        markdown_path.write_text(
            render_markdown(snapshot, max_evidence_per_section), encoding="utf-8"
        )
        written_md = markdown_path
    if write_json:
        json_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written_json = json_path
    return written_md, written_json


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output-dir", default="reports/compact_analysis_snapshots")
    parser.add_argument("--max-ops-per-subgraph", type=int, default=12)
    parser.add_argument("--max-evidence-per-section", type=int, default=8)
    parser.add_argument("--include-explanation-excerpts", action="store_true")
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = Path.cwd()
    try:
        snapshot = build_snapshot(
            root,
            args.model,
            args.layer,
            args.max_ops_per_subgraph,
            args.max_evidence_per_section,
            args.include_explanation_excerpts,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    markdown_path, json_path = write_outputs(
        snapshot,
        root / args.output_dir,
        write_markdown=not args.no_markdown,
        write_json=not args.no_json,
        max_evidence_per_section=args.max_evidence_per_section,
    )

    if args.verbose:
        if markdown_path:
            print(f"wrote {markdown_path} ({markdown_path.stat().st_size} bytes)")
        if json_path:
            print(f"wrote {json_path} ({json_path.stat().st_size} bytes)")
        if snapshot["missing_artifacts"]:
            print(f"missing artifacts: {len(snapshot['missing_artifacts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
