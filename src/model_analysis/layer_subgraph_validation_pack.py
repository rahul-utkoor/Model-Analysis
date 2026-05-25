"""Build per-layer subgraph evidence packs from full-model static analysis."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir, safe_model_name


LEARNER_ORDER = [
    "query projection",
    "key projection",
    "value projection",
    "attention",
    "attention score matmul",
    "attention mask add",
    "attention softmax",
    "attention context matmul",
    "attention output projection",
    "attention residual add",
    "feed forward",
    "ffn intermediate projection",
    "gelu",
    "ffn output projection",
    "ffn residual add",
    "layernorm",
]

AUXILIARY_REGION_TYPES = {"AxisTransformRegion", "ForkRegion", "JoinRegion", "BiasAddRegion"}
AUXILIARY_CATEGORIES = {"shape_axis_transform", "attention_mask_axis_transform", "attention_mask_join", "attention_mask_fork"}


@dataclass
class LayerSubgraphRecord:
    subgraph_id: str
    ordinal: int
    node_slug: str
    display_name: str
    layer_index: int
    region_id: str | None
    region_name: str
    source_region_type: str
    semantic_category: str
    section: str
    op_range: str
    primitive_ops: list[dict[str, Any]]
    boundary: dict[str, Any]
    local_op_semantics: list[dict[str, Any]]
    local_region_semantics: list[dict[str, Any]]
    local_ranking: list[dict[str, Any]]
    local_plans: list[dict[str, Any]]
    local_validations: list[dict[str, Any]]
    classification: dict[str, Any]
    onnx_export: dict[str, Any]
    explanation: str


@dataclass
class LayerSubgraphValidationPack:
    model_name: str
    layer_index: int
    generated_at: str
    source_paths: dict[str, str]
    subgraphs: list[LayerSubgraphRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def layer_subgraph_record_to_dict(record: LayerSubgraphRecord) -> dict[str, Any]:
    return asdict(record)


def layer_subgraph_pack_to_dict(pack: LayerSubgraphValidationPack) -> dict[str, Any]:
    return asdict(pack)


def write_layer_subgraph_pack_json(pack: LayerSubgraphValidationPack | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    data = layer_subgraph_pack_to_dict(pack) if isinstance(pack, LayerSubgraphValidationPack) else pack
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_range(value: str) -> tuple[int, int]:
    nums = [int(item) for item in re.findall(r"\d+", str(value))]
    if not nums:
        return (10**9, 10**9)
    if len(nums) == 1:
        return (nums[0], nums[0])
    return (nums[0], nums[1])


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned or "subgraph"


def _learner_key(name: str) -> int:
    lowered = name.lower()
    for idx, token in enumerate(LEARNER_ORDER):
        if token in lowered:
            return idx
    return len(LEARNER_ORDER)


def _safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _op_ref_from_leaf(leaf: dict[str, Any], tensor_op_by_id: dict[str, dict[str, Any]], tensor_op_by_source: dict[str, dict[str, Any]]) -> dict[str, Any]:
    op = tensor_op_by_id.get(leaf.get("id", "")) or tensor_op_by_source.get(leaf.get("source_name", "")) or {}
    return {
        "op_id": leaf.get("id") or op.get("op_id", ""),
        "source_name": leaf.get("source_name") or leaf.get("name") or op.get("source_node_name", op.get("name", "")),
        "op_type": op.get("op_type") or leaf.get("op_type", ""),
        "topological_index": leaf.get("op_index", op.get("source_location", {}).get("node_index")),
    }


def _primitive_ops_for_record(record: dict[str, Any], tensor_op_by_id: dict[str, dict[str, Any]], tensor_op_by_source: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    leaves = _safe_list(record.get("recursive_primitive_leaves"))
    if not leaves and record.get("kind") == "primitive":
        leaves = [record]
    ops = [_op_ref_from_leaf(leaf, tensor_op_by_id, tensor_op_by_source) for leaf in leaves]
    seen = set()
    out = []
    for op in ops:
        key = op.get("op_id") or op.get("source_name")
        if key and key not in seen:
            seen.add(key)
            out.append(op)
    out.sort(key=lambda item: item.get("topological_index") if item.get("topological_index") is not None else 10**9)
    return out


def _normalize_selected_record(record: dict[str, Any], parent: dict[str, Any] | None = None) -> dict[str, Any]:
    if record.get("kind") == "primitive":
        source = record.get("source_name", record.get("name", ""))
        return {
            "kind": "primitive",
            "region_id": record.get("id", ""),
            "region_type": "PrimitiveRegion",
            "source_region_type": "PrimitiveRegion",
            "semantic_category": "",
            "name": record.get("name", source),
            "section": record.get("section", parent.get("section", "") if parent else ""),
            "op_range": str(record.get("op_index", "")),
            "recursive_primitive_leaves": [record],
            "reason": "Primitive attention/activation node selected from parent expansion.",
        }
    return record


def select_expandable_layer_nodes(
    abstract_expansion: dict[str, Any] | None,
    region_semantics: dict[str, Any],
    layer_index: int,
    include_auxiliary: bool = False,
) -> list[dict[str, Any]]:
    section = f"Encoder Layer {layer_index}"
    selected: list[dict[str, Any]] = []
    if abstract_expansion:
        records = _safe_list(abstract_expansion.get("records"))
        by_id = {record.get("region_id", ""): record for record in records}
        for record in records:
            if record.get("section") != section:
                continue
            region_type = record.get("region_type", "")
            if not include_auxiliary and region_type in AUXILIARY_REGION_TYPES:
                continue
            if not record.get("recursive_primitive_leaves") and not record.get("immediate_expansion"):
                continue
            selected.append(record)
            if record.get("region_type") == "AttentionSkeletonRegion":
                for child in _safe_list(record.get("immediate_expansion")):
                    if child.get("kind") == "primitive" and f"Layer {layer_index}" in child.get("name", ""):
                        selected.append(_normalize_selected_record(child, record))
                    elif child.get("id") in by_id:
                        selected.append(by_id[child["id"]])
            if record.get("region_type") == "FeedForwardRegion":
                for child in _safe_list(record.get("immediate_expansion")):
                    if child.get("id") in by_id:
                        selected.append(by_id[child["id"]])
    else:
        for region in _safe_list(region_semantics.get("regions")):
            if region.get("section") != section and f"Layer {layer_index}" not in region.get("region_name", ""):
                continue
            region_type = region.get("source_region_type", region.get("region_type", ""))
            category = region.get("semantic_category", "")
            if not include_auxiliary and (region_type in AUXILIARY_REGION_TYPES or category in AUXILIARY_CATEGORIES):
                continue
            selected.append(
                {
                    "region_id": region.get("region_id"),
                    "region_type": region_type,
                    "source_region_type": region_type,
                    "semantic_category": category,
                    "name": region.get("region_name", region.get("region_id", "")),
                    "section": region.get("section", section),
                    "op_range": region.get("op_range", ""),
                    "recursive_primitive_leaves": [{"id": op_id} for op_id in region.get("evidence", {}).get("source_ops", [])],
                    "reason": "Selected from Region Pruning Semantics fallback.",
                }
            )
    dedup: dict[str, dict[str, Any]] = {}
    for record in selected:
        key = record.get("region_id") or record.get("id") or f"{record.get('name')}::{record.get('op_range')}::{record.get('region_type')}"
        dedup.setdefault(key, record)
    result = list(dedup.values())
    result.sort(key=lambda item: (_parse_range(item.get("op_range", ""))[0], _learner_key(item.get("name", "")), item.get("name", "")))
    return result


def _summarize_op_semantics(op: dict[str, Any]) -> dict[str, Any]:
    effect = op.get("pruning_effect", {})
    return {
        "op_id": op.get("op_id"),
        "source_name": op.get("source_name"),
        "semantic_kind": op.get("semantic_kind"),
        "semantic_category": op.get("semantic_category"),
        "parameterized": op.get("parameterized"),
        "index_behavior": op.get("index_behavior"),
        "direct_pruning": effect.get("direct_pruning"),
    }


def _summarize_region(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "region_id": region.get("region_id"),
        "region_name": region.get("region_name"),
        "source_region_type": region.get("source_region_type", region.get("region_type")),
        "semantic_category": region.get("semantic_category"),
        "pruning_role": region.get("pruning_role"),
        "dimensions": region.get("dimensions", []),
        "blockers": region.get("blockers", []),
        "repairs": region.get("repair_obligations", []),
    }


def _summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_kind": candidate.get("candidate_kind"),
        "pruning_class": candidate.get("pruning_class"),
        "rank_score": candidate.get("rank_score"),
        "confidence": candidate.get("confidence"),
        "target_dimension": candidate.get("target_dimension"),
        "blockers": candidate.get("blockers", []),
        "reason": candidate.get("reason", ""),
    }


def _summarize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_id": plan.get("plan_id"),
        "plan_kind": plan.get("plan_kind"),
        "plan_status": plan.get("plan_status"),
        "target_dimension": plan.get("target_dimension"),
        "actions": [
            {
                "action_type": action.get("action_type"),
                "target_source_name": action.get("target_source_name"),
                "target_axis": action.get("target_axis"),
                "dimension": action.get("dimension"),
            }
            for action in _safe_list(plan.get("actions"))
        ],
        "symbolic_index_set": plan.get("symbolic_index_set", {}),
    }


def _summarize_validation(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "validation_id": validation.get("validation_id"),
        "validation_status": validation.get("validation_status"),
        "validation_score": validation.get("validation_score"),
        "failed_checks": [check.get("check_type") for check in _safe_list(validation.get("checks")) if check.get("status") == "fail"],
        "warning_checks": [check.get("check_type") for check in _safe_list(validation.get("checks")) if check.get("status") == "warning"],
    }


def _boundary(primitive_ops: list[dict[str, Any]], tensor_ops: dict[str, dict[str, Any]]) -> dict[str, Any]:
    op_ids = {op.get("op_id") for op in primitive_ops}
    inputs: list[str] = []
    outputs: list[str] = []
    external_producers: list[str] = []
    external_consumers: list[str] = []
    for ref in primitive_ops:
        op = tensor_ops.get(ref.get("op_id", ""), {})
        for value in op.get("inputs", []):
            producers = set(op.get("predecessor_ops", []))
            if producers - op_ids:
                inputs.append(value)
                external_producers.extend(sorted(producers - op_ids))
        for value in op.get("outputs", []):
            consumers = set(op.get("successor_ops", []))
            if consumers - op_ids or not consumers:
                outputs.append(value)
                external_consumers.extend(sorted(consumers - op_ids))
    return {
        "input_tensors": sorted(set(inputs)),
        "output_tensors": sorted(set(outputs)),
        "external_producers": sorted(set(external_producers)),
        "external_consumers": sorted(set(external_consumers)),
    }


def _match_candidates(candidates: list[dict[str, Any]], region_id: str, name: str, source_names: set[str]) -> list[dict[str, Any]]:
    out = []
    for candidate in candidates:
        evidence_sources = {item.get("source_name", "") for item in _safe_list(candidate.get("op_semantics_evidence"))}
        if candidate.get("region_id") == region_id or candidate.get("region_name") == name or (not region_id and evidence_sources & source_names):
            out.append(candidate)
    return out


def _match_plans(plans: list[dict[str, Any]], region_id: str, candidates: list[dict[str, Any]], source_names: set[str]) -> list[dict[str, Any]]:
    del source_names
    candidate_ids = {item.get("candidate_id") for item in candidates}
    out = []
    for plan in plans:
        if plan.get("candidate_region_id") == region_id or plan.get("candidate_id") in candidate_ids:
            out.append(plan)
    return out


def _match_validations(validations: list[dict[str, Any]], plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan_ids = {item.get("plan_id") for item in plans}
    return [item for item in validations if item.get("plan_id") in plan_ids]


def _classification(name: str, category: str, candidates: list[dict[str, Any]], plans: list[dict[str, Any]], validations: list[dict[str, Any]]) -> dict[str, Any]:
    if category in {"attention_mask_add", "gelu_activation", "shape_axis_transform"} or "gelu" in name.lower():
        pruning_class = "auxiliary"
    elif category in {"attention_score_matmul", "attention_context_matmul", "attention_softmax", "attention_skeleton", "residual_merge", "layer_norm"}:
        pruning_class = "blocked"
    elif candidates and any(item.get("pruning_class") != "unknown" for item in candidates):
        classes = [item.get("pruning_class", "unknown") for item in candidates]
        priority = ["safe", "constrained", "blocked", "auxiliary", "unknown"]
        pruning_class = next((item for item in priority if item in classes), "mixed")
        if len(set(classes)) > 1 and pruning_class not in {"safe", "constrained"}:
            pruning_class = "mixed"
    elif category in {"query_projection", "key_projection", "value_projection", "attention_output_projection"}:
        pruning_class = "constrained"
    else:
        pruning_class = "unknown"
    if validations:
        if any(item.get("validation_status") == "valid" for item in validations):
            plan_status = "valid_plan"
            validation_status = "valid"
        elif any(item.get("validation_status") == "invalid" for item in validations):
            plan_status = "invalid_plan"
            validation_status = "invalid"
        else:
            plan_status = "unknown"
            validation_status = validations[0].get("validation_status", "unknown")
    elif plans:
        plan_status = "invalid_plan" if any(item.get("plan_status") != "ready_symbolic" for item in plans) else "unknown"
        validation_status = "unknown"
    elif category == "feed_forward_block" or pruning_class == "safe" and "feed forward" in name.lower():
        plan_status = "no_plan_but_expected"
        validation_status = "unknown"
    else:
        plan_status = "no_plan_expected"
        validation_status = "not_applicable"
    return {
        "pruning_class": pruning_class,
        "plan_status": plan_status,
        "validation_status": validation_status,
    }


def _explanation(name: str, category: str, classification: dict[str, Any], candidates: list[dict[str, Any]], validations: list[dict[str, Any]]) -> str:
    if category == "feed_forward_block":
        return f"{name} is the safe FFN intermediate_dim opportunity; symbolic plan status is {classification['plan_status']} and validation is {classification['validation_status']}."
    if category in {"query_projection", "key_projection", "value_projection"}:
        return f"{name} is a learned attention projection, but pruning is constrained by attention_head_mapping_unproven."
    if category == "attention_score_matmul":
        return f"{name} is blocked: Q x K^T is an attention contraction, not a learned parameter projection."
    if category == "attention_context_matmul":
        return f"{name} is blocked: Softmax(scores) x V is an attention contraction, not a learned parameter projection."
    if category == "attention_mask_add":
        return f"{name} is auxiliary/constraint-carrying mask score biasing, not a direct pruning target."
    if category == "residual_merge":
        return f"{name} is blocked because residual branches require hidden_dim agreement."
    if category == "layer_norm":
        return f"{name} is protected by default; hidden_dim pruning would require LayerNorm parameter repair."
    if "gelu" in name.lower():
        return f"{name} is index-preserving propagation inside the FFN plan, not an independent pruning target."
    if candidates:
        return candidates[0].get("reason", f"{name} classified as {classification['pruning_class']}.")
    if validations:
        return f"{name} has validation status {validations[0].get('validation_status')}."
    return f"{name} is classified as {classification['pruning_class']} from local static analysis slices."


def _onnx_record(primitive_ops: list[dict[str, Any]], record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    node_names = [op.get("source_name", "") for op in primitive_ops if op.get("source_name")]
    return {
        "subgraph_id": record.get("region_id") or record.get("id") or record.get("name", ""),
        "subgraph_kind": "layer_node",
        "node_names": node_names,
        "op_types": [op.get("op_type", "") for op in primitive_ops],
        "pattern": record.get("name", ""),
        "boundary_input_tensors": [],
        "boundary_output_tensors": [],
        "internal_tensors": [],
        "initializer_tensors": [],
        "source_onnx_path": str(source_path),
        "reason": "Layer subgraph validation pack visualization artifact.",
    }


def _export_onnx(
    primitive_ops: list[dict[str, Any]],
    record: dict[str, Any],
    source_path: Path | None,
    output_path: Path,
    model_name: str,
    strict: bool,
) -> dict[str, Any]:
    if source_path is None or not source_path.exists():
        return {"attempted": False, "status": "skipped", "source_model_path": str(source_path or ""), "output_path": str(output_path), "error": "ONNX source model missing."}
    try:
        import onnx
        from model_analysis.onnx_subgraph_extractor import extract_onnx_subgraph_model

        source_model = onnx.load(source_path)
        export = extract_onnx_subgraph_model(source_model, _onnx_record(primitive_ops, record, source_path), output_path, model_name)
        if export.status != "success" and strict:
            raise RuntimeError(export.reason)
        return {
            "attempted": True,
            "status": "exported" if export.status == "success" else "failed",
            "source_model_path": str(source_path),
            "output_path": str(output_path),
            "checker_status": export.metadata.get("checker_status"),
            "error": "" if export.status == "success" else export.reason,
        }
    except Exception as exc:
        if strict:
            raise
        return {"attempted": True, "status": "failed", "source_model_path": str(source_path), "output_path": str(output_path), "error": str(exc)}


def _write_dot(record: LayerSubgraphRecord, tensor_ops: dict[str, dict[str, Any]], path: Path, render_svg: bool) -> None:
    ensure_dir(path.parent)
    op_ids = [op.get("op_id", "") for op in record.primitive_ops]
    op_set = set(op_ids)
    lines = ["digraph layer_subgraph {", "  rankdir=LR;"]
    for op_id in op_ids:
        op = tensor_ops.get(op_id, {})
        label = (op.get("source_node_name") or op.get("name") or op_id).split("/")[-1]
        lines.append(f'  "{op_id}" [label="{label}\\n{op.get("op_type", "")}"];')
    for op_id in op_ids:
        op = tensor_ops.get(op_id, {})
        for succ in op.get("successor_ops", []):
            if succ in op_set:
                lines.append(f'  "{op_id}" -> "{succ}";')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")
    if render_svg and shutil.which("dot"):
        svg = path.with_suffix(".svg")
        subprocess.run(["dot", "-Tsvg", str(path), "-o", str(svg)], check=False)


def _write_node_files(record: LayerSubgraphRecord, report_dir: Path, artifact_dir: Path, tensor_ops: dict[str, dict[str, Any]], render_svg: bool) -> None:
    ensure_dir(report_dir)
    ensure_dir(artifact_dir)
    data = layer_subgraph_record_to_dict(record)
    (report_dir / "analysis.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    for name, key in [
        ("primitive_ops.json", "primitive_ops"),
        ("op_semantics.json", "local_op_semantics"),
        ("region_semantics.json", "local_region_semantics"),
        ("ranking.json", "local_ranking"),
        ("plan.json", "local_plans"),
        ("validation.json", "local_validations"),
    ]:
        (report_dir / name).write_text(json.dumps(data.get(key, []), indent=2), encoding="utf-8")
    (report_dir / "explanation.md").write_text(layer_subgraph_record_to_markdown(data), encoding="utf-8")
    _write_dot(record, tensor_ops, artifact_dir / "subgraph.dot", render_svg)
    netron = [
        f"# Netron: {record.display_name}",
        "",
        f"- ONNX status: `{record.onnx_export.get('status')}`",
        f"- File: `{record.onnx_export.get('output_path')}`",
        "",
        "```bash",
        f"netron {record.onnx_export.get('output_path')}",
        "```",
        "",
        "This ONNX fragment is a visualization artifact, not a standalone model for full re-analysis.",
        "",
    ]
    (artifact_dir / "netron.md").write_text("\n".join(netron), encoding="utf-8")


def build_layer_subgraph_validation_pack(
    *,
    model_name: str,
    layer_index: int,
    tensor_ir: dict[str, Any],
    op_semantics: dict[str, Any],
    structural_region_tree: dict[str, Any],
    region_pruning_semantics: dict[str, Any],
    ranking: dict[str, Any],
    plans: dict[str, Any],
    validations: dict[str, Any],
    abstract_expansion: dict[str, Any] | None = None,
    source_paths: dict[str, str] | None = None,
    report_root: Path | None = None,
    artifact_root: Path | None = None,
    source_onnx_path: Path | None = None,
    export_onnx: bool = True,
    render_svg: bool = False,
    max_subgraphs: int | None = None,
    include_auxiliary: bool = False,
    strict_onnx_export: bool = False,
) -> LayerSubgraphValidationPack:
    del structural_region_tree
    safe = safe_model_name(model_name)
    tensor_ops_list = _safe_list(tensor_ir.get("ops"))
    tensor_op_by_id = {op.get("op_id", ""): op for op in tensor_ops_list}
    tensor_op_by_source = {op.get("source_node_name", op.get("name", "")): op for op in tensor_ops_list}
    op_sem_by_id = {op.get("op_id", ""): op for op in _safe_list(op_semantics.get("ops"))}
    op_sem_by_source = {op.get("source_name", ""): op for op in _safe_list(op_semantics.get("ops"))}
    region_by_id = {item.get("region_id", ""): item for item in _safe_list(region_pruning_semantics.get("regions"))}
    selected = select_expandable_layer_nodes(abstract_expansion, region_pruning_semantics, layer_index, include_auxiliary)
    if max_subgraphs is not None:
        selected = selected[:max_subgraphs]
    if report_root:
        report_layer_root = report_root / safe / f"layer_{layer_index}"
        if report_layer_root.exists():
            shutil.rmtree(report_layer_root)
    if artifact_root:
        artifact_layer_root = artifact_root / safe / f"layer_{layer_index}"
        if artifact_layer_root.exists():
            shutil.rmtree(artifact_layer_root)
    records: list[LayerSubgraphRecord] = []
    for ordinal, raw in enumerate(selected, start=1):
        region_id = raw.get("region_id") or raw.get("id")
        display = raw.get("name", region_id or "subgraph")
        primitive_ops = _primitive_ops_for_record(raw, tensor_op_by_id, tensor_op_by_source)
        source_names = {op.get("source_name", "") for op in primitive_ops}
        op_ids = {op.get("op_id", "") for op in primitive_ops}
        local_op_sem = [_summarize_op_semantics(op_sem_by_id.get(op_id) or op_sem_by_source.get(src) or {}) for op_id, src in [(op.get("op_id", ""), op.get("source_name", "")) for op in primitive_ops]]
        local_op_sem = [item for item in local_op_sem if item.get("op_id") or item.get("source_name")]
        region = region_by_id.get(region_id, {})
        local_regions = [_summarize_region(region)] if region else []
        candidates = _match_candidates(_safe_list(ranking.get("candidates")), region_id or "", display, source_names)
        local_plans = _match_plans(_safe_list(plans.get("plans")), region_id or "", candidates, source_names)
        local_validations = _match_validations(_safe_list(validations.get("validations")), local_plans)
        category = region.get("semantic_category") or raw.get("semantic_category", "")
        if not category and candidates:
            category = candidates[0].get("semantic_category", "")
        if not category or category == "unknown":
            kinds = [item.get("semantic_kind") for item in local_op_sem if item.get("semantic_kind")]
            if len(set(kinds)) == 1:
                category = kinds[0]
        classification = _classification(display, category, candidates, local_plans, local_validations)
        slug = f"{ordinal:02d}_{_slug(display)}"
        onnx_path = (artifact_root / safe / f"layer_{layer_index}" / slug / "subgraph.onnx") if artifact_root else Path("")
        onnx_status = _export_onnx(primitive_ops, raw, source_onnx_path, onnx_path, model_name, strict_onnx_export) if export_onnx else {"attempted": False, "status": "skipped", "source_model_path": str(source_onnx_path or ""), "output_path": str(onnx_path), "error": "ONNX export disabled."}
        record = LayerSubgraphRecord(
            subgraph_id=f"layer_{layer_index}::{slug}",
            ordinal=ordinal,
            node_slug=slug,
            display_name=display,
            layer_index=layer_index,
            region_id=region_id,
            region_name=display,
            source_region_type=region.get("source_region_type", raw.get("source_region_type", raw.get("region_type", ""))),
            semantic_category=category or "unknown",
            section=raw.get("section", f"Encoder Layer {layer_index}"),
            op_range=str(raw.get("op_range", "")),
            primitive_ops=primitive_ops,
            boundary=_boundary(primitive_ops, tensor_op_by_id),
            local_op_semantics=local_op_sem,
            local_region_semantics=local_regions,
            local_ranking=[_summarize_candidate(item) for item in candidates],
            local_plans=[_summarize_plan(item) for item in local_plans],
            local_validations=[_summarize_validation(item) for item in local_validations],
            classification=classification,
            onnx_export=onnx_status,
            explanation=_explanation(display, category, classification, candidates, local_validations),
        )
        records.append(record)
        if report_root and artifact_root:
            _write_node_files(record, report_root / safe / f"layer_{layer_index}" / slug, artifact_root / safe / f"layer_{layer_index}" / slug, tensor_op_by_id, render_svg)
    pack = LayerSubgraphValidationPack(
        model_name=model_name,
        layer_index=layer_index,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_paths=source_paths or {},
        subgraphs=records,
        summary=_summary(records),
    )
    if report_root:
        out = report_root / safe / f"layer_{layer_index}"
        ensure_dir(out)
        write_layer_subgraph_pack_json(pack, out / "index.json")
        (out / "index.md").write_text(layer_subgraph_pack_to_markdown(pack, artifact_root=artifact_root), encoding="utf-8")
    return pack


def _summary(records: list[LayerSubgraphRecord]) -> dict[str, Any]:
    class_counts = Counter(record.classification.get("pruning_class", "unknown") for record in records)
    validation_counts = Counter(record.classification.get("validation_status", "unknown") for record in records)
    categories = Counter(record.semantic_category for record in records)
    onnx_status = Counter(record.onnx_export.get("status", "skipped") for record in records)
    return {
        "total_subgraphs": len(records),
        "onnx_exported": onnx_status.get("exported", 0),
        "onnx_failed": onnx_status.get("failed", 0),
        "onnx_skipped": onnx_status.get("skipped", 0),
        "safe_subgraphs": class_counts.get("safe", 0),
        "constrained_subgraphs": class_counts.get("constrained", 0),
        "blocked_subgraphs": class_counts.get("blocked", 0),
        "auxiliary_subgraphs": class_counts.get("auxiliary", 0),
        "unknown_subgraphs": class_counts.get("unknown", 0),
        "valid_plan_subgraphs": sum(1 for record in records if record.classification.get("plan_status") == "valid_plan"),
        "no_plan_expected_subgraphs": sum(1 for record in records if record.classification.get("plan_status") == "no_plan_expected"),
        "invalid_plan_subgraphs": sum(1 for record in records if record.classification.get("plan_status") == "invalid_plan"),
        "semantic_category_counts": dict(sorted(categories.items())),
        "pruning_class_counts": dict(sorted(class_counts.items())),
        "validation_status_counts": dict(sorted(validation_counts.items())),
    }


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def layer_subgraph_pack_to_markdown(pack: LayerSubgraphValidationPack | dict[str, Any], artifact_root: Path | None = None) -> str:
    data = layer_subgraph_pack_to_dict(pack) if isinstance(pack, LayerSubgraphValidationPack) else pack
    summary = data.get("summary", {})
    rows = []
    for item in data.get("subgraphs", []):
        rows.append({
            "#": item.get("ordinal"),
            "Abstract node": item.get("display_name"),
            "Semantic category": item.get("semantic_category"),
            "Primitive ops": len(item.get("primitive_ops", [])),
            "Class": item.get("classification", {}).get("pruning_class"),
            "Plan": item.get("classification", {}).get("plan_status"),
            "Validation": item.get("classification", {}).get("validation_status"),
            "ONNX": item.get("onnx_export", {}).get("status"),
        })
    lines = [
        f"# Layer {data.get('layer_index')} Subgraph Validation Pack: {data.get('model_name')}",
        "",
        "## Summary",
        "",
        f"- Total expandable nodes: `{summary.get('total_subgraphs', 0)}`",
        f"- ONNX exported: `{summary.get('onnx_exported', 0)}`",
        f"- ONNX failed: `{summary.get('onnx_failed', 0)}`",
        f"- ONNX skipped: `{summary.get('onnx_skipped', 0)}`",
        f"- Safe: `{summary.get('safe_subgraphs', 0)}`",
        f"- Constrained: `{summary.get('constrained_subgraphs', 0)}`",
        f"- Blocked: `{summary.get('blocked_subgraphs', 0)}`",
        f"- Auxiliary: `{summary.get('auxiliary_subgraphs', 0)}`",
        f"- Unknown: `{summary.get('unknown_subgraphs', 0)}`",
        f"- Valid plans: `{summary.get('valid_plan_subgraphs', 0)}`",
        "",
        "## Subgraph Table",
        "",
        _table(rows, ["#", "Abstract node", "Semantic category", "Primitive ops", "Class", "Plan", "Validation", "ONNX"]),
        "",
    ]
    for title, cls in [("Safe Subgraphs", "safe"), ("Constrained Subgraphs", "constrained"), ("Blocked Subgraphs", "blocked"), ("Auxiliary Subgraphs", "auxiliary")]:
        lines.extend(["## " + title, "", _table([row for row in rows if row["Class"] == cls], ["#", "Abstract node", "Semantic category", "Plan", "Validation", "ONNX"]), ""])
    lines.extend(["## How to Inspect in Netron", ""])
    if artifact_root:
        lines.append("```bash")
        for item in data.get("subgraphs", []):
            path = item.get("onnx_export", {}).get("output_path", "")
            if item.get("onnx_export", {}).get("status") == "exported":
                lines.append(f"netron {path}")
        lines.append("```")
    else:
        lines.append("Open `artifacts/layer_subgraphs/<model>/layer_<N>/<folder>/subgraph.onnx` when exported.")
    lines.extend(["", "These ONNX files are visualization artifacts and are not treated as standalone analysis sources.", ""])
    return "\n".join(lines)


def layer_subgraph_record_to_markdown(record: dict[str, Any]) -> str:
    lines = [
        f"# {record.get('display_name')}",
        "",
        "## What this subgraph is",
        "",
        record.get("explanation", ""),
        "",
        "## Primitive ONNX/TensorIR Ops",
        "",
        _table(record.get("primitive_ops", []), ["topological_index", "source_name", "op_type"]),
        "",
        "## Boundary Tensors",
        "",
        f"- Inputs: `{record.get('boundary', {}).get('input_tensors', [])}`",
        f"- Outputs: `{record.get('boundary', {}).get('output_tensors', [])}`",
        "",
        "## Op Semantics",
        "",
        _table(record.get("local_op_semantics", []), ["source_name", "semantic_kind", "semantic_category", "parameterized", "direct_pruning"]),
        "",
        "## Region Pruning Semantics",
        "",
        _table(record.get("local_region_semantics", []), ["region_name", "semantic_category", "pruning_role"]),
        "",
        "## Ranking Result",
        "",
        _table(record.get("local_ranking", []), ["candidate_kind", "pruning_class", "rank_score", "confidence", "target_dimension", "reason"]),
        "",
        "## Pruning Plan",
        "",
        _table(record.get("local_plans", []), ["plan_kind", "plan_status", "target_dimension", "symbolic_index_set"]),
        "",
        "## Plan Validation",
        "",
        _table(record.get("local_validations", []), ["validation_status", "validation_score", "failed_checks", "warning_checks"]),
        "",
        "## Verdict",
        "",
        f"- Class: `{record.get('classification', {}).get('pruning_class')}`",
        f"- Plan: `{record.get('classification', {}).get('plan_status')}`",
        f"- Validation: `{record.get('classification', {}).get('validation_status')}`",
        f"- ONNX export: `{record.get('onnx_export', {}).get('status')}`",
        "",
    ]
    return "\n".join(lines)
