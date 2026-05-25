"""Full-model structured static analysis reports."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.layer_subgraph_validation_pack import build_layer_subgraph_validation_pack
from model_analysis.paths import ensure_dir, safe_model_name
from model_analysis.reporting import write_json, write_markdown


REQUIRED_MODEL_ARTIFACTS = {
    "op_semantics": ("reports/op_semantics/{safe}.json", "Op Semantics"),
    "region_pruning_semantics": (
        "reports/region_pruning_semantics/{safe}.json",
        "Region Pruning Semantics",
    ),
    "ranking": ("reports/pruning_opportunity_rankings/{safe}.json", "Pruning Opportunity Ranking"),
}

RECOMMENDED_MODEL_ARTIFACTS = {
    "tensor_ir": ("reports/tensor_ir/{safe}.json", "Tensor IR"),
    "structural_region_tree": ("reports/structural_region_trees/{safe}.json", "Structural Region Tree"),
    "region_dimension_ir": ("reports/region_dimension_ir/{safe}.json", "Region Dimension IR"),
    "plans": ("reports/pruning_plans/{safe}.json", "Symbolic Pruning Plans"),
    "validations": ("reports/pruning_plan_validation/{safe}.json", "Pruning Plan Validation"),
    "abstract_expansion": (
        "reports/abstract_node_expansions/{safe}/abstract_node_expansions_main.json",
        "Abstract Node Expansion Report",
    ),
}

LAYER_BUILD_KEYS = [
    "tensor_ir",
    "op_semantics",
    "structural_region_tree",
    "region_pruning_semantics",
    "ranking",
    "plans",
    "validations",
]


def _safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_path(root: Path, safe: str, template: str) -> Path:
    return root / template.format(safe=safe)


def discover_model_artifacts(root: Path, model_name: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, str]]]:
    safe = safe_model_name(model_name)
    available: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    for key, (template, label) in {**REQUIRED_MODEL_ARTIFACTS, **RECOMMENDED_MODEL_ARTIFACTS}.items():
        path = _artifact_path(root, safe, template)
        present = path.exists()
        available[key] = {"present": present, "path": str(path), "label": label}
        if present:
            loaded_value = _load_json(path)
            if loaded_value is not None:
                loaded[key] = loaded_value
        else:
            missing.append({"artifact": key, "label": label, "path": str(path)})
    static_onnx = root / "data" / "models" / "onnx_static" / safe / "model.static.onnx"
    dynamic_onnx = root / "data" / "models" / "onnx" / safe / "model.onnx"
    source_onnx = static_onnx if static_onnx.exists() else dynamic_onnx if dynamic_onnx.exists() else None
    available["onnx_model"] = {
        "present": source_onnx is not None,
        "path": str(source_onnx or static_onnx),
        "label": "ONNX model for visualization subgraphs",
    }
    if source_onnx is None:
        missing.append({"artifact": "onnx_model", "label": "ONNX model", "path": str(static_onnx)})
    return available, loaded, missing


def missing_required_artifacts(missing: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in missing if item.get("artifact") in REQUIRED_MODEL_ARTIFACTS]


def detect_layers(region_semantics: dict[str, Any] | None, abstract_expansion: dict[str, Any] | None = None) -> list[int]:
    observed: set[int] = set()
    layer_pattern = re.compile(r"(?:layer\.|Layer\s+)(\d+)")
    sources: list[str] = []
    if region_semantics:
        for region in _safe_list(region_semantics.get("regions")):
            sources.append(str(region.get("region_name", "")))
            sources.append(str(region.get("section", "")))
            sources.append(" ".join(str(op) for op in region.get("evidence", {}).get("source_ops", [])))
    if abstract_expansion:
        for record in _safe_list(abstract_expansion.get("records")):
            sources.append(str(record.get("name", "")))
            sources.append(str(record.get("section", "")))
            for leaf in _safe_list(record.get("recursive_primitive_leaves")):
                sources.append(str(leaf.get("source_name", "")))
    for source in sources:
        for match in layer_pattern.finditer(source):
            observed.add(int(match.group(1)))
    return sorted(observed)


def parse_layer_selection(value: str, available_layers: list[int], max_layers: int | None = None) -> list[int]:
    if value == "all":
        layers = list(available_layers)
    else:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if max_layers is not None:
        layers = layers[:max_layers]
    return layers


def _source_onnx_path(root: Path, model_name: str) -> Path | None:
    safe = safe_model_name(model_name)
    static_onnx = root / "data" / "models" / "onnx_static" / safe / "model.static.onnx"
    dynamic_onnx = root / "data" / "models" / "onnx" / safe / "model.onnx"
    return static_onnx if static_onnx.exists() else dynamic_onnx if dynamic_onnx.exists() else None


def _layer_pack_from_existing(root: Path, model_name: str, layer_index: int) -> dict[str, Any] | None:
    safe = safe_model_name(model_name)
    path = root / "reports" / "layer_subgraph_validation" / safe / f"layer_{layer_index}" / "index.json"
    return _load_json(path)


def _can_build_layers(loaded: dict[str, dict[str, Any]]) -> bool:
    return all(key in loaded for key in LAYER_BUILD_KEYS)


def build_or_load_layer_pack(
    *,
    root: Path,
    model_name: str,
    layer_index: int,
    loaded_artifacts: dict[str, dict[str, Any]],
    artifact_root: Path,
    export_onnx_subgraphs: bool,
    render_svg: bool,
    include_auxiliary: bool,
    strict: bool,
) -> dict[str, Any] | None:
    if _can_build_layers(loaded_artifacts):
        source_onnx = _source_onnx_path(root, model_name)
        pack = build_layer_subgraph_validation_pack(
            model_name=model_name,
            layer_index=layer_index,
            tensor_ir=loaded_artifacts["tensor_ir"],
            op_semantics=loaded_artifacts["op_semantics"],
            structural_region_tree=loaded_artifacts["structural_region_tree"],
            region_pruning_semantics=loaded_artifacts["region_pruning_semantics"],
            ranking=loaded_artifacts["ranking"],
            plans=loaded_artifacts["plans"],
            validations=loaded_artifacts["validations"],
            abstract_expansion=loaded_artifacts.get("abstract_expansion"),
            source_paths={},
            report_root=None,
            artifact_root=artifact_root,
            source_onnx_path=source_onnx,
            export_onnx=export_onnx_subgraphs,
            render_svg=render_svg,
            include_auxiliary=include_auxiliary,
            strict_onnx_export=strict,
        )
        return pack if isinstance(pack, dict) else pack.__dict__ | {"subgraphs": [item.__dict__ for item in pack.subgraphs]}
    return _layer_pack_from_existing(root, model_name, layer_index)


def polished_display_name(item: dict[str, Any]) -> str:
    name = item.get("display_name", "")
    category = item.get("semantic_category", "")
    layer = item.get("layer_index", "")
    ordinal = int(item.get("ordinal") or 0)
    if category == "layer_norm" and name == f"Layer {layer} LayerNorm":
        if ordinal and ordinal < 12:
            return f"Layer {layer} Attention Output LayerNorm"
        return f"Layer {layer} FFN Output LayerNorm"
    return name


def why_no_plan(item: dict[str, Any]) -> str:
    category = item.get("semantic_category", "")
    classification = item.get("classification", {})
    pruning_class = classification.get("pruning_class", "")
    plan_status = classification.get("plan_status", "")
    if plan_status not in {"no_plan_expected", "no_plan_but_expected"}:
        return ""
    if category == "ffn_intermediate_projection":
        return "safe component; full plan belongs to enclosing FeedForwardRegion."
    if category == "ffn_output_projection":
        return "repair component; consumer input repair belongs to the enclosing FFN plan."
    if pruning_class == "constrained":
        return "missing proof or blocker prevents a ready symbolic plan."
    if pruning_class == "blocked":
        return "semantic blocker prevents pruning under conservative rules."
    if pruning_class == "auxiliary":
        return "auxiliary propagation/metadata node, not a direct pruning target."
    return "no standalone pruning plan expected for this subgraph."


def polished_verdict(item: dict[str, Any]) -> str:
    category = item.get("semantic_category", "")
    classification = item.get("classification", {})
    validation = classification.get("validation_status")
    if category == "feed_forward_block":
        if validation == "valid":
            return "safe; symbolic FFN intermediate_dim plan exists and validates."
        return "safe opportunity; expected symbolic FFN plan should be reviewed."
    if category == "attention_skeleton":
        return "blocked/constrained; attention pruning requires head-axis mapping proof."
    if category == "attention_softmax":
        return "auxiliary/propagation-only; attention probability normalization, not a direct pruning target."
    if category == "attention_output_projection":
        return "constrained; input depends on attention context/head-axis mapping and output hidden_dim feeds residual path."
    if category in {"query_projection", "key_projection", "value_projection"}:
        return "constrained; learned projection but attention_head_mapping_unproven blocks a full plan."
    if category == "attention_score_matmul":
        return "blocked; Q x K^T contraction, not a learned parameter projection."
    if category == "attention_context_matmul":
        return "blocked; Softmax(scores) x V contraction, not a learned parameter projection."
    if category == "attention_mask_add":
        return "auxiliary/constraint carrier; mask broadcast over attention scores, not residual merge."
    if category == "residual_merge":
        return "blocked/protected; hidden_dim branch agreement is required."
    if category == "layer_norm":
        return "blocked/protected; hidden_dim and LayerNorm gamma/beta semantics are protected."
    if category == "ffn_intermediate_projection":
        return "safe component; full executable plan belongs to the enclosing FeedForwardRegion."
    if category == "ffn_output_projection":
        return "repair component; consumer input columns are pruned by the enclosing FFN plan."
    if category == "gelu_activation":
        return "auxiliary/propagation-only; index-preserving activation inside the FFN plan."
    return item.get("explanation") or "needs review."


def polish_subgraph(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["display_name"] = polished_display_name(out)
    out["verdict"] = polished_verdict(out)
    out["why_no_plan"] = why_no_plan(out)
    return out


def summarize_subgraph(item: dict[str, Any], report_path: Path, artifact_path: Path) -> dict[str, Any]:
    classification = item.get("classification", {})
    return {
        "ordinal": item.get("ordinal"),
        "display_name": item.get("display_name"),
        "node_slug": item.get("node_slug"),
        "semantic_category": item.get("semantic_category"),
        "pruning_class": classification.get("pruning_class", "unknown"),
        "plan_status": classification.get("plan_status", "unknown"),
        "validation_status": classification.get("validation_status", "unknown"),
        "onnx_status": item.get("onnx_export", {}).get("status", "skipped"),
        "report_path": str(report_path),
        "artifact_path": str(artifact_path),
    }


def layer_summary_from_pack(
    *,
    model_report_root: Path,
    artifact_root: Path,
    model_name: str,
    layer_index: int,
    pack: dict[str, Any],
) -> dict[str, Any]:
    safe = safe_model_name(model_name)
    layer_dir = model_report_root / safe / "layers" / f"layer_{layer_index}"
    subgraph_root = layer_dir / "subgraphs"
    artifact_layer_root = artifact_root / safe / "layers" / f"layer_{layer_index}"
    subgraphs = []
    for raw in _safe_list(pack.get("subgraphs")):
        item = polish_subgraph(raw)
        slug = item.get("node_slug") or f"{int(item.get('ordinal') or 0):02d}_subgraph"
        report_path = subgraph_root / slug / "analysis.json"
        artifact_path = artifact_layer_root / slug / "subgraph.onnx"
        subgraphs.append(summarize_subgraph(item, report_path, artifact_path))
    summary = pack.get("summary", {})
    return {
        "layer_index": layer_index,
        "total_subgraphs": summary.get("total_subgraphs", len(subgraphs)),
        "onnx_exported": summary.get("onnx_exported", 0),
        "onnx_skipped": summary.get("onnx_skipped", 0),
        "onnx_failed": summary.get("onnx_failed", 0),
        "safe": summary.get("safe_subgraphs", 0),
        "constrained": summary.get("constrained_subgraphs", 0),
        "blocked": summary.get("blocked_subgraphs", 0),
        "auxiliary": summary.get("auxiliary_subgraphs", 0),
        "unknown": summary.get("unknown_subgraphs", 0),
        "valid_plan_subgraphs": summary.get("valid_plan_subgraphs", 0),
        "layer_report_path": str(layer_dir / "index.md"),
        "subgraphs": subgraphs,
    }


def _ranking_summary(loaded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = loaded.get("ranking", {}).get("summary", {})
    return {
        "total_candidates": summary.get("total_candidates", 0),
        "safe": summary.get("safe_candidates", 0),
        "constrained": summary.get("constrained_candidates", 0),
        "blocked": summary.get("blocked_candidates", 0),
        "auxiliary": summary.get("auxiliary_candidates", 0),
        "unknown": summary.get("unknown_candidates", 0),
        "mlp_safe_candidates": summary.get("mlp_safe_candidates", 0),
        "attention_constrained_candidates": summary.get("attention_constrained_candidates", 0),
        "residual_blocked_candidates": summary.get("residual_blocked_candidates", 0),
        "layernorm_blocked_candidates": summary.get("layernorm_blocked_candidates", 0),
    }


def _plan_summary(loaded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return dict(loaded.get("plans", {}).get("summary", {}))


def _validation_summary(loaded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = loaded.get("validations", {}).get("summary", {})
    status_counts = summary.get("validation_status_counts", {})
    if not isinstance(status_counts, dict):
        status_counts = {}
    return {
        "total_validations": summary.get("total_validations", summary.get("total_plans", 0)),
        "valid": summary.get("valid", summary.get("valid_plans", status_counts.get("valid", 0))),
        "warning": summary.get("warning", summary.get("warning_plans", status_counts.get("warning", 0))),
        "invalid": summary.get("invalid", summary.get("invalid_plans", status_counts.get("invalid", 0))),
        "unknown": summary.get("unknown", summary.get("unknown_plans", status_counts.get("unknown", 0))),
    }


def _op_semantic_counts(loaded: dict[str, dict[str, Any]]) -> dict[str, int]:
    return dict(loaded.get("op_semantics", {}).get("summary", {}).get("semantic_kind_counts", {}))


def _region_category_counts(loaded: dict[str, dict[str, Any]]) -> dict[str, int]:
    return dict(loaded.get("region_pruning_semantics", {}).get("summary", {}).get("semantic_category_counts", {}))


def _safe_opportunities(loaded: dict[str, dict[str, Any]], plans: dict[str, dict[str, Any]], validations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for candidate in _safe_list(loaded.get("ranking", {}).get("candidates")):
        if candidate.get("pruning_class") != "safe":
            continue
        plan = plans.get(candidate.get("candidate_id", ""), {})
        validation = validations.get(plan.get("plan_id", ""), {})
        out.append(
            {
                "layer": _infer_layer(candidate.get("region_name", "")),
                "region_name": candidate.get("region_name"),
                "semantic_category": candidate.get("semantic_category"),
                "target_dimension": candidate.get("target_dimension"),
                "candidate_id": candidate.get("candidate_id"),
                "rank_score": candidate.get("rank_score"),
                "plan_id": plan.get("plan_id", ""),
                "validation_status": validation.get("validation_status", ""),
            }
        )
    return out


def _infer_layer(text: str) -> int | None:
    match = re.search(r"Layer\s+(\d+)|layer\.(\d+)", str(text))
    if not match:
        return None
    return int(next(group for group in match.groups() if group is not None))


def _candidates_by_class(loaded: dict[str, dict[str, Any]], pruning_class: str, limit: int = 200) -> list[dict[str, Any]]:
    out = []
    for candidate in _safe_list(loaded.get("ranking", {}).get("candidates")):
        if candidate.get("pruning_class") != pruning_class:
            continue
        out.append(
            {
                "layer": _infer_layer(candidate.get("region_name", "")),
                "region_name": candidate.get("region_name"),
                "semantic_category": candidate.get("semantic_category"),
                "candidate_kind": candidate.get("candidate_kind"),
                "target_dimension": candidate.get("target_dimension"),
                "rank_score": candidate.get("rank_score"),
                "blockers": candidate.get("blockers", []),
                "reason": candidate.get("reason", ""),
            }
        )
    return out[:limit]


def _summed_layer_counts(layers: list[dict[str, Any]]) -> dict[str, int]:
    keys = ["total_subgraphs", "onnx_exported", "onnx_skipped", "onnx_failed", "safe", "constrained", "blocked", "auxiliary", "unknown", "valid_plan_subgraphs"]
    return {key: sum(int(layer.get(key, 0) or 0) for layer in layers) for key in keys}


def build_full_model_analysis_report(
    *,
    root: Path,
    model_name: str,
    layers: list[int],
    output_root: Path,
    artifact_root: Path,
    export_onnx_subgraphs: bool = True,
    render_svg: bool = False,
    include_auxiliary: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    from model_analysis.full_model_analysis_report_text import (
        compact_model_snapshot_to_markdown,
        layer_report_to_markdown,
        model_report_to_markdown,
        subgraph_explanation_to_markdown,
        summary_to_markdown,
    )

    safe = safe_model_name(model_name)
    available, loaded, missing = discover_model_artifacts(root, model_name)
    required_missing = missing_required_artifacts(missing)
    if required_missing:
        raise FileNotFoundError(
            "Required model artifacts missing: "
            + ", ".join(f"{item['artifact']}={item['path']}" for item in required_missing)
        )

    plan_by_candidate = {
        plan.get("candidate_id", ""): plan for plan in _safe_list(loaded.get("plans", {}).get("plans"))
    }
    validation_by_plan = {
        validation.get("plan_id", ""): validation
        for validation in _safe_list(loaded.get("validations", {}).get("validations"))
    }

    if (output_root / safe).exists():
        shutil.rmtree(output_root / safe)
    ensure_dir(output_root / safe)
    tmp_artifact_root = artifact_root / "_tmp_layer_subgraphs"
    if tmp_artifact_root.exists():
        shutil.rmtree(tmp_artifact_root)

    layer_packs: list[dict[str, Any]] = []
    layer_summaries: list[dict[str, Any]] = []
    for layer_index in layers:
        pack = build_or_load_layer_pack(
            root=root,
            model_name=model_name,
            layer_index=layer_index,
            loaded_artifacts=loaded,
            artifact_root=tmp_artifact_root,
            export_onnx_subgraphs=export_onnx_subgraphs,
            render_svg=render_svg,
            include_auxiliary=include_auxiliary,
            strict=strict,
        )
        if not pack:
            missing.append(
                {
                    "artifact": "layer_subgraph_pack",
                    "label": f"Layer {layer_index} Subgraph Pack",
                    "path": str(root / "reports" / "layer_subgraph_validation" / safe / f"layer_{layer_index}"),
                }
            )
            continue
        relocate_layer_artifacts(pack, tmp_artifact_root / safe / f"layer_{layer_index}", artifact_root / safe / "layers" / f"layer_{layer_index}")
        pack["subgraphs"] = [polish_subgraph(item) for item in _safe_list(pack.get("subgraphs"))]
        layer_packs.append(pack)
        layer_summary = layer_summary_from_pack(
            model_report_root=output_root,
            artifact_root=artifact_root,
            model_name=model_name,
            layer_index=layer_index,
            pack=pack,
        )
        layer_summaries.append(layer_summary)
        write_layer_report_files(output_root, artifact_root, model_name, pack, layer_summary, layer_report_to_markdown, subgraph_explanation_to_markdown)

    layer_totals = _summed_layer_counts(layer_summaries)
    model_summary = {
        "layers_generated": len(layer_summaries),
        **layer_totals,
        "ranking": _ranking_summary(loaded),
        "plans": _plan_summary(loaded),
        "plan_validation": _validation_summary(loaded),
        "op_semantic_counts": _op_semantic_counts(loaded),
        "region_semantic_counts": _region_category_counts(loaded),
    }
    report = {
        "model_name": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "available_artifacts": available,
        "missing_artifacts": missing,
        "model_summary": model_summary,
        "layers": layer_summaries,
        "safe_opportunities": _safe_opportunities(loaded, plan_by_candidate, validation_by_plan),
        "constrained_opportunities": _candidates_by_class(loaded, "constrained"),
        "blocked_structures": _candidates_by_class(loaded, "blocked"),
        "auxiliary_structures": _candidates_by_class(loaded, "auxiliary", limit=100),
        "report_paths": {
            "root": str(output_root / safe),
            "index_json": str(output_root / safe / "index.json"),
            "index_md": str(output_root / safe / "index.md"),
        },
        "artifact_paths": {"root": str(artifact_root / safe)},
    }
    write_json(report, output_root / safe / "index.json")
    write_markdown(model_report_to_markdown(report), output_root / safe / "index.md")
    write_json(report, output_root / safe / "compact_snapshot.json")
    write_markdown(compact_model_snapshot_to_markdown(report), output_root / safe / "compact_snapshot.md")
    write_summary_files(output_root / safe / "summaries", report, summary_to_markdown)
    if tmp_artifact_root.exists():
        shutil.rmtree(tmp_artifact_root)
    return report


def relocate_layer_artifacts(pack: dict[str, Any], source_dir: Path, target_dir: Path) -> None:
    if source_dir.exists():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        ensure_dir(target_dir.parent)
        shutil.move(str(source_dir), str(target_dir))
    for item in _safe_list(pack.get("subgraphs")):
        onnx = item.get("onnx_export", {})
        old_path = Path(str(onnx.get("output_path", "")))
        if old_path.name:
            new_path = target_dir / item.get("node_slug", "") / old_path.name
            if str(old_path):
                onnx["output_path"] = str(new_path)


def write_layer_report_files(
    output_root: Path,
    artifact_root: Path,
    model_name: str,
    pack: dict[str, Any],
    layer_summary: dict[str, Any],
    layer_markdown_fn,
    subgraph_markdown_fn,
) -> None:
    safe = safe_model_name(model_name)
    layer_index = int(pack.get("layer_index", layer_summary.get("layer_index", 0)))
    layer_dir = output_root / safe / "layers" / f"layer_{layer_index}"
    subgraph_root = layer_dir / "subgraphs"
    ensure_dir(subgraph_root)
    layer_data = {"pack": pack, "summary": layer_summary}
    write_json(layer_data, layer_dir / "index.json")
    write_markdown(layer_markdown_fn(layer_data), layer_dir / "index.md")
    for item in _safe_list(pack.get("subgraphs")):
        slug = item.get("node_slug") or f"{int(item.get('ordinal') or 0):02d}_subgraph"
        node_dir = subgraph_root / slug
        ensure_dir(node_dir)
        write_json(item, node_dir / "analysis.json")
        write_markdown(subgraph_markdown_fn(item), node_dir / "explanation.md")
    del artifact_root


def write_summary_files(summary_root: Path, report: dict[str, Any], summary_markdown_fn) -> None:
    summaries = {
        "opportunity_summary": {
            "safe": report.get("safe_opportunities", []),
            "constrained": report.get("constrained_opportunities", []),
            "blocked": report.get("blocked_structures", []),
            "auxiliary": report.get("auxiliary_structures", []),
            "ranking_summary": report.get("model_summary", {}).get("ranking", {}),
        },
        "plan_summary": report.get("model_summary", {}).get("plans", {}),
        "validation_summary": report.get("model_summary", {}).get("plan_validation", {}),
        "attention_summary": {
            "constrained": [
                item
                for item in report.get("constrained_opportunities", [])
                if "attention" in str(item.get("semantic_category", "")) or "Projection" in str(item.get("region_name", ""))
            ],
            "blocked": [
                item
                for item in report.get("blocked_structures", [])
                if "attention" in str(item.get("semantic_category", "")) or "Attention" in str(item.get("region_name", ""))
            ],
        },
        "feedforward_summary": {
            "safe": [
                item
                for item in report.get("safe_opportunities", [])
                if item.get("semantic_category") in {"feed_forward_block", "ffn_intermediate_projection"}
            ],
            "valid_ffn_plans": report.get("model_summary", {}).get("plan_validation", {}).get("valid", 0),
        },
    }
    for name, data in summaries.items():
        write_json(data, summary_root / f"{name}.json")
        write_markdown(summary_markdown_fn(name, data), summary_root / f"{name}.md")
