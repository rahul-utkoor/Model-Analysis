"""Pruning-relevant semantics for primitive Tensor IR operations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class OpPruningEffect:
    direct_pruning: str
    reason: str
    required_repairs: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass
class OpSemanticsRecord:
    op_id: str
    source_name: str
    op_type: str
    topological_index: int
    semantic_kind: str
    semantic_category: str
    parameterized: bool | str
    index_behavior: str
    dimension_roles: dict[str, str]
    pruning_effect: OpPruningEffect
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpSemanticsIR:
    model_name: str
    frontend: str
    source_tensor_ir_path: str
    source_region_tree_path: str | None
    source_region_pruning_semantics_path: str | None
    generated_at: str
    ops: list[OpSemanticsRecord]
    summary: dict[str, Any]


def op_pruning_effect_to_dict(value: OpPruningEffect) -> dict[str, Any]:
    return asdict(value)


def op_semantics_record_to_dict(value: OpSemanticsRecord) -> dict[str, Any]:
    return asdict(value)


def op_semantics_ir_to_dict(value: OpSemanticsIR) -> dict[str, Any]:
    return asdict(value)


def write_op_semantics_json(value: OpSemanticsIR | dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    data = op_semantics_ir_to_dict(value) if isinstance(value, OpSemanticsIR) else value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_op_semantics_json(path: Path) -> OpSemanticsIR:
    data = json.loads(path.read_text(encoding="utf-8"))
    return OpSemanticsIR(
        model_name=data.get("model_name", "model"),
        frontend=data.get("frontend", "unknown"),
        source_tensor_ir_path=data.get("source_tensor_ir_path", ""),
        source_region_tree_path=data.get("source_region_tree_path"),
        source_region_pruning_semantics_path=data.get("source_region_pruning_semantics_path"),
        generated_at=data.get("generated_at", ""),
        ops=[
            OpSemanticsRecord(
                op_id=item["op_id"],
                source_name=item.get("source_name", ""),
                op_type=item.get("op_type", ""),
                topological_index=int(item.get("topological_index", 0)),
                semantic_kind=item.get("semantic_kind", "unknown"),
                semantic_category=item.get("semantic_category", "unknown"),
                parameterized=item.get("parameterized", "unknown"),
                index_behavior=item.get("index_behavior", "unknown"),
                dimension_roles=item.get("dimension_roles", {}),
                pruning_effect=OpPruningEffect(**item.get("pruning_effect", {"direct_pruning": "unknown", "reason": ""})),
                evidence=item.get("evidence", {}),
            )
            for item in data.get("ops", [])
        ],
        summary=data.get("summary", {}),
    )


def _source_name(op: dict[str, Any]) -> str:
    return str(op.get("source_node_name") or op.get("name") or op.get("op_id") or "")


def _source_path(op: dict[str, Any]) -> str:
    return _source_name(op).lower()


def _topological_index(op: dict[str, Any], fallback: int) -> int:
    source_location = op.get("source_location") or {}
    if "node_index" in source_location:
        return int(source_location["node_index"])
    op_id = str(op.get("op_id", ""))
    for part in op_id.split("::"):
        if part.isdigit():
            return int(part)
    return fallback


def _effect(direct: str, reason: str, repairs: list[str] | None = None, blockers: list[str] | None = None) -> OpPruningEffect:
    return OpPruningEffect(direct, reason, repairs or [], blockers or [])


def _is_parameterized_projection_path(path: str) -> bool:
    return any(
        token in path
        for token in (
            "/attention/self/query/matmul",
            "/attention/self/key/matmul",
            "/attention/self/value/matmul",
            "/attention/output/dense/matmul",
            "/intermediate/dense/matmul",
            "/output/dense/matmul",
            "/cls/",
        )
    ) and path.endswith("matmul")


def _is_projection_bias_path(path: str) -> bool:
    return any(
        token in path
        for token in (
            "/attention/self/query/add",
            "/attention/self/key/add",
            "/attention/self/value/add",
            "/attention/output/dense/add",
            "/intermediate/dense/add",
            "/output/dense/add",
            "/cls/",
        )
    ) and path.endswith("add")


def _is_attention_score_matmul(path: str) -> bool:
    return (
        path.endswith("/attention/self/matmul")
        and "/attention/self/query/" not in path
        and "/attention/self/key/" not in path
        and "/attention/self/value/" not in path
    )


def _is_attention_context_matmul(path: str) -> bool:
    return path.endswith("/attention/self/matmul_1")


def _is_attention_mask_add(path: str) -> bool:
    return (
        path.endswith("/attention/self/add")
        and "/attention/output/add" not in path
        and "/output/add" not in path
        and "/embeddings/add" not in path
    )


def _is_true_residual_add(path: str) -> bool:
    return any(path.endswith(token) for token in ("/attention/output/add", "/output/add", "/embeddings/add", "/embeddings/add_1"))


def _is_gelu_path(path: str) -> bool:
    return "/intermediate/intermediate_act_fn/" in path or "gelu" in path


def _projection_roles(path: str) -> dict[str, str]:
    if "/intermediate/dense/" in path:
        return {"input": "hidden_dim", "output": "intermediate_dim"}
    if "/output/dense/" in path and "/attention/output/dense/" not in path:
        return {"input": "intermediate_dim", "output": "hidden_dim"}
    if "/attention/self/query/" in path:
        return {"input": "hidden_dim", "output": "head_dim"}
    if "/attention/self/key/" in path:
        return {"input": "hidden_dim", "output": "head_dim"}
    if "/attention/self/value/" in path:
        return {"input": "hidden_dim", "output": "head_dim"}
    if "/attention/output/dense/" in path:
        return {"input": "hidden_dim", "output": "hidden_dim"}
    if "/cls/" in path:
        return {"input": "hidden_dim", "output": "prediction_dim"}
    return {"input": "unknown", "output": "unknown"}


def _axis_transform_kind(op_type: str) -> str:
    return {
        "Reshape": "reshape",
        "Transpose": "transpose",
        "Shape": "shape",
        "Concat": "concat",
        "Slice": "slice",
        "Unsqueeze": "unsqueeze",
        "Squeeze": "squeeze",
        "Cast": "cast",
        "Constant": "constant",
        "ConstantOfShape": "constant",
        "Gather": "shape",
        "Range": "shape",
    }.get(op_type, "metadata_only")


def _classify_op(op: dict[str, Any]) -> tuple[str, str, bool | str, str, dict[str, str], OpPruningEffect, list[str], str]:
    op_type = str(op.get("op_type", ""))
    path = _source_path(op)
    matched: list[str] = []

    if _is_parameterized_projection_path(path):
        matched.append("parameterized_projection_path")
        roles = _projection_roles(path)
        return (
            "parameterized_linear_matmul",
            "parameterized_projection",
            True,
            "creates_prunable_output_axis",
            roles,
            _effect("allowed", "Learned projection MatMul exposes row/column pruning axes; region semantics decides legality."),
            matched,
            "high",
        )
    if _is_projection_bias_path(path):
        matched.append("projection_bias_add_path")
        return (
            "linear_bias_add",
            "parameterized_projection",
            True,
            "index_preserving",
            {"input": _projection_roles(path).get("output", "unknown"), "output": _projection_roles(path).get("output", "unknown")},
            _effect("not_applicable", "Bias follows the projection output axis.", ["prune_bias"]),
            matched,
            "high",
        )
    if op_type == "Gather" and "/embeddings/" in path and any(token in path for token in ("word_embeddings", "token_type_embeddings", "position_embeddings")):
        matched.append("embedding_gather_path")
        return (
            "embedding_gather",
            "embedding_lookup",
            True,
            "no_pruning_relevance",
            {"input": "vocab_dim", "output": "hidden_dim"},
            _effect("blocked", "Embedding table and lookup dimensions are protected by default.", blockers=["embedding_vocab_or_hidden_protected"]),
            matched,
            "high",
        )
    if _is_gelu_path(path) and op_type in {"Div", "Add", "Mul", "Erf"}:
        matched.append("gelu_decomposition_path")
        kind = "gelu_erf" if op_type == "Erf" else "gelu_mul" if op_type == "Mul" else "gelu_elementwise"
        return (
            kind,
            "elementwise_index_preserving",
            False,
            "index_preserving",
            {"input": "intermediate_dim", "output": "intermediate_dim"},
            _effect("not_applicable", "GELU decomposition preserves selected intermediate_dim indices."),
            matched,
            "high",
        )
    if op_type in {"LayerNormalization", "SkipLayerNormalization"} or "/layernorm/" in path or "/layernormalization" in path:
        matched.append("layernorm_path")
        return (
            "layernorm",
            "normalization",
            True,
            "index_preserving",
            {"input": "hidden_dim", "output": "hidden_dim"},
            _effect("blocked", "LayerNorm hidden width is protected unless gamma/beta and downstream hidden paths are repaired.", ["layernorm_parameter_repair"], ["layernorm_hidden_dim"]),
            matched,
            "high",
        )
    if op_type == "Add" and _is_true_residual_add(path):
        matched.append("residual_add_path")
        return (
            "residual_add",
            "branch_merge",
            False,
            "branch_agreement_required",
            {"input": "hidden_dim", "output": "hidden_dim"},
            _effect("blocked", "Residual branches must preserve compatible hidden dimensions.", ["residual_branch_repair"], ["residual_hidden_dim"]),
            matched,
            "high",
        )
    if op_type == "MatMul" and _is_attention_score_matmul(path):
        matched.append("attention_score_matmul_path")
        return (
            "attention_score_matmul",
            "attention_contraction",
            False,
            "axis_contraction",
            {"input": "head_dim", "output": "attention_score_dim", "sequence": "sequence_dim"},
            _effect("blocked", "Q x K^T contraction, not a learned parameter projection.", blockers=["attention_head_mapping_unproven"]),
            matched,
            "high",
        )
    if op_type == "MatMul" and _is_attention_context_matmul(path):
        matched.append("attention_context_matmul_path")
        return (
            "attention_context_matmul",
            "attention_contraction",
            False,
            "axis_contraction",
            {"input": "attention_score_dim", "output": "hidden_dim", "sequence": "sequence_dim", "head": "head_dim"},
            _effect("blocked", "Softmax(scores) x V contraction, not a learned parameter projection.", blockers=["attention_head_mapping_unproven"]),
            matched,
            "high",
        )
    if op_type == "Add" and _is_attention_mask_add(path):
        matched.append("attention_mask_add_path")
        return (
            "attention_mask_add",
            "attention_masking",
            False,
            "broadcast_metadata",
            {"input": "attention_score_dim", "mask": "mask_dim", "output": "attention_score_dim", "sequence": "sequence_dim"},
            _effect("not_applicable", "Attention mask Add biases attention scores; it is not residual hidden-state pruning.", ["shape_metadata_update"], ["unknown_axis_mapping"]),
            matched,
            "high",
        )
    if op_type == "Softmax" and "/attention/self/softmax" in path:
        matched.append("attention_softmax_path")
        return (
            "attention_softmax",
            "attention_contraction",
            False,
            "index_preserving",
            {"input": "attention_score_dim", "output": "attention_score_dim", "sequence": "sequence_dim"},
            _effect("not_applicable", "Softmax normalizes attention scores and carries head/axis mapping constraints.", blockers=["attention_head_mapping_unproven"]),
            matched,
            "high",
        )
    if op_type == "Where" and "/attention/self/where" in path:
        matched.append("attention_mask_select_path")
        return (
            "attention_mask_select",
            "attention_masking",
            False,
            "broadcast_metadata",
            {"input": "mask_dim", "output": "attention_score_dim"},
            _effect("not_applicable", "Attention mask selection carries broadcast metadata.", ["shape_metadata_update"]),
            matched,
            "high",
        )
    if op_type in {"GreaterOrEqual", "Greater", "Less", "LessOrEqual", "Equal", "And", "Or", "Not", "IsNaN"}:
        matched.append("predicate_op_type")
        return (
            "comparison_predicate",
            "metadata_flow",
            False,
            "metadata_only",
            {"input": "symbolic_axis", "output": "mask_dim"},
            _effect("not_applicable", "Predicate construction is metadata/mask flow, not a pruning target.", ["shape_metadata_update"]),
            matched,
            "medium",
        )
    if op_type == "Where":
        matched.append("where_op_type")
        return (
            "where",
            "metadata_flow",
            False,
            "broadcast_metadata",
            {"input": "mask_dim", "output": "unknown"},
            _effect("not_applicable", "Where/select carries mask metadata unless a region proves another semantic role.", ["shape_metadata_update"]),
            matched,
            "medium",
        )
    if op_type in {"Reshape", "Transpose", "Unsqueeze", "Squeeze", "Concat", "Slice"}:
        matched.append("axis_transform_op_type")
        return (
            _axis_transform_kind(op_type),
            "axis_transform",
            False,
            "axis_remap_required",
            {"input": "symbolic_axis", "output": "symbolic_axis"},
            _effect("not_applicable", "Axis transform may remap pruning indices and requires shape metadata tracking.", ["shape_metadata_update"], ["unknown_axis_mapping"]),
            matched,
            "high",
        )
    if op_type in {"Shape", "Cast", "Constant", "ConstantOfShape", "Range"} or (op_type == "Gather" and op.get("canonical_op_type") == "shape_op"):
        matched.append("metadata_op_type")
        return (
            _axis_transform_kind(op_type),
            "metadata_flow",
            False,
            "metadata_only",
            {"input": "symbolic_axis", "output": "symbolic_axis"},
            _effect("not_applicable", "Metadata-only operation constructs shape, constants, or indices.", ["shape_metadata_update"]),
            matched,
            "high",
        )
    return (
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        {"input": "unknown", "output": "unknown"},
        _effect("unknown", "No conservative pruning semantics rule matched this op.", blockers=["unsupported_or_unknown_op_semantics"]),
        ["fallback_unknown"],
        "low",
    )


def _region_maps(structural_region_tree: dict[str, Any] | None) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not structural_region_tree:
        return out
    for region in structural_region_tree.get("regions", []):
        for op_id in region.get("op_ids", []):
            out[op_id].append(
                {
                    "region_id": str(region.get("region_id", "")),
                    "region_name": str(region.get("name", region.get("region_id", ""))),
                    "region_type": str(region.get("region_type", "")),
                }
            )
    return out


def _region_semantics_maps(region_pruning_semantics: dict[str, Any] | None) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not region_pruning_semantics:
        return out
    for region in region_pruning_semantics.get("regions", []):
        evidence = region.get("evidence", {})
        for op_id in evidence.get("source_ops", []):
            out[op_id].append(
                {
                    "region_id": str(region.get("region_id", "")),
                    "region_name": str(region.get("region_name", "")),
                    "semantic_category": str(region.get("semantic_category", "")),
                    "pruning_role": str(region.get("pruning_role", "")),
                }
            )
    return out


def build_op_semantics_ir(
    tensor_ir: dict[str, Any],
    *,
    structural_region_tree: dict[str, Any] | None = None,
    region_pruning_semantics: dict[str, Any] | None = None,
    abstract_expansion_report: dict[str, Any] | None = None,
    source_tensor_ir_path: str = "",
    source_region_tree_path: str | None = None,
    source_region_pruning_semantics_path: str | None = None,
) -> OpSemanticsIR:
    del abstract_expansion_report
    nearby_regions = _region_maps(structural_region_tree)
    nearby_semantics = _region_semantics_maps(region_pruning_semantics)
    records: list[OpSemanticsRecord] = []
    for index, op in enumerate(tensor_ir.get("ops", [])):
        kind, category, parameterized, behavior, roles, effect, matched, confidence = _classify_op(op)
        op_id = str(op.get("op_id", ""))
        evidence = {
            "source_path": _source_name(op),
            "matched_patterns": matched,
            "nearby_region_ids": [item["region_id"] for item in nearby_regions.get(op_id, [])],
            "nearby_region_names": [item["region_name"] for item in nearby_regions.get(op_id, [])],
            "nearby_region_semantics": nearby_semantics.get(op_id, []),
            "confidence": confidence,
        }
        records.append(
            OpSemanticsRecord(
                op_id=op_id,
                source_name=_source_name(op),
                op_type=str(op.get("op_type", "")),
                topological_index=_topological_index(op, index),
                semantic_kind=kind,
                semantic_category=category,
                parameterized=parameterized,
                index_behavior=behavior,
                dimension_roles=roles,
                pruning_effect=effect,
                evidence=evidence,
            )
        )
    records.sort(key=lambda item: (item.topological_index, item.op_id))
    return OpSemanticsIR(
        model_name=tensor_ir.get("model_name", "model"),
        frontend=tensor_ir.get("source_frontend", "unknown"),
        source_tensor_ir_path=source_tensor_ir_path,
        source_region_tree_path=source_region_tree_path,
        source_region_pruning_semantics_path=source_region_pruning_semantics_path,
        generated_at=datetime.now(timezone.utc).isoformat(),
        ops=records,
        summary=_summary(records),
    )


def _summary(records: list[OpSemanticsRecord]) -> dict[str, Any]:
    kind_counts = Counter(item.semantic_kind for item in records)
    category_counts = Counter(item.semantic_category for item in records)
    parameterized_count = sum(1 for item in records if item.parameterized is True)
    direct_relevant = sum(1 for item in records if item.pruning_effect.direct_pruning in {"allowed", "blocked"})
    metadata_only = sum(1 for item in records if item.index_behavior == "metadata_only")
    unknown = kind_counts.get("unknown", 0)
    blocker_counts = Counter(blocker for item in records for blocker in item.pruning_effect.blockers)
    repair_counts = Counter(repair for item in records for repair in item.pruning_effect.required_repairs)
    behavior_counts = Counter(item.index_behavior for item in records)
    return {
        "num_ops": len(records),
        "semantic_kind_counts": dict(sorted(kind_counts.items())),
        "semantic_category_counts": dict(sorted(category_counts.items())),
        "index_behavior_counts": dict(sorted(behavior_counts.items())),
        "parameterized_ops": parameterized_count,
        "directly_pruning_relevant_ops": direct_relevant,
        "metadata_only_ops": metadata_only,
        "unknown_ops": unknown,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "repair_counts": dict(sorted(repair_counts.items())),
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


def _count_rows(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [{"item": key, "count": value} for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _row(op: dict[str, Any]) -> dict[str, Any]:
    effect = op.get("pruning_effect", {})
    return {
        "idx": op.get("topological_index"),
        "source": op.get("source_name"),
        "op_type": op.get("op_type"),
        "kind": op.get("semantic_kind"),
        "category": op.get("semantic_category"),
        "direct": effect.get("direct_pruning"),
        "blockers": ", ".join(effect.get("blockers", [])),
    }


def op_semantics_ir_to_markdown(value: OpSemanticsIR | dict[str, Any]) -> str:
    data = op_semantics_ir_to_dict(value) if isinstance(value, OpSemanticsIR) else value
    summary = data.get("summary", {})
    ops = data.get("ops", [])
    parameterized = [op for op in ops if op.get("semantic_category") == "parameterized_projection"]
    attention = [op for op in ops if op.get("semantic_category") == "attention_contraction"]
    elementwise = [op for op in ops if op.get("semantic_category") == "elementwise_index_preserving"]
    branch = [op for op in ops if op.get("semantic_category") == "branch_merge"]
    axis_metadata = [op for op in ops if op.get("semantic_category") in {"axis_transform", "metadata_flow", "attention_masking"}]
    unknown = [op for op in ops if op.get("semantic_kind") == "unknown"]
    return "\n".join(
        [
            f"# Op Semantics: {data.get('model_name', '')}",
            "",
            "## Summary",
            "",
            f"- Frontend: `{data.get('frontend', 'unknown')}`",
            f"- Total ops: `{summary.get('num_ops', 0)}`",
            f"- Parameterized ops: `{summary.get('parameterized_ops', 0)}`",
            f"- Directly pruning-relevant ops: `{summary.get('directly_pruning_relevant_ops', 0)}`",
            f"- Metadata-only ops: `{summary.get('metadata_only_ops', 0)}`",
            f"- Unknown ops: `{summary.get('unknown_ops', 0)}`",
            "",
            "## Semantic Kind Counts",
            "",
            _table(_count_rows(summary.get("semantic_kind_counts", {})), ["item", "count"], limit=120),
            "",
            "## Semantic Category Counts",
            "",
            _table(_count_rows(summary.get("semantic_category_counts", {})), ["item", "count"], limit=80),
            "",
            "## Parameterized Projection Ops",
            "",
            _table([_row(op) for op in parameterized], ["idx", "source", "op_type", "kind", "direct", "blockers"], limit=120),
            "",
            "## Attention Contraction Ops",
            "",
            _table([_row(op) for op in attention], ["idx", "source", "op_type", "kind", "direct", "blockers"], limit=80),
            "",
            "## Elementwise Index-Preserving Ops",
            "",
            _table([_row(op) for op in elementwise], ["idx", "source", "op_type", "kind", "direct"], limit=80),
            "",
            "## Branch Merge Ops",
            "",
            _table([_row(op) for op in branch], ["idx", "source", "op_type", "kind", "direct", "blockers"], limit=80),
            "",
            "## Axis/Metadata Ops",
            "",
            _table([_row(op) for op in axis_metadata], ["idx", "source", "op_type", "kind", "category", "direct"], limit=120),
            "",
            "## Unknown Ops",
            "",
            _table([_row(op) for op in unknown], ["idx", "source", "op_type", "kind", "direct", "blockers"], limit=120),
            "",
            "## Interpretation",
            "",
            "Op Semantics annotates primitive Tensor IR operations with local pruning-relevant behavior. It is a static analysis artifact and does not execute pruning or modify models.",
            "",
        ]
    )

