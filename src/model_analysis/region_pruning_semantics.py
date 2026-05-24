"""Static pruning propagation semantics for structural regions."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_analysis.paths import ensure_dir


@dataclass
class DimensionSemantics:
    dim_name: str
    symbolic_role: str
    status: str
    source: str
    reason: str


@dataclass
class PropagationRule:
    rule_id: str
    rule_type: str
    source_dimension: str
    target_dimensions: list[str]
    index_mapping: str
    direction: str
    explanation: str


@dataclass
class RepairObligation:
    obligation_id: str
    obligation_type: str
    affected_regions: list[str]
    affected_dimensions: list[str]
    required: bool
    explanation: str


@dataclass
class Blocker:
    blocker_id: str
    blocker_type: str
    severity: str
    explanation: str


@dataclass
class RegionSemanticsRecord:
    region_id: str
    region_name: str
    region_type: str
    section: str
    op_range: str
    primitive_leaf_count: int
    pruning_role: str
    dimensions: list[DimensionSemantics] = field(default_factory=list)
    propagation_rules: list[PropagationRule] = field(default_factory=list)
    repair_obligations: list[RepairObligation] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionPruningSemantics:
    model_name: str
    frontend: str
    source_region_tree_path: str
    source_region_dimension_ir_path: str | None
    generated_at: str
    regions: list[RegionSemanticsRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def dimension_semantics_to_dict(value: DimensionSemantics) -> dict[str, Any]:
    return asdict(value)


def propagation_rule_to_dict(value: PropagationRule) -> dict[str, Any]:
    return asdict(value)


def repair_obligation_to_dict(value: RepairObligation) -> dict[str, Any]:
    return asdict(value)


def blocker_to_dict(value: Blocker) -> dict[str, Any]:
    return asdict(value)


def region_semantics_record_to_dict(value: RegionSemanticsRecord) -> dict[str, Any]:
    return asdict(value)


def region_pruning_semantics_to_dict(value: RegionPruningSemantics) -> dict[str, Any]:
    return asdict(value)


def write_region_pruning_semantics_json(value: RegionPruningSemantics, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(region_pruning_semantics_to_dict(value), indent=2), encoding="utf-8")


def load_region_pruning_semantics_json(path: Path) -> RegionPruningSemantics:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RegionPruningSemantics(
        model_name=data["model_name"],
        frontend=data.get("frontend", "unknown"),
        source_region_tree_path=data.get("source_region_tree_path", ""),
        source_region_dimension_ir_path=data.get("source_region_dimension_ir_path"),
        generated_at=data.get("generated_at", ""),
        regions=[
            RegionSemanticsRecord(
                region_id=item["region_id"],
                region_name=item.get("region_name", item["region_id"]),
                region_type=item.get("region_type", "UnknownRegion"),
                section=item.get("section", "Other Main Flow"),
                op_range=item.get("op_range", "-"),
                primitive_leaf_count=item.get("primitive_leaf_count", 0),
                pruning_role=item.get("pruning_role", "unknown"),
                dimensions=[DimensionSemantics(**dim) for dim in item.get("dimensions", [])],
                propagation_rules=[PropagationRule(**rule) for rule in item.get("propagation_rules", [])],
                repair_obligations=[RepairObligation(**repair) for repair in item.get("repair_obligations", [])],
                blockers=[Blocker(**blocker) for blocker in item.get("blockers", [])],
                evidence=item.get("evidence", {}),
            )
            for item in data.get("regions", [])
        ],
        summary=data.get("summary", {}),
    )


def _norm(value: Any) -> str:
    return str(value or "").replace("\\", "/").lower()


def _compact_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.lower()).strip("_") or "unknown"


def _ops(tensor_ir: dict[str, Any]) -> list[dict[str, Any]]:
    return tensor_ir.get("ops") or tensor_ir.get("operations") or tensor_ir.get("tensor_ops") or []


def _op_id(op: dict[str, Any]) -> str:
    return str(op.get("op_id") or op.get("id") or op.get("name"))


def _op_name(op: dict[str, Any], fallback: str) -> str:
    return str(op.get("source_node_name") or op.get("onnx_node_name") or op.get("name") or fallback)


def _op_type(op: dict[str, Any]) -> str:
    return str(op.get("canonical_op_type") or op.get("op_type") or op.get("type") or "")


def _tensor_maps(tensor_ir: dict[str, Any]) -> dict[str, Any]:
    op_by_id: dict[str, dict[str, Any]] = {}
    op_order: dict[str, int] = {}
    for index, op in enumerate(_ops(tensor_ir)):
        oid = _op_id(op)
        if oid:
            op_by_id[oid] = op
            op_order[oid] = index
    return {"op_by_id": op_by_id, "op_order": op_order}


def _op_order(op_id: str, tm: dict[str, Any]) -> int:
    if op_id in tm["op_order"]:
        return tm["op_order"][op_id]
    match = re.findall(r"\d+", op_id)
    return int(match[-1]) if match else 10**12


def _regions_and_children(tree: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str | None, list[str]], dict[str, dict[str, Any]]]:
    regions = {item["region_id"]: item for item in tree.get("regions", []) if item.get("region_id")}
    children: dict[str | None, list[str]] = defaultdict(list)
    interfaces = {item["region_id"]: item for item in tree.get("interfaces", []) if item.get("region_id")}
    for region in regions.values():
        children[region.get("parent")].append(region["region_id"])
    return regions, children, interfaces


def _region_children(region_id: str, regions: dict[str, dict[str, Any]], children: dict[str | None, list[str]]) -> list[str]:
    explicit = regions[region_id].get("children")
    if isinstance(explicit, list):
        return [item for item in explicit if item in regions]
    return [item for item in children.get(region_id, []) if item in regions]


def _recursive_ops(regions: dict[str, dict[str, Any]], children: dict[str | None, list[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def visit(region_id: str) -> set[str]:
        if region_id in memo:
            return memo[region_id]
        ops = {str(item) for item in regions[region_id].get("op_ids", [])}
        for child in _region_children(region_id, regions, children):
            ops |= visit(child)
        memo[region_id] = ops
        return ops

    for region_id in regions:
        visit(region_id)
    return memo


def _op_range(op_ids: set[str], tm: dict[str, Any]) -> str:
    if not op_ids:
        return "-"
    values = [_op_order(item, tm) for item in op_ids]
    return f"{min(values)}-{max(values)}"


def _paths(op_ids: set[str], tm: dict[str, Any]) -> list[str]:
    return [_norm(_op_name(tm["op_by_id"].get(op_id, {}), op_id)) for op_id in sorted(op_ids, key=lambda item: _op_order(item, tm))]


def _layer(paths: list[str]) -> int | None:
    for path in paths:
        for pattern in (r"/encoder/layer\.(\d+)/", r"encoder\.layer\.(\d+)", r"layer[._/](\d+)"):
            match = re.search(pattern, path)
            if match:
                return int(match.group(1))
    return None


def _section(paths: list[str]) -> str:
    if any("/embeddings/" in item for item in paths):
        return "Embeddings"
    layer = _layer(paths)
    if layer is not None:
        return f"Encoder Layer {layer}"
    if any("/cls/" in item or "model.cls" in item for item in paths):
        return "Prediction Head"
    if paths and all(_is_shape_mask_path(item) for item in paths):
        return "Auxiliary Shape / Mask Flow"
    return "Other Main Flow"


def _is_shape_mask_path(path: str) -> bool:
    return any(
        token in path
        for token in (
            "/shape",
            "/reshape",
            "/transpose",
            "/unsqueeze",
            "/squeeze",
            "/concat",
            "/range",
            "/cast",
            "/constant",
            "/constantofshape",
            "/where",
            "/greater",
            "/less",
            "/equal",
            "/and",
            "/or",
            "/not",
            "/isnan",
            "attention_mask",
            "attention.mask",
        )
    )


def _projection_kind(paths: list[str]) -> str:
    blob = " ".join(paths)
    if "/attention/self/query/" in blob:
        return "query"
    if "/attention/self/key/" in blob:
        return "key"
    if "/attention/self/value/" in blob:
        return "value"
    if "/attention/output/dense/" in blob:
        return "attention_output"
    if "/intermediate/dense/" in blob:
        return "ffn_intermediate"
    if "/output/dense/" in blob:
        return "ffn_output"
    if "/cls/" in blob or "model.cls" in blob:
        return "prediction"
    if "/embeddings/" in blob:
        return "embedding"
    return "generic"


def _attention_internal_kind(paths: list[str]) -> str | None:
    blob = " ".join(paths)
    if "/attention/self/matmul_1" in blob:
        return "context_matmul"
    if (
        "/attention/self/matmul" in blob
        and "/attention/self/query/" not in blob
        and "/attention/self/key/" not in blob
        and "/attention/self/value/" not in blob
    ):
        return "score_matmul"
    if (
        "/attention/self/add" in blob
        and "/attention/self/query/" not in blob
        and "/attention/self/key/" not in blob
        and "/attention/self/value/" not in blob
        and "/attention/output/" not in blob
    ):
        return "mask_add"
    if "/attention/self/where" in blob:
        return "mask_select"
    return None


def _infer_name(region: dict[str, Any], paths: list[str], expansion: dict[str, dict[str, Any]]) -> str:
    region_id = region["region_id"]
    if region_id in expansion and expansion[region_id].get("name"):
        return str(expansion[region_id]["name"])
    region_type = region.get("region_type", "")
    layer = _layer(paths)
    prefix = f"Layer {layer} " if layer is not None else ""
    if region_type == "FeedForwardRegion":
        return prefix + "Feed Forward"
    if region_type == "AttentionSkeletonRegion":
        return prefix + "Attention"
    if region_type == "LinearProjectionRegion":
        attention_kind = _attention_internal_kind(paths)
        if attention_kind == "score_matmul":
            return prefix + "Attention Score MatMul"
        if attention_kind == "context_matmul":
            return prefix + "Attention Context MatMul"
        kind = _projection_kind(paths)
        return {
            "query": prefix + "Query Projection",
            "key": prefix + "Key Projection",
            "value": prefix + "Value Projection",
            "attention_output": prefix + "Attention Output Projection",
            "ffn_intermediate": prefix + "FFN Intermediate Projection",
            "ffn_output": prefix + "FFN Output Projection",
            "prediction": "Prediction Projection",
        }.get(kind, prefix + "Linear Projection")
    if region_type == "ActivationRegion":
        return prefix + ("GELU" if any("erf" in path or "gelu" in path for path in paths) else "Activation")
    if region_type == "ResidualMergeRegion":
        blob = " ".join(paths)
        if _attention_internal_kind(paths) == "mask_add":
            return prefix + "Attention Mask Add"
        if "/attention/output/add" in blob:
            return prefix + "Attention Residual Add"
        if "/output/add" in blob:
            return prefix + "FFN Residual Add"
        if "/embeddings/add" in blob:
            return "Embedding Add"
        return prefix + "Residual Add"
    if region_type == "LayerNormRegion":
        return "Embedding LayerNorm" if any("/embeddings/" in path for path in paths) else prefix + "LayerNorm"
    if region_type == "AttentionSkeletonRegion":
        return prefix + "Attention"
    return str(region.get("name") or region_id)


def _abstract_expansion_by_region(expansion_report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not expansion_report:
        return {}
    return {item["region_id"]: item for item in expansion_report.get("records", []) if item.get("region_id")}


def _rdim_maps(region_dimension_ir: dict[str, Any] | None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if not region_dimension_ir:
        return {}, {}
    dims: dict[str, list[dict[str, Any]]] = defaultdict(list)
    constraints: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dim in region_dimension_ir.get("dimension_variables", []):
        dims[dim.get("region_id", "")].append(dim)
    for constraint in region_dimension_ir.get("constraint_equations", []):
        constraints[constraint.get("region_id", "")].append(constraint)
    return dict(dims), dict(constraints)


def _role_from_rdim(dim: dict[str, Any]) -> tuple[str, str]:
    dim_name = str(dim.get("dim_name", "unknown"))
    axis = str(dim.get("axis_role", "unknown"))
    if dim_name in {"intermediate_dim"} or axis == "intermediate":
        role = "intermediate_dim"
    elif dim_name in {"hidden_dim", "out_features", "in_features", "bias_dim"} or axis == "hidden":
        role = "hidden_dim"
    elif dim_name in {"head_dim"}:
        role = "head_dim"
    elif dim_name in {"num_heads"}:
        role = "num_heads"
    elif dim_name in {"sequence_dim"} or axis == "sequence":
        role = "sequence_dim"
    elif axis == "vocab":
        role = "vocab_dim"
    elif axis == "shape" or dim_name == "symbolic_axis":
        role = "axis_mapping"
    elif dim_name in {"fanout_dim"}:
        role = "fanout_dim"
    else:
        role = "unknown"
    if dim.get("blocked"):
        status = "blocked"
    elif dim.get("protected"):
        status = "protected"
    elif dim.get("prunable"):
        status = "prunable"
    elif dim.get("propagated"):
        status = "propagated"
    else:
        status = "unknown"
    return role, status


def _add_dim(dimensions: list[DimensionSemantics], name: str, role: str, status: str, source: str, reason: str) -> None:
    if not any(item.dim_name == name and item.symbolic_role == role and item.status == status for item in dimensions):
        dimensions.append(DimensionSemantics(name, role, status, source, reason))


def _rule(region_id: str, index: int, rule_type: str, source: str, targets: list[str], mapping: str, direction: str, explanation: str) -> PropagationRule:
    return PropagationRule(f"rpsem_rule::{_compact_id(region_id)}::{index:03d}", rule_type, source, targets, mapping, direction, explanation)


def _repair(region_id: str, index: int, repair_type: str, regions: list[str], dimensions: list[str], required: bool, explanation: str) -> RepairObligation:
    return RepairObligation(f"rpsem_repair::{_compact_id(region_id)}::{index:03d}", repair_type, regions, dimensions, required, explanation)


def _blocker(region_id: str, index: int, blocker_type: str, severity: str, explanation: str) -> Blocker:
    return Blocker(f"rpsem_blocker::{_compact_id(region_id)}::{index:03d}", blocker_type, severity, explanation)


def _record_semantics(
    region: dict[str, Any],
    region_name: str,
    section: str,
    op_range: str,
    op_ids: set[str],
    paths: list[str],
    rdim_dimensions: list[dict[str, Any]],
    rdim_constraints: list[dict[str, Any]],
) -> RegionSemanticsRecord:
    region_id = region["region_id"]
    region_type = region.get("region_type", "UnknownRegion")
    projection_kind = _projection_kind(paths)
    attention_internal_kind = _attention_internal_kind(paths)
    dimensions: list[DimensionSemantics] = []
    rules: list[PropagationRule] = []
    repairs: list[RepairObligation] = []
    blockers: list[Blocker] = []

    for dim in rdim_dimensions:
        role, status = _role_from_rdim(dim)
        _add_dim(dimensions, str(dim.get("dim_name", "unknown")), role, status, "region_dimension_ir", str(dim.get("reason", "")))

    if attention_internal_kind == "score_matmul":
        dimensions = []
        pruning_role = "constraint_carrier"
        _add_dim(dimensions, "sequence_dim", "sequence_dim", "propagated", "inferred_from_op_path", "Attention score MatMul propagates sequence axes through Q x K^T.")
        _add_dim(dimensions, "head_dim", "head_dim", "blocked", "inferred_from_op_path", "Q/K contraction depends on the unproven attention head-axis mapping.")
        _add_dim(dimensions, "attention_score_dim", "unknown", "unknown", "inferred_from_op_path", "Attention scores are dataflow products, not independent parameter channels.")
        rules.append(_rule(region_id, 0, "attention_score_contraction", "query_key_axes", ["attention_scores"], "reshape_head_mapping_required", "bidirectional", "Attention score MatMul is Q x K^T, a dataflow contraction rather than a parameterized projection layer."))
        repairs.append(_repair(region_id, 0, "attention_axis_mapping_required", [region_id], ["head_dim", "sequence_dim"], True, "Prove Q/K head-axis mapping before pruning attention score paths."))
        blockers.append(_blocker(region_id, 0, "attention_head_mapping_unproven", "blocker", "Attention score MatMul has no independent prunable weight/channel dimension."))
    elif attention_internal_kind == "context_matmul":
        dimensions = []
        pruning_role = "constraint_carrier"
        _add_dim(dimensions, "sequence_dim", "sequence_dim", "propagated", "inferred_from_op_path", "Attention context MatMul propagates the softmax/value sequence axes.")
        _add_dim(dimensions, "head_dim", "head_dim", "blocked", "inferred_from_op_path", "Context contraction depends on unproven head-axis mapping.")
        _add_dim(dimensions, "context_dim", "hidden_dim", "propagated", "inferred_from_op_path", "Context output dimensions flow from V through attention axis transforms.")
        rules.append(_rule(region_id, 0, "attention_context_contraction", "softmax_value_axes", ["context"], "reshape_head_mapping_required", "bidirectional", "Attention context MatMul is Softmax(scores) x V, not a parameterized projection layer."))
        repairs.append(_repair(region_id, 0, "attention_axis_mapping_required", [region_id], ["head_dim", "context_dim"], True, "Prove Softmax/V head-axis mapping before pruning attention context paths."))
        blockers.append(_blocker(region_id, 0, "attention_head_mapping_unproven", "blocker", "Attention context MatMul has no independent prunable weight/channel dimension."))
    elif attention_internal_kind in {"mask_add", "mask_select"}:
        dimensions = []
        pruning_role = "constraint_carrier"
        _add_dim(dimensions, "attention_score_dim", "unknown", "propagated", "inferred_from_op_path", "Attention mask application follows the attention score tensor.")
        _add_dim(dimensions, "sequence_dim", "sequence_dim", "propagated", "inferred_from_op_path", "Mask axes must broadcast over sequence dimensions.")
        _add_dim(dimensions, "mask_dim", "axis_mapping", "propagated", "inferred_from_op_path", "Mask dimensions are metadata-like broadcast axes.")
        rules.append(_rule(region_id, 0, "attention_mask_application", "mask_dim", ["attention_score_dim"], "axis_remap_required", "bidirectional", "Attention mask add biases attention scores; it is not a residual hidden-state merge."))
        repairs.append(_repair(region_id, 0, "shape_metadata_update", [region_id], ["mask_dim", "sequence_dim"], False, "Mask broadcasting metadata may need updates if upstream axes change."))
        blockers.append(_blocker(region_id, 0, "unknown_axis_mapping", "warning", "Exact attention mask broadcast axes are not proven in this semantics pass."))
    elif region_type == "FeedForwardRegion":
        pruning_role = "directly_prunable"
        _add_dim(dimensions, "intermediate_dim", "intermediate_dim", "prunable", "inferred_from_region_type", "Feed-forward intermediate width is the canonical MLP pruning axis.")
        _add_dim(dimensions, "hidden_dim", "hidden_dim", "protected", "inferred_from_region_type", "Feed-forward boundary hidden width is kept unchanged.")
        rules.append(_rule(region_id, 0, "same_indices_across_mlp", "intermediate_dim", ["ffn_intermediate_projection.out_features", "gelu.intermediate_dim", "ffn_output_projection.in_features"], "same_indices", "bidirectional", "The same selected intermediate indices must flow through Linear -> GELU -> Linear."))
        repairs.append(_repair(region_id, 0, "same_indices_across_mlp", [region_id], ["intermediate_dim"], True, "Use one index set for the expansion output, activation, and projection input."))
        repairs.append(_repair(region_id, 1, "prune_consumer_input", [region_id], ["intermediate_dim"], True, "Prune FFN output projection input columns to match the pruned intermediate outputs."))
    elif region_type == "ActivationRegion":
        pruning_role = "propagation_only"
        dim_name = "intermediate_dim" if any("intermediate" in path or "gelu" in path or "erf" in path for path in paths) else "elementwise_dim"
        _add_dim(dimensions, dim_name, "intermediate_dim" if dim_name == "intermediate_dim" else "unknown", "propagated", "inferred_from_region_type", "Elementwise activation preserves its input index set.")
        rules.append(_rule(region_id, 0, "activation_preserves_indices", dim_name, [dim_name], "no_index_change", "local", "GELU/activation does not introduce a new pruning choice; it forwards the same indices."))
    elif region_type == "LinearProjectionRegion":
        if projection_kind == "ffn_intermediate":
            pruning_role = "directly_prunable"
            _add_dim(dimensions, "intermediate_dim", "intermediate_dim", "prunable", "inferred_from_op_path", "FFN intermediate projection output is locally prunable.")
            repairs.append(_repair(region_id, 0, "prune_bias", [region_id], ["intermediate_dim"], True, "Bias entries follow pruned output features when present."))
            repairs.append(_repair(region_id, 1, "prune_consumer_input", [region_id], ["intermediate_dim"], True, "The downstream FFN output projection input must use the same indices."))
            rules.append(_rule(region_id, 0, "projection_output_to_consumers", "intermediate_dim", ["consumer.in_features"], "same_indices", "forward", "Projection output pruning propagates to consumers."))
        elif projection_kind == "ffn_output":
            pruning_role = "propagation_only"
            _add_dim(dimensions, "intermediate_dim", "intermediate_dim", "repair_required", "inferred_from_op_path", "FFN output projection input receives propagated intermediate pruning.")
            _add_dim(dimensions, "hidden_dim", "hidden_dim", "protected", "inferred_from_op_path", "FFN output hidden width feeds residual/LayerNorm and is protected.")
            repairs.append(_repair(region_id, 0, "prune_consumer_input", [region_id], ["intermediate_dim"], True, "Prune input columns when the upstream intermediate projection is pruned."))
        elif projection_kind == "attention_output":
            pruning_role = "constraint_carrier"
            _add_dim(dimensions, "input_dim", "hidden_dim", "blocked", "inferred_from_op_path", "Attention context pruning requires proven attention axis mapping.")
            _add_dim(dimensions, "hidden_dim", "hidden_dim", "protected", "inferred_from_op_path", "Attention output hidden width feeds a residual branch.")
            blockers.append(_blocker(region_id, 0, "attention_head_mapping_unproven", "blocker", "Attention output pruning depends on context/head axis mapping."))
        elif projection_kind in {"query", "key", "value"}:
            pruning_role = "directly_prunable"
            _add_dim(dimensions, "out_features", "hidden_dim", "prunable", "inferred_from_op_path", f"{projection_kind.title()} projection output is locally prunable but must respect attention head mapping.")
            repairs.append(_repair(region_id, 0, "prune_bias", [region_id], ["out_features"], True, "Projection bias follows output features when present."))
            repairs.append(_repair(region_id, 1, "attention_axis_mapping_required", [region_id], ["out_features"], True, "Q/K/V pruning must be reconciled with attention head and reshape axes."))
            blockers.append(_blocker(region_id, 0, "attention_head_mapping_unproven", "warning", "Q/K/V output pruning is ambiguous until head-axis mapping is proven."))
        elif projection_kind == "prediction":
            pruning_role = "analysis_only"
            _add_dim(dimensions, "task_head_dim", "unknown", "protected", "inferred_from_op_path", "Prediction head pruning is not supported by default.")
            blockers.append(_blocker(region_id, 0, "unsupported_region_type", "warning", "Task-head pruning requires task-specific semantics."))
        else:
            pruning_role = "directly_prunable"
            _add_dim(dimensions, "out_features", "hidden_dim", "prunable", "inferred_from_region_type", "Generic projection output is locally prunable.")
            repairs.append(_repair(region_id, 0, "prune_bias", [region_id], ["out_features"], False, "Bias repair is needed if a bias tensor is present."))
            rules.append(_rule(region_id, 0, "projection_output_to_consumers", "out_features", ["consumer.in_features"], "same_indices", "forward", "Projection output indices propagate to consumer inputs."))
    elif region_type == "AttentionSkeletonRegion":
        pruning_role = "constraint_carrier"
        for dim_name, role in (("num_heads", "num_heads"), ("head_dim", "head_dim"), ("hidden_dim", "hidden_dim")):
            _add_dim(dimensions, dim_name, role, "blocked" if dim_name != "hidden_dim" else "protected", "inferred_from_region_type", "Attention pruning requires a proven hidden/head/axis mapping.")
        rules.append(_rule(region_id, 0, "attention_axis_mapping", "hidden_dim", ["num_heads", "head_dim", "sequence_dim"], "reshape_head_mapping_required", "bidirectional", "Attention uses MatMul/Softmax/MatMul plus shape transforms; head axes must be mapped before pruning."))
        repairs.append(_repair(region_id, 0, "attention_axis_mapping_required", [region_id], ["num_heads", "head_dim", "hidden_dim"], True, "Prove reshape/transpose head-axis mapping before executable attention pruning."))
        blockers.append(_blocker(region_id, 0, "attention_head_mapping_unproven", "blocker", "Head pruning is blocked until head-axis mapping evidence exists."))
    elif region_type == "ResidualMergeRegion":
        pruning_role = "blocked"
        _add_dim(dimensions, "hidden_dim", "hidden_dim", "protected", "inferred_from_region_type", "Residual merge requires branch hidden dimensions to agree.")
        rules.append(_rule(region_id, 0, "residual_branch_agreement", "hidden_dim", ["branch_a.hidden_dim", "branch_b.hidden_dim"], "branch_agreement_required", "bidirectional", "Residual branches must preserve compatible hidden dimensions."))
        repairs.append(_repair(region_id, 0, "residual_branch_repair", [region_id], ["hidden_dim"], True, "Any hidden pruning would require coordinated repair across every residual branch."))
        blockers.append(_blocker(region_id, 0, "residual_hidden_dim", "blocker", "Residual hidden dimension pruning is blocked by default."))
    elif region_type == "LayerNormRegion":
        pruning_role = "protected"
        _add_dim(dimensions, "hidden_dim", "hidden_dim", "protected", "inferred_from_region_type", "LayerNorm affine/statistics dimensions match hidden width.")
        repairs.append(_repair(region_id, 0, "layernorm_parameter_repair", [region_id], ["hidden_dim"], True, "Gamma and beta would need the same hidden-index repair if hidden width changed."))
        blockers.append(_blocker(region_id, 0, "layernorm_hidden_dim", "blocker", "LayerNorm hidden dimension is protected by default."))
    elif region_type in {"AxisTransformRegion", "ShapeMotifRegion", "ForkRegion", "JoinRegion"} or section == "Auxiliary Shape / Mask Flow":
        pruning_role = "propagation_only" if region_type in {"AxisTransformRegion", "ShapeMotifRegion", "ForkRegion"} else "constraint_carrier"
        _add_dim(dimensions, "symbolic_axis", "axis_mapping", "propagated", "inferred_from_region_type", "Shape/mask regions carry axis metadata rather than pruning choices.")
        rules.append(_rule(region_id, 0, "axis_or_shape_mapping", "symbolic_axis", ["mapped_axis"], "axis_remap_required", "bidirectional", "Index sets may need remapping through shape, mask, or branch plumbing."))
        repairs.append(_repair(region_id, 0, "shape_metadata_update", [region_id], ["symbolic_axis"], False, "Shape metadata may need to be updated after upstream pruning."))
        blockers.append(_blocker(region_id, 0, "unknown_axis_mapping", "warning", "Axis mapping is not proven for this auxiliary region."))
    elif section == "Embeddings" or any("/embeddings/" in path for path in paths):
        pruning_role = "protected"
        _add_dim(dimensions, "vocab_dim", "vocab_dim", "protected", "inferred_from_op_path", "Embedding vocabulary dimension is protected.")
        _add_dim(dimensions, "hidden_dim", "hidden_dim", "protected", "inferred_from_op_path", "Embedding hidden width feeds the model hidden path.")
        blockers.append(_blocker(region_id, 0, "graph_boundary", "warning", "Embedding pruning crosses input/table boundary semantics."))
    else:
        pruning_role = "analysis_only"
        _add_dim(dimensions, "unknown_dim", "unknown", "unknown", "unknown", "No specific pruning semantics are known for this region type.")
        blockers.append(_blocker(region_id, 0, "unsupported_region_type", "unknown", "No conservative pruning semantics rule matched this region."))

    for constraint in rdim_constraints:
        constraint_type = str(constraint.get("constraint_type", "unknown"))
        if constraint_type and constraint_type != "unknown":
            rules.append(_rule(region_id, len(rules), constraint_type, str(constraint.get("lhs", "")), [str(constraint.get("rhs", ""))], str(constraint.get("relation", "unknown")), "bidirectional", str(constraint.get("reason", ""))))

    if any(blocker.severity == "blocker" for blocker in blockers) and pruning_role == "directly_prunable":
        pruning_role = "constraint_carrier"

    evidence = {
        "source_ops": sorted(op_ids, key=lambda item: _compact_id(item))[:50],
        "source_region_type": region_type,
        "source_dimension_vars": [item.get("var_id") for item in rdim_dimensions],
        "source_constraints": [item.get("constraint_id") for item in rdim_constraints],
        "confidence": region.get("confidence", "unknown"),
    }
    return RegionSemanticsRecord(
        region_id=region_id,
        region_name=region_name,
        region_type=region_type,
        section=section,
        op_range=op_range,
        primitive_leaf_count=len(op_ids),
        pruning_role=pruning_role,
        dimensions=dimensions,
        propagation_rules=rules,
        repair_obligations=repairs,
        blockers=blockers,
        evidence=evidence,
    )


def _summary(records: list[RegionSemanticsRecord]) -> dict[str, Any]:
    role_counts = Counter(item.pruning_role for item in records)
    type_counts = Counter(item.region_type for item in records)
    blocker_counts = Counter(blocker.blocker_type for item in records for blocker in item.blockers)
    repair_counts = Counter(repair.obligation_type for item in records for repair in item.repair_obligations)
    dim_status_counts = Counter(dim.status for item in records for dim in item.dimensions)
    return {
        "num_regions": len(records),
        "pruning_role_counts": dict(sorted(role_counts.items())),
        "region_type_counts": dict(sorted(type_counts.items())),
        "blocker_type_counts": dict(sorted(blocker_counts.items())),
        "repair_obligation_counts": dict(sorted(repair_counts.items())),
        "dimension_status_counts": dict(sorted(dim_status_counts.items())),
        "directly_prunable_regions": role_counts.get("directly_prunable", 0),
        "propagation_only_regions": role_counts.get("propagation_only", 0),
        "blocked_or_protected_regions": role_counts.get("blocked", 0) + role_counts.get("protected", 0),
        "attention_blocked_regions": sum(1 for item in records if any(blocker.blocker_type == "attention_head_mapping_unproven" for blocker in item.blockers)),
        "residual_blocked_regions": sum(1 for item in records if any(blocker.blocker_type == "residual_hidden_dim" for blocker in item.blockers)),
        "layernorm_protected_regions": sum(1 for item in records if item.region_type == "LayerNormRegion"),
        "mlp_pruning_opportunities": sum(1 for item in records if item.region_type == "FeedForwardRegion" and item.pruning_role == "directly_prunable"),
    }


def build_region_pruning_semantics(
    structural_region_tree: dict[str, Any],
    tensor_ir: dict[str, Any],
    *,
    region_dimension_ir: dict[str, Any] | None = None,
    abstract_expansion_report: dict[str, Any] | None = None,
    source_region_tree_path: str = "",
    source_region_dimension_ir_path: str | None = None,
) -> RegionPruningSemantics:
    tm = _tensor_maps(tensor_ir)
    regions, children, _interfaces = _regions_and_children(structural_region_tree)
    recursive = _recursive_ops(regions, children)
    expansion = _abstract_expansion_by_region(abstract_expansion_report)
    dims_by_region, constraints_by_region = _rdim_maps(region_dimension_ir)
    records: list[RegionSemanticsRecord] = []

    for region_id, region in sorted(regions.items(), key=lambda item: (min(_op_order(op, tm) for op in recursive.get(item[0], set())) if recursive.get(item[0]) else 10**12, item[0])):
        region_type = region.get("region_type", "UnknownRegion")
        if region_type in {"ModelRegion", "PrimitiveRegion"}:
            continue
        op_ids = recursive.get(region_id, set())
        paths = _paths(op_ids, tm)
        name = _infer_name(region, paths, expansion)
        section = expansion.get(region_id, {}).get("section") or _section(paths)
        record = _record_semantics(
            region,
            name,
            section,
            expansion.get(region_id, {}).get("op_range") or _op_range(op_ids, tm),
            op_ids,
            paths,
            dims_by_region.get(region_id, []),
            constraints_by_region.get(region_id, []),
        )
        records.append(record)

    return RegionPruningSemantics(
        model_name=structural_region_tree.get("model_name", tensor_ir.get("model_name", "model")),
        frontend=structural_region_tree.get("source_frontend", tensor_ir.get("source_frontend", "unknown")),
        source_region_tree_path=source_region_tree_path,
        source_region_dimension_ir_path=source_region_dimension_ir_path,
        generated_at=datetime.now(timezone.utc).isoformat(),
        regions=records,
        summary=_summary(records),
    )


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _table(rows: list[dict[str, Any]], columns: list[str], limit: int = 120) -> str:
    if not rows:
        return "_None._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(_cell(row.get(column, "")) for column in columns) + " |")
    if len(rows) > limit:
        lines.append("| ... | " + f"{len(rows) - limit} more rows omitted" + " |" * (len(columns) - 2))
    return "\n".join(lines)


def _is_auxiliary_detail(region: dict[str, Any]) -> bool:
    return (
        region.get("region_type") in {"AxisTransformRegion", "ForkRegion", "JoinRegion", "ShapeMotifRegion"}
        or region.get("section") == "Auxiliary Shape / Mask Flow"
    )


def _count_rows(values: list[tuple[str, str]]) -> list[dict[str, Any]]:
    counts = Counter(values)
    return [
        {"type": key[0], "section_or_blocker": key[1], "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def region_pruning_semantics_to_markdown(value: RegionPruningSemantics | dict[str, Any], *, include_auxiliary_details: bool = False) -> str:
    data = region_pruning_semantics_to_dict(value) if isinstance(value, RegionPruningSemantics) else value
    summary = data.get("summary", {})
    regions = data.get("regions", [])
    direct = [item for item in regions if item.get("pruning_role") == "directly_prunable"]
    propagation = [
        item for item in regions
        if item.get("pruning_role") == "propagation_only"
        and (include_auxiliary_details or not _is_auxiliary_detail(item))
    ]
    blocked = [
        item for item in regions
        if (
            item.get("pruning_role") in {"blocked", "protected"}
            or any(blocker.get("severity") == "blocker" for blocker in item.get("blockers", []))
            or (
                item.get("pruning_role") == "constraint_carrier"
                and (include_auxiliary_details or not _is_auxiliary_detail(item))
            )
        )
    ]
    auxiliary = [item for item in regions if _is_auxiliary_detail(item)]
    auxiliary_type_rows = _count_rows([(item.get("region_type", "unknown"), item.get("section", "unknown")) for item in auxiliary])
    auxiliary_blocker_rows = _count_rows([
        (item.get("region_type", "unknown"), blocker.get("blocker_type", "none"))
        for item in auxiliary
        for blocker in item.get("blockers", [])
    ])
    important_types = {"FeedForwardRegion", "AttentionSkeletonRegion", "ResidualMergeRegion", "LayerNormRegion", "LinearProjectionRegion", "ActivationRegion"}
    details = [item for item in regions if item.get("region_type") in important_types]

    def row(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": item.get("region_name"),
            "type": item.get("region_type"),
            "role": item.get("pruning_role"),
            "section": item.get("section"),
            "dims": ", ".join(f"{d['dim_name']}:{d['status']}" for d in item.get("dimensions", [])),
            "blockers": ", ".join(b["blocker_type"] for b in item.get("blockers", [])) or "-",
        }

    lines = [
        f"# Region Pruning Semantics: {data.get('model_name', '')}",
        "",
        "## Summary",
        "",
        f"- Total regions: `{summary.get('num_regions', 0)}`",
        f"- Directly prunable regions: `{summary.get('directly_prunable_regions', 0)}`",
        f"- Propagation-only regions: `{summary.get('propagation_only_regions', 0)}`",
        f"- Blocked/protected regions: `{summary.get('blocked_or_protected_regions', 0)}`",
        f"- Attention-blocked regions: `{summary.get('attention_blocked_regions', 0)}`",
        f"- Residual-blocked regions: `{summary.get('residual_blocked_regions', 0)}`",
        f"- LayerNorm-protected regions: `{summary.get('layernorm_protected_regions', 0)}`",
        f"- MLP pruning opportunities: `{summary.get('mlp_pruning_opportunities', 0)}`",
        "",
        "## Interpretation Highlights",
        "",
        "- Clean executable pruning opportunity: FFN `intermediate_dim` pruning with same-index propagation through Linear -> GELU -> Linear.",
        "- Structurally visible but blocked: attention head/channel pruning until head-axis mapping is proven.",
        "- Protected by default: residual hidden dimensions, LayerNorm hidden dimensions, and embedding vocabulary/hidden dimensions.",
        "- Auxiliary shape/mask/axis flow carries metadata and axis propagation obligations; it is not directly prunable.",
        "",
        "## Directly Prunable Opportunities",
        "",
        _table([row(item) for item in direct], ["name", "type", "role", "section", "dims", "blockers"], limit=80),
        "",
        "## Propagation-Only Regions",
        "",
        _table([row(item) for item in propagation], ["name", "type", "role", "section", "dims", "blockers"], limit=80),
        "",
        "## Protected / Blocked Regions",
        "",
        _table([row(item) for item in blocked], ["name", "type", "role", "section", "dims", "blockers"], limit=120),
        "",
        "## Auxiliary Shape / Axis Propagation Summary",
        "",
        "Raw auxiliary regions are summarized here by default so they do not dominate the learner report.",
        "",
        "### Counts by Type and Section",
        "",
        _table(auxiliary_type_rows, ["type", "section_or_blocker", "count"], limit=80),
        "",
        "### Counts by Type and Blocker",
        "",
        _table(auxiliary_blocker_rows, ["type", "section_or_blocker", "count"], limit=80),
        "",
    ]
    if include_auxiliary_details:
        lines.extend([
            "### Auxiliary Detail Rows",
            "",
            _table([row(item) for item in auxiliary], ["name", "type", "role", "section", "dims", "blockers"], limit=300),
            "",
        ])
    lines.extend([
        "## Region Details",
        "",
    ])
    for item in details[:200]:
        lines.extend(
            [
                f"### {item.get('region_name')}",
                "",
                f"- Type: `{item.get('region_type')}`",
                f"- Role: `{item.get('pruning_role')}`",
                f"- Section: `{item.get('section')}`",
                f"- Op range: `{item.get('op_range')}`",
                f"- Primitive leaves: `{item.get('primitive_leaf_count')}`",
                "",
                "**Dimensions**",
                "",
                _table(item.get("dimensions", []), ["dim_name", "symbolic_role", "status", "source", "reason"], limit=20),
                "",
                "**Rules**",
                "",
                _table(item.get("propagation_rules", []), ["rule_type", "source_dimension", "target_dimensions", "index_mapping", "direction", "explanation"], limit=20),
                "",
                "**Repairs**",
                "",
                _table(item.get("repair_obligations", []), ["obligation_type", "affected_dimensions", "required", "explanation"], limit=20),
                "",
                "**Blockers**",
                "",
                _table(item.get("blockers", []), ["blocker_type", "severity", "explanation"], limit=20),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "This report is conservative static analysis over learner structural regions. It explains pruning information flow, required repairs, and blockers. It does not modify models, execute pruning, rewrite ONNX, export ONNX, or evaluate accuracy.",
            "",
        ]
    )
    return "\n".join(lines)
