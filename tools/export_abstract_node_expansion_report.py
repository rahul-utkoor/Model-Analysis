#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

LARGE = 10**12


# =============================================================================
# Generic helpers
# =============================================================================

def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def text(x: Any) -> str:
    return str(x if x is not None else "")


def compact(x: Any) -> str:
    return text(x).replace("\\", "/").lower()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def last_int(s: Any) -> int | None:
    nums = re.findall(r"\d+", text(s))
    return int(nums[-1]) if nums else None


def first_layer_from_path(path: str) -> int | None:
    patterns = [
        r"/encoder/layer\.(\d+)/",
        r"encoder\.layer\.(\d+)",
        r"encoder_layer_(\d+)",
        r"/layer\.(\d+)/",
        r"layer_(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, path)
        if m:
            return int(m.group(1))
    return None


def op_range_string(op_ids: set[str], op_order: dict[str, int]) -> str:
    if not op_ids:
        return "-"
    vals = [op_order.get(x, LARGE) for x in op_ids]
    return f"{min(vals)}-{max(vals)}"


def range_start(r: str) -> int:
    m = re.match(r"(\d+)", r or "")
    return int(m.group(1)) if m else LARGE


def section_sort_key(name: str) -> tuple[int, int, str]:
    if name == "Root":
        return (-1, -1, name)
    if name == "Model":
        return (0, -1, name)
    if name == "Embeddings":
        return (1, -1, name)
    m = re.match(r"Encoder Layer (\d+)", name)
    if m:
        return (2, int(m.group(1)), name)
    if name == "Prediction Head":
        return (3, -1, name)
    if name == "Other Main Flow":
        return (4, -1, name)
    if name == "Auxiliary Shape / Mask Flow":
        return (5, -1, name)
    return (9, -1, name)


# =============================================================================
# Tensor IR access
# =============================================================================

def tensor_ops(tensor_ir: dict[str, Any]) -> list[dict[str, Any]]:
    return tensor_ir.get("ops") or tensor_ir.get("operations") or tensor_ir.get("tensor_ops") or []


def op_id_of(op: dict[str, Any]) -> str | None:
    return op.get("op_id") or op.get("id") or op.get("name")


def op_type_of(op: dict[str, Any]) -> str:
    return text(op.get("canonical_op_type") or op.get("op_type") or op.get("type") or "")


def op_name_of(op: dict[str, Any], fallback: str) -> str:
    return text(
        op.get("source_node_name")
        or op.get("onnx_node_name")
        or op.get("name")
        or op.get("label")
        or fallback
    )


def op_inputs(op: dict[str, Any]) -> list[str]:
    vals = op.get("inputs") or op.get("input_values") or op.get("input_ids") or []
    return [text(v) for v in vals]


def op_outputs(op: dict[str, Any]) -> list[str]:
    vals = op.get("outputs") or op.get("output_values") or op.get("output_ids") or []
    return [text(v) for v in vals]


def build_tensor_maps(tensor_ir: dict[str, Any]) -> dict[str, Any]:
    ops = tensor_ops(tensor_ir)

    op_by_id: dict[str, dict[str, Any]] = {}
    op_order: dict[str, int] = {}
    value_producer: dict[str, str] = {}
    value_consumers: dict[str, list[str]] = defaultdict(list)

    for i, op in enumerate(ops):
        oid = op_id_of(op)
        if not oid:
            continue

        op_by_id[oid] = op
        op_order[oid] = i

        for out in op_outputs(op):
            value_producer[out] = oid

        for inp in op_inputs(op):
            value_consumers[inp].append(oid)

    return {
        "ops": ops,
        "op_by_id": op_by_id,
        "op_order": op_order,
        "value_producer": value_producer,
        "value_consumers": dict(value_consumers),
    }


def op_order_of(op_id: str, tm: dict[str, Any]) -> int:
    if op_id in tm["op_order"]:
        return tm["op_order"][op_id]
    n = last_int(op_id)
    return n if n is not None else LARGE


def op_name_by_id(op_id: str, tm: dict[str, Any]) -> str:
    op = tm["op_by_id"].get(op_id, {})
    return op_name_of(op, op_id)


def op_path_by_id(op_id: str, tm: dict[str, Any]) -> str:
    return compact(op_name_by_id(op_id, tm))


def op_type_by_id(op_id: str, tm: dict[str, Any]) -> str:
    return op_type_of(tm["op_by_id"].get(op_id, {}))


# =============================================================================
# Region tree access
# =============================================================================

def build_region_maps(tree: dict[str, Any]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str | None, list[str]],
    dict[str, dict[str, Any]],
]:
    region_by_id: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str | None, list[str]] = defaultdict(list)

    for r in tree.get("regions", []):
        rid = r.get("region_id")
        if not rid:
            continue
        region_by_id[rid] = r
        children_by_parent[r.get("parent")].append(rid)

    interface_by_region: dict[str, dict[str, Any]] = {}
    for iface in tree.get("interfaces", []):
        rid = iface.get("region_id")
        if rid:
            interface_by_region[rid] = iface

    return region_by_id, children_by_parent, interface_by_region


def own_ops(region: dict[str, Any]) -> set[str]:
    return set(text(x) for x in (region.get("op_ids") or []))


def region_children(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
) -> list[str]:
    r = region_by_id[rid]
    explicit = r.get("children")
    if isinstance(explicit, list):
        return [x for x in explicit if x in region_by_id]
    return [x for x in children_by_parent.get(rid, []) if x in region_by_id]


def compute_recursive_leaf_ops(
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def visit(rid: str) -> set[str]:
        if rid in memo:
            return memo[rid]

        r = region_by_id[rid]
        ops = set(own_ops(r))

        for cid in region_children(rid, region_by_id, children_by_parent):
            ops |= visit(cid)

        memo[rid] = ops
        return ops

    for rid in region_by_id:
        visit(rid)

    return memo


def direct_ops_not_covered_by_children(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    recursive_leaf_ops: dict[str, set[str]],
) -> set[str]:
    r = region_by_id[rid]
    covered: set[str] = set()

    for cid in region_children(rid, region_by_id, children_by_parent):
        covered |= recursive_leaf_ops.get(cid, set())

    return own_ops(r) - covered


def find_model_region(region_by_id: dict[str, dict[str, Any]]) -> str | None:
    for rid, r in region_by_id.items():
        if r.get("region_type") == "ModelRegion":
            return rid
    return None


# =============================================================================
# Auxiliary classification
# =============================================================================

AUX_OP_TYPES = {
    "and",
    "shape",
    "shape_op",
    "reshape",
    "transpose",
    "slice",
    "unsqueeze",
    "squeeze",
    "concat",
    "range",
    "cast",
    "constant",
    "constantofshape",
    "constant_of_shape",
    "equal",
    "greater",
    "greaterorequal",
    "greater_or_equal",
    "less",
    "lessorequal",
    "less_or_equal",
    "not",
    "or",
    "where",
    "isnan",
    "expand",
    "flatten",
}

AUX_PATH_WORDS = {
    "attention_mask",
    "attention.mask",
    "/attention_mask",
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
    "/greater",
    "/less",
    "/equal",
    "/and",
    "/or",
    "/not",
    "/where",
    "/isnan",
    "/expand",
    "/flatten",
}


def is_embedding_lookup_path(path: str) -> bool:
    return any(x in path for x in [
        "/embeddings/word_embeddings/gather",
        "/embeddings/token_type_embeddings/gather",
        "/embeddings/position_embeddings/gather",
    ])


def is_main_compute_path(path: str) -> bool:
    return any(x in path for x in [
        "/embeddings/word_embeddings/gather",
        "/embeddings/token_type_embeddings/gather",
        "/embeddings/position_embeddings/gather",
        "/embeddings/add",
        "/embeddings/add_1",
        "/layernorm/layernormalization",
        "/attention/self/query/matmul",
        "/attention/self/query/add",
        "/attention/self/key/matmul",
        "/attention/self/key/add",
        "/attention/self/value/matmul",
        "/attention/self/value/add",
        "/attention/self/matmul",
        "/attention/self/add",
        "/attention/self/softmax",
        "/attention/self/matmul_1",
        "/attention/output/dense/matmul",
        "/attention/output/dense/add",
        "/attention/output/add",
        "/intermediate/dense/matmul",
        "/intermediate/dense/add",
        "/intermediate/intermediate_act_fn/div",
        "/intermediate/intermediate_act_fn/erf",
        "/intermediate/intermediate_act_fn/add",
        "/intermediate/intermediate_act_fn/mul",
        "/intermediate/intermediate_act_fn/mul_1",
        "/output/dense/matmul",
        "/output/dense/add",
        "/output/add",
        "/cls/",
    ])


def is_auxiliary_op(op_id: str, tm: dict[str, Any]) -> bool:
    path = op_path_by_id(op_id, tm)
    typ = op_type_by_id(op_id, tm).lower().replace("_", "")

    if is_embedding_lookup_path(path):
        return False

    if is_main_compute_path(path):
        # Exception: attention masking/select is still auxiliary from a learner
        # dataflow perspective, even though it lies inside attention.
        if "/attention/self/where" in path:
            return True
        if "/attention/self/where_1" in path:
            return True
        return False

    if typ in AUX_OP_TYPES:
        return True

    return any(w in path for w in AUX_PATH_WORDS)


def is_auxiliary_region(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    recursive_leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
) -> bool:
    rt = region_by_id[rid].get("region_type", "")
    ops = recursive_leaf_ops.get(rid, set())

    if rt == "AxisTransformRegion":
        return True

    if not ops:
        return False

    return all(is_auxiliary_op(op, tm) for op in ops)


def is_auxiliary_only_fork_axis_join(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    recursive_leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
) -> bool:
    rt = region_by_id[rid].get("region_type", "")
    if rt not in {"ForkRegion", "AxisTransformRegion", "JoinRegion"}:
        return False
    ops = recursive_leaf_ops.get(rid, set())
    return bool(ops) and all(is_auxiliary_op(op, tm) for op in ops)


def op_section(op_id: str, tm: dict[str, Any]) -> str:
    path = op_path_by_id(op_id, tm)

    if is_auxiliary_op(op_id, tm):
        return "Auxiliary Shape / Mask Flow"

    if "/embeddings/" in path:
        return "Embeddings"

    layer = first_layer_from_path(path)
    if layer is not None:
        return f"Encoder Layer {layer}"

    if "/cls/" in path or "model.cls" in path:
        return "Prediction Head"

    return "Other Main Flow"


def region_section(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    recursive_leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
) -> str:
    ops = sorted(recursive_leaf_ops.get(rid, set()), key=lambda x: op_order_of(x, tm))
    if not ops:
        return "Other Main Flow"

    sections = [op_section(op, tm) for op in ops if not is_auxiliary_op(op, tm)]
    if sections:
        return sections[0]

    return "Auxiliary Shape / Mask Flow"


def region_layer(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    recursive_leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
) -> int | None:
    ops = sorted(recursive_leaf_ops.get(rid, set()), key=lambda x: op_order_of(x, tm))
    for op in ops:
        layer = first_layer_from_path(op_path_by_id(op, tm))
        if layer is not None:
            return layer
    return None


# =============================================================================
# Learner names
# =============================================================================

def attention_primitive_name(path: str, layer: int | None) -> str | None:
    prefix = f"Layer {layer} " if layer is not None else ""

    if "/attention/self/matmul_1" in path:
        return prefix + "Attention Context MatMul"
    if "/attention/self/matmul" in path and "/query/" not in path and "/key/" not in path and "/value/" not in path:
        return prefix + "Attention Score MatMul"
    if "/attention/self/add" in path and "/query/" not in path and "/key/" not in path and "/value/" not in path:
        return prefix + "Attention Mask Add"
    if "/attention/self/softmax" in path:
        return prefix + "Attention Softmax"
    if "/attention/self/where_1" in path:
        return prefix + "Attention Mask Select"
    if "/attention/self/where" in path:
        return prefix + "Attention Mask Apply"

    return None


def projection_name_from_path(path: str, layer: int | None) -> str:
    prefix = f"Layer {layer} " if layer is not None else ""

    if "/attention/self/query/" in path:
        return prefix + "Query Projection"
    if "/attention/self/key/" in path:
        return prefix + "Key Projection"
    if "/attention/self/value/" in path:
        return prefix + "Value Projection"
    if "/attention/output/dense/" in path:
        return prefix + "Attention Output Projection"
    if "/intermediate/dense/" in path:
        return prefix + "FFN Intermediate Projection"
    if "/output/dense/" in path:
        return prefix + "FFN Output Projection"
    if "/cls/" in path or "model.cls" in path:
        return "Prediction Projection"

    attn = attention_primitive_name(path, layer)
    if attn:
        return attn

    return prefix + "Linear Projection"


def primitive_display_name(op_id: str, tm: dict[str, Any], *, semantic_attention: bool) -> str:
    raw = op_name_by_id(op_id, tm)
    path = compact(raw)
    layer = first_layer_from_path(path)

    if semantic_attention:
        attn = attention_primitive_name(path, layer)
        if attn:
            return attn

    return raw


def region_display_name(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    recursive_leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
) -> str:
    r = region_by_id[rid]
    rt = r.get("region_type", "")
    ops = sorted(recursive_leaf_ops.get(rid, set()), key=lambda x: op_order_of(x, tm))
    paths = [op_path_by_id(op, tm) for op in ops]
    path_blob = " ".join(paths)
    layer = region_layer(rid, region_by_id, recursive_leaf_ops, tm)
    prefix = f"Layer {layer} " if layer is not None else ""

    if rt == "ModelRegion":
        return "Model"

    if rt == "FeedForwardRegion":
        return prefix + "Feed Forward"

    if rt == "AttentionSkeletonRegion":
        return prefix + "Attention"

    if rt == "LayerNormRegion":
        if "/embeddings/" in path_blob:
            return "Embedding LayerNorm"
        return prefix + "LayerNorm"

    if rt == "LinearProjectionRegion":
        for p in paths:
            name = projection_name_from_path(p, layer)
            if name != prefix + "Linear Projection":
                return name
        return prefix + "Linear Projection"

    if rt == "BiasAddRegion":
        return "Bias Add"

    if rt == "ActivationRegion":
        if "gelu" in path_blob or "erf" in path_blob or "intermediate_act_fn" in path_blob:
            return prefix + "GELU"
        return prefix + "Activation"

    if rt == "ResidualMergeRegion":
        # Attention-internal Add is masking, not a residual merge.
        if "/attention/self/add" in path_blob:
            return prefix + "Attention Mask Add"
        if "/embeddings/add" in path_blob:
            return "Embedding Add"
        if "/attention/output/add" in path_blob:
            return prefix + "Attention Residual Add"
        if "/output/add" in path_blob:
            return prefix + "FFN Residual Add"
        return prefix + "Residual Add"

    if rt == "AxisTransformRegion":
        if "attention_mask" in path_blob or "attention.mask" in path_blob:
            return prefix + "Attention Mask / Shape Transform"
        if "position" in path_blob:
            return "Position Shape Transform"
        return prefix + "Shape / Axis Transform"

    if rt == "ForkRegion":
        return prefix + "Fork"

    if rt == "JoinRegion":
        if "/attention/self/where_1" in path_blob:
            return prefix + "Attention Mask Select"
        return prefix + "Join"

    if rt == "PrimitiveRegion":
        if ops:
            return primitive_display_name(ops[0], tm, semantic_attention=True)
        return "Primitive"

    return rt.replace("Region", "") or "Region"


# =============================================================================
# Primitive leaves and dependency sorting
# =============================================================================

def primitive_leaves(
    ops: set[str],
    tm: dict[str, Any],
    *,
    limit: int | None,
    semantic_attention: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(ops, key=lambda x: op_order_of(x, tm))
    if limit is not None:
        ordered = ordered[:limit]

    out = []
    for op_id in ordered:
        op = tm["op_by_id"].get(op_id, {})
        out.append({
            "kind": "primitive",
            "id": op_id,
            "name": primitive_display_name(op_id, tm, semantic_attention=semantic_attention),
            "source_name": op_name_of(op, op_id),
            "op_type": op_type_of(op),
            "op_index": op_order_of(op_id, tm),
            "section": op_section(op_id, tm),
        })
    return out


def dependency_sort_region_children(
    child_ids: list[str],
    recursive_leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
) -> list[str]:
    if len(child_ids) <= 1:
        return child_ids

    owner: dict[str, str] = {}
    for cid in child_ids:
        for op in recursive_leaf_ops.get(cid, set()):
            owner.setdefault(op, cid)

    edges: dict[str, set[str]] = {cid: set() for cid in child_ids}
    indeg: dict[str, int] = {cid: 0 for cid in child_ids}

    for dst_cid in child_ids:
        for op_id in recursive_leaf_ops.get(dst_cid, set()):
            op = tm["op_by_id"].get(op_id)
            if not op:
                continue

            for inp in op_inputs(op):
                prod_op = tm["value_producer"].get(inp)
                if not prod_op:
                    continue
                src_cid = owner.get(prod_op)
                if src_cid and src_cid != dst_cid and dst_cid not in edges[src_cid]:
                    edges[src_cid].add(dst_cid)
                    indeg[dst_cid] += 1

    ready = [cid for cid in child_ids if indeg[cid] == 0]
    ready.sort(key=lambda cid: (
        range_start(op_range_string(recursive_leaf_ops.get(cid, set()), tm["op_order"])),
        cid,
    ))

    out: list[str] = []

    while ready:
        cur = ready.pop(0)
        out.append(cur)

        for dst in sorted(edges[cur], key=lambda cid: (
            range_start(op_range_string(recursive_leaf_ops.get(cid, set()), tm["op_order"])),
            cid,
        )):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)

        ready.sort(key=lambda cid: (
            range_start(op_range_string(recursive_leaf_ops.get(cid, set()), tm["op_order"])),
            cid,
        ))

    if len(out) != len(child_ids):
        seen = set(out)
        rest = [cid for cid in child_ids if cid not in seen]
        rest.sort(key=lambda cid: (
            range_start(op_range_string(recursive_leaf_ops.get(cid, set()), tm["op_order"])),
            cid,
        ))
        out.extend(rest)

    return out


# =============================================================================
# Shape motifs
# =============================================================================

def shape_motif_key(op_id: str, tm: dict[str, Any]) -> tuple[str, str]:
    path = op_path_by_id(op_id, tm)
    layer = first_layer_from_path(path)
    typ = op_type_by_id(op_id, tm).lower().replace("_", "")

    predicate_types = {
        "greater",
        "greaterorequal",
        "less",
        "lessorequal",
        "equal",
        "and",
        "or",
        "not",
        "where",
        "isnan",
        "constantofshape",
    }

    if typ in predicate_types:
        if layer is not None and "/attention/self/" in path:
            return (f"Encoder Layer {layer}", f"Layer {layer} attention mask application")
        return ("Auxiliary Shape / Mask Flow", "Global predicate / mask preprocessing")

    if "/embeddings/" in path:
        if "position" in path or "slice" in path or "gather" in path:
            return ("Embeddings", "Embedding position-id / lookup shape helpers")
        return ("Embeddings", "Embedding shape helpers")

    if "/model/bert/" in path and "/encoder/layer." not in path:
        return ("Auxiliary Shape / Mask Flow", "Global attention-mask preprocessing")

    if layer is not None and "/attention/self/" in path:
        if "/where" in path:
            return (f"Encoder Layer {layer}", f"Layer {layer} attention mask application")
        if any(x in path for x in ["/reshape", "/transpose", "/concat", "/unsqueeze", "/shape", "/gather"]):
            return (f"Encoder Layer {layer}", f"Layer {layer} attention Q/K/V shape plumbing")
        return (f"Encoder Layer {layer}", f"Layer {layer} attention scalar/constants")

    if layer is not None and "/intermediate/intermediate_act_fn/" in path:
        return (f"Encoder Layer {layer}", f"Layer {layer} GELU scalar constants")

    if layer is not None:
        return (f"Encoder Layer {layer}", f"Layer {layer} auxiliary shape/mask flow")

    if "/cls/" in path:
        return ("Prediction Head", "Prediction-head auxiliary shape/mask flow")

    return ("Auxiliary Shape / Mask Flow", "Miscellaneous auxiliary shape/mask flow")


def build_shape_motif_records(
    model: str,
    model_ops: set[str],
    tm: dict[str, Any],
    *,
    max_leaf_names: int,
    include_root_leaves: bool,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], set[str]] = defaultdict(set)

    for op_id in model_ops:
        if is_auxiliary_op(op_id, tm):
            buckets[shape_motif_key(op_id, tm)].add(op_id)

    records: list[dict[str, Any]] = []

    for (section, name), ops in buckets.items():
        if not ops:
            continue

        rid = "shape_motif::" + re.sub(r"[^a-z0-9]+", "_", compact(name)).strip("_")
        leaves = []
        truncated = False
        if include_root_leaves:
            leaves = primitive_leaves(ops, tm, limit=max_leaf_names, semantic_attention=True)
            truncated = len(ops) > max_leaf_names

        records.append({
            "kind": "shape_motif",
            "region_id": rid,
            "name": name,
            "region_type": "ShapeMotifRegion",
            "section": section,
            "op_range": op_range_string(ops, tm["op_order"]),
            "leaf_count": len(ops),
            "pruning_role": "propagation_only",
            "confidence": "medium",
            "reason": "Grouped auxiliary shape/mask operations into a learner-level motif.",
            "immediate_expansion": primitive_leaves(ops, tm, limit=25, semantic_attention=True),
            "recursive_primitive_leaves": leaves,
            "recursive_primitive_leaf_sample_truncated": truncated,
        })

    records.sort(key=lambda r: (
        section_sort_key(r["section"]),
        range_start(r["op_range"]),
        r["name"],
    ))
    return records


# =============================================================================
# Section records and immediate expansion
# =============================================================================

def make_section_records(
    model_rid: str,
    recursive_leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
    *,
    include_auxiliary: bool,
    include_root_leaves: bool,
    max_leaf_names: int,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    section_ops: dict[str, set[str]] = defaultdict(set)

    for op_id in recursive_leaf_ops.get(model_rid, set()):
        sec = op_section(op_id, tm)
        if sec == "Auxiliary Shape / Mask Flow" and not include_auxiliary:
            continue
        section_ops[sec].add(op_id)

    records = []
    for sec, ops in section_ops.items():
        leaves = []
        truncated = False
        if include_root_leaves:
            leaves = primitive_leaves(ops, tm, limit=max_leaf_names, semantic_attention=True)
            truncated = len(ops) > max_leaf_names

        records.append({
            "kind": "section",
            "region_id": f"section::{sec.lower().replace(' ', '_').replace('/', '_')}",
            "name": sec,
            "region_type": "SectionRegion",
            "section": "Model",
            "op_range": op_range_string(ops, tm["op_order"]),
            "leaf_count": len(ops),
            "pruning_role": "analysis_only",
            "confidence": "high",
            "reason": "Virtual learner section grouping source operations by model structure.",
            "immediate_expansion": [],
            "recursive_primitive_leaves": leaves,
            "recursive_primitive_leaf_sample_truncated": truncated,
        })

    records.sort(key=lambda r: section_sort_key(r["name"]))
    return records, dict(section_ops)


def _record_to_expansion_item(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": rec.get("kind", "abstract"),
        "id": rec.get("region_id"),
        "name": rec.get("name"),
        "region_type": rec.get("region_type"),
        "op_range": rec.get("op_range"),
        "leaf_count": rec.get("leaf_count", 0),
        "section": rec.get("section"),
    }


def _expansion_sort_key(item: dict[str, Any]) -> tuple[int, str, str]:
    return (
        range_start(item.get("op_range", "")),
        text(item.get("name", "")),
        text(item.get("id", "")),
    )


def populate_section_immediate_expansions(
    records: list[dict[str, Any]],
    section_records: list[dict[str, Any]],
    section_ops: dict[str, set[str]],
    tm: dict[str, Any],
    *,
    view: str,
    max_leaf_names: int,
) -> None:
    """Populate virtual SectionRegion immediate expansions with learner-level nodes."""
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nested_ids_by_section: dict[str, set[str]] = defaultdict(set)

    for rec in records:
        if rec.get("region_type") in {"ModelRegion", "SectionRegion"}:
            continue
        if view == "shape" and rec.get("region_type") != "ShapeMotifRegion":
            continue
        if view == "main" and rec.get("region_type") == "ShapeMotifRegion":
            continue
        by_section[rec.get("section", "Other Main Flow")].append(rec)

    for section, recs in by_section.items():
        rec_ids = {rec.get("region_id") for rec in recs}
        for rec in recs:
            for item in rec.get("immediate_expansion", []):
                item_id = item.get("id")
                if item_id in rec_ids:
                    nested_ids_by_section[section].add(item_id)

    for section in section_records:
        sec_name = section["name"]
        candidates = [
            _record_to_expansion_item(rec)
            for rec in by_section.get(sec_name, [])
            if rec.get("region_id") not in nested_ids_by_section.get(sec_name, set())
        ]
        candidates.sort(key=_expansion_sort_key)

        if not candidates:
            candidates = primitive_leaves(
                section_ops.get(sec_name, set()),
                tm,
                limit=max_leaf_names,
                semantic_attention=True,
            )

        section["immediate_expansion"] = candidates


def immediate_expansion_for_region(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    recursive_leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
    *,
    view: str,
    compress_single_op_wrappers: bool,
    include_single_op_shape_regions: bool,
) -> list[dict[str, Any]]:
    child_ids = region_children(rid, region_by_id, children_by_parent)
    child_ids = dependency_sort_region_children(child_ids, recursive_leaf_ops, tm)

    out: list[dict[str, Any]] = []
    covered: set[str] = set()

    for cid in child_ids:
        c_ops = recursive_leaf_ops.get(cid, set())

        if view == "main" and is_auxiliary_region(cid, region_by_id, recursive_leaf_ops, tm):
            continue

        if view == "shape" and not is_auxiliary_region(cid, region_by_id, recursive_leaf_ops, tm):
            continue

        if (
            view == "shape"
            and not include_single_op_shape_regions
            and len(c_ops) == 1
            and region_by_id[cid].get("region_type") in {"AxisTransformRegion", "ForkRegion", "JoinRegion"}
        ):
            continue

        if (
            compress_single_op_wrappers
            and len(c_ops) == 1
            and region_by_id[cid].get("region_type") in {
                "PrimitiveRegion",
                "AxisTransformRegion",
                "ForkRegion",
                "JoinRegion",
            }
        ):
            op_id = next(iter(c_ops))
            if view == "main" and is_auxiliary_op(op_id, tm):
                continue
            if view == "shape" and not is_auxiliary_op(op_id, tm):
                continue
            out.extend(primitive_leaves({op_id}, tm, limit=None, semantic_attention=True))
            covered.add(op_id)
            continue

        out.append({
            "kind": "abstract",
            "id": cid,
            "name": region_display_name(cid, region_by_id, recursive_leaf_ops, tm),
            "region_type": region_by_id[cid].get("region_type", ""),
            "op_range": op_range_string(c_ops, tm["op_order"]),
            "leaf_count": len(c_ops),
            "section": region_section(cid, region_by_id, recursive_leaf_ops, tm),
        })
        covered |= c_ops

    direct = direct_ops_not_covered_by_children(
        rid,
        region_by_id,
        children_by_parent,
        recursive_leaf_ops,
    )

    for op_id in sorted(direct - covered, key=lambda x: op_order_of(x, tm)):
        if view == "main" and is_auxiliary_op(op_id, tm):
            continue
        if view == "shape" and not is_auxiliary_op(op_id, tm):
            continue
        out.extend(primitive_leaves({op_id}, tm, limit=None, semantic_attention=True))

    return sorted(out, key=lambda item: item.get("op_index", range_start(item.get("op_range", ""))))


# =============================================================================
# Record construction
# =============================================================================

def build_records(
    model: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    interface_by_region: dict[str, dict[str, Any]],
    recursive_leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
    *,
    view: str,
    max_leaf_names: int,
    compress_single_op_wrappers: bool,
    include_root_leaves: bool,
    include_single_op_shape_regions: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model_rid = find_model_region(region_by_id)

    section_records: list[dict[str, Any]] = []
    section_ops: dict[str, set[str]] = {}
    if model_rid:
        section_records, section_ops = make_section_records(
            model_rid,
            recursive_leaf_ops,
            tm,
            include_auxiliary=(view in {"shape", "full"}),
            include_root_leaves=include_root_leaves,
            max_leaf_names=max_leaf_names,
        )

    if model_rid:
        model_ops = recursive_leaf_ops.get(model_rid, set())

        model_leaves = []
        truncated = False
        if include_root_leaves:
            model_leaves = primitive_leaves(model_ops, tm, limit=max_leaf_names, semantic_attention=True)
            truncated = len(model_ops) > max_leaf_names

        model_record = {
            "kind": "abstract",
            "region_id": model_rid,
            "name": "Model",
            "region_type": "ModelRegion",
            "section": "Root",
            "op_range": op_range_string(model_ops, tm["op_order"]),
            "leaf_count": len(model_ops),
            "pruning_role": "analysis_only",
            "confidence": region_by_id[model_rid].get("confidence", "high"),
            "reason": region_by_id[model_rid].get(
                "reason",
                "Root structural region containing all Tensor IR operations.",
            ),
            "immediate_expansion": [
                {
                    "kind": "section",
                    "id": r["region_id"],
                    "name": r["name"],
                    "region_type": "SectionRegion",
                    "op_range": r["op_range"],
                    "leaf_count": r["leaf_count"],
                }
                for r in section_records
            ],
            "recursive_primitive_leaves": model_leaves,
            "recursive_primitive_leaf_sample_truncated": truncated,
        }
        records.append(model_record)

    records.extend(section_records)

    # Shape motif summary comes before single-op shape details.
    if view in {"shape", "full"} and model_rid:
        records.extend(build_shape_motif_records(
            model,
            recursive_leaf_ops.get(model_rid, set()),
            tm,
            max_leaf_names=max_leaf_names,
            include_root_leaves=include_root_leaves,
        ))

    for rid, r in region_by_id.items():
        rt = r.get("region_type", "")

        if rt in {"PrimitiveRegion", "ModelRegion"}:
            continue

        ops = recursive_leaf_ops.get(rid, set())
        if not ops:
            continue

        if view == "main":
            # Correction 1:
            # Do not let auxiliary-only fork/axis/join scaffolding pollute the
            # main learner report.
            if is_auxiliary_only_fork_axis_join(rid, region_by_id, recursive_leaf_ops, tm):
                continue
            if is_auxiliary_region(rid, region_by_id, recursive_leaf_ops, tm):
                continue

        if view == "shape":
            if not is_auxiliary_region(rid, region_by_id, recursive_leaf_ops, tm):
                continue
            if (
                not include_single_op_shape_regions
                and rt in {"AxisTransformRegion", "ForkRegion", "JoinRegion"}
                and len(ops) == 1
            ):
                continue

        iface = interface_by_region.get(rid, {})
        section = region_section(rid, region_by_id, recursive_leaf_ops, tm)

        immediate = immediate_expansion_for_region(
            rid,
            region_by_id,
            children_by_parent,
            recursive_leaf_ops,
            tm,
            view=view,
            compress_single_op_wrappers=compress_single_op_wrappers,
            include_single_op_shape_regions=include_single_op_shape_regions,
        )

        records.append({
            "kind": "abstract",
            "region_id": rid,
            "name": region_display_name(rid, region_by_id, recursive_leaf_ops, tm),
            "region_type": rt,
            "section": section,
            "op_range": op_range_string(ops, tm["op_order"]),
            "leaf_count": len(ops),
            "pruning_role": iface.get("pruning_role", "unknown"),
            "confidence": r.get("confidence", "unknown"),
            "reason": r.get("reason", ""),
            "immediate_expansion": immediate,
            "recursive_primitive_leaves": primitive_leaves(
                ops,
                tm,
                limit=max_leaf_names,
                semantic_attention=True,
            ),
            "recursive_primitive_leaf_sample_truncated": len(ops) > max_leaf_names,
        })

    populate_section_immediate_expansions(
        records,
        section_records,
        section_ops,
        tm,
        view=view,
        max_leaf_names=max_leaf_names,
    )

    records.sort(key=lambda r: (
        0 if r["name"] == "Model" else 1,
        section_sort_key(r.get("section", "")),
        range_start(r.get("op_range", "")),
        r.get("name", ""),
        r.get("region_id", ""),
    ))

    return records


# =============================================================================
# Markdown writer
# =============================================================================

def write_markdown(path: Path, model: str, view: str, records: list[dict[str, Any]]) -> None:
    lines: list[str] = []

    lines.append(f"# Abstract Node Expansion Report: {model}")
    lines.append("")
    lines.append(f"- View: `{view}`")
    lines.append(f"- Records: `{len(records)}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    counts = defaultdict(int)
    for rec in records:
        counts[rec.get("section", "Unknown")] += 1

    lines.append("| Section | Records |")
    lines.append("|---|---:|")
    for sec, cnt in sorted(counts.items(), key=lambda kv: section_sort_key(kv[0])):
        lines.append(f"| {sec} | {cnt} |")
    lines.append("")

    current_section = None

    for rec in records:
        sec = rec.get("section", "Unknown")
        if sec != current_section:
            current_section = sec
            lines.append(f"## {sec}")
            lines.append("")

        lines.append(f"### {rec['name']}")
        lines.append("")
        lines.append(f"- Kind: `{rec.get('kind', 'abstract')}`")
        lines.append(f"- Region type: `{rec['region_type']}`")
        lines.append(f"- Region id: `{rec['region_id']}`")
        lines.append(f"- Op range: `{rec['op_range']}`")
        lines.append(f"- Recursive primitive leaves: `{rec['leaf_count']}`")
        lines.append(f"- Pruning role: `{rec.get('pruning_role', 'unknown')}`")
        lines.append(f"- Confidence: `{rec.get('confidence', 'unknown')}`")

        if rec.get("reason"):
            lines.append(f"- Reason: {rec['reason']}")

        lines.append("")
        lines.append("#### Immediate expansion")
        lines.append("")

        if rec.get("immediate_expansion"):
            for item in rec["immediate_expansion"]:
                if item["kind"] in {"abstract", "section", "shape_motif"}:
                    lines.append(
                        f"- **{item['name']}** "
                        f"(`{item['region_type']}`, ops `{item['op_range']}`, leaves `{item['leaf_count']}`)"
                    )
                else:
                    source = item.get("source_name", item["name"])
                    if source != item["name"]:
                        lines.append(
                            f"- `{item['name']}` "
                            f"({item.get('op_type', '')}, op `{item.get('op_index', '-')}`, source `{source}`)"
                        )
                    else:
                        lines.append(
                            f"- `{item['name']}` "
                            f"({item.get('op_type', '')}, op `{item.get('op_index', '-')}`)"
                        )
        else:
            lines.append("- _No immediate expansion items._")

        leaves = rec.get("recursive_primitive_leaves", [])
        if leaves:
            lines.append("")
            lines.append("#### Recursive primitive leaves")
            lines.append("")
            for item in leaves:
                source = item.get("source_name", item["name"])
                if source != item["name"]:
                    lines.append(
                        f"- `{item['name']}` "
                        f"({item.get('op_type', '')}, op `{item.get('op_index', '-')}`, source `{source}`)"
                    )
                else:
                    lines.append(
                        f"- `{item['name']}` "
                        f"({item.get('op_type', '')}, op `{item.get('op_index', '-')}`)"
                    )
            if rec.get("recursive_primitive_leaf_sample_truncated"):
                lines.append("- ...")

        lines.append("")

    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# PDF writer
# =============================================================================

def xml_escape(s: Any) -> str:
    from xml.sax.saxutils import escape
    return escape(text(s))


def para(s: Any, style):
    from reportlab.platypus import Paragraph
    return Paragraph(xml_escape(s), style)


def write_pdf(path: Path, model: str, view: str, records: list[dict[str, Any]]) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
            KeepTogether,
        )
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: reportlab\n"
            "Install with:\n"
            "  ./conda-env/bin/python -m pip install reportlab"
        ) from e

    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"], fontSize=17, leading=21)
    h1 = ParagraphStyle("H1x", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=7)
    h2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=11, leading=13, spaceBefore=8, spaceAfter=5)
    body = ParagraphStyle("Bodyx", parent=styles["BodyText"], fontSize=8, leading=10)
    small = ParagraphStyle("Smallx", parent=styles["BodyText"], fontSize=7, leading=8)
    code = ParagraphStyle("Codex", parent=styles["Code"], fontSize=6.3, leading=7.5, wordWrap="CJK")

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )

    story = []
    story.append(Paragraph(f"Abstract Node Expansion Report: {model}", title))
    story.append(Paragraph(f"View: {view}", body))
    story.append(Paragraph(f"Records: {len(records)}", body))
    story.append(Spacer(1, 8))

    counts = defaultdict(int)
    for rec in records:
        counts[rec.get("section", "Unknown")] += 1

    summary = [["Section", "Records"]]
    for sec, cnt in sorted(counts.items(), key=lambda kv: section_sort_key(kv[0])):
        summary.append([sec, str(cnt)])

    summary_table = Table(
        [[para(c, small) for c in row] for row in summary],
        colWidths=[4.8 * inch, 1.2 * inch],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    current_section = None

    for rec in records:
        sec = rec.get("section", "Unknown")
        if sec != current_section:
            current_section = sec
            story.append(Paragraph(sec, h1))

        block = []
        block.append(Paragraph(rec["name"], h2))

        meta = [
            ["Kind", rec.get("kind", "abstract")],
            ["Region type", rec["region_type"]],
            ["Region id", rec["region_id"]],
            ["Op range", rec["op_range"]],
            ["Recursive leaves", str(rec["leaf_count"])],
            ["Pruning role", rec.get("pruning_role", "unknown")],
            ["Confidence", rec.get("confidence", "unknown")],
        ]
        if rec.get("reason"):
            meta.append(["Reason", rec["reason"]])

        meta_table = Table(
            [[para(a, small), para(b, small)] for a, b in meta],
            colWidths=[1.25 * inch, 5.6 * inch],
        )
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        block.append(meta_table)
        block.append(Spacer(1, 5))

        block.append(Paragraph("Immediate expansion", body))
        exp = [["Kind", "Name", "Details"]]

        if rec.get("immediate_expansion"):
            for item in rec["immediate_expansion"]:
                if item["kind"] in {"abstract", "section", "shape_motif"}:
                    exp.append([
                        item["kind"],
                        item["name"],
                        f"{item['region_type']} | ops {item['op_range']} | leaves {item['leaf_count']}",
                    ])
                else:
                    source = item.get("source_name", item["name"])
                    detail = f"{item.get('op_type', '')} | op {item.get('op_index', '-')}"
                    if source != item["name"]:
                        detail += f" | source {source}"
                    exp.append(["primitive", item["name"], detail])
        else:
            exp.append(["-", "No immediate expansion items", "-"])

        exp_table = Table(
            [[para(c, code if j == 1 else small) for j, c in enumerate(row)] for row in exp],
            colWidths=[0.8 * inch, 3.9 * inch, 2.15 * inch],
            repeatRows=1,
        )
        exp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        block.append(exp_table)

        leaves = rec.get("recursive_primitive_leaves", [])
        if leaves:
            block.append(Spacer(1, 5))
            block.append(Paragraph("Recursive primitive leaves", body))
            for item in leaves:
                source = item.get("source_name", item["name"])
                if source != item["name"]:
                    line = (
                        f"• {item['name']} "
                        f"({item.get('op_type', '')}, op {item.get('op_index', '-')}, source {source})"
                    )
                else:
                    line = (
                        f"• {item['name']} "
                        f"({item.get('op_type', '')}, op {item.get('op_index', '-')})"
                    )
                block.append(Paragraph(line, code))

            if rec.get("recursive_primitive_leaf_sample_truncated"):
                block.append(Paragraph("• ...", code))

        block.append(Spacer(1, 9))
        story.append(KeepTogether(block))

    doc.build(story)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate corrected abstract-node expansion report with learner-level structure."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree-json", default=None)
    ap.add_argument("--tensor-ir", default=None)
    ap.add_argument("--out-dir", default="reports/abstract_node_expansions")
    ap.add_argument("--view", choices=["main", "shape", "full"], default="main")
    ap.add_argument("--max-leaf-names", type=int, default=30)
    ap.add_argument("--no-compress-single-op-wrappers", action="store_true")

    # Correction 5: explicit debug switches.
    ap.add_argument(
        "--include-root-leaves",
        action="store_true",
        help="Print recursive primitive leaves for Model/SectionRegion/ShapeMotifRegion records.",
    )
    ap.add_argument(
        "--include-single-op-shape-regions",
        action="store_true",
        help="In shape view, include one-op AxisTransform/Fork/Join records instead of only grouped shape motifs.",
    )

    args = ap.parse_args()

    safe = safe_model_name(args.model)

    tree_path = Path(args.tree_json or f"reports/structural_region_trees/{safe}.json")
    tensor_path = Path(args.tensor_ir or f"reports/tensor_ir/{safe}.json")

    if not tree_path.exists():
        raise FileNotFoundError(
            f"Missing Structural Region Tree: {tree_path}\n"
            f"Run:\n"
            f"  ./conda-env/bin/python scripts/build_structural_region_tree.py --model {args.model} --verbose"
        )

    if not tensor_path.exists():
        raise FileNotFoundError(
            f"Missing Tensor IR: {tensor_path}\n"
            f"Run:\n"
            f"  ./conda-env/bin/python scripts/build_tensor_ir.py --model {args.model} --verbose"
        )

    tree = load_json(tree_path)
    tensor_ir = load_json(tensor_path)

    tm = build_tensor_maps(tensor_ir)
    region_by_id, children_by_parent, interface_by_region = build_region_maps(tree)
    recursive_leaf_ops = compute_recursive_leaf_ops(region_by_id, children_by_parent)

    records = build_records(
        args.model,
        region_by_id,
        children_by_parent,
        interface_by_region,
        recursive_leaf_ops,
        tm,
        view=args.view,
        max_leaf_names=args.max_leaf_names,
        compress_single_op_wrappers=not args.no_compress_single_op_wrappers,
        include_root_leaves=args.include_root_leaves,
        include_single_op_shape_regions=args.include_single_op_shape_regions,
    )

    out_dir = Path(args.out_dir) / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    base = f"abstract_node_expansions_{args.view}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"
    pdf_path = out_dir / f"{base}.pdf"

    json_path.write_text(json.dumps({
        "model_name": args.model,
        "view": args.view,
        "num_records": len(records),
        "debug": {
            "include_root_leaves": args.include_root_leaves,
            "include_single_op_shape_regions": args.include_single_op_shape_regions,
            "compress_single_op_wrappers": not args.no_compress_single_op_wrappers,
        },
        "records": records,
    }, indent=2, sort_keys=True))

    write_markdown(md_path, args.model, args.view, records)
    write_pdf(pdf_path, args.model, args.view, records)

    print(f"[abstract-expansion-report] model={args.model}")
    print(f"[abstract-expansion-report] view={args.view}")
    print(f"[abstract-expansion-report] records={len(records)}")
    print(f"[abstract-expansion-report] json={json_path}")
    print(f"[abstract-expansion-report] markdown={md_path}")
    print(f"[abstract-expansion-report] pdf={pdf_path}")
    print("[abstract-expansion-report] corrections enabled:")
    print("  - main view removes auxiliary-only Fork/Axis/Join records")
    print("  - attention internals use semantic names")
    print("  - shape view groups one-op axis transforms into ShapeMotifRegion records")
    print("  - Model/SectionRegion records hide recursive primitive leaves by default")
    print("  - debug flags: --include-root-leaves and --include-single-op-shape-regions")


if __name__ == "__main__":
    main()
