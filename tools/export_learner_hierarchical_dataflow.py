#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LARGE = 10**12


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def safe_id(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(name))
    return s.strip("_") or "unknown"


def dot_escape(x: Any) -> str:
    s = str(x if x is not None else "")
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def last_int(s: Any) -> int | None:
    nums = re.findall(r"\d+", str(s))
    return int(nums[-1]) if nums else None


def compact(s: Any) -> str:
    return str(s or "").replace("/", ".").replace("_", ".").lower()


# ---------------------------------------------------------------------
# TensorIR helpers
# ---------------------------------------------------------------------

def tensor_ops(tensor_ir: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not tensor_ir:
        return []
    return (
        tensor_ir.get("ops")
        or tensor_ir.get("operations")
        or tensor_ir.get("tensor_ops")
        or []
    )


def op_id_of(op: dict[str, Any]) -> str | None:
    return op.get("op_id") or op.get("id") or op.get("name")


def op_type_of(op: dict[str, Any]) -> str:
    return str(
        op.get("canonical_op_type")
        or op.get("op_type")
        or op.get("type")
        or ""
    )


def op_name_of(op: dict[str, Any], fallback: str) -> str:
    return str(
        op.get("source_node_name")
        or op.get("onnx_node_name")
        or op.get("name")
        or op.get("label")
        or fallback
    )


def op_inputs(op: dict[str, Any]) -> list[str]:
    vals = op.get("inputs") or op.get("input_values") or op.get("input_ids") or []
    return [str(v) for v in vals]


def op_outputs(op: dict[str, Any]) -> list[str]:
    vals = op.get("outputs") or op.get("output_values") or op.get("output_ids") or []
    return [str(v) for v in vals]


def build_tensor_maps(tensor_ir: dict[str, Any] | None) -> dict[str, Any]:
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


def raw_op_order(op_id: str, tm: dict[str, Any]) -> int:
    if op_id in tm["op_order"]:
        return tm["op_order"][op_id]
    n = last_int(op_id)
    return n if n is not None else LARGE


# ---------------------------------------------------------------------
# Structural Region Tree helpers
# ---------------------------------------------------------------------

def build_region_maps(tree: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str | None, list[str]]]:
    region_by_id: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str | None, list[str]] = defaultdict(list)

    for r in tree.get("regions", []):
        rid = r.get("region_id")
        if not rid:
            continue
        region_by_id[rid] = r
        children_by_parent[r.get("parent")].append(rid)

    return region_by_id, children_by_parent


def region_children(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
) -> list[str]:
    r = region_by_id[rid]
    explicit = r.get("children")
    if isinstance(explicit, list):
        return [c for c in explicit if c in region_by_id]
    return [c for c in children_by_parent.get(rid, []) if c in region_by_id]


def own_region_ops(region: dict[str, Any]) -> set[str]:
    return set(str(x) for x in (region.get("op_ids") or []))


def compute_region_leaf_ops(
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def visit(rid: str) -> set[str]:
        if rid in memo:
            return memo[rid]

        r = region_by_id[rid]
        leaves = set(own_region_ops(r))

        for cid in region_children(rid, region_by_id, children_by_parent):
            leaves |= visit(cid)

        memo[rid] = leaves
        return leaves

    for rid in region_by_id:
        visit(rid)

    return memo


def min_topo_ops(ops: set[str], tm: dict[str, Any]) -> int:
    if not ops:
        return LARGE
    return min(raw_op_order(op, tm) for op in ops)


def max_topo_ops(ops: set[str], tm: dict[str, Any]) -> int:
    if not ops:
        return LARGE
    return max(raw_op_order(op, tm) for op in ops)


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

SHAPE_OP_TYPES = {
    "shape",
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
    "gather",       # only shape-helper Gather, not embedding Gather
    "flatten",
    "expand",
}

MASK_BOOL_OP_TYPES = {
    "equal",
    "greater",
    "greaterorequal",
    "less",
    "lessorequal",
    "and",
    "or",
    "not",
    "where",
    "isnan",
}


def op_text(op_id: str, tm: dict[str, Any]) -> str:
    op = tm["op_by_id"].get(op_id, {})
    return compact(" ".join([
        op_name_of(op, op_id),
        op_type_of(op),
        " ".join(op_inputs(op)),
        " ".join(op_outputs(op)),
    ]))


def is_embedding_op(op_id: str, tm: dict[str, Any]) -> bool:
    t = op_text(op_id, tm)
    return any(x in t for x in [
        "word.embeddings",
        "token.type.embeddings",
        "position.embeddings",
        "embeddings.word",
        "embeddings.token",
        "embeddings.position",
    ])


def is_prediction_head_op(op_id: str, tm: dict[str, Any]) -> bool:
    return "model.cls" in op_text(op_id, tm) or ".cls." in op_text(op_id, tm)


def detect_layer_from_text(t: str) -> int | None:
    patterns = [
        r"encoder[._/]layer[._/](\d+)",
        r"encoder.layer.(\d+)",
        r"encoder_layer_(\d+)",
        r"layer[._/](\d+)",
        r"layer_(\d+)",
    ]
    for p in patterns:
        m = re.search(p, t, re.I)
        if m:
            return int(m.group(1))
    return None


def op_layer(op_id: str, tm: dict[str, Any]) -> int | None:
    return detect_layer_from_text(op_text(op_id, tm))


def region_text(rid: str, region_by_id: dict[str, dict[str, Any]], leaf_ops: dict[str, set[str]], tm: dict[str, Any]) -> str:
    r = region_by_id[rid]
    parts = [r.get("region_type", ""), str(r.get("metadata") or {})]

    for oid in sorted(leaf_ops.get(rid, set()), key=lambda x: raw_op_order(x, tm)):
        parts.append(op_text(oid, tm))

    return compact(" ".join(parts))


def region_layer(rid: str, region_by_id: dict[str, dict[str, Any]], leaf_ops: dict[str, set[str]], tm: dict[str, Any]) -> int | None:
    t = region_text(rid, region_by_id, leaf_ops, tm)
    return detect_layer_from_text(t)


def is_shape_helper_op(op_id: str, tm: dict[str, Any]) -> bool:
    op = tm["op_by_id"].get(op_id, {})
    typ = op_type_of(op).lower()
    t = op_text(op_id, tm)

    if is_embedding_op(op_id, tm):
        return False

    if "attention.self.matmul" in t or "attention.self.softmax" in t:
        return False

    if "dense.matmul" in t or "dense.add" in t:
        return False

    if "layernormalization" in typ.lower() or "layernormalization" in t:
        return False

    if typ in SHAPE_OP_TYPES:
        return True

    if typ in MASK_BOOL_OP_TYPES:
        return True

    if any(x in t for x in [
        "attention.mask",
        "constantofshape",
        "shape",
        "unsqueeze",
        "squeeze",
        "reshape",
        "transpose",
        "range",
        "concat",
        "where",
    ]):
        # But do not hide true attention matmul/softmax.
        if any(y in t for y in ["matmul", "softmax", "layernormalization"]):
            return False
        return True

    return False


def is_shape_helper_region(rid: str, region_by_id: dict[str, dict[str, Any]], leaf_ops: dict[str, set[str]], tm: dict[str, Any]) -> bool:
    r = region_by_id[rid]
    rt = r.get("region_type", "")

    if rt == "AxisTransformRegion":
        return True

    ops = leaf_ops.get(rid, set())
    if not ops:
        return False

    # A region is shape-helper only if all of its primitive leaves are helper-ish.
    return all(is_shape_helper_op(op, tm) for op in ops)


def is_main_compute_op(op_id: str, tm: dict[str, Any]) -> bool:
    return not is_shape_helper_op(op_id, tm)


def is_main_compute_region(rid: str, region_by_id: dict[str, dict[str, Any]], leaf_ops: dict[str, set[str]], tm: dict[str, Any]) -> bool:
    ops = leaf_ops.get(rid, set())
    if not ops:
        return False
    return any(is_main_compute_op(op, tm) for op in ops)


def is_single_op_wrapper(rid: str, region_by_id: dict[str, dict[str, Any]], leaf_ops: dict[str, set[str]]) -> bool:
    r = region_by_id[rid]
    ops = leaf_ops.get(rid, set())
    return len(ops) == 1 and r.get("region_type") in {
        "AxisTransformRegion",
        "PrimitiveRegion",
        "JoinRegion",
        "ForkRegion",
    }


# ---------------------------------------------------------------------
# Learner tree model
# ---------------------------------------------------------------------

@dataclass
class LNode:
    id: str
    label: str
    kind: str
    ops: set[str] = field(default_factory=set)
    children: list["LNode"] = field(default_factory=list)
    color: str = "#eeeeee"
    border: str = "#333333"
    shape: str = "box"

    @property
    def min_topo(self) -> int:
        return getattr(self, "_min_topo", LARGE)

    @min_topo.setter
    def min_topo(self, v: int) -> None:
        setattr(self, "_min_topo", v)

    @property
    def max_topo(self) -> int:
        return getattr(self, "_max_topo", LARGE)

    @max_topo.setter
    def max_topo(self, v: int) -> None:
        setattr(self, "_max_topo", v)


def region_title(rid: str, region_by_id: dict[str, dict[str, Any]], leaf_ops: dict[str, set[str]], tm: dict[str, Any]) -> str:
    r = region_by_id[rid]
    rt = r.get("region_type", "")
    t = region_text(rid, region_by_id, leaf_ops, tm)
    layer = region_layer(rid, region_by_id, leaf_ops, tm)
    prefix = f"Layer {layer} " if layer is not None else ""

    if rt == "FeedForwardRegion":
        return prefix + "Feed Forward"

    if rt == "AttentionSkeletonRegion":
        return prefix + "Attention"

    if rt == "ResidualMergeRegion":
        if "embedding" in t:
            return "Embedding Add"
        if "attention.output" in t:
            return prefix + "Attention Residual Add"
        if ".output.add" in t:
            return prefix + "FFN Residual Add"
        return prefix + "Residual Add"

    if rt == "LayerNormRegion":
        if "embedding" in t:
            return "Embedding LayerNorm"
        return prefix + "LayerNorm"

    if rt == "LinearProjectionRegion":
        if "query" in t:
            return prefix + "Query Projection"
        if "key" in t:
            return prefix + "Key Projection"
        if "value" in t:
            return prefix + "Value Projection"
        if "attention.output" in t:
            return prefix + "Attention Output Projection"
        if "intermediate" in t:
            return prefix + "Intermediate Projection"
        if "output.dense" in t:
            return prefix + "FFN Output Projection"
        if "cls" in t:
            return "Prediction Projection"
        return prefix + "Linear Projection"

    if rt == "ActivationRegion":
        if "gelu" in t or "erf" in t:
            return prefix + "GELU"
        return prefix + "Activation"

    if rt == "AxisTransformRegion":
        return "Shape Transform"

    if rt == "PrimitiveRegion":
        ops = sorted(leaf_ops.get(rid, set()), key=lambda x: raw_op_order(x, tm))
        if ops:
            op = tm["op_by_id"].get(ops[0], {})
            return op_name_of(op, ops[0])
        return "Primitive"

    return rt.replace("Region", "") or "Region"


def region_style(rid: str, region_by_id: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    rt = region_by_id[rid].get("region_type", "")

    table = {
        "FeedForwardRegion": ("#c9f7d5", "#15803d", "component"),
        "AttentionSkeletonRegion": ("#d6e4ff", "#1d4ed8", "octagon"),
        "ResidualMergeRegion": ("#ffd6d6", "#b91c1c", "diamond"),
        "LayerNormRegion": ("#ffe2bf", "#c2410c", "box"),
        "LinearProjectionRegion": ("#d9f7be", "#15803d", "box"),
        "ActivationRegion": ("#eadcff", "#7e22ce", "box"),
        "AxisTransformRegion": ("#fff3bf", "#c48a00", "box"),
        "PrimitiveRegion": ("#ffffff", "#555555", "ellipse"),
    }
    return table.get(rt, ("#eeeeee", "#555555", "box"))


def make_op_node(op_id: str, tm: dict[str, Any]) -> LNode:
    op = tm["op_by_id"].get(op_id, {})
    label = op_name_of(op, op_id)
    n = LNode(
        id=f"op::{op_id}",
        label=label,
        kind="op",
        ops={op_id},
        color="#ffffff",
        border="#555555",
        shape="ellipse",
    )
    n.min_topo = raw_op_order(op_id, tm)
    n.max_topo = raw_op_order(op_id, tm)
    return n


def make_region_node(
    rid: str,
    region_by_id: dict[str, dict[str, Any]],
    leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
) -> LNode:
    color, border, shape = region_style(rid, region_by_id)
    ops = set(leaf_ops.get(rid, set()))
    n = LNode(
        id=f"region::{rid}",
        label=region_title(rid, region_by_id, leaf_ops, tm),
        kind=region_by_id[rid].get("region_type", "Region"),
        ops=ops,
        color=color,
        border=border,
        shape=shape,
    )
    n.min_topo = min_topo_ops(ops, tm)
    n.max_topo = max_topo_ops(ops, tm)
    return n


# ---------------------------------------------------------------------
# Build learner hierarchy
# ---------------------------------------------------------------------

def classify_top_section(op_id: str, tm: dict[str, Any]) -> str:
    if is_shape_helper_op(op_id, tm):
        return "aux"

    if is_embedding_op(op_id, tm):
        return "embeddings"

    layer = op_layer(op_id, tm)
    if layer is not None:
        return f"layer::{layer}"

    if is_prediction_head_op(op_id, tm):
        return "prediction"

    return "other"


def section_label(section: str) -> str:
    if section == "embeddings":
        return "Embeddings"
    if section.startswith("layer::"):
        return f"Encoder Layer {section.split('::', 1)[1]}"
    if section == "prediction":
        return "Prediction Head"
    if section == "aux":
        return "Auxiliary Shape / Mask Flow"
    return "Other Main Flow"


def section_style(section: str) -> tuple[str, str, str]:
    if section == "embeddings":
        return "#e0f2fe", "#0369a1", "folder"
    if section.startswith("layer::"):
        return "#f0fdf4", "#15803d", "folder"
    if section == "prediction":
        return "#fef3c7", "#b45309", "folder"
    if section == "aux":
        return "#f3f4f6", "#6b7280", "folder"
    return "#eeeeee", "#444444", "folder"


def collect_best_region_items_for_ops(
    candidate_rids: list[str],
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
    *,
    view: str,
    section: str,
    include_primitives: bool,
    compress_single_op_wrappers: bool,
) -> list[LNode]:
    """
    Select display items from regions without duplicating primitives.

    Strategy:
    - Prefer meaningful semantic regions.
    - Hide shape-only regions in main view.
    - In shape view, show shape regions but compress single-op wrappers.
    - Add uncovered primitive ops exactly once.
    """
    section_ops: set[str] = set()

    for rid in candidate_rids:
        section_ops |= leaf_ops.get(rid, set())

    # Restrict by view/section.
    def op_allowed(op: str) -> bool:
        cls = classify_top_section(op, tm)
        if view == "main":
            return cls != "aux" and cls == section
        if view == "shape":
            return cls == "aux" and section == "aux"
        if view == "full":
            return cls == section
        return True

    section_ops = {op for op in section_ops if op_allowed(op)}

    # Meaningful region = more than just a single low-level wrapper,
    # and not shape-only in main view.
    selected: list[str] = []

    for rid in candidate_rids:
        ops = set(leaf_ops.get(rid, set())) & section_ops
        if not ops:
            continue

        if view == "main" and is_shape_helper_region(rid, region_by_id, leaf_ops, tm):
            continue

        if compress_single_op_wrappers and is_single_op_wrapper(rid, region_by_id, leaf_ops):
            continue

        rt = region_by_id[rid].get("region_type", "")
        if rt in {
            "FeedForwardRegion",
            "AttentionSkeletonRegion",
            "ResidualMergeRegion",
            "LayerNormRegion",
            "LinearProjectionRegion",
            "ActivationRegion",
        }:
            selected.append(rid)
        elif view in {"shape", "full"} and rt == "AxisTransformRegion":
            selected.append(rid)
        elif not include_primitives and rt != "PrimitiveRegion":
            selected.append(rid)

    # Remove region overlaps by keeping larger semantic regions first, then non-overlapping.
    selected.sort(
        key=lambda rid: (
            min_topo_ops(leaf_ops.get(rid, set()) & section_ops, tm),
            -len(leaf_ops.get(rid, set()) & section_ops),
            rid,
        )
    )

    covered: set[str] = set()
    items: list[LNode] = []

    for rid in selected:
        ops = set(leaf_ops.get(rid, set())) & section_ops
        if not ops:
            continue

        # Avoid duplicates: if this region is completely already covered, skip.
        if ops <= covered:
            continue

        # If it overlaps partially, still keep it only if it is a meaningful larger region.
        if ops & covered and len(ops - covered) == 0:
            continue

        node = make_region_node(rid, region_by_id, {rid: ops}, tm)
        items.append(node)
        covered |= ops

    if include_primitives:
        for op in sorted(section_ops - covered, key=lambda x: raw_op_order(x, tm)):
            if view == "main" and is_shape_helper_op(op, tm):
                continue
            if view == "shape" and not is_shape_helper_op(op, tm):
                continue
            items.append(make_op_node(op, tm))

    items.sort(key=lambda n: (n.min_topo, n.max_topo, n.label))
    return items


def dependency_sort(nodes: list[LNode], tm: dict[str, Any]) -> list[LNode]:
    if len(nodes) <= 1:
        return nodes

    op_owner: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for op in node.ops:
            op_owner.setdefault(op, i)

    edges: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    indeg = {i: 0 for i in range(len(nodes))}

    for j, node in enumerate(nodes):
        for op_id in node.ops:
            op = tm["op_by_id"].get(op_id)
            if not op:
                continue
            for inp in op_inputs(op):
                prod = tm["value_producer"].get(inp)
                if not prod:
                    continue
                i = op_owner.get(prod)
                if i is None or i == j:
                    continue
                if j not in edges[i]:
                    edges[i].add(j)
                    indeg[j] += 1

    ready = [i for i in range(len(nodes)) if indeg[i] == 0]
    ready.sort(key=lambda i: (nodes[i].min_topo, nodes[i].max_topo, nodes[i].label))

    out: list[int] = []

    while ready:
        cur = ready.pop(0)
        out.append(cur)
        for dst in sorted(edges[cur], key=lambda i: (nodes[i].min_topo, nodes[i].label)):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
        ready.sort(key=lambda i: (nodes[i].min_topo, nodes[i].max_topo, nodes[i].label))

    if len(out) != len(nodes):
        seen = set(out)
        rest = [i for i in range(len(nodes)) if i not in seen]
        rest.sort(key=lambda i: (nodes[i].min_topo, nodes[i].max_topo, nodes[i].label))
        out.extend(rest)

    return [nodes[i] for i in out]


def build_learner_tree(
    model: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
    *,
    view: str,
    include_primitives: bool,
    compress_single_op_wrappers: bool,
) -> LNode:
    root = LNode(
        id="section::model",
        label="Model",
        kind="Model",
        color="#eeeeee",
        border="#333333",
        shape="box3d",
    )

    all_ops = set(tm["op_by_id"].keys())

    # Root children are explicit learner sections.
    section_ops: dict[str, set[str]] = defaultdict(set)

    for op in all_ops:
        sec = classify_top_section(op, tm)

        if view == "main" and sec == "aux":
            continue
        if view == "shape" and sec != "aux":
            continue

        section_ops[sec].add(op)

    def section_key(sec: str) -> tuple[int, int, str]:
        if sec == "embeddings":
            rank = 0
        elif sec.startswith("layer::"):
            rank = 1
        elif sec == "prediction":
            rank = 2
        elif sec == "other":
            rank = 3
        elif sec == "aux":
            rank = 4
        else:
            rank = 5

        mn = min_topo_ops(section_ops[sec], tm)
        return (rank, mn, sec)

    # Candidate regions are all regions. The collector will filter by section.
    all_rids = list(region_by_id.keys())

    for sec in sorted(section_ops.keys(), key=section_key):
        if not section_ops[sec]:
            continue

        color, border, shape = section_style(sec)
        s = LNode(
            id=f"section::{sec}",
            label=section_label(sec),
            kind="Section",
            ops=set(section_ops[sec]),
            color=color,
            border=border,
            shape=shape,
        )
        s.min_topo = min_topo_ops(s.ops, tm)
        s.max_topo = max_topo_ops(s.ops, tm)

        children = collect_best_region_items_for_ops(
            all_rids,
            region_by_id,
            children_by_parent,
            leaf_ops,
            tm,
            view=view,
            section=sec,
            include_primitives=include_primitives,
            compress_single_op_wrappers=compress_single_op_wrappers,
        )

        children = dependency_sort(children, tm)
        s.children = children

        root.children.append(s)
        root.ops |= s.ops

    root.min_topo = min_topo_ops(root.ops, tm)
    root.max_topo = max_topo_ops(root.ops, tm)
    root.children = dependency_sort(root.children, tm)

    return root


# ---------------------------------------------------------------------
# DOT emission
# ---------------------------------------------------------------------

def dot_node_id(node: LNode) -> str:
    return safe_id(node.id)


def emit_dot(root: LNode, *, title: str, max_depth: int, show_data_edges: bool, tm: dict[str, Any]) -> str:
    lines = [
        "digraph LearnerHierarchicalDataflow {",
        "  graph [rankdir=LR, bgcolor=\"white\", fontname=\"Helvetica\", labelloc=\"t\", labeljust=\"l\", nodesep=0.35, ranksep=0.65];",
        "  node [fontname=\"Helvetica\", fontsize=10, style=\"filled,rounded\", margin=0.08];",
        "  edge [fontname=\"Helvetica\", fontsize=8, color=\"#444444\", arrowsize=0.7];",
        f"  label=\"{dot_escape(title)}\";",
        "",
    ]

    emitted: set[str] = set()
    containment_edges: set[tuple[str, str]] = set()
    data_edges: set[tuple[str, str]] = set()

    def emit_node(node: LNode) -> str:
        nid = dot_node_id(node)
        if nid in emitted:
            return nid

        label = node.label
        if node.kind in {"Section", "Model"} and node.min_topo < LARGE:
            label = f"{label}\\nops {node.min_topo}-{node.max_topo}"

        lines.append(
            f'  "{nid}" [label="{dot_escape(label)}", '
            f'shape="{node.shape}", fillcolor="{node.color}", color="{node.border}", penwidth=1.8];'
        )
        emitted.add(nid)
        return nid

    def walk(node: LNode, depth: int) -> None:
        if depth > max_depth:
            return

        parent_id = emit_node(node)

        for child in node.children:
            if depth + 1 > max_depth:
                continue

            child_id = emit_node(child)
            e = (parent_id, child_id)
            if e not in containment_edges:
                lines.append(f'  "{parent_id}" -> "{child_id}" [label="contains"];')
                containment_edges.add(e)

            walk(child, depth + 1)

        if show_data_edges and len(node.children) > 1:
            owner: dict[str, str] = {}
            for child in node.children:
                for op in child.ops:
                    owner.setdefault(op, dot_node_id(child))

            for child in node.children:
                child_id = dot_node_id(child)
                for op_id in child.ops:
                    op = tm["op_by_id"].get(op_id)
                    if not op:
                        continue
                    for inp in op_inputs(op):
                        prod = tm["value_producer"].get(inp)
                        if not prod:
                            continue
                        src_id = owner.get(prod)
                        if not src_id or src_id == child_id:
                            continue
                        e = (src_id, child_id)
                        if e not in data_edges:
                            lines.append(
                                f'  "{src_id}" -> "{child_id}" '
                                f'[style="dashed", color="#888888", label="data"];'
                            )
                            data_edges.add(e)

    walk(root, 0)
    lines.append("}")
    return "\n".join(lines) + "\n"


def render(dot_path: Path, fmt: str) -> Path | None:
    if fmt == "none":
        return None

    dot = shutil.which("dot")
    if not dot:
        print("[warn] Graphviz `dot` not found. Install with: brew install graphviz")
        return None

    out = dot_path.with_suffix(f".{fmt}")
    subprocess.run([dot, f"-T{fmt}", str(dot_path), "-o", str(out)], check=True)
    return out


# ---------------------------------------------------------------------
# Text outline
# ---------------------------------------------------------------------

def write_outline(path: Path, root: LNode) -> None:
    lines: list[str] = []

    def walk(node: LNode, depth: int) -> None:
        span = ""
        if node.min_topo < LARGE:
            span = f" [ops {node.min_topo}-{node.max_topo}]"
        lines.append("  " * depth + f"- {node.label}{span}")
        for c in node.children:
            walk(c, depth + 1)

    walk(root, 0)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export learner-friendly hierarchical dataflow region tree."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree-json", default=None)
    ap.add_argument("--tensor-ir", default=None)
    ap.add_argument("--out-dir", default="reports/learner_hierarchical_dataflow")
    ap.add_argument("--view", choices=["main", "shape", "full"], default="main")
    ap.add_argument("--include-primitives", action="store_true")
    ap.add_argument("--no-compress-single-op-wrappers", action="store_true")
    ap.add_argument("--show-data-edges", action="store_true")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--render", choices=["svg", "pdf", "png", "none"], default="pdf")
    args = ap.parse_args()

    model = args.model
    safe = safe_model_name(model)

    tree_path = Path(args.tree_json or f"reports/structural_region_trees/{safe}.json")
    tensor_path = Path(args.tensor_ir or f"reports/tensor_ir/{safe}.json")

    if not tree_path.exists():
        raise FileNotFoundError(
            f"Missing Structural Region Tree: {tree_path}\n"
            f"Run: python scripts/build_structural_region_tree.py --model {model}"
        )

    if not tensor_path.exists():
        raise FileNotFoundError(
            f"Missing Tensor IR: {tensor_path}\n"
            f"Run: python scripts/build_tensor_ir.py --model {model}"
        )

    tree = load_json(tree_path)
    tensor_ir = maybe_load_json(tensor_path)

    tm = build_tensor_maps(tensor_ir)
    region_by_id, children_by_parent = build_region_maps(tree)
    leaf_ops = compute_region_leaf_ops(region_by_id, children_by_parent)

    learner_root = build_learner_tree(
        model,
        region_by_id,
        children_by_parent,
        leaf_ops,
        tm,
        view=args.view,
        include_primitives=args.include_primitives,
        compress_single_op_wrappers=not args.no_compress_single_op_wrappers,
    )

    out_dir = Path(args.out_dir) / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.view
    if args.include_primitives:
        suffix += "_with_leaves"
    else:
        suffix += "_abstract"
    if args.show_data_edges:
        suffix += "_data_edges"

    dot_path = out_dir / f"learner_hierarchical_dataflow_{suffix}.dot"
    outline_path = out_dir / f"learner_hierarchical_dataflow_{suffix}.outline.txt"

    title = f"{model} :: learner hierarchical dataflow tree [{args.view}]"

    dot_path.write_text(
        emit_dot(
            learner_root,
            title=title,
            max_depth=args.max_depth,
            show_data_edges=args.show_data_edges,
            tm=tm,
        )
    )
    write_outline(outline_path, learner_root)

    rendered = render(dot_path, args.render)

    print(f"[learner-dataflow] model={model}")
    print(f"[learner-dataflow] view={args.view}")
    print(f"[learner-dataflow] sections={len(learner_root.children)}")
    print(f"[learner-dataflow] dot={dot_path}")
    print(f"[learner-dataflow] outline={outline_path}")
    if rendered:
        print(f"[learner-dataflow] rendered={rendered}")

    print("[learner-dataflow] default main view hides auxiliary shape/mask flow")
    print("[learner-dataflow] single-op wrappers are compressed unless --no-compress-single-op-wrappers is used")


if __name__ == "__main__":
    main()
