#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

LARGE = 10**12


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def safe_file_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(name))
    return s.strip("_") or "unknown"


def dot_escape(x: Any) -> str:
    s = str(x if x is not None else "")
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def first_int(s: Any) -> int | None:
    nums = re.findall(r"\d+", str(s))
    if not nums:
        return None
    return int(nums[-1])


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


def op_inputs(op: dict[str, Any]) -> list[str]:
    vals = (
        op.get("inputs")
        or op.get("input_values")
        or op.get("input_ids")
        or []
    )
    return [str(v) for v in vals]


def op_outputs(op: dict[str, Any]) -> list[str]:
    vals = (
        op.get("outputs")
        or op.get("output_values")
        or op.get("output_ids")
        or []
    )
    return [str(v) for v in vals]


def op_kind(op: dict[str, Any]) -> str:
    return str(
        op.get("canonical_op_type")
        or op.get("op_type")
        or op.get("type")
        or ""
    )


def exact_onnx_name(op: dict[str, Any], fallback: str) -> str:
    """
    Leaf label.

    Prefer the exact ONNX-style node name, matching Netron's node property panel.
    """
    return str(
        op.get("source_node_name")
        or op.get("onnx_node_name")
        or op.get("name")
        or op.get("label")
        or fallback
    )


def build_tensor_maps(tensor_ir: dict[str, Any] | None) -> dict[str, Any]:
    ops = tensor_ops(tensor_ir)

    op_by_id: dict[str, dict[str, Any]] = {}
    op_order: dict[str, int] = {}
    value_producer: dict[str, str] = {}
    value_consumers: dict[str, list[str]] = defaultdict(list)

    for idx, op in enumerate(ops):
        oid = op_id_of(op)
        if not oid:
            continue

        op_by_id[oid] = op
        op_order[oid] = idx

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
    n = first_int(op_id)
    return n if n is not None else LARGE


# ---------------------------------------------------------------------
# Region-tree helpers
# ---------------------------------------------------------------------

def build_region_maps(tree: dict[str, Any]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str | None, list[str]],
]:
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
    region: dict[str, Any],
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
) -> list[str]:
    explicit = region.get("children")
    if isinstance(explicit, list):
        return [c for c in explicit if c in region_by_id]
    return [c for c in children_by_parent.get(region.get("region_id"), []) if c in region_by_id]


def own_region_ops(region: dict[str, Any]) -> set[str]:
    return set(str(x) for x in (region.get("op_ids") or []))


def compute_region_leaf_ops(
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
) -> dict[str, set[str]]:
    """
    leaf_ops(region) = own op_ids U leaf_ops(children).

    This is the key invariant used for ordering.
    """
    memo: dict[str, set[str]] = {}

    def visit(rid: str) -> set[str]:
        if rid in memo:
            return memo[rid]

        region = region_by_id[rid]
        leaves = set(own_region_ops(region))

        for cid in region_children(region, region_by_id, children_by_parent):
            leaves |= visit(cid)

        memo[rid] = leaves
        return leaves

    for rid in region_by_id:
        visit(rid)

    return memo


def region_min_topo(rid: str, leaf_ops: dict[str, set[str]], tm: dict[str, Any]) -> int:
    ops = leaf_ops.get(rid, set())
    if ops:
        return min(raw_op_order(op, tm) for op in ops)

    n = first_int(rid)
    return n if n is not None else LARGE


def region_max_topo(rid: str, leaf_ops: dict[str, set[str]], tm: dict[str, Any]) -> int:
    ops = leaf_ops.get(rid, set())
    if ops:
        return max(raw_op_order(op, tm) for op in ops)

    n = first_int(rid)
    return n if n is not None else LARGE


# ---------------------------------------------------------------------
# Display naming
# ---------------------------------------------------------------------

def compact_text(s: Any) -> str:
    return str(s or "").replace("/", ".").replace("_", ".").lower()


def region_text(region: dict[str, Any], tm: dict[str, Any]) -> str:
    parts = [region.get("region_type", ""), str(region.get("metadata") or {})]

    for oid in region.get("op_ids") or []:
        op = tm["op_by_id"].get(oid, {})
        parts.append(exact_onnx_name(op, oid))
        parts.append(op_kind(op))
        parts.extend(op_inputs(op))
        parts.extend(op_outputs(op))

    return compact_text(" ".join(map(str, parts)))


def detect_layer(text: str) -> str | None:
    patterns = [
        r"encoder[._/]layer[._/](\d+)",
        r"encoder.layer.(\d+)",
        r"encoder_layer_(\d+)",
        r"layer[._/](\d+)",
        r"layer_(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return f"Layer {m.group(1)}"
    return None


def abstract_region_title(region: dict[str, Any], tm: dict[str, Any]) -> str:
    rt = region.get("region_type", "")
    t = region_text(region, tm)
    layer = detect_layer(t)
    prefix = f"{layer} " if layer else ""

    if rt == "ModelRegion":
        return "Model"

    if rt == "FeedForwardRegion":
        return prefix + "Feed Forward"

    if rt == "AttentionSkeletonRegion":
        return prefix + "Attention"

    if rt == "ResidualMergeRegion":
        if "embedding" in t:
            return "Embedding Add"
        if "attention" in t:
            return prefix + "Attention Add"
        if "output" in t:
            return prefix + "Output Add"
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
        return prefix + "Linear Projection"

    if rt == "ActivationRegion":
        if "gelu" in t or "erf" in t:
            return prefix + "GELU"
        return prefix + "Activation"

    if rt == "AxisTransformRegion":
        if "attention.mask" in t:
            return "Attention Mask Shape"
        if "position" in t:
            return "Position Shape"
        return "Shape Transform"

    if rt == "PrimitiveRegion":
        op_ids = list(region.get("op_ids") or [])
        if op_ids:
            op = tm["op_by_id"].get(op_ids[0], {})
            return exact_onnx_name(op, op_ids[0])
        return "Primitive"

    if rt == "ForkRegion":
        return prefix + "Fork"

    if rt == "JoinRegion":
        return prefix + "Join"

    return rt.replace("Region", "") or "Region"


def primitive_label(op_id: str, tm: dict[str, Any]) -> str:
    op = tm["op_by_id"].get(op_id, {})
    return exact_onnx_name(op, op_id)


def region_color(region: dict[str, Any]) -> tuple[str, str]:
    rt = region.get("region_type", "")

    table = {
        "ModelRegion": ("#eeeeee", "#333333"),
        "FeedForwardRegion": ("#c9f7d5", "#15803d"),
        "AttentionSkeletonRegion": ("#d6e4ff", "#1d4ed8"),
        "ResidualMergeRegion": ("#ffd6d6", "#b91c1c"),
        "LayerNormRegion": ("#ffe2bf", "#c2410c"),
        "LinearProjectionRegion": ("#d9f7be", "#15803d"),
        "ActivationRegion": ("#eadcff", "#7e22ce"),
        "AxisTransformRegion": ("#fff3bf", "#c48a00"),
        "PrimitiveRegion": ("#ffffff", "#555555"),
        "ForkRegion": ("#eeeeee", "#555555"),
        "JoinRegion": ("#eeeeee", "#555555"),
    }

    return table.get(rt, ("#eeeeee", "#555555"))


def region_shape(region: dict[str, Any]) -> str:
    rt = region.get("region_type", "")
    if rt == "ModelRegion":
        return "box3d"
    if rt == "PrimitiveRegion":
        return "ellipse"
    if rt == "ResidualMergeRegion":
        return "diamond"
    if rt == "AttentionSkeletonRegion":
        return "octagon"
    if rt == "FeedForwardRegion":
        return "component"
    return "box"


# ---------------------------------------------------------------------
# Item abstraction: child region or direct primitive op
# ---------------------------------------------------------------------

def item_id_region(rid: str) -> str:
    return f"region::{rid}"


def item_id_op(op_id: str) -> str:
    return f"op::{op_id}"


def item_is_region(item: str) -> bool:
    return item.startswith("region::")


def item_region_id(item: str) -> str:
    return item[len("region::"):]


def item_op_id(item: str) -> str:
    return item[len("op::"):]


def item_leaf_ops(item: str, leaf_ops: dict[str, set[str]]) -> set[str]:
    if item_is_region(item):
        return set(leaf_ops.get(item_region_id(item), set()))
    return {item_op_id(item)}


def item_min_topo(item: str, leaf_ops: dict[str, set[str]], tm: dict[str, Any]) -> int:
    ops = item_leaf_ops(item, leaf_ops)
    if not ops:
        return LARGE
    return min(raw_op_order(op, tm) for op in ops)


def item_max_topo(item: str, leaf_ops: dict[str, set[str]], tm: dict[str, Any]) -> int:
    ops = item_leaf_ops(item, leaf_ops)
    if not ops:
        return LARGE
    return max(raw_op_order(op, tm) for op in ops)


def parent_items(
    parent_rid: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    leaf_ops: dict[str, set[str]],
    include_primitives: bool,
) -> list[str]:
    """
    Immediate tree items under a parent:
    - child regions;
    - if include_primitives, direct primitive ops owned by parent and not covered by children.
    """
    parent = region_by_id[parent_rid]

    child_rids = region_children(parent, region_by_id, children_by_parent)
    child_items = [item_id_region(cid) for cid in child_rids]

    if not include_primitives:
        return child_items

    covered_by_children: set[str] = set()
    for cid in child_rids:
        covered_by_children |= leaf_ops.get(cid, set())

    direct_ops = sorted(
        own_region_ops(parent) - covered_by_children,
        key=lambda op: raw_op_order(op, {"op_order": {}}),
    )

    return child_items + [item_id_op(op) for op in direct_ops]


# ---------------------------------------------------------------------
# Dependency-aware topological ordering among siblings
# ---------------------------------------------------------------------

def build_item_dependency_edges(
    items: list[str],
    leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
) -> dict[str, set[str]]:
    """
    Edge item A -> item B if any op in B consumes a value produced by any op in A.

    This is what fixes cases like:

      position_embeddings/Gather -> embeddings/Add_1

    even if a heuristic sorting key would otherwise place Add_1 earlier.
    """
    item_for_op: dict[str, str] = {}

    for item in items:
        for op_id in item_leaf_ops(item, leaf_ops):
            # If regions overlap, keep first assignment and avoid overclaiming.
            item_for_op.setdefault(op_id, item)

    edges: dict[str, set[str]] = {item: set() for item in items}

    for b_item in items:
        for b_op_id in item_leaf_ops(b_item, leaf_ops):
            b_op = tm["op_by_id"].get(b_op_id)
            if not b_op:
                continue

            for inp in op_inputs(b_op):
                producer = tm["value_producer"].get(inp)
                if not producer:
                    continue

                a_item = item_for_op.get(producer)
                if not a_item:
                    continue

                if a_item == b_item:
                    continue

                edges[a_item].add(b_item)

    return edges


def topo_sort_items(
    items: list[str],
    leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
) -> list[str]:
    """
    Dependency-aware sibling order.

    Primary rule:
      producer item must appear before consumer item.

    Tie-breaker:
      min primitive topo index.
    """
    if len(items) <= 1:
        return list(items)

    item_set = set(items)
    edges = build_item_dependency_edges(items, leaf_ops, tm)

    indeg = {item: 0 for item in items}
    for src, dsts in edges.items():
        for dst in dsts:
            if dst in item_set:
                indeg[dst] += 1

    ready = [x for x in items if indeg[x] == 0]
    ready.sort(key=lambda x: (item_min_topo(x, leaf_ops, tm), item_max_topo(x, leaf_ops, tm), x))

    out = []

    while ready:
        cur = ready.pop(0)
        out.append(cur)

        for dst in sorted(edges.get(cur, []), key=lambda x: (item_min_topo(x, leaf_ops, tm), x)):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)

        ready.sort(key=lambda x: (item_min_topo(x, leaf_ops, tm), item_max_topo(x, leaf_ops, tm), x))

    if len(out) != len(items):
        # Overlapping regions can create apparent cycles.
        # Keep already sorted acyclic prefix, then append remaining by raw topo span.
        remaining = [x for x in items if x not in set(out)]
        remaining.sort(key=lambda x: (item_min_topo(x, leaf_ops, tm), item_max_topo(x, leaf_ops, tm), x))
        out.extend(remaining)

    return out


# ---------------------------------------------------------------------
# DOT emission
# ---------------------------------------------------------------------

def emit_dot(
    *,
    model: str,
    root_id: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
    max_depth: int,
    include_primitives: bool,
    show_sibling_dataflow_edges: bool,
) -> str:
    lines = [
        "digraph TopologicalRegionTree {",
        "  graph [rankdir=LR, bgcolor=\"white\", fontname=\"Helvetica\", labelloc=\"t\", labeljust=\"l\", nodesep=0.35, ranksep=0.70];",
        "  node [fontname=\"Helvetica\", fontsize=10, style=\"filled,rounded\", margin=0.08];",
        "  edge [fontname=\"Helvetica\", fontsize=8, color=\"#444444\", arrowsize=0.7];",
        f"  label=\"{dot_escape(model)} :: topologically ordered hierarchical dataflow region tree\";",
        "",
    ]

    emitted_nodes: set[str] = set()
    emitted_containment_edges: set[tuple[str, str]] = set()
    emitted_dataflow_edges: set[tuple[str, str]] = set()

    def emit_region_node(rid: str) -> str:
        node_name = safe_file_name("region_" + rid)

        if node_name in emitted_nodes:
            return node_name

        region = region_by_id[rid]
        fill, border = region_color(region)

        label = abstract_region_title(region, tm)

        # Keep graph clean. Add small span only for abstract nodes if useful.
        mn = region_min_topo(rid, leaf_ops, tm)
        mx = region_max_topo(rid, leaf_ops, tm)
        if mn < LARGE and mx < LARGE and region.get("region_type") != "PrimitiveRegion":
            label = f"{label}\\nops {mn}-{mx}"

        lines.append(
            f'  "{node_name}" [label="{dot_escape(label)}", '
            f'shape="{region_shape(region)}", fillcolor="{fill}", color="{border}", penwidth=1.8];'
        )
        emitted_nodes.add(node_name)
        return node_name

    def emit_op_node(op_id: str) -> str:
        node_name = safe_file_name("op_" + op_id)

        if node_name in emitted_nodes:
            return node_name

        op = tm["op_by_id"].get(op_id, {})
        label = primitive_label(op_id, tm)

        # Exact ONNX node names can be long; keep them exact but not decorated.
        lines.append(
            f'  "{node_name}" [label="{dot_escape(label)}", '
            f'shape="ellipse", fillcolor="#ffffff", color="#555555", penwidth=1.3];'
        )
        emitted_nodes.add(node_name)
        return node_name

    def emit_item(item: str) -> str:
        if item_is_region(item):
            return emit_region_node(item_region_id(item))
        return emit_op_node(item_op_id(item))

    def walk_region(rid: str, depth: int) -> None:
        if depth > max_depth:
            return

        parent_node = emit_region_node(rid)

        items = parent_items(
            rid,
            region_by_id,
            children_by_parent,
            leaf_ops,
            include_primitives=include_primitives,
        )

        ordered = topo_sort_items(items, leaf_ops, tm)

        # Containment edges: tree structure.
        for item in ordered:
            if item_is_region(item):
                cid = item_region_id(item)
                if depth + 1 > max_depth:
                    continue
                child_node = emit_region_node(cid)
                edge = (parent_node, child_node)
                if edge not in emitted_containment_edges:
                    lines.append(f'  "{parent_node}" -> "{child_node}" [label="contains"];')
                    emitted_containment_edges.add(edge)
                walk_region(cid, depth + 1)
            else:
                if not include_primitives:
                    continue
                op_id = item_op_id(item)
                child_node = emit_op_node(op_id)
                edge = (parent_node, child_node)
                if edge not in emitted_containment_edges:
                    lines.append(f'  "{parent_node}" -> "{child_node}" [label="contains"];')
                    emitted_containment_edges.add(edge)

        # Optional sibling dataflow edges. These are dashed because they are not containment.
        if show_sibling_dataflow_edges and len(ordered) > 1:
            deps = build_item_dependency_edges(ordered, leaf_ops, tm)
            for src_item, dst_items in deps.items():
                if src_item not in ordered:
                    continue
                src_node = emit_item(src_item)
                for dst_item in dst_items:
                    if dst_item not in ordered:
                        continue
                    dst_node = emit_item(dst_item)
                    edge = (src_node, dst_node)
                    if edge in emitted_dataflow_edges:
                        continue
                    lines.append(
                        f'  "{src_node}" -> "{dst_node}" '
                        f'[style="dashed", color="#888888", label="data"];'
                    )
                    emitted_dataflow_edges.add(edge)

    walk_region(root_id, 0)

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_dot(dot_path: Path, fmt: str) -> Path | None:
    if fmt == "none":
        return None

    dot_bin = shutil.which("dot")
    if not dot_bin:
        print("[warn] Graphviz `dot` not found. Install with: brew install graphviz")
        return None

    out = dot_path.with_suffix(f".{fmt}")
    subprocess.run([dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out)], check=True)
    return out


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def write_order_report(
    path: Path,
    root_id: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    leaf_ops: dict[str, set[str]],
    tm: dict[str, Any],
    include_primitives: bool,
) -> None:
    """
    Writes a text report showing the actual sibling ordering chosen.
    Useful for checking local cases such as:
      position_embeddings/Gather before embeddings/Add_1.
    """
    lines = []

    def item_label(item: str) -> str:
        if item_is_region(item):
            rid = item_region_id(item)
            return abstract_region_title(region_by_id[rid], tm)
        return primitive_label(item_op_id(item), tm)

    def walk(rid: str, indent: int) -> None:
        region = region_by_id[rid]
        lines.append(
            "  " * indent
            + f"- {abstract_region_title(region, tm)} "
            + f"[{rid}, ops {region_min_topo(rid, leaf_ops, tm)}-{region_max_topo(rid, leaf_ops, tm)}]"
        )

        items = parent_items(rid, region_by_id, children_by_parent, leaf_ops, include_primitives)
        ordered = topo_sort_items(items, leaf_ops, tm)

        for item in ordered:
            if item_is_region(item):
                walk(item_region_id(item), indent + 1)
            elif include_primitives:
                op = item_op_id(item)
                lines.append(
                    "  " * (indent + 1)
                    + f"- {primitive_label(op, tm)} [op {raw_op_order(op, tm)}]"
                )

    walk(root_id, 0)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export a dependency-aware topologically ordered hierarchical dataflow region tree."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree-json", default=None)
    ap.add_argument("--tensor-ir", default=None)
    ap.add_argument("--out-dir", default="reports/topological_region_tree_graphviz")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--include-primitives", action="store_true")
    ap.add_argument("--show-sibling-dataflow-edges", action="store_true")
    ap.add_argument("--render", choices=["svg", "pdf", "png", "none"], default="svg")
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

    tree = load_json(tree_path)
    tensor_ir = load_json_if_exists(tensor_path)

    tm = build_tensor_maps(tensor_ir)
    region_by_id, children_by_parent = build_region_maps(tree)
    leaf_ops = compute_region_leaf_ops(region_by_id, children_by_parent)

    root_id = tree.get("root_region_id")
    if not root_id:
        roots = children_by_parent.get(None) or []
        if not roots:
            raise RuntimeError("Could not find root region.")
        root_id = roots[0]

    out_dir = Path(args.out_dir) / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "with_leaves" if args.include_primitives else "abstract"
    if args.show_sibling_dataflow_edges:
        suffix += "_with_data_edges"

    dot_text = emit_dot(
        model=model,
        root_id=root_id,
        region_by_id=region_by_id,
        children_by_parent=children_by_parent,
        leaf_ops=leaf_ops,
        tm=tm,
        max_depth=args.max_depth,
        include_primitives=args.include_primitives,
        show_sibling_dataflow_edges=args.show_sibling_dataflow_edges,
    )

    dot_path = out_dir / f"topological_region_tree_{suffix}.dot"
    dot_path.write_text(dot_text)

    rendered = render_dot(dot_path, args.render)

    report_path = out_dir / f"topological_region_tree_{suffix}.order.txt"
    write_order_report(
        report_path,
        root_id,
        region_by_id,
        children_by_parent,
        leaf_ops,
        tm,
        include_primitives=args.include_primitives,
    )

    print(f"[topological-region-tree] model={model}")
    print(f"[topological-region-tree] regions={len(region_by_id)}")
    print(f"[topological-region-tree] dot={dot_path}")
    print(f"[topological-region-tree] order_report={report_path}")
    if rendered:
        print(f"[topological-region-tree] rendered={rendered}")
    print("[topological-region-tree] ordering rule: sibling data dependencies first, raw op order tie-breaker")
    print("[topological-region-tree] primitive leaves use exact ONNX/source node names when available")


if __name__ == "__main__":
    main()
