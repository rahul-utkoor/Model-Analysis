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

LARGE = 1_000_000


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


def tensor_ops(tensor_ir: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not tensor_ir:
        return []
    return tensor_ir.get("ops") or tensor_ir.get("operations") or tensor_ir.get("tensor_ops") or []


def op_id_of(op: dict[str, Any]) -> str | None:
    return op.get("op_id") or op.get("id") or op.get("name")


def op_kind(op: dict[str, Any]) -> str:
    return str(op.get("canonical_op_type") or op.get("op_type") or op.get("type") or "").lower()


def op_inputs(op: dict[str, Any]) -> list[str]:
    return [str(v) for v in (op.get("inputs") or op.get("input_values") or op.get("input_ids") or [])]


def op_outputs(op: dict[str, Any]) -> list[str]:
    return [str(v) for v in (op.get("outputs") or op.get("output_values") or op.get("output_ids") or [])]


def exact_onnx_name(op: dict[str, Any], fallback: str) -> str:
    """
    Leaf-node label.

    Prefer the exact ONNX-style source node name because this should match
    Netron's node property panel.
    """
    return str(
        op.get("source_node_name")
        or op.get("onnx_node_name")
        or op.get("name")
        or op.get("label")
        or fallback
    )


def compact_text_for_detection(s: Any) -> str:
    return str(s or "").replace("/", ".").replace("_", ".").lower()


def detect_layer(text: str) -> str | None:
    pats = [
        r"encoder[._/]layer[._/](\d+)",
        r"encoder_layer_(\d+)",
        r"layer[._/](\d+)",
        r"layer_(\d+)",
    ]
    for p in pats:
        m = re.search(p, text, re.I)
        if m:
            return f"Layer {m.group(1)}"
    return None


def build_tensor_maps(tensor_ir: dict[str, Any] | None) -> dict[str, Any]:
    ops = tensor_ops(tensor_ir)
    op_by_id = {}
    op_order = {}
    value_producer = {}
    graph_inputs = set()
    initializers = set()

    for i, op in enumerate(ops):
        oid = op_id_of(op)
        if not oid:
            continue
        op_by_id[oid] = op
        op_order[oid] = i
        for out in op_outputs(op):
            value_producer[out] = oid

    for op in ops:
        for v in op_inputs(op):
            low = v.lower()
            if any(x in low for x in ["input_ids", "token_type_ids", "attention_mask", "pixel_values", "input_features"]):
                graph_inputs.add(v)

    for v in graph_inputs:
        initializers.discard(v)

    return {
        "ops": ops,
        "op_by_id": op_by_id,
        "op_order": op_order,
        "value_producer": value_producer,
        "graph_inputs": graph_inputs,
        "initializers": initializers,
    }


def compute_op_distance(tm: dict[str, Any]) -> dict[str, int]:
    value_dist: dict[str, int] = {}
    for v in tm["graph_inputs"]:
        value_dist[v] = 0

    op_dist = {}

    for oid, op in sorted(tm["op_by_id"].items(), key=lambda kv: tm["op_order"].get(kv[0], LARGE)):
        dists = []
        for v in op_inputs(op):
            if v in value_dist:
                dists.append(value_dist[v])
            elif any(x in v.lower() for x in ["input_ids", "token_type_ids", "attention_mask", "pixel_values", "input_features"]):
                dists.append(0)

        usable = [d for d in dists if d < LARGE]
        dist = min(usable) + 1 if usable else LARGE
        op_dist[oid] = dist

        for out in op_outputs(op):
            value_dist[out] = min(value_dist.get(out, LARGE), dist)

    return op_dist


def build_region_maps(tree: dict[str, Any]):
    region_by_id = {}
    children_by_parent = defaultdict(list)

    for r in tree.get("regions", []):
        rid = r.get("region_id")
        if not rid:
            continue
        region_by_id[rid] = r
        children_by_parent[r.get("parent")].append(rid)

    return region_by_id, children_by_parent


def region_text(region: dict[str, Any], tm: dict[str, Any]) -> str:
    parts = [region.get("region_type", ""), str(region.get("metadata") or {})]
    for oid in region.get("op_ids") or []:
        op = tm["op_by_id"].get(oid, {})
        parts.append(exact_onnx_name(op, oid))
        parts.append(op_kind(op))
        parts.extend(op_inputs(op))
        parts.extend(op_outputs(op))
    return compact_text_for_detection(" ".join(map(str, parts)))


def region_raw_order(region: dict[str, Any], tm: dict[str, Any]) -> int:
    md = region.get("metadata") or {}
    for k in ["order", "region_order", "source_order", "topological_order", "first_op_index", "min_op_index"]:
        v = md.get(k)
        if isinstance(v, int):
            return v

    op_ids = region.get("op_ids") or []
    if op_ids:
        return min(tm["op_order"].get(op, LARGE) for op in op_ids)

    nums = re.findall(r"\d+", region.get("region_id", ""))
    return int(nums[-1]) if nums else LARGE


def region_input_distance(region: dict[str, Any], op_dist: dict[str, int]) -> int:
    op_ids = region.get("op_ids") or []
    if not op_ids:
        return LARGE
    return min(op_dist.get(op, LARGE) for op in op_ids)


def consumes_graph_input(region: dict[str, Any], tm: dict[str, Any]) -> bool:
    graph_inputs = tm["graph_inputs"]
    for oid in region.get("op_ids") or []:
        op = tm["op_by_id"].get(oid, {})
        for v in op_inputs(op):
            low = v.lower()
            if v in graph_inputs:
                return True
            if any(x in low for x in ["input_ids", "token_type_ids", "attention_mask", "pixel_values", "input_features"]):
                return True
    return False


def is_embedding_path(region: dict[str, Any], tm: dict[str, Any]) -> bool:
    t = region_text(region, tm)
    return any(x in t for x in [
        "word.embeddings",
        "token.type.embeddings",
        "position.embeddings",
        "input.ids",
        "token.type.ids",
        "embedding",
        "embeddings",
    ])


def is_shape_helper(region: dict[str, Any], tm: dict[str, Any]) -> bool:
    rt = region.get("region_type", "")
    t = region_text(region, tm)

    if rt == "AxisTransformRegion":
        return True

    if is_embedding_path(region, tm):
        return False

    return any(x in t for x in [
        "shape",
        "reshape",
        "transpose",
        "slice",
        "unsqueeze",
        "squeeze",
        "concat",
        "range",
        "cast",
        "constantofshape",
        "where",
        "equal",
        "greaterorequal",
        "axis",
    ])


def semantic_rank(region: dict[str, Any], tm: dict[str, Any]) -> int:
    rt = region.get("region_type", "")

    if consumes_graph_input(region, tm) and not is_shape_helper(region, tm):
        return 0

    if is_embedding_path(region, tm) and not is_shape_helper(region, tm):
        return 1

    if rt in {
        "LayerNormRegion",
        "ResidualMergeRegion",
        "AttentionSkeletonRegion",
        "FeedForwardRegion",
        "LinearProjectionRegion",
        "ActivationRegion",
    }:
        return 2

    if rt == "PrimitiveRegion":
        t = region_text(region, tm)
        if any(x in t for x in ["gather", "matmul", "add", "layernormalization", "softmax", "erf", "mul"]):
            return 2

    if rt == "ModelRegion":
        return 3

    if is_shape_helper(region, tm):
        return 4

    return 5


def order_key(region: dict[str, Any], tm: dict[str, Any], op_dist: dict[str, int], mode: str):
    raw = region_raw_order(region, tm)
    dist = region_input_distance(region, op_dist)
    rank = semantic_rank(region, tm)
    rid = region.get("region_id", "")

    if mode == "raw":
        return (raw, rid)
    if mode == "hybrid":
        return (rank, raw, dist, rid)
    return (rank, dist, raw, rid)


def ordered_children(region, region_by_id, children_by_parent, tm, op_dist, mode):
    explicit = region.get("children")
    if isinstance(explicit, list):
        kids = [c for c in explicit if c in region_by_id]
    else:
        kids = [c for c in children_by_parent.get(region.get("region_id"), []) if c in region_by_id]
    return sorted(kids, key=lambda cid: order_key(region_by_id[cid], tm, op_dist, mode))


def abstract_title(region: dict[str, Any], tm: dict[str, Any]) -> str:
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
        if "intermediate" in t:
            return prefix + "Intermediate Projection"
        if "attention.output" in t:
            return prefix + "Attention Output Projection"
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

    if rt == "ForkRegion":
        return prefix + "Fork"

    if rt == "JoinRegion":
        return prefix + "Join"

    if rt == "PrimitiveRegion":
        op_ids = region.get("op_ids") or []
        if op_ids:
            op = tm["op_by_id"].get(op_ids[0], {})
            return exact_onnx_name(op, op_ids[0])
        return "Primitive"

    return rt.replace("Region", "")


def node_label(region: dict[str, Any], tm: dict[str, Any], label_mode: str) -> str:
    """
    clean:
      abstract nodes: short semantic name
      primitive leaves: exact ONNX node name

    minimal:
      abstract nodes: even shorter names
      primitive leaves: exact ONNX node name

    debug:
      clean name + tiny region type
    """
    title = abstract_title(region, tm)

    if label_mode == "debug":
        rt = region.get("region_type", "")
        raw = region_raw_order(region, tm)
        return f"{title}\\n{rt}\\nraw={raw}"

    return title


def fill_color(region: dict[str, Any], tm: dict[str, Any]) -> str:
    rt = region.get("region_type", "")
    if rt == "ModelRegion":
        return "#eeeeee"
    if rt == "PrimitiveRegion":
        return "#ffffff"
    if rt == "FeedForwardRegion":
        return "#c9f7d5"
    if rt == "AttentionSkeletonRegion":
        return "#d6e4ff"
    if rt == "ResidualMergeRegion":
        return "#ffd6d6"
    if rt == "LayerNormRegion":
        return "#ffe2bf"
    if rt == "LinearProjectionRegion":
        return "#d9f7be"
    if rt == "ActivationRegion":
        return "#eadcff"
    if rt == "AxisTransformRegion":
        return "#fff3bf"
    return "#eeeeee"


def border_color(region: dict[str, Any], tm: dict[str, Any]) -> str:
    if is_shape_helper(region, tm):
        return "#c48a00"
    rt = region.get("region_type", "")
    return {
        "ModelRegion": "#333333",
        "PrimitiveRegion": "#555555",
        "FeedForwardRegion": "#237a3b",
        "AttentionSkeletonRegion": "#1d4ed8",
        "ResidualMergeRegion": "#b91c1c",
        "LayerNormRegion": "#c2410c",
        "LinearProjectionRegion": "#15803d",
        "ActivationRegion": "#7e22ce",
    }.get(rt, "#555555")


def shape(region: dict[str, Any]) -> str:
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


def emit_dot(
    *,
    model: str,
    root_id: str,
    region_by_id,
    children_by_parent,
    tm,
    op_dist,
    order: str,
    max_depth: int,
    include_primitives: bool,
    label_mode: str,
    title: str,
):
    lines = [
        "digraph CleanLearnerGraph {",
        "  graph [rankdir=LR, bgcolor=\"white\", fontname=\"Helvetica\", labelloc=\"t\", labeljust=\"l\", nodesep=0.35, ranksep=0.65];",
        "  node [fontname=\"Helvetica\", fontsize=10, style=\"filled,rounded\", margin=0.08];",
        "  edge [fontname=\"Helvetica\", fontsize=8, color=\"#444444\", arrowsize=0.7];",
        f"  label=\"{dot_escape(title)}\";",
        "",
    ]

    seen = set()

    def walk(rid: str, depth: int):
        if rid in seen or depth > max_depth:
            return

        region = region_by_id.get(rid)
        if not region:
            return

        if region.get("region_type") == "PrimitiveRegion" and not include_primitives:
            return

        seen.add(rid)

        nid = safe_file_name(rid)
        label = node_label(region, tm, label_mode)

        lines.append(
            f'  "{nid}" [label="{dot_escape(label)}", '
            f'shape="{shape(region)}", fillcolor="{fill_color(region, tm)}", '
            f'color="{border_color(region, tm)}", penwidth=1.8];'
        )

        kids = ordered_children(region, region_by_id, children_by_parent, tm, op_dist, order)
        for cid in kids:
            child = region_by_id[cid]
            if child.get("region_type") == "PrimitiveRegion" and not include_primitives:
                continue
            walk(cid, depth + 1)
            if cid in seen:
                lines.append(f'  "{nid}" -> "{safe_file_name(cid)}";')

    walk(root_id, 0)
    lines.append("}")
    return "\n".join(lines) + "\n"


def render(dot_path: Path, fmt: str):
    if fmt == "none":
        return None
    dot = shutil.which("dot")
    if not dot:
        print("[warn] Graphviz `dot` not found. Install with: brew install graphviz")
        return None
    out = dot_path.with_suffix(f".{fmt}")
    subprocess.run([dot, f"-T{fmt}", str(dot_path), "-o", str(out)], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree-json", default=None)
    ap.add_argument("--tensor-ir", default=None)
    ap.add_argument("--out-dir", default="reports/clean_learner_graphviz")
    ap.add_argument("--order", choices=["semantic", "raw", "hybrid"], default="semantic")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--include-primitives", action="store_true")
    ap.add_argument("--label-mode", choices=["clean", "minimal", "debug"], default="clean")
    ap.add_argument("--render", choices=["svg", "pdf", "png", "none"], default="svg")
    args = ap.parse_args()

    model = args.model
    safe = safe_model_name(model)

    tree_path = Path(args.tree_json or f"reports/structural_region_trees/{safe}.json")
    tensor_path = Path(args.tensor_ir or f"reports/tensor_ir/{safe}.json")

    if not tree_path.exists():
        raise FileNotFoundError(f"Missing structural region tree: {tree_path}")

    tree = load_json(tree_path)
    tensor_ir = load_json_if_exists(tensor_path)

    tm = build_tensor_maps(tensor_ir)
    op_dist = compute_op_distance(tm)
    region_by_id, children_by_parent = build_region_maps(tree)

    root_id = tree.get("root_region_id")
    if not root_id:
        roots = children_by_parent.get(None) or []
        if not roots:
            raise RuntimeError("Could not find root region.")
        root_id = roots[0]

    out_dir = Path(args.out_dir) / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    dot_text = emit_dot(
        model=model,
        root_id=root_id,
        region_by_id=region_by_id,
        children_by_parent=children_by_parent,
        tm=tm,
        op_dist=op_dist,
        order=args.order,
        max_depth=args.max_depth,
        include_primitives=args.include_primitives,
        label_mode=args.label_mode,
        title=f"{model} clean learner dataflow tree ({args.order})",
    )

    suffix = "with_leaves" if args.include_primitives else "abstract"
    dot_path = out_dir / f"full_model_{args.order}_{suffix}_{args.label_mode}.dot"
    dot_path.write_text(dot_text)

    rendered = render(dot_path, args.render)

    print(f"[clean-graphviz] model={model}")
    print(f"[clean-graphviz] dot={dot_path}")
    if rendered:
        print(f"[clean-graphviz] rendered={rendered}")
    print("[clean-graphviz] leaf labels use exact ONNX/source node names when available")


if __name__ == "__main__":
    main()
