#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict, Counter, deque
from pathlib import Path
from typing import Any


LARGE = 1_000_000


def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def safe_file_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(name))
    return name.strip("_") or "unknown"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def dot_escape(s: Any) -> str:
    s = str(s if s is not None else "")
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return s


def first_numeric_index(s: Any) -> int | None:
    nums = re.findall(r"\d+", str(s))
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def clean_name(name: Any) -> str:
    s = str(name or "")

    replacements = [
        ("/model/bert/", ""),
        ("model_bert_", ""),
        ("bert.encoder.layer.", "layer"),
        ("bert_encoder_layer_", "layer"),
        ("encoder.layer.", "layer"),
        ("encoder_layer_", "layer"),
        ("attention_self_", "attention.self."),
        ("attention.output.", "attention.output."),
        ("attention_output_", "attention.output."),
        ("intermediate_dense", "intermediate.dense"),
        ("output_dense", "output.dense"),
        ("LayerNorm_LayerNormalization", "LayerNorm"),
        ("embeddings_word_embeddings", "word_embeddings"),
        ("embeddings_token_type_embeddings", "token_type_embeddings"),
        ("embeddings_position_embeddings", "position_embeddings"),
    ]

    for a, b in replacements:
        s = s.replace(a, b)

    s = s.replace("/", ".")
    s = re.sub(r"\.+", ".", s)
    s = s.strip(".")
    return s


def tensor_ops(tensor_ir: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not tensor_ir:
        return []
    return (
        tensor_ir.get("ops")
        or tensor_ir.get("operations")
        or tensor_ir.get("tensor_ops")
        or []
    )


def tensor_values(tensor_ir: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not tensor_ir:
        return []
    return (
        tensor_ir.get("values")
        or tensor_ir.get("tensor_values")
        or []
    )


def op_id_of(op: dict[str, Any]) -> str | None:
    return op.get("op_id") or op.get("id") or op.get("name")


def op_inputs(op: dict[str, Any]) -> list[str]:
    vals = op.get("inputs") or op.get("input_values") or op.get("input_ids") or []
    return [str(v) for v in vals]


def op_outputs(op: dict[str, Any]) -> list[str]:
    vals = op.get("outputs") or op.get("output_values") or op.get("output_ids") or []
    return [str(v) for v in vals]


def op_name(op: dict[str, Any], fallback: str = "") -> str:
    return clean_name(
        op.get("source_node_name")
        or op.get("name")
        or op.get("label")
        or op.get("op_name")
        or fallback
    )


def op_kind(op: dict[str, Any]) -> str:
    return str(
        op.get("canonical_op_type")
        or op.get("op_type")
        or op.get("type")
        or ""
    ).lower()


def build_tensor_maps(tensor_ir: dict[str, Any] | None) -> dict[str, Any]:
    ops = tensor_ops(tensor_ir)
    vals = tensor_values(tensor_ir)

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

        for v in op_outputs(op):
            value_producer[v] = oid

        for v in op_inputs(op):
            value_consumers[v].append(oid)

    graph_inputs = set()
    initializers = set()

    for v in vals:
        vid = str(v.get("value_id") or v.get("id") or v.get("name") or "")
        if not vid:
            continue
        role = str(v.get("role") or v.get("kind") or "").lower()
        if v.get("is_graph_input") or role in {"graph_input", "input"}:
            graph_inputs.add(vid)
        if v.get("is_initializer") or role in {"initializer", "parameter", "constant"}:
            initializers.add(vid)

    # Fallback by name.
    for vid in list(value_consumers.keys()) + list(value_producer.keys()):
        low = vid.lower()
        if any(x in low for x in ["input_ids", "token_type_ids", "attention_mask", "pixel_values", "input_features"]):
            graph_inputs.add(vid)

    return {
        "ops": ops,
        "op_by_id": op_by_id,
        "op_order": op_order,
        "value_producer": value_producer,
        "value_consumers": dict(value_consumers),
        "graph_inputs": graph_inputs,
        "initializers": initializers,
    }


def compute_op_distance_from_graph_inputs(tm: dict[str, Any]) -> dict[str, int]:
    op_by_id = tm["op_by_id"]
    op_order = tm["op_order"]
    graph_inputs = tm["graph_inputs"]
    initializers = tm["initializers"]

    value_dist: dict[str, int] = {}
    for v in graph_inputs:
        value_dist[v] = 0
    for v in initializers:
        value_dist[v] = LARGE

    op_dist: dict[str, int] = {}

    for oid, op in sorted(op_by_id.items(), key=lambda kv: op_order.get(kv[0], LARGE)):
        in_dists = []
        for v in op_inputs(op):
            if v in graph_inputs:
                in_dists.append(0)
            elif v in value_dist:
                in_dists.append(value_dist[v])
            elif v in initializers:
                in_dists.append(LARGE)

        usable = [d for d in in_dists if d < LARGE]
        dist = min(usable) + 1 if usable else LARGE
        op_dist[oid] = dist

        for out in op_outputs(op):
            value_dist[out] = min(value_dist.get(out, LARGE), dist)

    return op_dist


def build_region_maps(tree: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str | None, list[str]], dict[str, dict[str, Any]]]:
    regions = tree.get("regions", [])
    interfaces = tree.get("interfaces", [])

    region_by_id = {r.get("region_id"): r for r in regions if r.get("region_id")}
    children_by_parent: dict[str | None, list[str]] = defaultdict(list)

    for r in regions:
        rid = r.get("region_id")
        children_by_parent[r.get("parent")].append(rid)

    interface_by_region = {i.get("region_id"): i for i in interfaces if i.get("region_id")}
    return region_by_id, children_by_parent, interface_by_region


def build_dims_by_region(region_dim_ir: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not region_dim_ir:
        return out
    for d in region_dim_ir.get("dimension_variables", []):
        rid = d.get("region_id")
        if rid:
            out[rid].append(d)
    return out


def region_raw_order(region: dict[str, Any], tm: dict[str, Any]) -> int:
    metadata = region.get("metadata") or {}
    for key in ["order", "region_order", "source_order", "topological_order", "first_op_index", "min_op_index"]:
        v = metadata.get(key)
        if isinstance(v, int):
            return v

    op_ids = region.get("op_ids") or []
    if op_ids:
        return min(tm["op_order"].get(op, first_numeric_index(op) or LARGE) for op in op_ids)

    n = first_numeric_index(region.get("region_id", ""))
    return n if n is not None else LARGE


def region_source_text(region: dict[str, Any], tm: dict[str, Any]) -> str:
    texts = []
    for oid in region.get("op_ids") or []:
        op = tm["op_by_id"].get(oid, {})
        texts.append(op_name(op, oid))
        texts.extend(op_inputs(op))
        texts.extend(op_outputs(op))
        texts.append(op_kind(op))
    texts.append(region.get("region_type", ""))
    texts.append(str(region.get("metadata") or {}))
    return " ".join(map(str, texts)).lower()


def region_graph_input_distance(region: dict[str, Any], op_dist: dict[str, int]) -> int:
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


def is_embedding_signal(region: dict[str, Any], tm: dict[str, Any]) -> bool:
    text = region_source_text(region, tm)
    return any(x in text for x in [
        "embedding",
        "embeddings",
        "word_embeddings",
        "token_type_embeddings",
        "position_embeddings",
        "input_ids",
        "token_type_ids",
    ])


def is_shape_helper(region: dict[str, Any], tm: dict[str, Any], dims: list[dict[str, Any]]) -> bool:
    rt = region.get("region_type")
    text = region_source_text(region, tm)

    if rt == "AxisTransformRegion":
        return True

    shape_words = [
        "shape", "reshape", "transpose", "slice", "unsqueeze", "squeeze",
        "concat", "range", "cast", "constantofshape", "where", "equal",
        "greaterorequal", "less", "mask", "axis"
    ]

    if any(w in text for w in shape_words):
        # Do not mark embedding Gather as shape helper.
        if is_embedding_signal(region, tm):
            return False
        return True

    for d in dims:
        axis = str(d.get("axis_role") or d.get("dim_name") or "").lower()
        if "symbolic_axis" in axis or "shape" in axis or "axis" in axis:
            return True

    return False


def region_semantic_class(
    region: dict[str, Any],
    tm: dict[str, Any],
    dims_by_region: dict[str, list[dict[str, Any]]],
    op_dist: dict[str, int],
) -> tuple[int, str, str]:
    rid = region.get("region_id")
    rt = region.get("region_type", "")
    dims = dims_by_region.get(rid, [])
    text = region_source_text(region, tm)

    if consumes_graph_input(region, tm) and not is_shape_helper(region, tm, dims):
        return (0, "graph_input_consumer", "Consumes graph input value such as input_ids/token_type_ids.")

    if is_embedding_signal(region, tm) and not is_shape_helper(region, tm, dims):
        return (1, "embedding_lookup_path", "Embedding/input lookup path.")

    if rt in {
        "FeedForwardRegion",
        "AttentionSkeletonRegion",
        "LinearProjectionRegion",
        "ResidualMergeRegion",
        "LayerNormRegion",
        "ActivationRegion",
    }:
        return (2, "main_activation_path", f"{rt} is a main activation/dataflow region.")

    if rt == "PrimitiveRegion":
        kinds = []
        for oid in region.get("op_ids") or []:
            kinds.append(op_kind(tm["op_by_id"].get(oid, {})))
        joined = " ".join(kinds + [text])
        if any(x in joined for x in ["gather", "matmul", "linear", "add", "layer_norm", "layernormalization", "softmax", "gelu", "erf", "mul"]):
            if is_shape_helper(region, tm, dims):
                return (4, "shape_axis_support", "Primitive op appears to support shape/axis computation.")
            return (2, "main_activation_path", "Primitive op is on the activation/data path.")

    if rt == "ModelRegion":
        return (3, "structural_region", "Model/root structural region.")

    if is_shape_helper(region, tm, dims):
        return (4, "shape_axis_support", "Shape/axis helper region deferred after main dataflow.")

    if "constant" in text or "initializer" in text:
        return (5, "constant_initializer_support", "Constant/initializer support region.")

    return (6, "unknown", "Unknown/analysis-only region.")


def region_order_key(
    region: dict[str, Any],
    tm: dict[str, Any],
    dims_by_region: dict[str, list[dict[str, Any]]],
    op_dist: dict[str, int],
    mode: str,
) -> tuple:
    raw = region_raw_order(region, tm)
    dist = region_graph_input_distance(region, op_dist)
    rank, cls, _ = region_semantic_class(region, tm, dims_by_region, op_dist)
    rid = region.get("region_id", "")

    if mode == "raw":
        return (raw, rid)
    if mode == "hybrid":
        return (rank, raw, dist, rid)
    return (rank, dist, raw, rid)


def role_of(region_id: str, interfaces: dict[str, dict[str, Any]]) -> str:
    return (interfaces.get(region_id) or {}).get("pruning_role") or "unknown"


def region_dim_summary(region_id: str, dims_by_region: dict[str, list[dict[str, Any]]]) -> str:
    dims = dims_by_region.get(region_id, [])
    if not dims:
        return ""
    names = sorted({d.get("dim_name", "dim") for d in dims})
    return ",".join(names[:4])


def detect_layer(text: str) -> str | None:
    patterns = [
        r"layer[._]?(\d+)",
        r"encoder[._]?layer[._]?(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return f"Layer {m.group(1)}"
    return None


def display_title(region: dict[str, Any], tm: dict[str, Any]) -> str:
    rt = region.get("region_type", "")
    text = region_source_text(region, tm)
    layer = detect_layer(text)

    prefix = f"{layer} " if layer else ""

    if rt == "ModelRegion":
        return "Model"
    if rt == "FeedForwardRegion":
        return prefix + "Feed-forward block"
    if rt == "AttentionSkeletonRegion":
        return prefix + "Attention skeleton"
    if rt == "ResidualMergeRegion":
        if "embedding" in text:
            return "Embedding residual merge"
        if "attention" in text:
            return prefix + "Attention residual merge"
        if "output" in text or "layernorm" in text:
            return prefix + "Residual merge"
        return "Residual merge"
    if rt == "AxisTransformRegion":
        if "attention_mask" in text or "mask" in text:
            return "Attention mask shape helper"
        if "position" in text:
            return "Position id shape helper"
        return "Shape / axis helper"
    if rt == "LinearProjectionRegion":
        if "query" in text:
            return prefix + "Query projection"
        if "key" in text:
            return prefix + "Key projection"
        if "value" in text:
            return prefix + "Value projection"
        if "attention.output" in text:
            return prefix + "Attention output projection"
        if "intermediate" in text:
            return prefix + "Intermediate projection"
        if "output.dense" in text:
            return prefix + "Feed-forward output projection"
        return prefix + "Linear projection"
    if rt == "ActivationRegion":
        if "gelu" in text or "erf" in text:
            return prefix + "GELU activation"
        return prefix + "Activation"
    if rt == "LayerNormRegion":
        if "embedding" in text:
            return "Embedding LayerNorm"
        return prefix + "LayerNorm"
    if rt == "ForkRegion":
        return prefix + "Fork"
    if rt == "JoinRegion":
        return prefix + "Join"
    if rt == "PrimitiveRegion":
        if "word_embeddings" in text or ("input_ids" in text and "gather" in text):
            return "Word embedding lookup"
        if "token_type_embeddings" in text or ("token_type_ids" in text and "gather" in text):
            return "Token type embedding lookup"
        if "position_embeddings" in text:
            return "Position embedding lookup"
        for oid in region.get("op_ids") or []:
            op = tm["op_by_id"].get(oid, {})
            k = op_kind(op)
            if k:
                return f"Primitive: {k}"
        return "Primitive op"

    return rt or "Region"


def display_subtitle(region: dict[str, Any], tm: dict[str, Any], interfaces: dict[str, dict[str, Any]], dims_by_region: dict[str, list[dict[str, Any]]], op_dist: dict[str, int]) -> str:
    rid = region.get("region_id")
    rank, cls, reason = region_semantic_class(region, tm, dims_by_region, op_dist)
    role = role_of(rid, interfaces)
    raw = region_raw_order(region, tm)
    dist = region_graph_input_distance(region, op_dist)
    dim = region_dim_summary(rid, dims_by_region)

    parts = [f"class={cls}", f"role={role}", f"raw={raw}"]
    if dist < LARGE:
        parts.append(f"input_dist={dist}")
    if dim:
        parts.append(f"dims={dim}")
    return " · ".join(parts)


def color_for_region(region: dict[str, Any], tm: dict[str, Any], dims_by_region: dict[str, list[dict[str, Any]]], op_dist: dict[str, int]) -> tuple[str, str]:
    rank, cls, _ = region_semantic_class(region, tm, dims_by_region, op_dist)
    if cls in {"graph_input_consumer", "embedding_lookup_path"}:
        return "#d7f7d7", "#218838"
    if cls == "main_activation_path":
        return "#d9ecff", "#2b6cb0"
    if cls == "shape_axis_support":
        return "#fff3cd", "#c48a00"
    if region.get("region_type") == "ResidualMergeRegion":
        return "#ffd6d6", "#b91c1c"
    return "#eeeeee", "#555555"


def shape_for_region(region: dict[str, Any]) -> str:
    rt = region.get("region_type")
    if rt == "ModelRegion":
        return "box3d"
    if rt == "FeedForwardRegion":
        return "component"
    if rt == "ResidualMergeRegion":
        return "diamond"
    if rt == "AttentionSkeletonRegion":
        return "octagon"
    if rt == "PrimitiveRegion":
        return "ellipse"
    return "box"


def ordered_children(
    region: dict[str, Any],
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    tm: dict[str, Any],
    dims_by_region: dict[str, list[dict[str, Any]]],
    op_dist: dict[str, int],
    mode: str,
) -> list[str]:
    explicit = region.get("children")
    if isinstance(explicit, list):
        kids = [k for k in explicit if k in region_by_id]
    else:
        kids = [k for k in children_by_parent.get(region.get("region_id"), []) if k in region_by_id]

    return sorted(kids, key=lambda cid: region_order_key(region_by_id[cid], tm, dims_by_region, op_dist, mode))


def emit_tree_dot(
    model: str,
    root_id: str,
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    interfaces: dict[str, dict[str, Any]],
    dims_by_region: dict[str, list[dict[str, Any]]],
    tm: dict[str, Any],
    op_dist: dict[str, int],
    mode: str,
    max_depth: int,
    include_primitives: bool,
    focus_region_type: str | None,
) -> str:
    lines = [
        "digraph LearnerDataflowTree {",
        "  graph [rankdir=LR, bgcolor=\"white\", fontname=\"Helvetica\", labelloc=\"t\", labeljust=\"l\", nodesep=0.35, ranksep=0.65];",
        "  node [fontname=\"Helvetica\", fontsize=10, style=\"filled,rounded\"];",
        "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#444444\", arrowsize=0.7];",
        f"  label=\"{dot_escape(model)} :: learner dataflow order ({mode})\";",
        "",
    ]

    seen: set[str] = set()

    def walk(rid: str, depth: int) -> None:
        if rid in seen or depth > max_depth:
            return
        region = region_by_id.get(rid)
        if not region:
            return
        if region.get("region_type") == "PrimitiveRegion" and not include_primitives:
            return
        if focus_region_type and depth == 0 and region.get("region_type") != focus_region_type:
            return

        seen.add(rid)

        fill, border = color_for_region(region, tm, dims_by_region, op_dist)
        title = display_title(region, tm)
        subtitle = display_subtitle(region, tm, interfaces, dims_by_region, op_dist)
        reason = region_semantic_class(region, tm, dims_by_region, op_dist)[2]

        label = f"{title}\\n{subtitle}\\n{reason}"
        node_id = safe_file_name(rid)

        lines.append(
            f'  "{node_id}" [label="{dot_escape(label)}", '
            f'shape="{shape_for_region(region)}", fillcolor="{fill}", color="{border}", penwidth=2];'
        )

        kids = ordered_children(region, region_by_id, children_by_parent, tm, dims_by_region, op_dist, mode)
        for i, cid in enumerate(kids):
            child = region_by_id[cid]
            if child.get("region_type") == "PrimitiveRegion" and not include_primitives:
                continue
            walk(cid, depth + 1)
            if cid in seen:
                lines.append(
                    f'  "{node_id}" -> "{safe_file_name(cid)}" [label="child[{i}]"];'
                )

    walk(root_id, 0)
    lines.append("}")
    return "\n".join(lines) + "\n"


def collect_unique_structures(region_by_id, interfaces, dims_by_region, tm, op_dist) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)

    for rid, region in region_by_id.items():
        rt = region.get("region_type", "Region")
        role = role_of(rid, interfaces)
        rank, cls, _ = region_semantic_class(region, tm, dims_by_region, op_dist)
        child_types = Counter(region_by_id[c].get("region_type", "Region") for c in region.get("children", []) if c in region_by_id)
        dim_names = ",".join(sorted({d.get("dim_name", "dim") for d in dims_by_region.get(rid, [])})) or "no_dims"
        sig = f"{rt}|role={role}|class={cls}|dims={dim_names}|children=" + ",".join(f"{k}:{v}" for k, v in sorted(child_types.items()))
        groups[sig].append(rid)

    structures = []
    for i, (sig, rids) in enumerate(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
        rep_id = sorted(rids, key=lambda rid: region_order_key(region_by_id[rid], tm, dims_by_region, op_dist, "semantic"))[0]
        rep = region_by_id[rep_id]
        structures.append({
            "structure_id": f"abs::{i:05d}::{safe_file_name(rep.get('region_type')).lower()}",
            "signature": sig,
            "count": len(rids),
            "representative_region_id": rep_id,
            "region_type": rep.get("region_type"),
            "display_title": display_title(rep, tm),
            "semantic_class": region_semantic_class(rep, tm, dims_by_region, op_dist)[1],
            "ordering_reason": region_semantic_class(rep, tm, dims_by_region, op_dist)[2],
        })
    return structures


def write_index(out_dir: Path, model: str, mode: str, full_svg: str, structures: list[dict[str, Any]]) -> None:
    lines = [
        f"# Learner Dataflow Graphviz Index: {model}",
        "",
        f"- Ordering mode: `{mode}`",
        f"- Full model SVG: `{full_svg}`",
        "",
        "| # | Structure | Count | Semantic class | Reason | SVG |",
        "|---:|---|---:|---|---|---|",
    ]
    for i, s in enumerate(structures):
        svg = f"{i:04d}_{safe_file_name(s['display_title']).lower()}_{safe_file_name(s['structure_id'])}.svg"
        lines.append(
            f"| {i} | {s['display_title']} | {s['count']} | {s['semantic_class']} | "
            f"{s['ordering_reason']} | `{svg}` |"
        )
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")


def render_dot(dot_path: Path, fmt: str) -> Path | None:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return None
    out = dot_path.with_suffix(f".{fmt}")
    subprocess.run([dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out)], check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export learner-dataflow ordered Graphviz/SVG views.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree-json", default=None)
    ap.add_argument("--tensor-ir", default=None)
    ap.add_argument("--region-dim-ir", default=None)
    ap.add_argument("--out-dir", default="reports/learner_dataflow_graphviz")
    ap.add_argument("--order", choices=["semantic", "raw", "hybrid"], default="semantic")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--include-primitives", action="store_true")
    ap.add_argument("--render", choices=["none", "svg", "pdf", "png"], default="svg")
    ap.add_argument("--structure-limit", type=int, default=50)
    args = ap.parse_args()

    model = args.model
    safe = safe_model_name(model)

    tree_path = Path(args.tree_json or f"reports/structural_region_trees/{safe}.json")
    tensor_path = Path(args.tensor_ir or f"reports/tensor_ir/{safe}.json")
    dim_path = Path(args.region_dim_ir or f"reports/region_dimension_ir/{safe}.json")

    if not tree_path.exists():
        raise FileNotFoundError(
            f"Missing Structural Region Tree: {tree_path}\n"
            f"Run: python scripts/build_structural_region_tree.py --model {model}"
        )

    tree = load_json(tree_path)
    tensor_ir = load_json_if_exists(tensor_path)
    region_dim_ir = load_json_if_exists(dim_path)

    tm = build_tensor_maps(tensor_ir)
    op_dist = compute_op_distance_from_graph_inputs(tm)
    region_by_id, children_by_parent, interfaces = build_region_maps(tree)
    dims_by_region = build_dims_by_region(region_dim_ir)

    root_id = tree.get("root_region_id")
    if not root_id:
        roots = children_by_parent.get(None) or []
        if not roots:
            raise ValueError("Could not determine root region.")
        root_id = roots[0]

    out_dir = Path(args.out_dir) / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    full_dot = emit_tree_dot(
        model=model,
        root_id=root_id,
        region_by_id=region_by_id,
        children_by_parent=children_by_parent,
        interfaces=interfaces,
        dims_by_region=dims_by_region,
        tm=tm,
        op_dist=op_dist,
        mode=args.order,
        max_depth=args.max_depth,
        include_primitives=args.include_primitives,
        focus_region_type=None,
    )

    full_dot_path = out_dir / f"full_model_{args.order}.dot"
    full_dot_path.write_text(full_dot)
    full_svg_name = full_dot_path.with_suffix(f".{args.render}").name if args.render != "none" else full_dot_path.name

    rendered = 0
    if args.render != "none":
        out = render_dot(full_dot_path, args.render)
        if out:
            rendered += 1

    structures = collect_unique_structures(region_by_id, interfaces, dims_by_region, tm, op_dist)
    structures = structures[: args.structure_limit]

    catalog = {
        "model_name": model,
        "order": args.order,
        "num_regions": len(region_by_id),
        "num_structures_exported": len(structures),
        "structures": structures,
    }
    (out_dir / "catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True))

    for i, s in enumerate(structures):
        rep_id = s["representative_region_id"]
        base = f"{i:04d}_{safe_file_name(s['display_title']).lower()}_{safe_file_name(s['structure_id'])}"
        dot_path = out_dir / f"{base}.dot"

        dot = emit_tree_dot(
            model=model,
            root_id=rep_id,
            region_by_id=region_by_id,
            children_by_parent=children_by_parent,
            interfaces=interfaces,
            dims_by_region=dims_by_region,
            tm=tm,
            op_dist=op_dist,
            mode=args.order,
            max_depth=args.max_depth,
            include_primitives=args.include_primitives,
            focus_region_type=None,
        )
        dot_path.write_text(dot)

        if args.render != "none":
            out = render_dot(dot_path, args.render)
            if out:
                rendered += 1

    write_index(out_dir, model, args.order, full_svg_name, structures)

    print(f"[learner-dataflow-graphviz] model={model}")
    print(f"[learner-dataflow-graphviz] order={args.order}")
    print(f"[learner-dataflow-graphviz] regions={len(region_by_id)}")
    print(f"[learner-dataflow-graphviz] structures={len(structures)}")
    print(f"[learner-dataflow-graphviz] out_dir={out_dir}")
    print(f"[learner-dataflow-graphviz] full_model={full_dot_path}")
    print(f"[learner-dataflow-graphviz] index={out_dir / 'index.md'}")
    if args.render != "none":
        print(f"[learner-dataflow-graphviz] rendered={rendered} {args.render} files")
        if not shutil.which("dot"):
            print("[warn] Graphviz `dot` not found. Install with: brew install graphviz")


if __name__ == "__main__":
    main()
