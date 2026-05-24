#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any


def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def safe_file_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", name)
    name = name.strip("_")
    return name or "unknown"


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def dot_escape(s: Any) -> str:
    s = str(s if s is not None else "")
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\n", "\\n")
    return s


def first_numeric_index(s: str) -> int | None:
    nums = re.findall(r"\d+", str(s))
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def build_tensor_op_maps(tensor_ir: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    if not tensor_ir:
        return {}, {}

    ops = tensor_ir.get("ops") or tensor_ir.get("operations") or tensor_ir.get("tensor_ops") or []
    op_by_id = {}
    op_order = {}

    for i, op in enumerate(ops):
        op_id = op.get("op_id") or op.get("id") or op.get("name")
        if not op_id:
            continue
        op_by_id[op_id] = op
        op_order[op_id] = i

    return op_by_id, op_order


def get_op_order(op_id: str, op_order: dict[str, int]) -> int:
    if op_id in op_order:
        return op_order[op_id]
    n = first_numeric_index(op_id)
    return n if n is not None else 10**12


def region_topological_key(region: dict[str, Any], op_order: dict[str, int]) -> tuple:
    metadata = region.get("metadata") or {}

    for k in ["order", "region_order", "source_order", "topological_order", "first_op_index", "min_op_index"]:
        v = metadata.get(k)
        if isinstance(v, int):
            return (0, v, region.get("region_id", ""))

    op_ids = region.get("op_ids") or []
    if op_ids:
        return (1, min(get_op_order(op, op_order) for op in op_ids), region.get("region_id", ""))

    n = first_numeric_index(region.get("region_id", ""))
    if n is not None:
        return (2, n, region.get("region_id", ""))

    return (3, region.get("region_id", ""))


def clean_source_name(name: str) -> str:
    if not name:
        return ""

    s = str(name)

    replacements = [
        ("/model/bert/encoder/layer.", "layer"),
        ("model_bert_encoder_layer_", "layer"),
        ("model_bert_embeddings_", "embeddings."),
        ("model_bert_", ""),
        ("bert.encoder.layer.", "layer"),
        ("bert.embeddings.", "embeddings."),
        ("attention_self_", "attention.self."),
        ("attention_output_", "attention.output."),
        ("intermediate_", "intermediate."),
        ("output_", "output."),
        ("LayerNorm_LayerNormalization", "LayerNorm"),
    ]

    for a, b in replacements:
        s = s.replace(a, b)

    s = s.replace("/", ".")
    s = re.sub(r"\.+", ".", s)
    s = s.strip(".")
    return s


def op_display_name(op_id: str, op_by_id: dict[str, dict[str, Any]]) -> str:
    op = op_by_id.get(op_id, {})
    candidates = [
        op.get("source_node_name"),
        op.get("name"),
        op.get("label"),
        op.get("op_name"),
        op_id,
    ]
    base = next((c for c in candidates if c), op_id)
    base = clean_source_name(str(base))

    canonical = op.get("canonical_op_type") or op.get("op_type") or op.get("type")
    if canonical and canonical not in base:
        return f"{base}\\n{canonical}"
    return base


def region_display_title(region: dict[str, Any], interface: dict[str, Any] | None = None) -> str:
    rt = region.get("region_type", "Region")
    md = region.get("metadata") or {}

    if rt == "ModelRegion":
        return "Model"
    if rt == "FeedForwardRegion":
        return "Feed-Forward Block"
    if rt == "AttentionSkeletonRegion":
        return "Attention Skeleton"
    if rt == "ResidualMergeRegion":
        return "Residual Merge"
    if rt == "AxisTransformRegion":
        return "Axis / Shape Transform"
    if rt == "LinearProjectionRegion":
        return "Linear Projection"
    if rt == "LayerNormRegion":
        return "LayerNorm"
    if rt == "ActivationRegion":
        if md.get("activation_kind") == "gelu" or "gelu" in str(md).lower():
            return "GELU Activation"
        return "Activation"
    if rt == "PrimitiveRegion":
        return "Primitive Op"
    if rt == "ForkRegion":
        return "Fork Region"
    if rt == "JoinRegion":
        return "Join Region"
    return rt


def region_pruning_role(region_id: str, interface_by_region: dict[str, dict[str, Any]]) -> str:
    iface = interface_by_region.get(region_id) or {}
    return iface.get("pruning_role") or "unknown"


def region_dimension_summary(region_id: str, dims_by_region: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    dims = dims_by_region.get(region_id, [])
    return {
        "num_dimensions": len(dims),
        "dim_names": sorted({d.get("dim_name", "unknown") for d in dims}),
        "axis_roles": sorted({d.get("axis_role", "unknown") for d in dims}),
        "num_prunable": sum(1 for d in dims if d.get("prunable")),
        "num_protected": sum(1 for d in dims if d.get("protected")),
        "num_blocked": sum(1 for d in dims if d.get("blocked")),
    }


def build_region_maps(tree: dict[str, Any]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str | None, list[str]],
    dict[str, dict[str, Any]],
]:
    regions = tree.get("regions", [])
    interfaces = tree.get("interfaces", [])

    region_by_id = {r.get("region_id"): r for r in regions if r.get("region_id")}
    children_by_parent: dict[str | None, list[str]] = defaultdict(list)

    for r in regions:
        children_by_parent[r.get("parent")].append(r.get("region_id"))

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


def ordered_children(
    region: dict[str, Any],
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    op_order: dict[str, int],
) -> list[str]:
    explicit = region.get("children")
    if isinstance(explicit, list):
        child_ids = [c for c in explicit if c in region_by_id]
    else:
        child_ids = [c for c in children_by_parent.get(region.get("region_id"), []) if c in region_by_id]

    return sorted(child_ids, key=lambda cid: region_topological_key(region_by_id[cid], op_order))


def abstract_signature(
    region: dict[str, Any],
    interface_by_region: dict[str, dict[str, Any]],
    dims_by_region: dict[str, list[dict[str, Any]]],
    region_by_id: dict[str, dict[str, Any]],
) -> str:
    rid = region.get("region_id")
    rt = region.get("region_type", "Region")
    role = region_pruning_role(rid, interface_by_region)
    conf = region.get("confidence", "unknown")

    child_types = Counter()
    for cid in region.get("children", []) or []:
        c = region_by_id.get(cid)
        if c:
            child_types[c.get("region_type", "Region")] += 1

    dim_summary = region_dimension_summary(rid, dims_by_region)

    child_part = ",".join(f"{k}:{v}" for k, v in sorted(child_types.items())) or "leaf"
    dim_part = ",".join(dim_summary["dim_names"]) or "no_dims"
    axis_part = ",".join(dim_summary["axis_roles"]) or "no_axes"

    return f"abs::{rt}|role={role}|conf={conf}|dims={dim_part}|axes={axis_part}|children={child_part}"


def collect_abstract_structures(
    tree: dict[str, Any],
    region_dim_ir: dict[str, Any] | None,
    tensor_ir: dict[str, Any] | None,
) -> dict[str, Any]:
    region_by_id, children_by_parent, interface_by_region = build_region_maps(tree)
    dims_by_region = build_dims_by_region(region_dim_ir)
    op_by_id, op_order = build_tensor_op_maps(tensor_ir)

    groups: dict[str, list[str]] = defaultdict(list)

    for rid, region in region_by_id.items():
        sig = abstract_signature(region, interface_by_region, dims_by_region, region_by_id)
        groups[sig].append(rid)

    structures = []

    for idx, (sig, region_ids) in enumerate(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
        representative_id = sorted(
            region_ids,
            key=lambda rid: region_topological_key(region_by_id[rid], op_order),
        )[0]
        rep = region_by_id[representative_id]
        role = region_pruning_role(representative_id, interface_by_region)
        dim_summary = region_dimension_summary(representative_id, dims_by_region)

        child_type_counts = Counter()
        for cid in rep.get("children", []) or []:
            c = region_by_id.get(cid)
            if c:
                child_type_counts[c.get("region_type", "Region")] += 1

        structures.append({
            "structure_id": f"abs::{idx:05d}::{safe_file_name(rep.get('region_type', 'Region')).lower()}",
            "signature": sig,
            "structure_type": rep.get("region_type", "Region"),
            "display_title": region_display_title(rep, interface_by_region.get(representative_id)),
            "count": len(region_ids),
            "representative_region_id": representative_id,
            "region_ids": sorted(region_ids, key=lambda rid: region_topological_key(region_by_id[rid], op_order)),
            "pruning_role": role,
            "confidence": rep.get("confidence", "unknown"),
            "dimension_summary": dim_summary,
            "child_type_counts": dict(sorted(child_type_counts.items())),
            "reason": rep.get("reason", ""),
        })

    return {
        "structures": structures,
        "region_by_id": region_by_id,
        "children_by_parent": children_by_parent,
        "interface_by_region": interface_by_region,
        "dims_by_region": dims_by_region,
        "op_by_id": op_by_id,
        "op_order": op_order,
    }


def color_for_role(role: str) -> str:
    return {
        "directly_prunable": "#b7f7c5",
        "propagation_only": "#cceeff",
        "constraint_carrier": "#ffe0aa",
        "blocked": "#ffcccc",
        "analysis_only": "#eeeeee",
        "unknown": "#ffffff",
    }.get(role, "#ffffff")


def border_for_region_type(region_type: str) -> str:
    return {
        "FeedForwardRegion": "#0b7a35",
        "ResidualMergeRegion": "#aa2222",
        "AttentionSkeletonRegion": "#7145d6",
        "AxisTransformRegion": "#0070aa",
        "LinearProjectionRegion": "#238b45",
        "ActivationRegion": "#6a51a3",
        "LayerNormRegion": "#e6550d",
        "PrimitiveRegion": "#555555",
        "ModelRegion": "#111111",
    }.get(region_type, "#333333")


def node_shape(region_type: str) -> str:
    if region_type == "ModelRegion":
        return "box3d"
    if region_type == "ResidualMergeRegion":
        return "diamond"
    if region_type == "AttentionSkeletonRegion":
        return "octagon"
    if region_type == "FeedForwardRegion":
        return "component"
    if region_type == "PrimitiveRegion":
        return "ellipse"
    return "box"


def region_node_label(
    region: dict[str, Any],
    interface_by_region: dict[str, dict[str, Any]],
    dims_by_region: dict[str, list[dict[str, Any]]],
    op_by_id: dict[str, dict[str, Any]],
) -> str:
    rid = region.get("region_id")
    rt = region.get("region_type", "Region")
    role = region_pruning_role(rid, interface_by_region)
    conf = region.get("confidence", "unknown")
    op_ids = region.get("op_ids") or []
    dim_summary = region_dimension_summary(rid, dims_by_region)

    title = region_display_title(region, interface_by_region.get(rid))

    if rt == "PrimitiveRegion" and op_ids:
        title = op_display_name(op_ids[0], op_by_id)

    lines = [
        title,
        f"{rt}",
        f"role={role} | conf={conf}",
    ]

    if op_ids:
        if len(op_ids) == 1:
            lines.append(f"op={clean_source_name(op_ids[0])}")
        else:
            lines.append(f"ops={len(op_ids)}")

    if dim_summary["dim_names"]:
        lines.append("dims=" + ",".join(dim_summary["dim_names"][:4]))

    return "\\n".join(lines)


def emit_structure_dot(
    *,
    model: str,
    structure: dict[str, Any],
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    interface_by_region: dict[str, dict[str, Any]],
    dims_by_region: dict[str, list[dict[str, Any]]],
    op_by_id: dict[str, dict[str, Any]],
    op_order: dict[str, int],
    max_depth: int,
    include_primitives: bool,
    representative_only: bool,
) -> str:
    rep_id = structure["representative_region_id"]
    rep = region_by_id[rep_id]

    lines = [
        "digraph AbstractStructure {",
        "  graph [rankdir=LR, bgcolor=\"white\", fontname=\"Helvetica\", labelloc=\"t\", labeljust=\"l\"];",
        "  node [fontname=\"Helvetica\", fontsize=10, style=\"filled,rounded\"];",
        "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#555555\", arrowsize=0.7];",
        "",
        f"  label=\"{dot_escape(model)} :: {dot_escape(structure['display_title'])} "
        f"(count={structure['count']})\\n"
        f"signature: {dot_escape(structure['signature'])}\";",
        "",
    ]

    seen: set[str] = set()

    def emit_region(rid: str, depth: int) -> None:
        if rid in seen:
            return
        if depth > max_depth:
            return

        region = region_by_id.get(rid)
        if not region:
            return

        rt = region.get("region_type", "Region")
        if rt == "PrimitiveRegion" and not include_primitives:
            return

        seen.add(rid)

        role = region_pruning_role(rid, interface_by_region)
        fill = color_for_role(role)
        border = border_for_region_type(rt)
        shape = node_shape(rt)
        label = region_node_label(region, interface_by_region, dims_by_region, op_by_id)
        node_id = safe_file_name(rid)

        lines.append(
            f'  "{node_id}" [label="{dot_escape(label)}", shape="{shape}", '
            f'fillcolor="{fill}", color="{border}", penwidth=2];'
        )

        children = ordered_children(region, region_by_id, children_by_parent, op_order)
        for i, cid in enumerate(children):
            child = region_by_id.get(cid)
            if not child:
                continue

            if child.get("region_type") == "PrimitiveRegion" and not include_primitives:
                continue

            emit_region(cid, depth + 1)

            if cid in seen:
                child_node_id = safe_file_name(cid)
                edge_label = f"child[{i}]"
                lines.append(f'  "{node_id}" -> "{child_node_id}" [label="{dot_escape(edge_label)}"];')

    emit_region(rep_id, 0)

    if not representative_only and len(structure["region_ids"]) > 1:
        lines.extend([
            "",
            "  subgraph cluster_instances {",
            "    label=\"Other instances with same abstract signature\";",
            "    color=\"#dddddd\";",
            "    style=\"rounded,dashed\";",
        ])
        for rid in structure["region_ids"][:25]:
            r = region_by_id[rid]
            key = region_topological_key(r, op_order)
            label = f"{rid}\\nops={key[1] if len(key) > 1 else '?'}"
            lines.append(
                f'    "inst_{safe_file_name(rid)}" [label="{dot_escape(label)}", '
                f'shape="note", fillcolor="#f8f8f8", color="#999999"];'
            )
        if len(structure["region_ids"]) > 25:
            lines.append(
                f'    "inst_more" [label="{len(structure["region_ids"]) - 25} more instances", '
                f'shape="note", fillcolor="#f8f8f8", color="#999999"];'
            )
        lines.append("  }")

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_index_markdown(model: str, structures: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        f"# Abstract Structure Graphviz Index: {model}",
        "",
        "| # | Structure | Count | Role | Confidence | Dims | Child types | DOT | SVG |",
        "|---:|---|---:|---|---|---|---|---|---|",
    ]

    for i, s in enumerate(structures):
        base = f"{i:04d}_{safe_file_name(s['display_title']).lower()}_{safe_file_name(s['structure_id'])}"
        dims = ", ".join(s["dimension_summary"].get("dim_names") or [])
        child_types = ", ".join(f"{k}:{v}" for k, v in s["child_type_counts"].items())
        dot = f"{base}.dot"
        svg = f"{base}.svg"
        lines.append(
            f"| {i} | {s['display_title']} | {s['count']} | {s['pruning_role']} | "
            f"{s['confidence']} | {dims or '-'} | {child_types or '-'} | "
            f"`{dot}` | `{svg}` |"
        )

    out_path.write_text("\n".join(lines) + "\n")


def maybe_render_dot(dot_path: Path, fmt: str) -> Path | None:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return None

    out_path = dot_path.with_suffix(f".{fmt}")
    subprocess.run(
        [dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out_path)],
        check=True,
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export unique abstract structures as Graphviz DOT/SVG for comparison with Netron."
    )
    ap.add_argument("--model", required=True, help="Model name, e.g. bert-base-uncased")
    ap.add_argument("--tree-json", default=None)
    ap.add_argument("--region-dim-ir", default=None)
    ap.add_argument("--tensor-ir", default=None)
    ap.add_argument("--out-dir", default="reports/abstract_structure_graphviz")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--include-primitives", action="store_true")
    ap.add_argument("--representative-only", action="store_true")
    ap.add_argument("--render", choices=["none", "svg", "pdf", "png"], default="svg")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of unique structures exported.")
    args = ap.parse_args()

    model = args.model
    safe = safe_model_name(model)

    tree_path = Path(args.tree_json or f"reports/structural_region_trees/{safe}.json")
    dim_path = Path(args.region_dim_ir or f"reports/region_dimension_ir/{safe}.json")
    tensor_path = Path(args.tensor_ir or f"reports/tensor_ir/{safe}.json")

    if not tree_path.exists():
        raise FileNotFoundError(
            f"Missing Structural Region Tree: {tree_path}\n"
            f"Run: python scripts/build_structural_region_tree.py --model {model}"
        )

    tree = json.loads(tree_path.read_text())
    dim_ir = load_json_if_exists(dim_path)
    tensor_ir = load_json_if_exists(tensor_path)

    collected = collect_abstract_structures(tree, dim_ir, tensor_ir)
    structures = collected["structures"]
    if args.limit is not None:
        structures = structures[: args.limit]

    out_dir = Path(args.out_dir) / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = {
        "model_name": model,
        "num_structures": len(structures),
        "structures": structures,
    }
    (out_dir / "catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True))

    rendered = 0

    for i, structure in enumerate(structures):
        base = f"{i:04d}_{safe_file_name(structure['display_title']).lower()}_{safe_file_name(structure['structure_id'])}"
        dot_path = out_dir / f"{base}.dot"

        dot = emit_structure_dot(
            model=model,
            structure=structure,
            region_by_id=collected["region_by_id"],
            children_by_parent=collected["children_by_parent"],
            interface_by_region=collected["interface_by_region"],
            dims_by_region=collected["dims_by_region"],
            op_by_id=collected["op_by_id"],
            op_order=collected["op_order"],
            max_depth=args.max_depth,
            include_primitives=args.include_primitives,
            representative_only=args.representative_only,
        )
        dot_path.write_text(dot)

        if args.render != "none":
            try:
                out = maybe_render_dot(dot_path, args.render)
                if out:
                    rendered += 1
            except Exception as e:
                print(f"[warn] failed to render {dot_path}: {e}")

    write_index_markdown(model, structures, out_dir / "index.md")

    print(f"[abstract-graphviz] model={model}")
    print(f"[abstract-graphviz] structures={len(structures)}")
    print(f"[abstract-graphviz] out_dir={out_dir}")
    print(f"[abstract-graphviz] catalog={out_dir / 'catalog.json'}")
    print(f"[abstract-graphviz] index={out_dir / 'index.md'}")
    if args.render == "none":
        print("[abstract-graphviz] render=none")
    else:
        print(f"[abstract-graphviz] rendered={rendered} format={args.render}")
        if rendered == 0:
            print("[abstract-graphviz] Graphviz `dot` may be missing. Install with: brew install graphviz")


if __name__ == "__main__":
    main()
