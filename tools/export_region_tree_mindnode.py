#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def region_label(
    region: dict[str, Any],
    interface_by_region: dict[str, dict[str, Any]],
    *,
    label_mode: str,
    include_id: bool,
    include_role: bool,
    include_confidence: bool,
    include_counts: bool,
) -> str:
    region_type = region.get("region_type", "Region")
    region_id = region.get("region_id", "")
    name = region.get("name") or ""

    iface = interface_by_region.get(region_id, {})
    role = iface.get("pruning_role") or region.get("metadata", {}).get("pruning_role") or "unknown"
    confidence = region.get("confidence", "unknown")

    op_count = len(region.get("op_ids", []))
    child_count = len(region.get("children", []))

    if label_mode == "compact":
        label = region_type
    elif label_mode == "name":
        label = f"{region_type}: {name}" if name and name != region_type else region_type
    elif label_mode == "semantic":
        pieces = [region_type]
        if role and role != "unknown":
            pieces.append(role)
        if confidence and confidence != "unknown":
            pieces.append(confidence)
        label = " | ".join(pieces)
    else:
        raise ValueError(f"unknown label mode: {label_mode}")

    suffix = []

    if include_role and role:
        suffix.append(f"role={role}")

    if include_confidence and confidence:
        suffix.append(f"conf={confidence}")

    if include_counts:
        suffix.append(f"ops={op_count}")
        suffix.append(f"children={child_count}")

    if include_id and region_id:
        suffix.append(region_id)

    if suffix:
        label = f"{label}  [{', '.join(suffix)}]"

    return label


def build_maps(tree: dict[str, Any]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str | None, list[str]],
    dict[str, dict[str, Any]],
]:
    regions = tree.get("regions", [])
    interfaces = tree.get("interfaces", [])

    region_by_id = {r.get("region_id"): r for r in regions}
    interface_by_region = {i.get("region_id"): i for i in interfaces}

    children_by_parent: dict[str | None, list[str]] = {}
    for r in regions:
        parent = r.get("parent")
        rid = r.get("region_id")
        children_by_parent.setdefault(parent, []).append(rid)

    return region_by_id, children_by_parent, interface_by_region


def region_order_key(region: dict[str, Any]) -> tuple:
    """
    Preserve the compiler/control-tree order as much as possible.

    Priority:
    1. explicit metadata order if present
    2. first TensorOp numeric id if present
    3. first op_id lexical fallback
    4. region_id fallback

    This avoids alphabetical sorting by region type, because that destroys
    the structural ordering of the model.
    """
    metadata = region.get("metadata", {}) or {}

    for key in [
        "order",
        "region_order",
        "source_order",
        "topological_order",
        "first_op_index",
        "min_op_index",
    ]:
        value = metadata.get(key)
        if isinstance(value, int):
            return (0, value)

    op_ids = region.get("op_ids", []) or []
    numeric_indices = []

    for op in op_ids:
        # Common patterns:
        # op::000123::...
        # op:000123
        # ..._000123
        nums = re.findall(r"\d+", str(op))
        if nums:
            try:
                numeric_indices.append(int(nums[-1]))
            except ValueError:
                pass

    if numeric_indices:
        return (1, min(numeric_indices))

    if op_ids:
        return (2, str(op_ids[0]))

    return (3, str(region.get("region_id", "")))


def ordered_children(
    region: dict[str, Any],
    region_by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
) -> list[str]:
    """
    Respect region.children if present. That is the tree builder's intended
    order. Only fall back to parent map if children is absent.
    """
    explicit_children = region.get("children")
    if isinstance(explicit_children, list):
        child_ids = [c for c in explicit_children if c in region_by_id]
    else:
        child_ids = [c for c in children_by_parent.get(region.get("region_id"), []) if c in region_by_id]

    # Keep explicit ordering mostly intact, but if all children look unordered,
    # topological key makes the output stable and meaningful.
    return sorted(child_ids, key=lambda cid: region_order_key(region_by_id[cid]))


def make_outline_lines(
    tree: dict[str, Any],
    *,
    max_depth: int | None,
    include_primitives: bool,
    label_mode: str,
    include_id: bool,
    include_role: bool,
    include_confidence: bool,
    include_counts: bool,
) -> list[str]:
    region_by_id, children_by_parent, interface_by_region = build_maps(tree)

    root_id = tree.get("root_region_id")
    if not root_id:
        roots = children_by_parent.get(None, [])
        if not roots:
            raise ValueError("Could not determine root region.")
        root_id = roots[0]

    if root_id not in region_by_id:
        raise ValueError(f"Root region not found in regions: {root_id}")

    lines: list[str] = []
    seen: set[str] = set()

    def walk(region_id: str, depth: int) -> None:
        if region_id in seen:
            lines.append("\t" * depth + f"[cycle/reference] {region_id}")
            return

        region = region_by_id[region_id]
        region_type = region.get("region_type")

        if not include_primitives and region_type == "PrimitiveRegion":
            return

        if max_depth is not None and depth > max_depth:
            return

        seen.add(region_id)

        label = region_label(
            region,
            interface_by_region,
            label_mode=label_mode,
            include_id=include_id,
            include_role=include_role,
            include_confidence=include_confidence,
            include_counts=include_counts,
        )
        lines.append("\t" * depth + label)

        for child_id in ordered_children(region, region_by_id, children_by_parent):
            walk(child_id, depth + 1)

        seen.remove(region_id)

    walk(root_id, 0)
    return lines


def make_opml(lines: list[str], title: str) -> str:
    """
    Convert tab-indented outline to OPML.

    MindNode can import OPML on macOS. The plain .txt is better for copy-paste;
    OPML is better for file import.
    """
    root = {"text": title, "children": []}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line in lines:
        depth = len(line) - len(line.lstrip("\t"))
        text = line.lstrip("\t")
        node = {"text": text, "children": []}

        while stack and stack[-1][0] >= depth:
            stack.pop()

        stack[-1][1]["children"].append(node)
        stack.append((depth, node))

    def emit_node(node: dict[str, Any], indent: int) -> list[str]:
        text = html.escape(node["text"], quote=True)
        children = node.get("children", [])
        pad = "  " * indent

        if not children:
            return [f'{pad}<outline text="{text}"/>']

        out = [f'{pad}<outline text="{text}">']
        for child in children:
            out.extend(emit_node(child, indent + 1))
        out.append(f"{pad}</outline>")
        return out

    body_lines = []
    for child in root["children"]:
        body_lines.extend(emit_node(child, 3))

    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<opml version="2.0">',
        '  <head>',
        f'    <title>{html.escape(title)}</title>',
        '  </head>',
        '  <body>',
        *body_lines,
        '  </body>',
        '</opml>',
        '',
    ])


def write_markdown_preview(lines: list[str], path: Path, title: str) -> None:
    path.write_text(
        "# " + title + "\n\n"
        "Copy the tab-indented outline from the `.mindnode.txt` file into MindNode.\n\n"
        "```text\n"
        + "\n".join(lines[:300])
        + ("\n...\n" if len(lines) > 300 else "\n")
        + "```\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export Structural Region Tree as a MindNode-compatible outline."
    )
    ap.add_argument("--model", required=True, help="Model name, e.g. bert-base-uncased")
    ap.add_argument(
        "--tree-json",
        default=None,
        help="Path to structural region tree JSON. Defaults to reports/structural_region_trees/<model>.json",
    )
    ap.add_argument(
        "--out-dir",
        default="reports/mindnode_outlines",
        help="Output directory.",
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum tree depth to export.",
    )
    ap.add_argument(
        "--include-primitives",
        action="store_true",
        help="Include PrimitiveRegion leaves. By default they are omitted to keep MindNode readable.",
    )
    ap.add_argument(
        "--label-mode",
        choices=["compact", "name", "semantic"],
        default="semantic",
        help="Node label style.",
    )
    ap.add_argument("--include-id", action="store_true")
    ap.add_argument("--include-role", action="store_true", default=True)
    ap.add_argument("--no-role", action="store_false", dest="include_role")
    ap.add_argument("--include-confidence", action="store_true", default=True)
    ap.add_argument("--no-confidence", action="store_false", dest="include_confidence")
    ap.add_argument("--include-counts", action="store_true")
    args = ap.parse_args()

    model = args.model
    safe = safe_model_name(model)

    tree_path = Path(args.tree_json or f"reports/structural_region_trees/{safe}.json")
    if not tree_path.exists():
        raise FileNotFoundError(
            f"Structural Region Tree missing: {tree_path}\n"
            f"Run: python scripts/build_structural_region_tree.py --model {model}"
        )

    tree = load_json(tree_path)
    lines = make_outline_lines(
        tree,
        max_depth=args.max_depth,
        include_primitives=args.include_primitives,
        label_mode=args.label_mode,
        include_id=args.include_id,
        include_role=args.include_role,
        include_confidence=args.include_confidence,
        include_counts=args.include_counts,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"{safe}.mindnode.txt"
    opml_path = out_dir / f"{safe}.opml"
    md_path = out_dir / f"{safe}.mindnode_preview.md"

    title = f"Structural Region Tree: {model}"

    txt_path.write_text("\n".join(lines) + "\n")
    opml_path.write_text(make_opml(lines, title))
    write_markdown_preview(lines, md_path, title)

    print(f"[mindnode] model={model}")
    print(f"[mindnode] nodes={len(lines)}")
    print(f"[mindnode] outline={txt_path}")
    print(f"[mindnode] opml={opml_path}")
    print(f"[mindnode] preview={md_path}")
    print()
    print("Copy-paste this file into MindNode:")
    print(f"  {txt_path}")
    print()
    print("Or import this OPML file:")
    print(f"  {opml_path}")


if __name__ == "__main__":
    main()
