#!/usr/bin/env python3
"""Export a control-tree construction trace as a MindNode-compatible outline."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def make_outline_lines(trace: dict[str, Any], max_steps: int | None = None) -> list[str]:
    model_name = trace.get("model_name", "")
    lines = [f"Control Tree Trace: {model_name}"]
    steps = trace.get("steps", [])
    if max_steps is not None:
        steps = steps[:max_steps]
    for step in steps:
        index = int(step.get("step_index", 0))
        action = step.get("action", "")
        created = step.get("created_region_type") or step.get("pass_name", "")
        lines.append(f"\tStep {index:03d} {action} {created}".rstrip())
        before = step.get("before_summary", {}).get("num_active_nodes", 0)
        after = step.get("after_summary", {}).get("num_active_nodes", 0)
        lines.append(f"\t\tActive nodes: {before} -> {after}")
        if step.get("collapsed_op_ids"):
            lines.append(f"\t\tCollapsed ops: {', '.join(step.get('collapsed_op_ids', [])[:12])}")
        if step.get("collapsed_region_ids"):
            lines.append(f"\t\tCollapsed regions: {', '.join(step.get('collapsed_region_ids', [])[:12])}")
        reason = step.get("reason", "")
        if reason:
            lines.append(f"\t\tReason: {reason}")
    return lines


def make_opml(lines: list[str], title: str) -> str:
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

    def emit(node: dict[str, Any], indent: int) -> list[str]:
        text = html.escape(node["text"], quote=True)
        children = node.get("children", [])
        pad = "  " * indent
        if not children:
            return [f'{pad}<outline text="{text}"/>']
        out = [f'{pad}<outline text="{text}">']
        for child in children:
            out.extend(emit(child, indent + 1))
        out.append(f"{pad}</outline>")
        return out

    body: list[str] = []
    for child in root["children"]:
        body.extend(emit(child, 3))
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<opml version="2.0">',
            "  <head>",
            f"    <title>{html.escape(title)}</title>",
            "  </head>",
            "  <body>",
            *body,
            "  </body>",
            "</opml>",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export control-tree trace as MindNode outline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-json", default=None)
    parser.add_argument("--out-dir", default="reports/mindnode_outlines")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    safe = safe_model_name(args.model)
    trace_path = Path(args.trace_json or f"reports/control_tree_steps/{safe}.json")
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Control tree trace missing: {trace_path}\n"
            f"Run: python scripts/build_control_tree_trace.py --model {args.model}"
        )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    lines = make_outline_lines(trace, max_steps=args.max_steps)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{safe}.control_tree_steps.mindnode.txt"
    opml_path = out_dir / f"{safe}.control_tree_steps.opml"
    title = f"Control Tree Trace: {args.model}"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    opml_path.write_text(make_opml(lines, title), encoding="utf-8")
    print(f"[control-tree-trace-mindnode] model={args.model}")
    print(f"[control-tree-trace-mindnode] steps={len(trace.get('steps', []))}")
    print(f"[control-tree-trace-mindnode] outline={txt_path}")
    print(f"[control-tree-trace-mindnode] opml={opml_path}")


if __name__ == "__main__":
    main()
