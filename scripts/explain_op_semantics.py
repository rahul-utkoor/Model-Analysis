#!/usr/bin/env python
"""Explain selected primitive op semantics records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain selected Op Semantics records.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--contains", help="Case-insensitive substring over source name, kind, category, or reason.")
    parser.add_argument("--semantic-kind")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def _matches(op: dict, args: argparse.Namespace) -> bool:
    if args.semantic_kind and op.get("semantic_kind") != args.semantic_kind:
        return False
    if args.category and op.get("semantic_category") != args.category:
        return False
    if args.contains:
        needle = args.contains.lower()
        haystack = " ".join(
            [
                str(op.get("source_name", "")),
                str(op.get("semantic_kind", "")),
                str(op.get("semantic_category", "")),
                str(op.get("pruning_effect", {}).get("reason", "")),
            ]
        ).lower()
        if needle not in haystack:
            return False
    return True


def _markdown(model: str, rows: list[dict]) -> str:
    lines = [f"# Op Semantics Selection: {model}", "", "| idx | source | kind | category | direct | blockers |", "| --- | --- | --- | --- | --- | --- |"]
    for op in rows:
        effect = op.get("pruning_effect", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(op.get("topological_index", "")),
                    str(op.get("source_name", "")).replace("|", "\\|"),
                    str(op.get("semantic_kind", "")),
                    str(op.get("semantic_category", "")),
                    str(effect.get("direct_pruning", "")),
                    ", ".join(effect.get("blockers", [])),
                ]
            )
            + " |"
        )
    lines.extend(["", "This is a static explanation over Op Semantics IR. It does not modify models.", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    safe = safe_model_name(config["hf_id"])
    path = root / "reports" / "op_semantics" / f"{safe}.json"
    if not path.exists():
        print(f"[missing] Op Semantics missing. Run: python scripts/build_op_semantics.py --model {config['name']}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [op for op in data.get("ops", []) if _matches(op, args)][: args.limit]
    suffix_parts = []
    if args.semantic_kind:
        suffix_parts.append(f"kind_{args.semantic_kind}")
    if args.category:
        suffix_parts.append(f"category_{args.category}")
    if args.contains:
        suffix_parts.append("contains_" + "_".join(args.contains.lower().split())[:60])
    suffix = "__selection__" + "__".join(suffix_parts) if suffix_parts else "__selection"
    out_json = root / "reports" / "op_semantics_explanations" / f"{safe}{suffix}.json"
    out_md = root / "reports" / "op_semantics_explanations" / f"{safe}{suffix}.md"
    write_json({"model_name": data.get("model_name"), "matches": rows}, out_json)
    write_markdown(_markdown(data.get("model_name", safe), rows), out_md)
    print(f"[op-semantics-explain] matches={len(rows)}")
    for op in rows:
        effect = op.get("pruning_effect", {})
        print(f"- {op.get('source_name')} [{op.get('semantic_kind')} => {op.get('semantic_category')}] direct={effect.get('direct_pruning')}")
    print(f"[op-semantics-explain] json={out_json}")
    print(f"[op-semantics-explain] markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

