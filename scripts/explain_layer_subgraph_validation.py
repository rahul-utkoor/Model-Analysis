#!/usr/bin/env python
"""Explain selected records from a layer subgraph validation pack."""

from __future__ import annotations

import argparse
import json
import sys

from model_analysis.layer_subgraph_validation_pack import _table
from model_analysis.paths import get_project_root, safe_model_name
from model_analysis.registry import get_model_config
from model_analysis.reporting import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain layer subgraph validation records.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--contains")
    parser.add_argument("--class", dest="pruning_class")
    parser.add_argument("--no-plan", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def _matches(item: dict, args: argparse.Namespace) -> bool:
    cls = item.get("classification", {})
    if args.pruning_class and cls.get("pruning_class") != args.pruning_class:
        return False
    if args.no_plan and cls.get("plan_status") != "no_plan_expected":
        return False
    if args.contains and args.contains.lower() not in item.get("display_name", "").lower():
        return False
    return True


def _markdown(model: str, layer: int, rows: list[dict]) -> str:
    table_rows = [
        {
            "node": item.get("display_name"),
            "category": item.get("semantic_category"),
            "class": item.get("classification", {}).get("pruning_class"),
            "plan": item.get("classification", {}).get("plan_status"),
            "validation": item.get("classification", {}).get("validation_status"),
            "onnx": item.get("onnx_export", {}).get("status"),
            "reason": item.get("explanation", ""),
        }
        for item in rows
    ]
    return "\n".join([
        f"# Layer {layer} Subgraph Validation Selection: {model}",
        "",
        _table(table_rows, ["node", "category", "class", "plan", "validation", "onnx", "reason"]),
        "",
    ])


def main() -> int:
    args = parse_args()
    root = get_project_root()
    try:
        config = get_model_config(args.model)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    safe = safe_model_name(config["hf_id"])
    path = root / "reports" / "layer_subgraph_validation" / safe / f"layer_{args.layer}" / "index.json"
    if not path.exists():
        print(f"[missing] Layer subgraph pack missing. Run: python scripts/build_layer_subgraph_validation_pack.py --model {config['name']} --layer {args.layer}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [item for item in data.get("subgraphs", []) if _matches(item, args)][: args.limit]
    suffix = []
    if args.pruning_class:
        suffix.append(f"class_{args.pruning_class}")
    if args.no_plan:
        suffix.append("no_plan")
    if args.contains:
        suffix.append("contains_" + "_".join(args.contains.lower().split())[:60])
    suffix_text = "__selection__" + "__".join(suffix) if suffix else "__selection"
    out_json = root / "reports" / "layer_subgraph_validation_explanations" / f"{safe}__layer_{args.layer}{suffix_text}.json"
    out_md = root / "reports" / "layer_subgraph_validation_explanations" / f"{safe}__layer_{args.layer}{suffix_text}.md"
    write_json({"model_name": data.get("model_name"), "layer_index": args.layer, "matches": rows}, out_json)
    write_markdown(_markdown(data.get("model_name", safe), args.layer, rows), out_md)
    print(f"[layer-subgraph-explain] matches={len(rows)}")
    for item in rows:
        cls = item.get("classification", {})
        print(f"- {item.get('display_name')} [{cls.get('pruning_class')}] plan={cls.get('plan_status')} validation={cls.get('validation_status')}")
    print(f"[layer-subgraph-explain] json={out_json}")
    print(f"[layer-subgraph-explain] markdown={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
