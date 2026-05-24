#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def safe_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", s)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def summarize_region(region: dict[str, Any], interface: dict[str, Any] | None, dim_count: int) -> dict[str, Any]:
    return {
        "region_id": region.get("region_id"),
        "region_type": region.get("region_type"),
        "name": region.get("name"),
        "parent": region.get("parent"),
        "children": region.get("children", []),
        "child_count": len(region.get("children", [])),
        "op_count": len(region.get("op_ids", [])),
        "value_count": len(region.get("value_ids", [])),
        "contains_fork": region.get("contains_fork"),
        "contains_join": region.get("contains_join"),
        "single_entry": region.get("single_entry"),
        "single_exit": region.get("single_exit"),
        "confidence": region.get("confidence"),
        "reason": region.get("reason"),
        "pruning_role": (interface or {}).get("pruning_role", "unknown"),
        "interface_reason": (interface or {}).get("reason"),
        "dim_count": dim_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--region-tree",
        default=None,
        help="Path to reports/structural_region_trees/<model>.json",
    )
    ap.add_argument(
        "--region-dim-ir",
        default=None,
        help="Optional path to reports/region_dimension_ir/<model>.json",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Default: viewer_data/<model>",
    )
    args = ap.parse_args()

    model = args.model
    tree_path = Path(args.region_tree or f"reports/structural_region_trees/{model}.json")
    dim_path = Path(args.region_dim_ir or f"reports/region_dimension_ir/{model}.json")
    out_dir = Path(args.out_dir or f"viewer_data/{model}")

    tree = load_json(tree_path)
    dim_ir = load_json(dim_path) if dim_path.exists() else None

    regions = tree.get("regions", [])
    interfaces = tree.get("interfaces", [])

    iface_by_region = {i.get("region_id"): i for i in interfaces}
    region_by_id = {r.get("region_id"): r for r in regions}

    dims_by_region: dict[str, list[dict[str, Any]]] = {}
    if dim_ir:
        for d in dim_ir.get("dimension_variables", []):
            dims_by_region.setdefault(d.get("region_id"), []).append(d)

    children_by_parent: dict[str | None, list[str]] = {}
    for r in regions:
        children_by_parent.setdefault(r.get("parent"), []).append(r.get("region_id"))

    for k in children_by_parent:
        children_by_parent[k] = sorted(children_by_parent[k])

    root_id = tree.get("root_region_id")
    if not root_id and None in children_by_parent:
        root_id = children_by_parent[None][0]

    region_summaries = {}
    for rid, region in region_by_id.items():
        region_summaries[rid] = summarize_region(
            region,
            iface_by_region.get(rid),
            len(dims_by_region.get(rid, [])),
        )

    # Small global index only.
    index = {
        "model_name": tree.get("model_name", model),
        "source_frontend": tree.get("source_frontend"),
        "root_region_id": root_id,
        "summary": tree.get("summary", {}),
        "num_regions": len(regions),
        "has_region_dimension_ir": dim_ir is not None,
        "region_summaries": region_summaries,
    }

    write_json(out_dir / "index.json", index)

    # One file per region.
    for rid, region in region_by_id.items():
        payload = {
            "region": region,
            "interface": iface_by_region.get(rid),
            "dimensions": dims_by_region.get(rid, []),
            "children": [
                region_summaries[cid]
                for cid in region.get("children", [])
                if cid in region_summaries
            ],
        }
        write_json(out_dir / "regions" / f"{safe_id(rid)}.json", payload)

    print(f"[lazy-region-tree] model={model}")
    print(f"[lazy-region-tree] regions={len(regions)}")
    print(f"[lazy-region-tree] output={out_dir}")
    print(f"[lazy-region-tree] index={out_dir / 'index.json'}")
    print("[lazy-region-tree] serve with:")
    print("  python -m http.server 8000")
    print("[lazy-region-tree] open:")
    print("  http://localhost:8000/tools/lazy_region_tree_viewer.html")


if __name__ == "__main__":
    main()
