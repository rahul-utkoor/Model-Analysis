#!/usr/bin/env python3
"""
Collect unique abstract structures from a Structural Region Tree.

This script groups StructuralRegion nodes into reusable abstract structure
signatures. It is deliberately frontend-independent: it operates on the
Structural Region Tree JSON produced by:

  scripts/build_structural_region_tree.py --model <model>

Outputs:

  reports/abstract_structures/<model>.json
  reports/abstract_structures/<model>.md

Usage:

  python abstract_structure_collector.py --model bert-base-uncased
  python abstract_structure_collector.py --model bert-base-uncased --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def safe_model_name(name: str) -> str:
    # Keep this compatible with common project-safe names.
    return name.replace("/", "__")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def normalize_token(s: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(s or "unknown")).strip("_") or "unknown"


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def interface_by_region(tree: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {i.get("region_id"): i for i in tree.get("interfaces", [])}


def dims_by_region(dim_ir: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not dim_ir:
        return out
    for d in dim_ir.get("dimension_variables", []):
        out[d.get("region_id")].append(d)
    return dict(out)


def make_structure_signature(
    region: dict[str, Any],
    child_regions: list[dict[str, Any]],
    interface: dict[str, Any] | None,
    dimensions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a canonical, stable abstract-structure signature.

    The signature intentionally ignores region ids and raw op ids. It captures
    the semantic *shape* of the region: region type, pruning role, child region
    type multiset, op type multiset, dimension names/roles, and constraint types.
    """
    region_type = region.get("region_type", "UnknownRegion")
    pruning_role = (interface or {}).get("pruning_role", "unknown")
    confidence = region.get("confidence", "unknown")

    child_type_counts = Counter(c.get("region_type", "UnknownRegion") for c in child_regions)
    op_type_counts = Counter(region.get("op_types", []) or [])

    dim_roles = sorted({
        f"{d.get('dim_name', 'unknown')}:{d.get('axis_role', 'unknown')}"
        for d in dimensions
    })

    constraints = []
    if interface:
        for c in interface.get("constraints", []) or []:
            if isinstance(c, dict):
                constraints.append(c.get("constraint_type") or c.get("type") or c.get("relation") or "constraint")
            else:
                constraints.append(str(c))
    constraints = sorted(set(constraints))

    signature = {
        "region_type": region_type,
        "pruning_role": pruning_role,
        "confidence": confidence,
        "contains_fork": bool(region.get("contains_fork")),
        "contains_join": bool(region.get("contains_join")),
        "single_entry": bool(region.get("single_entry")),
        "single_exit": bool(region.get("single_exit")),
        "child_type_counts": dict(sorted(child_type_counts.items())),
        "op_type_counts": dict(sorted(op_type_counts.items())),
        "dimension_roles": dim_roles,
        "constraint_types": constraints,
    }
    return signature


def signature_key(signature: dict[str, Any]) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def summarize_instance(
    region: dict[str, Any],
    interface: dict[str, Any] | None,
    dimensions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "region_id": region.get("region_id"),
        "region_type": region.get("region_type"),
        "name": region.get("name"),
        "parent": region.get("parent"),
        "children": region.get("children", []),
        "op_ids": region.get("op_ids", []),
        "op_types": region.get("op_types", []),
        "op_count": len(region.get("op_ids", [])),
        "child_count": len(region.get("children", [])),
        "contains_fork": region.get("contains_fork"),
        "contains_join": region.get("contains_join"),
        "single_entry": region.get("single_entry"),
        "single_exit": region.get("single_exit"),
        "confidence": region.get("confidence"),
        "reason": region.get("reason"),
        "pruning_role": (interface or {}).get("pruning_role", "unknown"),
        "interface_reason": (interface or {}).get("reason"),
        "dimension_count": len(dimensions),
        "dimension_names": sorted({d.get("dim_name") for d in dimensions if d.get("dim_name")}),
        "axis_roles": sorted({d.get("axis_role") for d in dimensions if d.get("axis_role")}),
    }


def collect_abstract_structures(
    model: str,
    tree: dict[str, Any],
    dim_ir: dict[str, Any] | None = None,
) -> dict[str, Any]:
    regions = tree.get("regions", [])
    by_id = {r.get("region_id"): r for r in regions}
    ifaces = interface_by_region(tree)
    dims = dims_by_region(dim_ir)

    buckets: dict[str, dict[str, Any]] = {}

    for region in regions:
        rid = region.get("region_id")
        child_regions = [by_id[cid] for cid in region.get("children", []) if cid in by_id]
        iface = ifaces.get(rid)
        rdims = dims.get(rid, [])
        sig = make_structure_signature(region, child_regions, iface, rdims)
        key = signature_key(sig)
        sid = "abs::" + normalize_token(sig["region_type"]).lower() + "::" + short_hash(key)

        if sid not in buckets:
            buckets[sid] = {
                "structure_id": sid,
                "structure_type": sig["region_type"],
                "signature": sig,
                "count": 0,
                "example_region_ids": [],
                "instances": [],
            }

        b = buckets[sid]
        b["count"] += 1
        if len(b["example_region_ids"]) < 10:
            b["example_region_ids"].append(rid)
        b["instances"].append(summarize_instance(region, iface, rdims))

    structures = sorted(
        buckets.values(),
        key=lambda x: (-x["count"], x["structure_type"], x["structure_id"]),
    )

    type_counts = Counter(s["structure_type"] for s in structures)
    role_counts = Counter(s["signature"].get("pruning_role", "unknown") for s in structures)

    return {
        "model_name": tree.get("model_name", model),
        "source_frontend": tree.get("source_frontend"),
        "num_regions": len(regions),
        "num_abstract_structures": len(structures),
        "structures": structures,
        "summary": {
            "structure_type_counts": dict(sorted(type_counts.items())),
            "pruning_role_counts": dict(sorted(role_counts.items())),
            "top_structures": [
                {
                    "structure_id": s["structure_id"],
                    "structure_type": s["structure_type"],
                    "count": s["count"],
                    "pruning_role": s["signature"].get("pruning_role"),
                }
                for s in structures[:25]
            ],
        },
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Abstract Structure Catalog: {report.get('model_name')}\n")
    lines.append("## Summary\n")
    lines.append(f"- Regions: {report.get('num_regions')}")
    lines.append(f"- Unique abstract structures: {report.get('num_abstract_structures')}")
    lines.append(f"- Source frontend: {report.get('source_frontend')}\n")

    lines.append("## Structure Type Counts\n")
    lines.append("| Structure type | Count |")
    lines.append("|---|---:|")
    for k, v in report.get("summary", {}).get("structure_type_counts", {}).items():
        lines.append(f"| `{k}` | {v} |")

    lines.append("\n## Top Abstract Structures\n")
    lines.append("| Structure ID | Type | Count | Role | Example regions |")
    lines.append("|---|---|---:|---|---|")
    for s in report.get("structures", [])[:50]:
        examples = ", ".join(f"`{x}`" for x in s.get("example_region_ids", [])[:5])
        role = s.get("signature", {}).get("pruning_role", "unknown")
        lines.append(f"| `{s['structure_id']}` | `{s['structure_type']}` | {s['count']} | `{role}` | {examples} |")

    lines.append("\n## Interpretation\n")
    lines.append(
        "Each abstract structure is a canonical signature over region type, pruning role, "
        "child region-type multiset, op-type multiset, dimension roles, and constraint types. "
        "Instances point back to concrete regions in the Structural Region Tree."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree", default=None)
    ap.add_argument("--region-dim-ir", default=None)
    ap.add_argument("--out-dir", default="reports/abstract_structures")
    ap.add_argument("--write", action="store_true", help="Write JSON/Markdown report")
    args = ap.parse_args()

    safe = safe_model_name(args.model)
    tree_path = Path(args.tree or f"reports/structural_region_trees/{safe}.json")
    dim_path = Path(args.region_dim_ir or f"reports/region_dimension_ir/{safe}.json")

    if not tree_path.exists():
        raise FileNotFoundError(f"Structural Region Tree missing: {tree_path}")

    tree = read_json(tree_path)
    dim_ir = read_json(dim_path) if dim_path.exists() else None

    report = collect_abstract_structures(args.model, tree, dim_ir)

    print(f"[abstract-structures] model={report['model_name']}")
    print(f"[abstract-structures] regions={report['num_regions']}")
    print(f"[abstract-structures] unique_structures={report['num_abstract_structures']}")
    for item in report["summary"]["top_structures"][:10]:
        print(f"  {item['count']:5d}  {item['structure_type']:28s}  {item['pruning_role']:20s}  {item['structure_id']}")

    if args.write:
        out_dir = Path(args.out_dir)
        write_json(out_dir / f"{safe}.json", report)
        (out_dir / f"{safe}.md").write_text(to_markdown(report))
        print(f"[abstract-structures] wrote {out_dir / f'{safe}.json'}")
        print(f"[abstract-structures] wrote {out_dir / f'{safe}.md'}")


if __name__ == "__main__":
    main()
