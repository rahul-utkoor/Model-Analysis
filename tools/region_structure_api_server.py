#!/usr/bin/env python3
"""
API-backed Structural Region Tree + Abstract Structure viewer.

Run from repository root:

  python region_structure_api_server.py --model bert-base-uncased --port 8765

Open:

  http://localhost:8765/

The frontend initially fetches only:
  GET /api/index
  GET /api/structures

When the user clicks a structure type, it fetches only the matching abstract
structure instances. When a concrete region is expanded, it fetches only that
region and its direct children. When collapsed, descendants are removed from the
frontend state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path.cwd()


def safe_model_name(name: str) -> str:
    return name.replace("/", "__")


def normalize_token(s: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(s or "unknown")).strip("_") or "unknown"


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def signature_key(signature: dict[str, Any]) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def make_structure_signature(region: dict[str, Any], child_regions: list[dict[str, Any]], interface: dict[str, Any] | None, dimensions: list[dict[str, Any]]) -> dict[str, Any]:
    child_type_counts = Counter(c.get("region_type", "UnknownRegion") for c in child_regions)
    op_type_counts = Counter(region.get("op_types", []) or [])
    dim_roles = sorted({f"{d.get('dim_name', 'unknown')}:{d.get('axis_role', 'unknown')}" for d in dimensions})

    constraints = []
    if interface:
        for c in interface.get("constraints", []) or []:
            if isinstance(c, dict):
                constraints.append(c.get("constraint_type") or c.get("type") or c.get("relation") or "constraint")
            else:
                constraints.append(str(c))

    return {
        "region_type": region.get("region_type", "UnknownRegion"),
        "pruning_role": (interface or {}).get("pruning_role", "unknown"),
        "confidence": region.get("confidence", "unknown"),
        "contains_fork": bool(region.get("contains_fork")),
        "contains_join": bool(region.get("contains_join")),
        "single_entry": bool(region.get("single_entry")),
        "single_exit": bool(region.get("single_exit")),
        "child_type_counts": dict(sorted(child_type_counts.items())),
        "op_type_counts": dict(sorted(op_type_counts.items())),
        "dimension_roles": dim_roles,
        "constraint_types": sorted(set(constraints)),
    }


def region_summary(region: dict[str, Any], interface: dict[str, Any] | None, dims: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "region_id": region.get("region_id"),
        "region_type": region.get("region_type"),
        "name": region.get("name"),
        "parent": region.get("parent"),
        "children": region.get("children", []),
        "child_count": len(region.get("children", [])),
        "op_count": len(region.get("op_ids", [])),
        "op_types": region.get("op_types", []),
        "contains_fork": region.get("contains_fork"),
        "contains_join": region.get("contains_join"),
        "single_entry": region.get("single_entry"),
        "single_exit": region.get("single_exit"),
        "confidence": region.get("confidence"),
        "reason": region.get("reason"),
        "pruning_role": (interface or {}).get("pruning_role", "unknown"),
        "interface_reason": (interface or {}).get("reason"),
        "dimension_count": len(dims),
        "dimension_names": sorted({d.get("dim_name") for d in dims if d.get("dim_name")}),
        "axis_roles": sorted({d.get("axis_role") for d in dims if d.get("axis_role")}),
    }


class RegionDatabase:
    def __init__(self, model: str, root: Path):
        self.model = model
        self.safe = safe_model_name(model)
        self.root = root
        self.tree_path = root / "reports" / "structural_region_trees" / f"{self.safe}.json"
        self.dim_path = root / "reports" / "region_dimension_ir" / f"{self.safe}.json"
        if not self.tree_path.exists():
            raise FileNotFoundError(f"Missing {self.tree_path}. Run scripts/build_structural_region_tree.py --model {model}")
        self.tree = read_json(self.tree_path)
        self.dim_ir = read_json(self.dim_path) if self.dim_path.exists() else None

        self.regions = self.tree.get("regions", [])
        self.region_by_id = {r.get("region_id"): r for r in self.regions}
        self.interfaces = {i.get("region_id"): i for i in self.tree.get("interfaces", [])}
        self.dims_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if self.dim_ir:
            for d in self.dim_ir.get("dimension_variables", []):
                self.dims_by_region[d.get("region_id")].append(d)
        self.region_summaries = {
            rid: region_summary(r, self.interfaces.get(rid), self.dims_by_region.get(rid, []))
            for rid, r in self.region_by_id.items()
        }
        self.root_region_id = self.tree.get("root_region_id") or self._infer_root()
        self.structures = self._collect_structures()
        self.structures_by_id = {s["structure_id"]: s for s in self.structures}

    def _infer_root(self) -> str | None:
        for r in self.regions:
            if r.get("parent") is None:
                return r.get("region_id")
        return self.regions[0].get("region_id") if self.regions else None

    def _collect_structures(self) -> list[dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for rid, region in self.region_by_id.items():
            children = [self.region_by_id[c] for c in region.get("children", []) if c in self.region_by_id]
            sig = make_structure_signature(region, children, self.interfaces.get(rid), self.dims_by_region.get(rid, []))
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
            bucket = buckets[sid]
            bucket["count"] += 1
            if len(bucket["example_region_ids"]) < 12:
                bucket["example_region_ids"].append(rid)
            bucket["instances"].append(self.region_summaries[rid])
        return sorted(buckets.values(), key=lambda s: (-s["count"], s["structure_type"], s["structure_id"]))

    def index(self) -> dict[str, Any]:
        type_counts = Counter(s["structure_type"] for s in self.structures)
        role_counts = Counter(s["signature"].get("pruning_role", "unknown") for s in self.structures)
        return {
            "model_name": self.tree.get("model_name", self.model),
            "source_frontend": self.tree.get("source_frontend"),
            "root_region_id": self.root_region_id,
            "num_regions": len(self.regions),
            "num_abstract_structures": len(self.structures),
            "has_region_dimension_ir": self.dim_ir is not None,
            "summary": self.tree.get("summary", {}),
            "structure_type_counts": dict(sorted(type_counts.items())),
            "pruning_role_counts": dict(sorted(role_counts.items())),
        }

    def structure_catalog(self) -> dict[str, Any]:
        return {
            "model_name": self.model,
            "structures": [
                {
                    "structure_id": s["structure_id"],
                    "structure_type": s["structure_type"],
                    "count": s["count"],
                    "pruning_role": s["signature"].get("pruning_role", "unknown"),
                    "confidence": s["signature"].get("confidence", "unknown"),
                    "dimension_roles": s["signature"].get("dimension_roles", []),
                    "constraint_types": s["signature"].get("constraint_types", []),
                    "child_type_counts": s["signature"].get("child_type_counts", {}),
                    "example_region_ids": s["example_region_ids"],
                }
                for s in self.structures
            ],
        }

    def structure_instances(self, structure_id: str, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        s = self.structures_by_id.get(structure_id)
        if not s:
            raise KeyError(f"Unknown structure_id: {structure_id}")
        instances = s["instances"][offset: offset + limit]
        return {
            "structure_id": structure_id,
            "structure_type": s["structure_type"],
            "signature": s["signature"],
            "total_instances": len(s["instances"]),
            "offset": offset,
            "limit": limit,
            "instances": instances,
        }

    def region_payload(self, region_id: str) -> dict[str, Any]:
        region = self.region_by_id.get(region_id)
        if not region:
            raise KeyError(f"Unknown region_id: {region_id}")
        return {
            "region": region,
            "summary": self.region_summaries[region_id],
            "interface": self.interfaces.get(region_id),
            "dimensions": self.dims_by_region.get(region_id, []),
            "children": [self.region_summaries[c] for c in region.get("children", []) if c in self.region_summaries],
        }

    def children(self, region_id: str) -> dict[str, Any]:
        region = self.region_by_id.get(region_id)
        if not region:
            raise KeyError(f"Unknown region_id: {region_id}")
        return {
            "region_id": region_id,
            "children": [self.region_summaries[c] for c in region.get("children", []) if c in self.region_summaries],
        }

    def search(self, q: str, limit: int = 100) -> dict[str, Any]:
        ql = q.lower()
        hits = []
        for r in self.region_summaries.values():
            blob = json.dumps(r, sort_keys=True).lower()
            if ql in blob:
                hits.append(r)
                if len(hits) >= limit:
                    break
        return {"query": q, "hits": hits, "count": len(hits)}

    def blocked(self, limit: int = 200) -> dict[str, Any]:
        hits = [
            item for item in self.region_summaries.values()
            if item.get("pruning_role") == "blocked"
        ]
        return {"count": len(hits), "limit": limit, "hits": hits[:limit]}


HTML_PATH = Path(__file__).with_name("region_structure_viewer.html")


class Handler(BaseHTTPRequestHandler):
    db: RegionDatabase

    def _send_json(self, data: Any, code: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, f"File not found: {path}")
            return
        data = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in {"/", "/index.html"}:
                self._send_file(HTML_PATH)
            elif path == "/api/index":
                self._send_json(self.db.index())
            elif path == "/api/structures":
                self._send_json(self.db.structure_catalog())
            elif path.startswith("/api/structures/") and path.endswith("/instances"):
                sid = unquote(path[len("/api/structures/") : -len("/instances")])
                offset = int(qs.get("offset", [0])[0])
                limit = int(qs.get("limit", [100])[0])
                self._send_json(self.db.structure_instances(sid, offset, limit))
            elif path.startswith("/api/region/"):
                rid = unquote(path[len("/api/region/") :])
                self._send_json(self.db.region_payload(rid))
            elif path.startswith("/api/children/"):
                rid = unquote(path[len("/api/children/") :])
                self._send_json(self.db.children(rid))
            elif path == "/api/search":
                q = qs.get("q", [""])[0]
                limit = int(qs.get("limit", [100])[0])
                self._send_json(self.db.search(q, limit))
            elif path == "/api/blocked":
                limit = int(qs.get("limit", [200])[0])
                self._send_json(self.db.blocked(limit))
            else:
                self.send_error(404, "Not found")
        except KeyError as e:
            self._send_json({"error": str(e)}, code=404)
        except Exception as e:  # useful during local research tooling
            self._send_json({"error": str(e)}, code=500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    db = RegionDatabase(args.model, Path(args.root).resolve())
    Handler.db = db
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[region-structure-api] model={args.model}")
    print(f"[region-structure-api] regions={len(db.regions)} abstract_structures={len(db.structures)}")
    print(f"[region-structure-api] open http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[region-structure-api] stopped")


if __name__ == "__main__":
    main()
