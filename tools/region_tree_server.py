#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote


def load_json(path: Path):
    return json.loads(path.read_text())


def json_response(handler: BaseHTTPRequestHandler, obj, status: int = 200):
    data = json.dumps(obj, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200, content_type: str = "text/plain"):
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", f"{content_type}; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def error_response(handler: BaseHTTPRequestHandler, msg: str, status: int = 400):
    json_response(handler, {"error": msg, "status": status}, status=status)


def safe_contains(obj, q: str) -> bool:
    return q.lower() in json.dumps(obj, sort_keys=True).lower()


class RegionTreeStore:
    def __init__(self, model: str, tree_path: Path, dim_path: Path | None):
        self.model = model
        self.tree_path = tree_path
        self.dim_path = dim_path

        self.tree = load_json(tree_path)
        self.dim_ir = load_json(dim_path) if dim_path and dim_path.exists() else None

        self.regions = self.tree.get("regions", [])
        self.interfaces = self.tree.get("interfaces", [])

        self.region_by_id = {r.get("region_id"): r for r in self.regions}
        self.interface_by_region = {i.get("region_id"): i for i in self.interfaces}

        self.dims_by_region: dict[str, list[dict]] = {}
        if self.dim_ir:
            for d in self.dim_ir.get("dimension_variables", []):
                rid = d.get("region_id")
                self.dims_by_region.setdefault(rid, []).append(d)

        self.children_by_parent: dict[str | None, list[str]] = {}
        for r in self.regions:
            parent = r.get("parent")
            self.children_by_parent.setdefault(parent, []).append(r.get("region_id"))

        for k in self.children_by_parent:
            self.children_by_parent[k] = sorted(
                self.children_by_parent[k],
                key=lambda rid: (
                    self.region_by_id.get(rid, {}).get("region_type", ""),
                    self.region_by_id.get(rid, {}).get("region_id", ""),
                ),
            )

        self.root_id = self.tree.get("root_region_id")
        if not self.root_id:
            roots = self.children_by_parent.get(None, [])
            self.root_id = roots[0] if roots else None

    def region_summary(self, rid: str) -> dict:
        r = self.region_by_id.get(rid)
        if not r:
            raise KeyError(rid)

        iface = self.interface_by_region.get(rid) or {}
        dims = self.dims_by_region.get(rid, [])
        children = r.get("children", []) or self.children_by_parent.get(rid, [])

        return {
            "region_id": r.get("region_id"),
            "region_type": r.get("region_type"),
            "name": r.get("name"),
            "parent": r.get("parent"),
            "child_count": len(children),
            "op_count": len(r.get("op_ids", [])),
            "value_count": len(r.get("value_ids", [])),
            "contains_fork": r.get("contains_fork"),
            "contains_join": r.get("contains_join"),
            "single_entry": r.get("single_entry"),
            "single_exit": r.get("single_exit"),
            "confidence": r.get("confidence"),
            "reason": r.get("reason"),
            "pruning_role": iface.get("pruning_role", "unknown"),
            "dim_count": len(dims),
        }

    def index(self) -> dict:
        root_summary = self.region_summary(self.root_id) if self.root_id else None
        return {
            "model_name": self.tree.get("model_name", self.model),
            "source_frontend": self.tree.get("source_frontend"),
            "root_region_id": self.root_id,
            "summary": self.tree.get("summary", {}),
            "num_regions": len(self.regions),
            "has_region_dimension_ir": self.dim_ir is not None,
            "root": root_summary,
        }

    def region_detail(self, rid: str) -> dict:
        r = self.region_by_id.get(rid)
        if not r:
            raise KeyError(rid)

        return {
            "summary": self.region_summary(rid),
            "region": r,
            "interface": self.interface_by_region.get(rid),
            "dimensions": self.dims_by_region.get(rid, []),
        }

    def children(self, rid: str) -> dict:
        if rid not in self.region_by_id:
            raise KeyError(rid)

        child_ids = self.region_by_id[rid].get("children", []) or self.children_by_parent.get(rid, [])
        return {
            "region_id": rid,
            "children": [self.region_summary(cid) for cid in child_ids if cid in self.region_by_id],
        }

    def search(self, q: str, limit: int = 50) -> dict:
        q = q.strip()
        if not q:
            return {"query": q, "matches": []}

        matches = []
        for r in self.regions:
            rid = r.get("region_id")
            summary = self.region_summary(rid)
            iface = self.interface_by_region.get(rid) or {}
            dims = self.dims_by_region.get(rid, [])

            haystack = {
                "summary": summary,
                "interface": iface,
                "dims": dims[:5],
            }

            if safe_contains(haystack, q):
                matches.append(summary)

            if len(matches) >= limit:
                break

        return {
            "query": q,
            "limit": limit,
            "matches": matches,
        }


def make_handler(store: RegionTreeStore, html_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), fmt % args))

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            try:
                if path in {"/", "/index.html"}:
                    if not html_path.exists():
                        return error_response(self, f"Viewer HTML not found: {html_path}", 500)
                    return text_response(self, html_path.read_text(), content_type="text/html")

                if path == "/api/index":
                    return json_response(self, store.index())

                if path == "/api/region":
                    rid = qs.get("id", [None])[0]
                    if not rid:
                        return error_response(self, "Missing required query parameter: id")
                    return json_response(self, store.region_detail(unquote(rid)))

                if path == "/api/children":
                    rid = qs.get("id", [None])[0]
                    if not rid:
                        return error_response(self, "Missing required query parameter: id")
                    return json_response(self, store.children(unquote(rid)))

                if path == "/api/search":
                    q = qs.get("q", [""])[0]
                    limit = int(qs.get("limit", ["50"])[0])
                    return json_response(self, store.search(q, limit=limit))

                if path == "/api/health":
                    return json_response(self, {"ok": True, "model": store.model})

                return error_response(self, f"Unknown path: {path}", 404)

            except KeyError as e:
                return error_response(self, f"Unknown region id: {e}", 404)
            except Exception as e:
                return error_response(self, repr(e), 500)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Region Tree API server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree", default=None, help="Path to structural region tree JSON")
    ap.add_argument("--dim-ir", default=None, help="Optional path to region dimension IR JSON")
    ap.add_argument("--html", default="tools/region_tree_api_viewer.html")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    tree_path = Path(args.tree or f"reports/structural_region_trees/{args.model}.json")
    dim_path = Path(args.dim_ir or f"reports/region_dimension_ir/{args.model}.json")
    html_path = Path(args.html)

    if not tree_path.exists():
        raise FileNotFoundError(
            f"Structural Region Tree missing: {tree_path}\n"
            f"Run: python scripts/build_structural_region_tree.py --model {args.model}"
        )

    store = RegionTreeStore(args.model, tree_path, dim_path if dim_path.exists() else None)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, html_path))

    print(f"[region-tree-server] model={args.model}")
    print(f"[region-tree-server] tree={tree_path}")
    print(f"[region-tree-server] dim_ir={dim_path if dim_path.exists() else 'missing'}")
    print(f"[region-tree-server] open http://{args.host}:{args.port}/")
    print("[region-tree-server] API:")
    print("  GET /api/index")
    print("  GET /api/region?id=<region_id>")
    print("  GET /api/children?id=<region_id>")
    print("  GET /api/search?q=<query>&limit=50")

    server.serve_forever()


if __name__ == "__main__":
    main()
