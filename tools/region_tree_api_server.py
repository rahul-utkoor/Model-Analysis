#!/usr/bin/env python3
"""
Small backend API for lazy Structural Region Tree browsing.

Run from the Model-Analysis repository root:

  python region_tree_api_server.py --model bert-base-uncased --port 8765

Then open:

  http://localhost:8765/

The frontend never receives the full region-tree JSON. It requests:
- summary/index
- one region at a time
- children summaries for a region
- optional region dimensions/interfaces

When a node is collapsed in the frontend, descendants are removed from the browser state.
They are fetched again only if the user expands the node again.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


HTML_FILE = "region_tree_api_viewer.html"


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


def summarize_region(region: dict[str, Any], interface: dict[str, Any] | None, dim_count: int) -> dict[str, Any]:
    return {
        "region_id": region.get("region_id"),
        "region_type": region.get("region_type"),
        "name": region.get("name"),
        "parent": region.get("parent"),
        "child_count": len(region.get("children") or []),
        "op_count": len(region.get("op_ids") or []),
        "value_count": len(region.get("value_ids") or []),
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


@dataclass
class ModelStore:
    repo_root: Path
    model: str
    safe_name: str
    tree_path: Path
    dim_path: Path | None
    tree: dict[str, Any]
    dim_ir: dict[str, Any] | None
    region_by_id: dict[str, dict[str, Any]]
    interface_by_region: dict[str, dict[str, Any]]
    dims_by_region: dict[str, list[dict[str, Any]]]
    summary_by_region: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, repo_root: Path, model: str) -> "ModelStore":
        safe = safe_model_name(model)
        tree_path = repo_root / "reports" / "structural_region_trees" / f"{safe}.json"
        dim_path = repo_root / "reports" / "region_dimension_ir" / f"{safe}.json"

        if not tree_path.exists():
            raise FileNotFoundError(
                f"Structural Region Tree missing: {tree_path}\n"
                f"Run: python scripts/build_structural_region_tree.py --model {model}"
            )

        tree = read_json(tree_path)
        dim_ir = read_json(dim_path) if dim_path.exists() else None

        regions = tree.get("regions") or []
        interfaces = tree.get("interfaces") or []
        region_by_id = {r.get("region_id"): r for r in regions if r.get("region_id")}
        interface_by_region = {i.get("region_id"): i for i in interfaces if i.get("region_id")}

        dims_by_region: dict[str, list[dict[str, Any]]] = {}
        if dim_ir:
            for d in dim_ir.get("dimension_variables") or []:
                rid = d.get("region_id")
                if rid:
                    dims_by_region.setdefault(rid, []).append(d)

        summary_by_region = {
            rid: summarize_region(region, interface_by_region.get(rid), len(dims_by_region.get(rid, [])))
            for rid, region in region_by_id.items()
        }

        return cls(
            repo_root=repo_root,
            model=model,
            safe_name=safe,
            tree_path=tree_path,
            dim_path=dim_path if dim_path.exists() else None,
            tree=tree,
            dim_ir=dim_ir,
            region_by_id=region_by_id,
            interface_by_region=interface_by_region,
            dims_by_region=dims_by_region,
            summary_by_region=summary_by_region,
        )

    def index(self) -> dict[str, Any]:
        return {
            "model_name": self.tree.get("model_name", self.model),
            "safe_name": self.safe_name,
            "source_frontend": self.tree.get("source_frontend"),
            "root_region_id": self.tree.get("root_region_id"),
            "summary": self.tree.get("summary", {}),
            "num_regions": len(self.region_by_id),
            "has_region_dimension_ir": self.dim_ir is not None,
            "tree_path": str(self.tree_path),
            "dim_path": str(self.dim_path) if self.dim_path else None,
        }

    def region_payload(self, region_id: str, include_details: bool = True) -> dict[str, Any]:
        region = self.region_by_id.get(region_id)
        if not region:
            raise KeyError(f"Unknown region_id: {region_id}")

        children_ids = region.get("children") or []
        children = [self.summary_by_region[cid] for cid in children_ids if cid in self.summary_by_region]

        payload = {
            "summary": self.summary_by_region.get(region_id),
            "children": children,
            "has_more_children": False,
        }
        if include_details:
            payload.update(
                {
                    "region": region,
                    "interface": self.interface_by_region.get(region_id),
                    "dimensions": self.dims_by_region.get(region_id, []),
                }
            )
        return payload

    def children_payload(self, region_id: str) -> dict[str, Any]:
        region = self.region_by_id.get(region_id)
        if not region:
            raise KeyError(f"Unknown region_id: {region_id}")
        children_ids = region.get("children") or []
        return {
            "region_id": region_id,
            "children": [self.summary_by_region[cid] for cid in children_ids if cid in self.summary_by_region],
        }

    def search(self, query: str, limit: int = 50) -> dict[str, Any]:
        q = query.lower().strip()
        hits = []
        if not q:
            return {"query": query, "hits": []}
        for rid, summary in self.summary_by_region.items():
            hay = json.dumps(summary, sort_keys=True).lower()
            if q in hay:
                hits.append(summary)
                if len(hits) >= limit:
                    break
        return {"query": query, "limit": limit, "hits": hits}

    def blocked(self, limit: int = 200) -> dict[str, Any]:
        hits = [s for s in self.summary_by_region.values() if s.get("pruning_role") == "blocked"]
        return {"count": len(hits), "limit": limit, "hits": hits[:limit]}


class ApiHandler(BaseHTTPRequestHandler):
    store: ModelStore
    html_path: Path

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_bytes(self, data: bytes, status: int = 200, content_type: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj: Any, status: int = 200) -> None:
        self.send_bytes(json_dumps(obj), status=status, content_type="application/json; charset=utf-8")

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message, "status": status}, status=status)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path in {"/", "/index.html", "/region_tree_api_viewer.html"}:
                data = self.html_path.read_bytes()
                self.send_bytes(data, content_type="text/html; charset=utf-8")
                return

            if path == "/api/index":
                self.send_json(self.store.index())
                return

            if path.startswith("/api/region/"):
                rid = unquote(path[len("/api/region/") :])
                include = qs.get("details", ["1"])[0] not in {"0", "false", "False"}
                self.send_json(self.store.region_payload(rid, include_details=include))
                return

            if path.startswith("/api/children/"):
                rid = unquote(path[len("/api/children/") :])
                self.send_json(self.store.children_payload(rid))
                return

            if path == "/api/search":
                query = qs.get("q", [""])[0]
                limit = int(qs.get("limit", ["50"])[0])
                self.send_json(self.store.search(query, limit=limit))
                return

            if path == "/api/blocked":
                limit = int(qs.get("limit", ["200"])[0])
                self.send_json(self.store.blocked(limit=limit))
                return

            if path == "/favicon.ico":
                self.send_bytes(b"", status=204, content_type="image/x-icon")
                return

            self.send_error_json(f"Not found: {path}", status=404)
        except KeyError as e:
            self.send_error_json(str(e), status=404)
        except Exception as e:  # pragma: no cover - server safety
            self.send_error_json(f"Internal server error: {e}", status=500)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lazy API backend for Structural Region Tree viewer.")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--html", default=HTML_FILE, help="Path to region_tree_api_viewer.html")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    store = ModelStore.load(repo_root, args.model)

    html_path = Path(args.html)
    if not html_path.is_absolute():
        html_path = Path(__file__).resolve().parent / html_path
    if not html_path.exists():
        raise FileNotFoundError(f"Viewer HTML not found: {html_path}")

    ApiHandler.store = store
    ApiHandler.html_path = html_path

    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    print(f"[region-tree-api] model={args.model}")
    print(f"[region-tree-api] regions={len(store.region_by_id)}")
    print(f"[region-tree-api] open: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[region-tree-api] stopping")


if __name__ == "__main__":
    main()
