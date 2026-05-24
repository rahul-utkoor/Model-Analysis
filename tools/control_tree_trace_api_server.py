#!/usr/bin/env python3
"""Lazy API backend for stepwise control-tree trace browsing."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


HTML_FILE = "control_tree_trace_viewer.html"


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


def _int_arg(values: list[str] | None, default: int) -> int:
    if not values:
        return default
    try:
        return int(values[0])
    except (TypeError, ValueError):
        return default


def make_step_summary(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_id": step.get("step_id"),
        "step_index": step.get("step_index"),
        "pass_name": step.get("pass_name"),
        "action": step.get("action"),
        "created_region_id": step.get("created_region_id"),
        "created_region_type": step.get("created_region_type"),
        "collapsed_node_count": len(step.get("collapsed_node_ids", []) or []),
        "collapsed_op_count": len(step.get("collapsed_op_ids", []) or []),
        "collapsed_region_count": len(step.get("collapsed_region_ids", []) or []),
        "confidence": step.get("confidence"),
        "reason": step.get("reason"),
        "before_active_node_count": (step.get("before_summary") or {}).get("num_active_nodes"),
        "after_active_node_count": (step.get("after_summary") or {}).get("num_active_nodes"),
    }


def step_without_snapshot(step: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in step.items() if key != "graph_snapshot"}
    snapshot = step.get("graph_snapshot", {}) or {}
    payload["graph_snapshot_summary"] = {
        "active_node_count": snapshot.get("active_node_count"),
        "returned_node_count": snapshot.get("returned_node_count"),
        "edge_count": len(snapshot.get("edges", []) or []),
        "truncated": snapshot.get("truncated", False),
    }
    return payload


def filter_step_summaries(
    steps: list[dict[str, Any]],
    *,
    pass_name: str | None = None,
    action: str | None = None,
    region_type: str | None = None,
    q: str | None = None,
    hide_skips: bool = False,
) -> list[dict[str, Any]]:
    query = (q or "").strip().lower()
    out: list[dict[str, Any]] = []
    for step in steps:
        summary = make_step_summary(step)
        if pass_name and summary.get("pass_name") != pass_name:
            continue
        if action and summary.get("action") != action:
            continue
        if region_type and summary.get("created_region_type") != region_type:
            continue
        if hide_skips and summary.get("action") == "skip":
            continue
        if query and query not in json.dumps(summary, sort_keys=True).lower():
            continue
        out.append(summary)
    return out


def paginate(items: list[dict[str, Any]], offset: int, limit: int) -> dict[str, Any]:
    offset = max(0, offset)
    limit = max(1, min(limit, 500))
    return {
        "offset": offset,
        "limit": limit,
        "total": len(items),
        "items": items[offset: offset + limit],
    }


def _node_payload(node: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_kind": "created_region" if role == "created" else node.get("node_kind", "boundary"),
        "label": node.get("label") or node.get("node_id"),
        "region_type": node.get("region_type"),
        "op_type": node.get("op_type"),
        "canonical_op_type": node.get("canonical_op_type"),
        "role": role,
        "confidence": node.get("confidence"),
        "pruning_role": node.get("pruning_role"),
        "metadata": node.get("metadata", {}),
    }


def _synthetic_node(node_id: str, role: str, *, label: str | None = None) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_kind": "boundary" if "boundary" in role else "abstract_region",
        "label": label or node_id,
        "region_type": None,
        "op_type": None,
        "canonical_op_type": None,
        "role": role,
        "confidence": None,
        "pruning_role": None,
        "metadata": {"synthetic": True},
    }


def _edge_payload(edge: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "src": edge.get("src"),
        "dst": edge.get("dst"),
        "edge_kind": edge.get("edge_kind", "dataflow"),
        "label": edge.get("label") or edge.get("tensor_or_value_id"),
        "role": role,
        "metadata": edge.get("metadata", {}),
    }


def build_local_step_graph(step: dict[str, Any], radius: int = 1) -> dict[str, Any]:
    """Extract a small, browser-safe neighborhood for one trace step."""
    snapshot = step.get("graph_snapshot", {}) or {}
    nodes_by_id = {node.get("node_id"): node for node in snapshot.get("nodes", []) if node.get("node_id")}
    edges = snapshot.get("edges", []) or []
    created_id = step.get("created_region_id")
    collapsed_ids = list(step.get("collapsed_node_ids", []) or [])
    selected_ids: set[str] = set()
    local_nodes: dict[str, dict[str, Any]] = {}
    local_edges: list[dict[str, Any]] = []

    if step.get("action") in {"initialize", "finalize"}:
        return {
            "step_id": step.get("step_id"),
            "step_index": step.get("step_index"),
            "mode": "summary",
            "nodes": [],
            "edges": [],
            "summary": {
                "before": step.get("before_summary", {}),
                "after": step.get("after_summary", {}),
                "reason": step.get("reason"),
            },
        }

    if created_id:
        selected_ids.add(created_id)
        if created_id in nodes_by_id:
            local_nodes[created_id] = _node_payload(nodes_by_id[created_id], "created")
        else:
            local_nodes[created_id] = _synthetic_node(created_id, "created", label=step.get("created_region_type"))

    for node_id in collapsed_ids:
        selected_ids.add(node_id)
        if node_id in nodes_by_id:
            local_nodes[node_id] = _node_payload(nodes_by_id[node_id], "collapsed")
        else:
            local_nodes[node_id] = _synthetic_node(node_id, "collapsed")
        if created_id:
            local_edges.append(
                {
                    "src": node_id,
                    "dst": created_id,
                    "edge_kind": "abstraction",
                    "label": "collapsed_into",
                    "role": "abstraction",
                    "metadata": {"synthetic": True},
                }
            )

    if step.get("action") == "skip" and not collapsed_ids:
        for op_id in (step.get("collapsed_op_ids") or [])[:20]:
            node_id = f"candidate::{op_id}"
            selected_ids.add(node_id)
            local_nodes[node_id] = _synthetic_node(node_id, "context", label=op_id)

    for edge in edges:
        src = edge.get("src")
        dst = edge.get("dst")
        if not src or not dst:
            continue
        if created_id and dst == created_id:
            selected_ids.update({src, dst})
            if src not in local_nodes:
                local_nodes[src] = _node_payload(nodes_by_id[src], "incoming_boundary") if src in nodes_by_id else _synthetic_node(src, "incoming_boundary")
            local_edges.append(_edge_payload(edge, "incoming"))
        elif created_id and src == created_id:
            selected_ids.update({src, dst})
            if dst not in local_nodes:
                local_nodes[dst] = _node_payload(nodes_by_id[dst], "outgoing_boundary") if dst in nodes_by_id else _synthetic_node(dst, "outgoing_boundary")
            local_edges.append(_edge_payload(edge, "outgoing"))
        elif src in selected_ids and dst in selected_ids:
            local_edges.append(_edge_payload(edge, "internal"))

    return {
        "step_id": step.get("step_id"),
        "step_index": step.get("step_index"),
        "mode": "local",
        "radius": radius,
        "nodes": sorted(local_nodes.values(), key=lambda item: (item.get("role", ""), item.get("node_id", ""))),
        "edges": local_edges,
        "summary": {
            "created_region_id": created_id,
            "created_region_type": step.get("created_region_type"),
            "collapsed_node_count": len(collapsed_ids),
            "action": step.get("action"),
            "reason": step.get("reason"),
        },
    }


def find_matching_step(
    steps: list[dict[str, Any]],
    *,
    step_index: int,
    direction: str,
    pass_name: str | None = None,
    action: str | None = None,
    region_type: str | None = None,
) -> dict[str, Any] | None:
    candidates = steps if direction == "next" else list(reversed(steps))
    for step in candidates:
        idx = int(step.get("step_index", -1))
        if direction == "next" and idx <= step_index:
            continue
        if direction == "prev" and idx >= step_index:
            continue
        summary = make_step_summary(step)
        if pass_name and summary.get("pass_name") != pass_name:
            continue
        if action and summary.get("action") != action:
            continue
        if region_type and summary.get("created_region_type") != region_type:
            continue
        return summary
    return None


class TraceStore:
    def __init__(self, model: str, trace_path: Path, summary_path: Path | None = None):
        self.model = model
        self.safe_name = safe_model_name(model)
        self.trace_path = trace_path
        self.summary_path = summary_path
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Control-tree trace missing: {trace_path}\n"
                f"Run: python scripts/build_control_tree_trace.py --model {model}"
            )
        self.trace = read_json(trace_path)
        self.summary_report = read_json(summary_path) if summary_path and summary_path.exists() else None
        self.steps = self.trace.get("steps", []) or []
        self.step_by_id = {step.get("step_id"): step for step in self.steps if step.get("step_id")}

    @classmethod
    def load(cls, repo_root: Path, model: str, trace_json: str | None = None) -> "TraceStore":
        safe = safe_model_name(model)
        trace_path = Path(trace_json) if trace_json else repo_root / "reports" / "control_tree_steps" / f"{safe}.json"
        if not trace_path.is_absolute():
            trace_path = (repo_root / trace_path).resolve()
        summary_path = repo_root / "reports" / "control_tree_step_summaries" / f"{safe}.json"
        return cls(model, trace_path, summary_path if summary_path.exists() else None)

    def index(self) -> dict[str, Any]:
        summary = self.trace.get("summary", {}) or {}
        return {
            "model_name": self.trace.get("model_name", self.model),
            "safe_name": self.safe_name,
            "source_frontend": self.trace.get("source_frontend", "unknown"),
            "num_steps": summary.get("num_steps", len(self.steps)),
            "num_collapse_steps": summary.get("num_collapse_steps", 0),
            "num_skip_steps": summary.get("num_skip_steps", 0),
            "initial_tensor_op_count": summary.get("initial_tensor_op_count", 0),
            "final_active_node_count": summary.get("final_active_node_count", 0),
            "pass_name_counts": summary.get("pass_name_counts", {}),
            "created_region_type_counts": summary.get("created_region_type_counts", {}),
            "trace_path": str(self.trace_path),
        }

    def steps_page(self, query: dict[str, list[str]]) -> dict[str, Any]:
        filtered = filter_step_summaries(
            self.steps,
            pass_name=(query.get("pass_name") or [""])[0] or None,
            action=(query.get("action") or [""])[0] or None,
            region_type=(query.get("region_type") or [""])[0] or None,
            q=(query.get("q") or [""])[0] or None,
            hide_skips=(query.get("hide_skips") or [""])[0].lower() in {"1", "true", "yes"},
        )
        return paginate(filtered, _int_arg(query.get("offset"), 0), _int_arg(query.get("limit"), 100))


class ApiHandler(BaseHTTPRequestHandler):
    store: TraceStore
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

            if path in {"/", "/index.html", "/control_tree_trace_viewer.html"}:
                self.send_bytes(self.html_path.read_bytes(), content_type="text/html; charset=utf-8")
                return
            if path == "/api/trace/index":
                self.send_json(self.store.index())
                return
            if path == "/api/trace/steps":
                self.send_json(self.store.steps_page(qs))
                return
            if path.startswith("/api/trace/steps/") and path.endswith("/local-graph"):
                sid = unquote(path[len("/api/trace/steps/"): -len("/local-graph")])
                step = self.store.step_by_id.get(sid)
                if not step:
                    raise KeyError(f"Unknown step_id: {sid}")
                self.send_json(build_local_step_graph(step, radius=_int_arg(qs.get("radius"), 1)))
                return
            if path.startswith("/api/trace/steps/"):
                sid = unquote(path[len("/api/trace/steps/"):])
                step = self.store.step_by_id.get(sid)
                if not step:
                    raise KeyError(f"Unknown step_id: {sid}")
                self.send_json(step_without_snapshot(step))
                return
            if path == "/api/trace/search":
                q = (qs.get("q") or [""])[0]
                self.send_json({"query": q, **paginate(filter_step_summaries(self.store.steps, q=q), 0, _int_arg(qs.get("limit"), 50))})
                return
            if path in {"/api/trace/next", "/api/trace/prev"}:
                direction = "next" if path.endswith("/next") else "prev"
                match = find_matching_step(
                    self.store.steps,
                    step_index=_int_arg(qs.get("step_index"), -1),
                    direction=direction,
                    pass_name=(qs.get("pass_name") or [""])[0] or None,
                    action=(qs.get("action") or [""])[0] or None,
                    region_type=(qs.get("region_type") or [""])[0] or None,
                )
                self.send_json({"match": match})
                return
            if path == "/favicon.ico":
                self.send_bytes(b"", status=204, content_type="image/x-icon")
                return
            self.send_error_json(f"Not found: {path}", status=404)
        except KeyError as exc:
            self.send_error_json(str(exc), status=404)
        except Exception as exc:  # pragma: no cover - server safety
            self.send_error_json(f"Internal server error: {exc}", status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lazy API backend for control-tree trace viewer.")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--trace-json", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--viewer", default=HTML_FILE)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    store = TraceStore.load(repo_root, args.model, trace_json=args.trace_json)
    viewer = Path(args.viewer)
    if not viewer.is_absolute():
        viewer = Path(__file__).resolve().parent / viewer
    if not viewer.exists():
        raise FileNotFoundError(f"Viewer HTML not found: {viewer}")

    ApiHandler.store = store
    ApiHandler.html_path = viewer
    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    print(f"[control-tree-trace-api] model={store.index()['model_name']}")
    print(f"[control-tree-trace-api] steps={len(store.steps)}")
    print(f"[control-tree-trace-api] open: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[control-tree-trace-api] stopping")


if __name__ == "__main__":
    main()
