#!/usr/bin/env python3
"""Lazy API backend for an ordered Structural Region Tree browser."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


HTML_FILE = "ordered_control_tree_viewer.html"


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


def _clean_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^/model/", "", text)
    text = re.sub(r"^model_bert_", "", text)
    text = text.replace("node::op::", "")
    text = text.replace("region::", "")
    text = text.replace("op::", "")
    text = text.replace("/", ".")
    text = re.sub(r"bert\.encoder\.layer\.(\d+)", r"layer\1", text)
    text = re.sub(r"encoder\.layer\.(\d+)", r"layer\1", text)
    text = re.sub(r"\.{2,}", ".", text).strip(".")
    return text or str(value or "")


def _split_words(value: str) -> str:
    text = re.sub(r"(?<!^)([A-Z])", r" \1", value).replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def _numeric_suffix(value: Any) -> int | None:
    nums = re.findall(r"\d+", str(value or ""))
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def humanize_region_type(region_type: str | None, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    if region_type == "ActivationRegion" and metadata.get("activation_kind") == "gelu":
        return "GELU Activation"
    return {
        "ModelRegion": "Model",
        "FeedForwardRegion": "Feed-Forward Block",
        "AttentionSkeletonRegion": "Attention Skeleton",
        "ResidualMergeRegion": "Residual Merge",
        "AxisTransformRegion": "Axis / Shape Transform",
        "LinearProjectionRegion": "Linear Projection",
        "ActivationRegion": "Activation",
        "LayerNormRegion": "LayerNorm",
        "PrimitiveRegion": "Primitive Op",
        "BiasAddRegion": "Bias Add",
        "ForkRegion": "Fork Region",
        "JoinRegion": "Join Region",
        "ProperAcyclicRegion": "Acyclic Region",
    }.get(region_type or "", _split_words(region_type or "Unknown Region"))


def _op_order_map(tensor_ir: dict[str, Any] | None) -> dict[str, int]:
    if not tensor_ir:
        return {}
    return {op.get("op_id", ""): index for index, op in enumerate(tensor_ir.get("ops", []) or [])}


def _ops_by_id(tensor_ir: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not tensor_ir:
        return {}
    return {op.get("op_id", ""): op for op in tensor_ir.get("ops", []) or []}


def compute_region_topological_key(region: dict[str, Any], tensor_ir: dict[str, Any] | None = None) -> tuple[int, int | str, str]:
    metadata = region.get("metadata", {}) or {}
    for key in ("order", "region_order", "source_order", "topological_order", "first_op_index", "min_op_index"):
        value = metadata.get(key)
        if isinstance(value, int):
            return (0, value, region.get("region_id", ""))

    order = _op_order_map(tensor_ir)
    op_ids = region.get("op_ids", []) or []
    indexed = [order[op_id] for op_id in op_ids if op_id in order]
    if indexed:
        return (1, min(indexed), region.get("region_id", ""))

    numeric = [_numeric_suffix(op_id) for op_id in op_ids]
    numeric = [item for item in numeric if item is not None]
    if numeric:
        return (2, min(numeric), region.get("region_id", ""))

    if op_ids:
        return (3, str(op_ids[0]), region.get("region_id", ""))
    return (4, region.get("region_id", ""), region.get("region_id", ""))


def source_op_stats(region: dict[str, Any], tensor_ir: dict[str, Any] | None = None) -> dict[str, Any]:
    op_ids = region.get("op_ids", []) or []
    order = _op_order_map(tensor_ir)
    indexed = [(order[op_id], op_id) for op_id in op_ids if op_id in order]
    if indexed:
        indexed.sort()
        first_idx, first_op = indexed[0]
        last_idx, last_op = indexed[-1]
        return {
            "topological_order": first_idx,
            "source_op_first": first_op,
            "source_op_last": last_op,
            "source_op_range": f"{first_idx}-{last_idx}" if first_idx != last_idx else str(first_idx),
            "source_op_count": len(op_ids),
        }
    nums = [item for item in (_numeric_suffix(op_id) for op_id in op_ids) if item is not None]
    if nums:
        return {
            "topological_order": min(nums),
            "source_op_first": op_ids[0] if op_ids else None,
            "source_op_last": op_ids[-1] if op_ids else None,
            "source_op_range": f"{min(nums)}-{max(nums)}" if min(nums) != max(nums) else str(min(nums)),
            "source_op_count": len(op_ids),
        }
    return {
        "topological_order": None,
        "source_op_first": op_ids[0] if op_ids else None,
        "source_op_last": op_ids[-1] if op_ids else None,
        "source_op_range": "n/a",
        "source_op_count": len(op_ids),
    }


def humanize_source_op(op_id: str, tensor_ir: dict[str, Any] | None = None) -> str:
    op = _ops_by_id(tensor_ir).get(op_id, {})
    raw = (
        op.get("source_node_name")
        or op.get("name")
        or op.get("op_type")
        or op.get("canonical_op_type")
        or op_id
    )
    cleaned = _clean_name(raw)
    op_type = op.get("op_type") or op.get("canonical_op_type")
    if op_type and str(op_type).lower() not in cleaned.lower():
        cleaned = f"{cleaned}.{op_type}"
    return cleaned


def _source_names(region: dict[str, Any], tensor_ir: dict[str, Any] | None, limit: int = 12) -> list[str]:
    return [humanize_source_op(op_id, tensor_ir) for op_id in (region.get("op_ids", []) or [])[:limit]]


def _dimension_summary(dimensions: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({dim.get("dim_name") for dim in dimensions if dim.get("dim_name")})
    roles = sorted({dim.get("axis_role") for dim in dimensions if dim.get("axis_role")})
    return {
        "count": len(dimensions),
        "dim_names": names,
        "axis_roles": roles,
        "prunable_count": sum(1 for dim in dimensions if dim.get("prunable")),
        "blocked_count": sum(1 for dim in dimensions if dim.get("blocked")),
        "protected_count": sum(1 for dim in dimensions if dim.get("protected")),
    }


def _teaching(region_type: str | None, metadata: dict[str, Any] | None = None) -> tuple[str, str]:
    if region_type == "FeedForwardRegion":
        return (
            "Compound region built from already recognized subregions.",
            "Exposes intermediate_dim pruning with same-index propagation.",
        )
    if region_type == "ResidualMergeRegion":
        return (
            "Join region in a dataflow graph.",
            "Hidden dimensions are protected because branches must agree.",
        )
    if region_type == "AttentionSkeletonRegion":
        return (
            "High-level dataflow idiom recovered from MatMul/Softmax/MatMul.",
            "Head pruning requires axis/head mapping before execution.",
        )
    if region_type == "AxisTransformRegion":
        return (
            "Index/axis transformation region.",
            "Pruning index sets must be remapped through reshape/transpose.",
        )
    if region_type == "LinearProjectionRegion":
        return (
            "Local arithmetic idiom collapsed into a semantic op.",
            "Output feature dimension may be prunable.",
        )
    if region_type == "ActivationRegion":
        return (
            "Semantic idiom recovery for an elementwise activation.",
            "Activation recognition enables feed-forward block recovery.",
        )
    if region_type == "PrimitiveRegion":
        return (
            "Leaf operation, like an instruction/basic block element.",
            "Primitive leaves provide source evidence for higher regions.",
        )
    return (
        "Structural region in the final dataflow control tree.",
        "The region may carry propagation, protection, or blocking evidence.",
    )


def _layer_prefix(source_names: list[str]) -> str | None:
    joined = " ".join(source_names)
    match = re.search(r"layer(\d+)", joined)
    if match:
        return f"Layer {match.group(1)}"
    return None


def display_title(region: dict[str, Any], tensor_ir: dict[str, Any] | None = None) -> str:
    region_type = region.get("region_type")
    metadata = region.get("metadata", {}) or {}
    base = humanize_region_type(region_type, metadata)
    source_names = _source_names(region, tensor_ir, limit=6)
    text = " ".join(source_names).lower()
    layer = _layer_prefix(source_names)
    if region_type == "ModelRegion":
        return "Model"
    if region_type == "PrimitiveRegion":
        if source_names:
            op = _ops_by_id(tensor_ir).get((region.get("op_ids") or [""])[0], {})
            return f"Primitive: {op.get('op_type') or op.get('canonical_op_type') or source_names[0]}"
        return "Primitive Op"
    if region_type == "LinearProjectionRegion":
        qualifier = ""
        if "query" in text:
            qualifier = " Query"
        elif "key" in text:
            qualifier = " Key"
        elif "value" in text:
            qualifier = " Value"
        elif "intermediate" in text:
            qualifier = " Intermediate"
        elif "output" in text:
            qualifier = " Output"
        return f"{layer + ' ' if layer else ''}{qualifier.strip() + ' ' if qualifier else ''}Linear Projection".strip()
    if region_type in {"FeedForwardRegion", "AttentionSkeletonRegion", "ResidualMergeRegion", "ActivationRegion"}:
        return f"{layer + ' ' if layer else ''}{base}".strip()
    return base


def display_subtitle(
    region: dict[str, Any],
    interface: dict[str, Any] | None,
    dimensions: list[dict[str, Any]],
    tensor_ir: dict[str, Any] | None = None,
) -> str:
    stats = source_op_stats(region, tensor_ir)
    role = (interface or {}).get("pruning_role", "unknown")
    dim_summary = _dimension_summary(dimensions)
    source = " + ".join(_source_names(region, tensor_ir, limit=2))
    pieces = []
    if stats["source_op_range"] != "n/a":
        pieces.append(f"ops {stats['source_op_range']}")
    if source:
        pieces.append(source)
    pieces.append(f"role: {role}")
    if dim_summary["blocked_count"]:
        pieces.append(f"{dim_summary['blocked_count']} blocked dims")
    elif dim_summary["prunable_count"]:
        pieces.append(f"{dim_summary['prunable_count']} prunable dims")
    return " · ".join(pieces)


class OrderedTreeStore:
    def __init__(
        self,
        model: str,
        tree: dict[str, Any],
        *,
        tensor_ir: dict[str, Any] | None = None,
        dim_ir: dict[str, Any] | None = None,
        tree_path: Path | None = None,
        tensor_path: Path | None = None,
        dim_path: Path | None = None,
    ):
        self.model = model
        self.tree = tree
        self.tensor_ir = tensor_ir
        self.dim_ir = dim_ir
        self.tree_path = tree_path
        self.tensor_path = tensor_path
        self.dim_path = dim_path
        self.regions = tree.get("regions", []) or []
        self.region_by_id = {region.get("region_id"): region for region in self.regions if region.get("region_id")}
        self.children_by_parent: dict[str | None, list[str]] = defaultdict(list)
        for region in self.regions:
            self.children_by_parent[region.get("parent")].append(region.get("region_id"))
        self.interface_by_region = {
            item.get("region_id"): item
            for item in tree.get("interfaces", []) or []
            if item.get("region_id")
        }
        self.dims_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if dim_ir:
            for dim in dim_ir.get("dimension_variables", []) or []:
                if dim.get("region_id"):
                    self.dims_by_region[dim["region_id"]].append(dim)
        self.root_region_id = tree.get("root_region_id") or self._infer_root()

    @classmethod
    def load(cls, repo_root: Path, model: str) -> "OrderedTreeStore":
        safe = safe_model_name(model)
        tree_path = repo_root / "reports" / "structural_region_trees" / f"{safe}.json"
        tensor_path = repo_root / "reports" / "tensor_ir" / f"{safe}.json"
        dim_path = repo_root / "reports" / "region_dimension_ir" / f"{safe}.json"
        if not tree_path.exists():
            raise FileNotFoundError(
                f"Structural Region Tree missing: {tree_path}\n"
                f"Run: python scripts/build_structural_region_tree.py --model {model}"
            )
        return cls(
            model,
            read_json(tree_path),
            tensor_ir=read_json(tensor_path) if tensor_path.exists() else None,
            dim_ir=read_json(dim_path) if dim_path.exists() else None,
            tree_path=tree_path,
            tensor_path=tensor_path if tensor_path.exists() else None,
            dim_path=dim_path if dim_path.exists() else None,
        )

    def _infer_root(self) -> str | None:
        roots = [region.get("region_id") for region in self.regions if region.get("parent") is None]
        return roots[0] if roots else (self.regions[0].get("region_id") if self.regions else None)

    def ordered_child_ids(self, region_id: str) -> list[str]:
        region = self.region_by_id.get(region_id, {})
        child_ids = region.get("children") if isinstance(region.get("children"), list) else self.children_by_parent.get(region_id, [])
        return sorted(
            [child_id for child_id in child_ids if child_id in self.region_by_id],
            key=lambda child_id: compute_region_topological_key(self.region_by_id[child_id], self.tensor_ir),
        )

    def region_summary(self, region_id: str) -> dict[str, Any]:
        region = self.region_by_id[region_id]
        interface = self.interface_by_region.get(region_id)
        dimensions = self.dims_by_region.get(region_id, [])
        stats = source_op_stats(region, self.tensor_ir)
        compiler_analogy, pruning_relevance = _teaching(region.get("region_type"), region.get("metadata"))
        child_ids = self.ordered_child_ids(region_id)
        source_names = _source_names(region, self.tensor_ir, limit=10)
        dim_summary = _dimension_summary(dimensions)
        return {
            "region_id": region_id,
            "region_type": region.get("region_type"),
            "display_title": display_title(region, self.tensor_ir),
            "display_subtitle": display_subtitle(region, interface, dimensions, self.tensor_ir),
            "technical_label": region_id,
            "topological_order": stats["topological_order"],
            "source_op_first": stats["source_op_first"],
            "source_op_last": stats["source_op_last"],
            "source_op_range": stats["source_op_range"],
            "source_op_count": stats["source_op_count"],
            "source_op_names": source_names,
            "pruning_role": (interface or {}).get("pruning_role", "unknown"),
            "confidence": region.get("confidence", "unknown"),
            "child_count": len(child_ids),
            "leaf": not child_ids,
            "has_dimensions": bool(dimensions),
            "dimension_summary": dim_summary,
            "reason": region.get("reason", ""),
            "compiler_analogy": compiler_analogy,
            "pruning_relevance": pruning_relevance,
        }

    def index(self) -> dict[str, Any]:
        role_counts = Counter((self.interface_by_region.get(rid) or {}).get("pruning_role", "unknown") for rid in self.region_by_id)
        return {
            "model_name": self.tree.get("model_name", self.model),
            "source_frontend": self.tree.get("source_frontend", "unknown"),
            "root_region_id": self.root_region_id,
            "num_regions": len(self.region_by_id),
            "summary": self.tree.get("summary", {}),
            "region_type_counts": self.tree.get("summary", {}).get("region_type_counts", {}),
            "pruning_role_counts": dict(sorted(role_counts.items())),
            "has_tensor_ir": self.tensor_ir is not None,
            "has_region_dimension_ir": self.dim_ir is not None,
        }

    def node_payload(self, region_id: str) -> dict[str, Any]:
        if region_id not in self.region_by_id:
            raise KeyError(f"Unknown region_id: {region_id}")
        region = self.region_by_id[region_id]
        return {
            "region": region,
            "display": self.region_summary(region_id),
            "interface": self.interface_by_region.get(region_id),
            "dimensions": self.dims_by_region.get(region_id, []),
            "source_ops": [
                {"op_id": op_id, "display_name": humanize_source_op(op_id, self.tensor_ir)}
                for op_id in region.get("op_ids", []) or []
            ],
            "children_summary": [self.region_summary(child_id) for child_id in self.ordered_child_ids(region_id)],
        }

    def children_payload(self, region_id: str) -> dict[str, Any]:
        if region_id not in self.region_by_id:
            raise KeyError(f"Unknown region_id: {region_id}")
        return {
            "parent_region_id": region_id,
            "children": [self.region_summary(child_id) for child_id in self.ordered_child_ids(region_id)],
        }

    def path_to_root(self, region_id: str) -> list[dict[str, Any]]:
        if region_id not in self.region_by_id:
            raise KeyError(f"Unknown region_id: {region_id}")
        path: list[str] = []
        seen: set[str] = set()
        current = region_id
        while current and current in self.region_by_id and current not in seen:
            seen.add(current)
            path.append(current)
            current = self.region_by_id[current].get("parent")
        path.reverse()
        return [self.region_summary(item) for item in path]

    def leaf_ops(self, region_id: str) -> list[dict[str, Any]]:
        if region_id not in self.region_by_id:
            raise KeyError(f"Unknown region_id: {region_id}")
        leaves: list[str] = []

        def walk(rid: str) -> None:
            children = self.ordered_child_ids(rid)
            if not children or self.region_by_id[rid].get("region_type") == "PrimitiveRegion":
                leaves.append(rid)
                return
            for child in children:
                walk(child)

        walk(region_id)
        return sorted([self.region_summary(rid) for rid in leaves], key=lambda item: (item["topological_order"] is None, item["topological_order"], item["region_id"]))

    def search(self, query: str, limit: int = 100) -> dict[str, Any]:
        q = query.lower().strip()
        hits: list[dict[str, Any]] = []
        if not q:
            return {"query": query, "hits": []}
        ordered = sorted(self.region_by_id, key=lambda rid: compute_region_topological_key(self.region_by_id[rid], self.tensor_ir))
        for rid in ordered:
            summary = self.region_summary(rid)
            hay = json.dumps(summary, sort_keys=True).lower()
            if q in hay:
                hits.append({**summary, "path": self.path_to_root(rid)})
                if len(hits) >= limit:
                    break
        return {"query": query, "limit": limit, "hits": hits}


class ApiHandler(BaseHTTPRequestHandler):
    store: OrderedTreeStore
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
            if path in {"/", "/index.html", "/ordered_control_tree_viewer.html"}:
                self.send_bytes(self.html_path.read_bytes(), content_type="text/html; charset=utf-8")
                return
            if path == "/api/tree/index":
                self.send_json(self.store.index())
                return
            if path.startswith("/api/tree/node/"):
                rid = unquote(path[len("/api/tree/node/"):])
                self.send_json(self.store.node_payload(rid))
                return
            if path.startswith("/api/tree/children/"):
                rid = unquote(path[len("/api/tree/children/"):])
                self.send_json(self.store.children_payload(rid))
                return
            if path.startswith("/api/tree/path/"):
                rid = unquote(path[len("/api/tree/path/"):])
                self.send_json({"region_id": rid, "path": self.store.path_to_root(rid)})
                return
            if path.startswith("/api/tree/leaf-ops/"):
                rid = unquote(path[len("/api/tree/leaf-ops/"):])
                self.send_json({"region_id": rid, "leaf_ops": self.store.leaf_ops(rid)})
                return
            if path == "/api/tree/search":
                query = (qs.get("q") or [""])[0]
                limit = int((qs.get("limit") or ["100"])[0])
                self.send_json(self.store.search(query, limit=limit))
                return
            if path == "/favicon.ico":
                self.send_bytes(b"", status=204, content_type="image/x-icon")
                return
            self.send_error_json(f"Not found: {path}", status=404)
        except KeyError as exc:
            self.send_error_json(str(exc), status=404)
        except Exception as exc:  # pragma: no cover
            self.send_error_json(f"Internal server error: {exc}", status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lazy ordered dataflow control-tree browser backend.")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--viewer", default=HTML_FILE)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    store = OrderedTreeStore.load(repo_root, args.model)
    viewer = Path(args.viewer)
    if not viewer.is_absolute():
        viewer = Path(__file__).resolve().parent / viewer
    if not viewer.exists():
        raise FileNotFoundError(f"Viewer HTML not found: {viewer}")
    ApiHandler.store = store
    ApiHandler.html_path = viewer
    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    print(f"[ordered-control-tree-api] model={store.index()['model_name']}")
    print(f"[ordered-control-tree-api] regions={len(store.region_by_id)}")
    print(f"[ordered-control-tree-api] open: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ordered-control-tree-api] stopping")


if __name__ == "__main__":
    main()
