#!/usr/bin/env python3
"""Lazy API backend for stepwise control-tree trace browsing."""

from __future__ import annotations

import argparse
import json
import re
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


def _clean_label(value: Any) -> str:
    text = str(value or "").strip()
    for prefix in ("node::op::", "region::", "region:", "trace::", "candidate::", "op::"):
        text = text.replace(prefix, "")
    text = re.sub(r"^/model/[^/]+/", "", text)
    text = re.sub(r"^model_[a-zA-Z0-9]+_", "", text)
    text = text.replace("/", ".")
    text = text.replace("::", ".")
    text = text.strip("._")
    return text or str(value or "")


def _split_words(value: str) -> str:
    text = re.sub(r"(?<!^)([A-Z])", r" \1", value).replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def humanize_op_name(node: dict[str, Any]) -> str:
    metadata = node.get("metadata", {}) or {}
    source_ids = node.get("source_op_ids") or []
    for candidate in (
        metadata.get("source_node_name"),
        source_ids[0] if source_ids else None,
        node.get("label"),
        node.get("node_id"),
    ):
        cleaned = _clean_label(candidate)
        if cleaned:
            return cleaned
    return "unknown"


def humanize_region_type(region_type: str | None, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    if region_type == "ActivationRegion" and metadata.get("activation_kind") == "gelu":
        return "Activation / GELU"
    return {
        "LinearProjectionRegion": "Linear Projection",
        "FeedForwardRegion": "Feed-Forward Block",
        "ResidualMergeRegion": "Residual Merge",
        "AttentionSkeletonRegion": "Attention Skeleton",
        "AxisTransformRegion": "Axis Transform",
        "ActivationRegion": "Activation",
        "LayerNormRegion": "LayerNorm",
        "PrimitiveRegion": "Primitive Op",
        "BiasAddRegion": "Bias Add",
        "ForkRegion": "Fork",
        "JoinRegion": "Join",
        "ProperAcyclicRegion": "Acyclic Region",
        "ModelRegion": "Model Region",
    }.get(region_type or "", _split_words(region_type or "Unknown"))


def infer_semantic_node_title(node: dict[str, Any]) -> str:
    region_type = node.get("region_type")
    metadata = node.get("metadata", {}) or {}
    if region_type:
        return humanize_region_type(region_type, metadata)
    name = humanize_op_name(node).lower()
    canonical = node.get("canonical_op_type")
    op_type = node.get("op_type")
    if canonical in {"linear", "matmul"} or str(op_type).lower() in {"matmul", "gemm"}:
        if "query" in name:
            return "Query MatMul"
        if "key" in name:
            return "Key MatMul"
        if "value" in name:
            return "Value MatMul"
        if "intermediate" in name:
            return "Intermediate Projection"
        if "output" in name:
            return "Output Projection"
        return "MatMul Projection"
    if canonical == "softmax":
        return "Softmax"
    if canonical == "bias_add":
        return "Bias Add"
    if canonical == "residual_add":
        return "Residual Add"
    if canonical == "layer_norm":
        return "LayerNorm"
    if canonical == "activation" or str(op_type).lower() in {"erf", "gelu", "relu", "tanh", "sigmoid"}:
        return "Activation / GELU" if metadata.get("activation_kind") == "gelu" or "erf" in name else "Activation"
    if canonical in {"shape_op", "axis_transform", "reshape", "transpose"}:
        return "Axis / Shape Transform"
    return _split_words(op_type or humanize_op_name(node))


def _node_explanation(node: dict[str, Any], role: str) -> str:
    title = infer_semantic_node_title(node)
    if role == "created":
        return f"This is the new abstract {title} node created by this reduction step."
    if role == "collapsed":
        return f"This node is part of the concrete subgraph being collapsed into a higher-level {title if node.get('region_type') else 'region'}."
    if role == "incoming_boundary":
        return "This node provides data flowing into the region being collapsed."
    if role == "outgoing_boundary":
        return "This node consumes data produced by the newly created region."
    return "This node gives local context for the selected trace step."


def humanize_edge_label(edge: dict[str, Any], role: str) -> tuple[str, str]:
    raw = edge.get("label") or edge.get("tensor_or_value_id") or ""
    clean = _clean_label(raw)
    base = {
        "incoming": "input to collapsed region",
        "internal": "internal dataflow",
        "outgoing": "output from collapsed region",
        "abstraction": "collapsed into",
    }.get(role)
    if not base:
        base = {
            "containment": "contains",
            "dataflow": "tensor flow",
            "dependency": "depends on",
            "abstraction": "collapsed into",
        }.get(edge.get("edge_kind"), "edge")
    display = f"{base}: {clean}" if clean and role != "abstraction" else base
    return display, clean or str(raw or edge.get("edge_kind", ""))


def _edge_explanation(edge: dict[str, Any], role: str) -> str:
    return {
        "incoming": "This tensor dependency crosses from surrounding context into the region being collapsed.",
        "internal": "This tensor dependency is internal to the region being collapsed.",
        "outgoing": "This tensor dependency leaves the newly abstracted region and flows to later computation.",
        "abstraction": "This shows that the selected concrete node is represented by the new abstract region after the collapse.",
    }.get(role, "This edge is part of the selected step's local structural context.")


def _step_display(pass_name: str | None, region_type: str | None, action: str | None) -> tuple[str, str]:
    region = humanize_region_type(region_type)
    if pass_name == "semantic_fusion_gelu":
        return "GELU semantic fusion", "ActivationRegion"
    if action == "skip":
        return "Skipped candidate collapse", region_type or pass_name or ""
    if region_type:
        return f"{region} collapse", region_type
    return _split_words(pass_name or action or "Step"), action or ""


def _short_reason(reason: str | None, limit: int = 140) -> str:
    text = str(reason or "")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def make_step_summary(step: dict[str, Any]) -> dict[str, Any]:
    title, subtitle = _step_display(step.get("pass_name"), step.get("created_region_type"), step.get("action"))
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
        "display_title": title,
        "display_subtitle": subtitle,
        "display_reason_short": _short_reason(step.get("reason")),
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
    payload["teaching_explanation"] = build_step_teaching_explanation(step, build_local_step_graph(step))
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
    metadata = node.get("metadata", {}) or {}
    title = infer_semantic_node_title(node)
    subtitle = humanize_op_name(node)
    return {
        "node_id": node.get("node_id"),
        "display_title": title,
        "display_subtitle": subtitle,
        "technical_label": node.get("node_id"),
        "node_kind": "created_region" if role == "created" else node.get("node_kind", "boundary"),
        "label": node.get("label") or node.get("node_id"),
        "region_type": node.get("region_type"),
        "op_type": node.get("op_type"),
        "canonical_op_type": node.get("canonical_op_type"),
        "role": role,
        "explanation": _node_explanation(node, role),
        "confidence": node.get("confidence"),
        "pruning_role": node.get("pruning_role"),
        "source_op_ids": node.get("source_op_ids", []),
        "metadata": metadata,
    }


def _synthetic_node(node_id: str, role: str, *, label: str | None = None) -> dict[str, Any]:
    node = {
        "node_id": node_id,
        "label": label or node_id,
        "region_type": None,
        "op_type": None,
        "canonical_op_type": None,
        "metadata": {"synthetic": True},
        "source_op_ids": [label or node_id] if role == "context" else [],
    }
    return {
        "node_id": node_id,
        "display_title": infer_semantic_node_title(node) if role != "created" else _split_words(label or node_id),
        "display_subtitle": _clean_label(label or node_id),
        "technical_label": node_id,
        "node_kind": "boundary" if "boundary" in role else "abstract_region",
        "label": label or node_id,
        "region_type": None,
        "op_type": None,
        "canonical_op_type": None,
        "role": role,
        "explanation": _node_explanation(node, role),
        "confidence": None,
        "pruning_role": None,
        "source_op_ids": node.get("source_op_ids", []),
        "metadata": {"synthetic": True},
    }


def _edge_payload(edge: dict[str, Any], role: str) -> dict[str, Any]:
    display, technical = humanize_edge_label(edge, role)
    return {
        "src": edge.get("src"),
        "dst": edge.get("dst"),
        "edge_kind": edge.get("edge_kind", "dataflow"),
        "label": edge.get("label") or edge.get("tensor_or_value_id"),
        "display_label": display,
        "technical_label": technical,
        "tensor_or_value_id": edge.get("tensor_or_value_id"),
        "role": role,
        "explanation": _edge_explanation(edge, role),
        "metadata": edge.get("metadata", {}),
    }


def _abstraction_edge(src: str, dst: str) -> dict[str, Any]:
    edge = {
        "src": src,
        "dst": dst,
        "edge_kind": "abstraction",
        "label": "collapsed_into",
        "tensor_or_value_id": None,
        "metadata": {"synthetic": True},
    }
    return _edge_payload(edge, "abstraction")


def _group_nodes(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {key: [] for key in ("incoming_boundary", "collapsed", "created", "outgoing_boundary", "context")}
    for node in nodes:
        groups.setdefault(node.get("role", "context"), []).append(node)
    return groups


def build_step_teaching_explanation(step: dict[str, Any], local_graph: dict[str, Any]) -> dict[str, str]:
    pass_name = step.get("pass_name")
    action = step.get("action")
    region_type = step.get("created_region_type")
    before = (step.get("before_summary") or {}).get("num_active_nodes")
    after = (step.get("after_summary") or {}).get("num_active_nodes")
    before_after = f"active nodes {before} -> {after}"
    collapsed_count = len(step.get("collapsed_node_ids", []) or [])
    if action == "skip":
        return {
            "headline": "Skip candidate collapse",
            "what_happened": "The analyzer considered this candidate but did not create a new abstract node for it.",
            "why_it_matters": "Skip steps make overlap and ambiguity visible instead of silently hiding detector decisions.",
            "compiler_analogy": "This is like rejecting an interval candidate because it was already covered by an earlier reduction.",
            "pruning_relevance": "No new pruning interface is introduced by this skipped candidate.",
            "before_after": before_after,
            "reading_hint": "Gray/context nodes show the candidate evidence when it is available.",
        }
    if pass_name == "semantic_fusion_gelu":
        return {
            "headline": "Recognize decomposed GELU activation",
            "what_happened": "Several primitive arithmetic ops around Erf are collapsed into one ActivationRegion.",
            "why_it_matters": "ONNX may decompose GELU into arithmetic; semantic fusion recovers the high-level operation.",
            "compiler_analogy": "This is like recognizing an instruction idiom and replacing it with a semantic node.",
            "pruning_relevance": "Recovering GELU exposes the feed-forward block that contains the MLP intermediate dimension.",
            "before_after": before_after,
            "reading_hint": "Yellow nodes are collapsed; the green node is the recovered activation region.",
        }
    if pass_name == "collapse_linear_projection" or region_type == "LinearProjectionRegion":
        return {
            "headline": "Collapse projection arithmetic into Linear Projection",
            "what_happened": f"{collapsed_count} primitive or lower-level nodes are collapsed into a LinearProjectionRegion.",
            "why_it_matters": "The tree now treats projection arithmetic as one semantic computation region.",
            "compiler_analogy": "This is a local structural reduction from low-level operations to a higher-level IR node.",
            "pruning_relevance": "The output feature dimension may become a prunable region dimension.",
            "before_after": before_after,
            "reading_hint": "Yellow nodes are the projection implementation; green is the abstract projection.",
        }
    if pass_name == "collapse_feedforward" or region_type == "FeedForwardRegion":
        return {
            "headline": "Collapse Linear + GELU + Linear into Feed-Forward Block",
            "what_happened": "Projection, activation, and projection regions are grouped as one feed-forward block.",
            "why_it_matters": "This recovers the semantic transformer MLP block from lower-level Tensor IR.",
            "compiler_analogy": "This is a compound-region collapse: previously abstracted regions compose into a larger region.",
            "pruning_relevance": "This is the canonical MLP pruning opportunity: prune intermediate_dim and propagate the same indices.",
            "before_after": before_after,
            "reading_hint": "Look for the two projection regions and the activation region feeding the new block.",
        }
    if pass_name == "collapse_residual_merge" or region_type == "ResidualMergeRegion":
        return {
            "headline": "Collapse residual branch merge",
            "what_happened": "A branch merge is represented as a ResidualMergeRegion.",
            "why_it_matters": "The region tree records that separate paths must agree at this join.",
            "compiler_analogy": "This is analogous to identifying a join region in a control-flow graph.",
            "pruning_relevance": "Residual hidden dimensions are usually protected or blocked because branches must agree.",
            "before_after": before_after,
            "reading_hint": "Boundary nodes show values entering or leaving the residual merge.",
        }
    if pass_name == "collapse_attention_skeleton" or region_type == "AttentionSkeletonRegion":
        return {
            "headline": "Collapse attention skeleton",
            "what_happened": "MatMul/Softmax/MatMul and nearby axis transforms are grouped as an attention skeleton.",
            "why_it_matters": "The tree records attention structure without claiming executable head pruning.",
            "compiler_analogy": "This is recognition of a high-level dataflow idiom from lower-level operations.",
            "pruning_relevance": "Attention pruning requires head-axis mapping before it can be considered executable.",
            "before_after": before_after,
            "reading_hint": "Attention skeletons often contain shape and axis context around the core attention MatMuls.",
        }
    if pass_name == "collapse_axis_transform" or region_type == "AxisTransformRegion":
        return {
            "headline": "Collapse axis/shape transform",
            "what_happened": "Shape, reshape, transpose, or related axis operations are grouped as an AxisTransformRegion.",
            "why_it_matters": "Axis movement is tracked as structural evidence instead of being hidden in primitive ops.",
            "compiler_analogy": "This resembles preserving index-map information through layout transformations.",
            "pruning_relevance": "Shape transforms carry axis-mapping constraints for pruning propagation.",
            "before_after": before_after,
            "reading_hint": "These regions are usually propagation evidence rather than direct pruning targets.",
        }
    region = humanize_region_type(region_type)
    return {
        "headline": f"Collapse into {region}",
        "what_happened": f"The trace collapsed {collapsed_count} active nodes into a {region}.",
        "why_it_matters": "This reduces low-level graph detail into a semantic structural region.",
        "compiler_analogy": "This is one reduction step in a structural-analysis control-tree construction.",
        "pruning_relevance": "The region may carry pruning roles, protected dimensions, or propagation constraints in later analysis.",
        "before_after": before_after,
        "reading_hint": "Yellow nodes are collapsed; green is the newly created abstract region.",
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
        graph = {
            "step_id": step.get("step_id"),
            "step_index": step.get("step_index"),
            "mode": "summary",
            "nodes": [],
            "edges": [],
            "groups": _group_nodes([]),
            "summary": {
                "before": step.get("before_summary", {}),
                "after": step.get("after_summary", {}),
                "reason": step.get("reason"),
            },
        }
        graph["teaching_explanation"] = build_step_teaching_explanation(step, graph)
        return graph

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
            local_edges.append(_abstraction_edge(node_id, created_id))

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

    nodes = sorted(local_nodes.values(), key=lambda item: (item.get("role", ""), item.get("node_id", "")))
    graph = {
        "step_id": step.get("step_id"),
        "step_index": step.get("step_index"),
        "mode": "local",
        "radius": radius,
        "nodes": nodes,
        "edges": local_edges,
        "groups": _group_nodes(nodes),
        "summary": {
            "created_region_id": created_id,
            "created_region_type": step.get("created_region_type"),
            "collapsed_node_count": len(collapsed_ids),
            "action": step.get("action"),
            "reason": step.get("reason"),
        },
    }
    graph["teaching_explanation"] = build_step_teaching_explanation(step, graph)
    return graph


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
