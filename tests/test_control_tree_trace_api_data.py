from __future__ import annotations

import importlib.util
from pathlib import Path


def load_api_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "control_tree_trace_api_server.py"
    spec = importlib.util.spec_from_file_location("control_tree_trace_api_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def synthetic_steps() -> list[dict]:
    return [
        {
            "step_id": "step_000000",
            "step_index": 0,
            "pass_name": "initialize_primitives",
            "action": "initialize",
            "created_region_id": None,
            "created_region_type": None,
            "collapsed_node_ids": [],
            "collapsed_op_ids": [],
            "collapsed_region_ids": [],
            "confidence": "high",
            "reason": "initialize",
            "before_summary": {},
            "after_summary": {"num_active_nodes": 4},
            "graph_snapshot": {"nodes": [], "edges": []},
        },
        {
            "step_id": "step_000001",
            "step_index": 1,
            "pass_name": "collapse_linear_projection",
            "action": "collapse",
            "created_region_id": "region::linear",
            "created_region_type": "LinearProjectionRegion",
            "collapsed_node_ids": ["node::op::matmul", "node::op::add"],
            "collapsed_op_ids": ["op::matmul", "op::add"],
            "collapsed_region_ids": [],
            "confidence": "high",
            "reason": "MatMul followed by bias Add",
            "before_summary": {"num_active_nodes": 4},
            "after_summary": {"num_active_nodes": 3},
            "graph_snapshot": {
                "nodes": [
                    {
                        "node_id": "node::op::input",
                        "node_kind": "tensor_op",
                        "label": "input",
                        "region_type": None,
                        "op_type": "Input",
                        "canonical_op_type": "unknown",
                        "confidence": "high",
                        "pruning_role": None,
                        "metadata": {},
                    },
                    {
                        "node_id": "region::linear",
                        "node_kind": "abstract_region",
                        "label": "LinearProjectionRegion",
                        "region_type": "LinearProjectionRegion",
                        "op_type": None,
                        "canonical_op_type": None,
                        "confidence": "high",
                        "pruning_role": "directly_prunable",
                        "metadata": {},
                    },
                    {
                        "node_id": "node::op::relu",
                        "node_kind": "tensor_op",
                        "label": "relu",
                        "region_type": None,
                        "op_type": "Relu",
                        "canonical_op_type": "activation",
                        "confidence": "high",
                        "pruning_role": None,
                        "metadata": {},
                    },
                ],
                "edges": [
                    {"src": "node::op::input", "dst": "region::linear", "edge_kind": "dataflow", "label": "x", "metadata": {}},
                    {"src": "region::linear", "dst": "node::op::relu", "edge_kind": "dataflow", "label": "y", "metadata": {}},
                ],
            },
        },
        {
            "step_id": "step_000002",
            "step_index": 2,
            "pass_name": "collapse_feedforward",
            "action": "skip",
            "created_region_id": None,
            "created_region_type": None,
            "collapsed_node_ids": [],
            "collapsed_op_ids": ["op::matmul", "op::add"],
            "collapsed_region_ids": [],
            "confidence": "medium",
            "reason": "Candidate is already represented by one active abstract region.",
            "before_summary": {"num_active_nodes": 3},
            "after_summary": {"num_active_nodes": 3},
            "graph_snapshot": {"nodes": [], "edges": []},
        },
        {
            "step_id": "step_000003",
            "step_index": 3,
            "pass_name": "collapse_feedforward",
            "action": "collapse",
            "created_region_id": "region::ffn",
            "created_region_type": "FeedForwardRegion",
            "collapsed_node_ids": ["region::linear", "node::op::relu"],
            "collapsed_op_ids": ["op::matmul", "op::add", "op::relu"],
            "collapsed_region_ids": ["region::linear"],
            "confidence": "high",
            "reason": "projection activation projection",
            "before_summary": {"num_active_nodes": 3},
            "after_summary": {"num_active_nodes": 2},
            "graph_snapshot": {"nodes": [], "edges": []},
        },
    ]


def test_step_summaries_omit_graph_snapshot() -> None:
    api = load_api_module()
    summary = api.make_step_summary(synthetic_steps()[1])
    payload = api.step_without_snapshot(synthetic_steps()[1])

    assert summary["collapsed_node_count"] == 2
    assert "graph_snapshot" not in payload
    assert payload["graph_snapshot_summary"]["edge_count"] == 2


def test_filters_and_pagination() -> None:
    api = load_api_module()
    steps = synthetic_steps()

    collapses = api.filter_step_summaries(steps, action="collapse")
    feedforward = api.filter_step_summaries(steps, region_type="FeedForwardRegion")
    page = api.paginate(collapses, 1, 1)

    assert len(collapses) == 2
    assert feedforward[0]["step_id"] == "step_000003"
    assert page["total"] == 2
    assert page["items"][0]["step_id"] == "step_000003"


def test_local_graph_contains_created_collapsed_and_boundary_nodes() -> None:
    api = load_api_module()
    graph = api.build_local_step_graph(synthetic_steps()[1])
    roles = {node["role"] for node in graph["nodes"]}
    edge_roles = {edge["role"] for edge in graph["edges"]}

    assert "created" in roles
    assert "collapsed" in roles
    assert "incoming_boundary" in roles
    assert "outgoing_boundary" in roles
    assert "abstraction" in edge_roles
    assert "incoming" in edge_roles
    assert "outgoing" in edge_roles


def test_skip_step_local_graph_uses_candidate_nodes() -> None:
    api = load_api_module()
    graph = api.build_local_step_graph(synthetic_steps()[2])

    assert graph["mode"] == "local"
    assert any(node["role"] == "context" for node in graph["nodes"])


def test_next_prev_matching_step_search() -> None:
    api = load_api_module()
    steps = synthetic_steps()

    next_collapse = api.find_matching_step(steps, step_index=0, direction="next", action="collapse")
    next_ffn = api.find_matching_step(steps, step_index=0, direction="next", region_type="FeedForwardRegion")
    prev_collapse = api.find_matching_step(steps, step_index=3, direction="prev", action="collapse")

    assert next_collapse["step_id"] == "step_000001"
    assert next_ffn["step_id"] == "step_000003"
    assert prev_collapse["step_id"] == "step_000001"
