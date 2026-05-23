from __future__ import annotations

from model_analysis.dag_region_analysis import (
    build_dag_region_analysis_report,
    enumerate_diamond_regions,
    enumerate_fork_regions,
    enumerate_join_fork_join_regions,
    find_reconvergence,
    is_fork_node,
)
from model_analysis.subgraph_analysis import build_onnx_adjacency, is_join_node


def join_fork_join_summary() -> dict:
    return {
        "model_name": "motif",
        "inputs": [{"name": "x_a", "shape": [1, 4]}, {"name": "x_b", "shape": [1, 4]}],
        "outputs": [{"name": "f_out", "shape": [1, 4]}],
        "initializers": [
            {"name": "w_a", "dims": [4, 4]},
            {"name": "w_b", "dims": [4, 4]},
            {"name": "w_d", "dims": [4, 4]},
            {"name": "w_e", "dims": [4, 4]},
        ],
        "tensor_shape_map": {
            "a_out": [1, 4],
            "b_out": [1, 4],
            "c_out": [1, 4],
            "d_out": [1, 4],
            "e_out": [1, 4],
            "f_out": [1, 4],
        },
        "nodes": [
            {"name": "A", "op_type": "MatMul", "inputs": ["x_a", "w_a"], "outputs": ["a_out"]},
            {"name": "B", "op_type": "MatMul", "inputs": ["x_b", "w_b"], "outputs": ["b_out"]},
            {"name": "C", "op_type": "Add", "inputs": ["a_out", "b_out"], "outputs": ["c_out"]},
            {"name": "D", "op_type": "MatMul", "inputs": ["c_out", "w_d"], "outputs": ["d_out"]},
            {"name": "E", "op_type": "MatMul", "inputs": ["c_out", "w_e"], "outputs": ["e_out"]},
            {"name": "F", "op_type": "Add", "inputs": ["d_out", "e_out"], "outputs": ["f_out"]},
        ],
    }


def non_reconvergent_summary() -> dict:
    return {
        "model_name": "no-join",
        "inputs": [{"name": "x"}],
        "outputs": [{"name": "d_out"}, {"name": "e_out"}],
        "initializers": [],
        "nodes": [
            {"name": "C", "op_type": "Identity", "inputs": ["x"], "outputs": ["c_out"]},
            {"name": "D", "op_type": "Relu", "inputs": ["c_out"], "outputs": ["d_out"]},
            {"name": "E", "op_type": "Sigmoid", "inputs": ["c_out"], "outputs": ["e_out"]},
        ],
    }


def test_exact_motif_detects_join_fork_and_reconvergent_join() -> None:
    summary = join_fork_join_summary()
    adjacency = build_onnx_adjacency(summary)

    assert is_join_node(adjacency.node_by_name["C"])
    assert is_fork_node("C", adjacency)
    convergence, first_path, second_path = find_reconvergence(adjacency, "D", "E", max_depth=4)
    assert convergence == "F"
    assert first_path == ["D", "F"]
    assert second_path == ["E", "F"]

    regions = enumerate_join_fork_join_regions(adjacency, summary, "motif", max_branch_depth=4)
    assert len(regions) == 1
    region = regions[0]
    assert region.region_kind == "join_fork_join"
    assert region.branch_paths == [["C", "D", "F"], ["C", "E", "F"]]
    assert region.join_nodes == ["C", "F"]
    assert region.fork_nodes == ["C"]
    assert "JoinForkJoin" in region.pattern


def test_simple_fork_and_diamond_regions_are_emitted() -> None:
    summary = join_fork_join_summary()
    adjacency = build_onnx_adjacency(summary)

    forks = enumerate_fork_regions(adjacency, summary, "motif")
    diamonds = enumerate_diamond_regions(adjacency, summary, "motif", max_branch_depth=4)
    assert any(region.fork_nodes == ["C"] and region.region_kind == "fork" for region in forks)
    assert any(region.fork_nodes == ["C"] and region.join_nodes == ["F"] for region in diamonds)


def test_no_false_diamond_when_branches_do_not_reconverge() -> None:
    summary = non_reconvergent_summary()
    adjacency = build_onnx_adjacency(summary)

    assert is_fork_node("C", adjacency)
    assert enumerate_diamond_regions(adjacency, summary, "no-join", max_branch_depth=4) == []


def test_dag_region_classification_and_report_summary() -> None:
    report = build_dag_region_analysis_report(
        join_fork_join_summary(),
        {"name": "motif", "hf_id": "local/motif", "task": "unit-test"},
        max_branch_depth=4,
    )
    regions = [region for region in report.regions if region.region_kind == "join_fork_join"]

    assert len(regions) == 1
    assert regions[0].pruning_class in {"multi_branch_constraint", "residual_like"}
    assert regions[0].risk_level == "high"
    assert report.summary["num_join_fork_join_regions"] == 1
    assert any(
        item["evidence_type"] in {"residual_equal_shape", "branch_output_compatibility"}
        for item in report.pruning_evidence
    )

