from __future__ import annotations

from model_analysis.deadbranch_propagation import analyze_deadbranch_propagation, deadbranch_report_to_dict


def op(index: int, source: str, op_type: str, *, category: str = "unknown", kind: str = "unknown", parameterized: bool = False, input_dim: str = "unknown", output_dim: str = "unknown") -> dict:
    return {
        "op_id": f"op::{index:06d}",
        "topological_index": index,
        "source_name": source,
        "op_type": op_type,
        "semantic_category": category,
        "semantic_kind": kind,
        "parameterized": parameterized,
        "dimension_roles": {"input": input_dim, "output": output_dim},
    }


def opt_ops() -> list[dict]:
    prefix = "/model/decoder/layers.0"
    return [
        op(1, prefix + "/self_attn/q_proj/MatMul", "MatMul"),
        op(2, prefix + "/self_attn/k_proj/MatMul", "MatMul"),
        op(3, prefix + "/self_attn/v_proj/MatMul", "MatMul"),
        op(4, prefix + "/self_attn/Reshape_2", "Reshape"),
        op(5, prefix + "/self_attn/Transpose_2", "Transpose"),
        op(6, prefix + "/self_attn/MatMul", "MatMul"),
        op(7, prefix + "/self_attn/Softmax", "Softmax"),
        op(8, prefix + "/self_attn/MatMul_1", "MatMul"),
        op(9, prefix + "/self_attn/Transpose_3", "Transpose"),
        op(10, prefix + "/self_attn/Reshape_3", "Reshape"),
        op(11, prefix + "/self_attn/out_proj/MatMul", "MatMul"),
        op(12, prefix + "/fc1/Gemm", "Gemm", category="parameterized_projection", kind="parameterized_linear_matmul", parameterized=True, input_dim="hidden_dim", output_dim="intermediate_dim"),
        op(13, prefix + "/activation_fn/Relu", "Relu", category="elementwise_index_preserving", kind="gelu_elementwise", input_dim="intermediate_dim", output_dim="intermediate_dim"),
        op(14, prefix + "/fc2/Gemm", "Gemm", category="parameterized_projection", kind="parameterized_linear_matmul", parameterized=True, input_dim="intermediate_dim", output_dim="hidden_dim"),
    ]


def test_opt_ffn_and_attention_value_pairs_are_detected() -> None:
    report = deadbranch_report_to_dict(analyze_deadbranch_propagation("facebook/opt-125m", {"ops": opt_ops()}))

    assert report["summary"]["ffn_pairs"] == 1
    assert report["summary"]["attention_value_pairs"] == 1
    assert report["summary"]["total_pairs"] == 2
    value = next(pair for pair in report["pairs"] if pair["pair_kind"] == "attention_value_deadness")
    assert value["producer_op_name"].endswith("/self_attn/v_proj/MatMul")
    assert value["consumer_op_name"].endswith("/self_attn/out_proj/MatMul")
    assert value["mapping_status"] == "proven"
    assert value["status"] == "propagatable"


def test_qk_pairs_are_blocked_not_propagatable() -> None:
    report = deadbranch_report_to_dict(analyze_deadbranch_propagation("facebook/opt-125m", {"ops": opt_ops()}))

    assert {row["pair_kind"] for row in report["blocked_pairs"]} == {"query_score_deadness", "key_score_deadness"}
    assert all(row["blocker_type"] == "qk_score_contraction_mixes_channels" for row in report["blocked_pairs"])
    assert all("q_proj" not in row["producer_op_name"] and "k_proj" not in row["producer_op_name"] for row in report["pairs"])


def test_opt_twelve_layer_alignment_matches_sparsegpt_observation() -> None:
    ops = []
    for layer in range(12):
        for item in opt_ops():
            row = dict(item)
            row["topological_index"] = item["topological_index"] + layer * 100
            row["op_id"] = f"op::{row['topological_index']:06d}"
            row["source_name"] = item["source_name"].replace("layers.0", f"layers.{layer}")
            ops.append(row)

    summary = deadbranch_report_to_dict(analyze_deadbranch_propagation("facebook/opt-125m", {"ops": ops}))["summary"]

    assert summary["total_pairs"] == 24
    assert summary["ffn_pairs"] == 12
    assert summary["attention_value_pairs"] == 12
    assert summary["query_key_blocked_pairs"] == 24
    assert summary["sparsegpt_alignment_status"] == "matches_expected"


def test_value_path_without_softmax_context_proof_is_constrained() -> None:
    report = deadbranch_report_to_dict(
        analyze_deadbranch_propagation(
            "facebook/opt-125m",
            {"ops": [item for item in opt_ops() if item["op_type"] != "Softmax"]},
        )
    )
    value = next(pair for pair in report["pairs"] if pair["pair_kind"] == "attention_value_deadness")

    assert value["mapping_status"] == "assumed_by_pattern"
    assert value["status"] == "constrained"


def test_gpt2_attention_and_mlp_c_proj_are_not_confused() -> None:
    prefix = "/model/transformer/h.0"
    ops = [
        op(1, prefix + "/attn/c_proj/Gemm", "Gemm"),
        op(2, prefix + "/mlp/c_fc/Gemm", "Gemm", category="parameterized_projection", kind="parameterized_linear_matmul", parameterized=True, input_dim="hidden_dim", output_dim="intermediate_dim"),
        op(3, prefix + "/mlp/act/Tanh", "Tanh", category="elementwise_index_preserving", kind="gelu_elementwise", input_dim="intermediate_dim", output_dim="intermediate_dim"),
        op(4, prefix + "/mlp/c_proj/Gemm", "Gemm", category="parameterized_projection", kind="parameterized_linear_matmul", parameterized=True, input_dim="intermediate_dim", output_dim="hidden_dim"),
    ]
    report = deadbranch_report_to_dict(analyze_deadbranch_propagation("gpt2", {"ops": ops}))

    assert report["summary"]["ffn_pairs"] == 1
    pair = report["pairs"][0]
    assert pair["producer_op_name"].endswith("/mlp/c_fc/Gemm")
    assert pair["consumer_op_name"].endswith("/mlp/c_proj/Gemm")
    assert not any(row["pair_kind"] == "attention_value_deadness" for row in report["pairs"])
