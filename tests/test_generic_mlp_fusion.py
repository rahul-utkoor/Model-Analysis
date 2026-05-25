from __future__ import annotations

from model_analysis.generic_mlp_fusion import detect_generic_mlp_matches
from model_analysis.pruning_opportunity_ranking import build_pruning_opportunity_ranking, pruning_opportunity_ranking_to_dict
from model_analysis.region_pruning_semantics import build_region_pruning_semantics, region_pruning_semantics_to_dict


def op(op_id: str, source: str, kind: str, roles: dict[str, str], *, op_type: str = "MatMul", category: str = "parameterized_projection", parameterized: bool = True, index: int = 0) -> dict:
    return {
        "op_id": op_id,
        "source_name": source,
        "op_type": op_type,
        "semantic_kind": kind,
        "semantic_category": category,
        "parameterized": parameterized,
        "dimension_roles": roles,
        "topological_index": index,
        "pruning_effect": {"direct_pruning": "allowed"},
    }


def activation(op_id: str, source: str, index: int) -> dict:
    return op(
        op_id,
        source,
        "gelu_elementwise",
        {"input": "intermediate_dim", "output": "intermediate_dim"},
        op_type="Relu",
        category="elementwise_index_preserving",
        parameterized=False,
        index=index,
    )


def distilbert_ops(layers: int = 1) -> dict:
    ops = []
    for layer in range(layers):
        base = f"/model/distilbert/transformer/layer.{layer}/ffn"
        ops.extend(
            [
                op(f"l{layer}_lin1", f"{base}/lin1/MatMul", "parameterized_linear_matmul", {"input": "hidden_dim", "output": "intermediate_dim"}, index=layer * 10),
                op(f"l{layer}_lin1_bias", f"{base}/lin1/Add", "linear_bias_add", {"input": "intermediate_dim", "output": "intermediate_dim"}, op_type="Add", index=layer * 10 + 1),
                activation(f"l{layer}_act", f"{base}/activation/Erf", layer * 10 + 2),
                op(f"l{layer}_lin2", f"{base}/lin2/MatMul", "parameterized_linear_matmul", {"input": "intermediate_dim", "output": "hidden_dim"}, index=layer * 10 + 3),
                op(f"l{layer}_lin2_bias", f"{base}/lin2/Add", "linear_bias_add", {"input": "hidden_dim", "output": "hidden_dim"}, op_type="Add", index=layer * 10 + 4),
            ]
        )
    return {"model_name": "distilbert-base-uncased", "ops": ops}


def vit_ops(layers: int = 1) -> dict:
    ops = []
    for layer in range(layers):
        base = f"/model/vit/layers.{layer}/mlp"
        ops.extend(
            [
                op(f"l{layer}_fc1", f"{base}/fc1/MatMul", "parameterized_linear_matmul", {"input": "hidden_dim", "output": "intermediate_dim"}, index=layer * 10),
                op(f"l{layer}_fc1_bias", f"{base}/fc1/Add", "linear_bias_add", {"input": "intermediate_dim", "output": "intermediate_dim"}, op_type="Add", index=layer * 10 + 1),
                activation(f"l{layer}_act", f"{base}/activation_fn/Erf", layer * 10 + 2),
                op(f"l{layer}_fc2", f"{base}/fc2/MatMul", "parameterized_linear_matmul", {"input": "intermediate_dim", "output": "hidden_dim"}, index=layer * 10 + 3),
                op(f"l{layer}_fc2_bias", f"{base}/fc2/Add", "linear_bias_add", {"input": "hidden_dim", "output": "hidden_dim"}, op_type="Add", index=layer * 10 + 4),
            ]
        )
    return {"model_name": "google/vit-base-patch16-224", "ops": ops}


def gpt2_ops(layers: int = 1) -> dict:
    ops = []
    for layer in range(layers):
        base = f"/model/transformer/h.{layer}"
        ops.extend(
            [
                op(f"l{layer}_attn_cproj", f"{base}/attn/c_proj/Gemm", "parameterized_linear_matmul", {"input": "hidden_dim", "output": "hidden_dim"}, op_type="Gemm", index=layer * 10),
                op(f"l{layer}_cfc", f"{base}/mlp/c_fc/Gemm", "parameterized_linear_matmul", {"input": "hidden_dim", "output": "intermediate_dim"}, op_type="Gemm", index=layer * 10 + 1),
                activation(f"l{layer}_act", f"{base}/mlp/act/Tanh", layer * 10 + 2),
                op(f"l{layer}_cproj", f"{base}/mlp/c_proj/Gemm", "parameterized_linear_matmul", {"input": "intermediate_dim", "output": "hidden_dim"}, op_type="Gemm", index=layer * 10 + 3),
            ]
        )
    return {"model_name": "gpt2", "ops": ops}


def test_detects_distilbert_mlp_matches() -> None:
    matches = detect_generic_mlp_matches("distilbert-base-uncased", distilbert_ops(6))

    assert len(matches) == 6
    assert all(match.family == "distilbert" for match in matches)
    assert all(match.evidence_status == "complete" for match in matches)


def test_detects_vit_mlp_matches() -> None:
    matches = detect_generic_mlp_matches("google/vit-base-patch16-224", vit_ops(12))

    assert len(matches) == 12
    assert all(match.family == "vit" for match in matches)


def test_detects_gpt2_mlp_without_attn_cproj_confusion() -> None:
    matches = detect_generic_mlp_matches("gpt2", gpt2_ops(12))

    assert len(matches) == 12
    assert all(match.contraction_op and "/mlp/c_proj/" in match.contraction_op["source_name"] for match in matches)
    assert all("attn/c_proj" not in " ".join(op["source_name"] for op in match.source_ops) for match in matches)


def test_generic_mlp_match_creates_region_semantics() -> None:
    semantics = region_pruning_semantics_to_dict(build_region_pruning_semantics({"model_name": "distilbert-base-uncased", "regions": []}, {"model_name": "distilbert-base-uncased", "ops": []}, op_semantics=distilbert_ops(1)))
    record = semantics["regions"][0]

    assert record["source_region_type"] == "GenericMLPRegion"
    assert record["semantic_category"] == "feed_forward_block"
    assert record["pruning_role"] == "directly_prunable"


def test_generic_mlp_region_ranks_as_safe_candidate() -> None:
    semantics = region_pruning_semantics_to_dict(build_region_pruning_semantics({"model_name": "google/vit-base-patch16-224", "regions": []}, {"model_name": "google/vit-base-patch16-224", "ops": []}, op_semantics=vit_ops(1)))
    ranking = pruning_opportunity_ranking_to_dict(build_pruning_opportunity_ranking(semantics, op_semantics=vit_ops(1)))

    assert ranking["summary"]["generic_mlp_safe_candidates"] == 1
    assert ranking["candidates"][0]["candidate_kind"] == "feedforward_intermediate_pruning"
    assert ranking["candidates"][0]["pruning_class"] == "safe"


def test_partial_generic_mlp_region_ranks_as_constrained() -> None:
    data = distilbert_ops(1)
    data["ops"] = [item for item in data["ops"] if "/activation/" not in item["source_name"]]
    semantics = region_pruning_semantics_to_dict(build_region_pruning_semantics({"model_name": "distilbert-base-uncased", "regions": []}, {"model_name": "distilbert-base-uncased", "ops": []}, op_semantics=data))
    ranking = pruning_opportunity_ranking_to_dict(build_pruning_opportunity_ranking(semantics, op_semantics=data))

    assert semantics["regions"][0]["pruning_role"] == "constraint_carrier"
    assert ranking["summary"]["generic_mlp_constrained_candidates"] == 1
