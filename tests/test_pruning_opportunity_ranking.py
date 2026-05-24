from __future__ import annotations

from model_analysis.pruning_opportunity_ranking import build_pruning_opportunity_ranking, pruning_opportunity_ranking_to_dict


def region(
    rid: str,
    category: str,
    role: str,
    *,
    dims: list[dict] | None = None,
    repairs: list[str] | None = None,
    blockers: list[str] | None = None,
    ops: list[str] | None = None,
) -> dict:
    return {
        "region_id": rid,
        "region_name": rid,
        "source_region_type": "FeedForwardRegion" if category == "feed_forward_block" else "LinearProjectionRegion",
        "semantic_category": category,
        "section": "Layer 0",
        "op_range": "1-1",
        "pruning_role": role,
        "dimensions": dims or [],
        "repair_obligations": [{"obligation_type": item, "required": True} for item in (repairs or [])],
        "blockers": [{"blocker_type": item, "severity": "blocker"} for item in (blockers or [])],
        "propagation_rules": [{"rule_type": "rule", "source_dimension": "x", "target_dimensions": ["y"]}],
        "evidence": {"source_ops": ops or []},
    }


def op(op_id: str, kind: str, direct: str = "allowed") -> dict:
    return {
        "op_id": op_id,
        "source_name": f"/{op_id}",
        "op_type": "MatMul",
        "semantic_kind": kind,
        "semantic_category": "parameterized_projection",
        "parameterized": True,
        "pruning_effect": {"direct_pruning": direct, "reason": "", "required_repairs": [], "blockers": []},
    }


def build_candidates(regions: list[dict], ops: list[dict] | None = None) -> list[dict]:
    ranking = build_pruning_opportunity_ranking(
        {"model_name": "synthetic", "regions": regions},
        op_semantics={"model_name": "synthetic", "ops": ops or []},
    )
    return pruning_opportunity_ranking_to_dict(ranking)["candidates"]


def test_feedforward_region_produces_safe_candidate() -> None:
    candidates = build_candidates(
        [
            region(
                "Layer 0 Feed Forward",
                "feed_forward_block",
                "directly_prunable",
                dims=[{"dim_name": "intermediate_dim", "status": "prunable"}],
                repairs=["same_indices_across_mlp", "prune_consumer_input"],
                ops=["a", "b", "c"],
            )
        ],
        [op("a", "parameterized_linear_matmul"), op("b", "gelu_erf", "not_applicable"), op("c", "parameterized_linear_matmul")],
    )

    candidate = candidates[0]
    assert candidate["pruning_class"] == "safe"
    assert candidate["rank_score"] >= 90
    assert candidate["candidate_kind"] == "feedforward_intermediate_pruning"


def test_ffn_intermediate_projection_candidate() -> None:
    candidate = build_candidates([region("ffn_int", "ffn_intermediate_projection", "directly_prunable", ops=["a"])], [op("a", "parameterized_linear_matmul")])[0]

    assert candidate["candidate_kind"] == "projection_output_pruning"
    assert candidate["pruning_class"] == "safe"


def test_ffn_output_projection_is_repair_candidate() -> None:
    candidate = build_candidates([region("ffn_out", "ffn_output_projection", "propagation_only")])[0]

    assert candidate["candidate_kind"] == "projection_input_repair"
    assert candidate["pruning_class"] == "constrained"


def test_query_projection_with_attention_blocker_is_constrained() -> None:
    candidate = build_candidates([region("query", "query_projection", "directly_prunable", blockers=["attention_head_mapping_unproven"])])[0]

    assert candidate["candidate_kind"] == "attention_projection_constrained_pruning"
    assert candidate["pruning_class"] == "constrained"


def test_attention_score_and_context_are_blocked() -> None:
    candidates = build_candidates(
        [
            region("score", "attention_score_matmul", "constraint_carrier"),
            region("context", "attention_context_matmul", "constraint_carrier"),
        ]
    )

    assert {item["candidate_kind"] for item in candidates} == {"attention_contraction_blocked"}
    assert all(item["pruning_class"] == "blocked" for item in candidates)
    assert all(item["rank_score"] <= 10 for item in candidates)
    assert all("not a learned parameter projection" in item["reason"] for item in candidates)


def test_attention_mask_add_is_auxiliary() -> None:
    candidate = build_candidates([region("mask", "attention_mask_add", "constraint_carrier")])[0]

    assert candidate["candidate_kind"] == "auxiliary_metadata_flow"
    assert candidate["pruning_class"] == "auxiliary"


def test_residual_and_layernorm_are_blocked() -> None:
    candidates = build_candidates(
        [
            region("res", "residual_merge", "blocked", blockers=["residual_hidden_dim"]),
            region("ln", "layer_norm", "protected", blockers=["layernorm_hidden_dim"]),
        ]
    )

    by_name = {item["region_name"]: item for item in candidates}
    assert by_name["res"]["candidate_kind"] == "residual_hidden_blocked"
    assert by_name["res"]["pruning_class"] == "blocked"
    assert by_name["ln"]["candidate_kind"] == "layernorm_hidden_blocked"
    assert by_name["ln"]["pruning_class"] == "blocked"


def test_shape_axis_transform_is_auxiliary() -> None:
    candidate = build_candidates([region("shape", "shape_axis_transform", "propagation_only")])[0]

    assert candidate["pruning_class"] == "auxiliary"


def test_unknown_region_is_unknown() -> None:
    candidate = build_candidates([region("unknown", "unknown", "unknown")])[0]

    assert candidate["pruning_class"] == "unknown"


def test_missing_op_semantics_lowers_confidence() -> None:
    candidate = build_candidates(
        [
            region(
                "ffn",
                "feed_forward_block",
                "directly_prunable",
                dims=[{"dim_name": "intermediate_dim", "status": "prunable"}],
                repairs=["same_indices_across_mlp", "prune_consumer_input"],
                ops=["missing"],
            )
        ],
        [],
    )[0]

    assert candidate["confidence"] == "medium"
    assert "missing_op_semantics_evidence" in candidate["warnings"]


def test_op_region_disagreement_constrains_candidate() -> None:
    candidate = build_candidates([region("ffn_int", "ffn_intermediate_projection", "directly_prunable", ops=["a"])], [op("a", "attention_score_matmul", "blocked")])[0]

    assert candidate["pruning_class"] == "constrained"
    assert "op_region_semantics_disagreement" in candidate["warnings"]


def test_summary_counts_classes() -> None:
    ranking = build_pruning_opportunity_ranking(
        {
            "model_name": "synthetic",
            "regions": [
                region("safe", "ffn_intermediate_projection", "directly_prunable"),
                region("blocked", "attention_score_matmul", "constraint_carrier"),
                region("aux", "attention_mask_add", "constraint_carrier"),
            ],
        }
    )
    summary = pruning_opportunity_ranking_to_dict(ranking)["summary"]

    assert summary["safe_candidates"] == 1
    assert summary["blocked_candidates"] == 1
    assert summary["auxiliary_candidates"] == 1

