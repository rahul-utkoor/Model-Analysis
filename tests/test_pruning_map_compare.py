from __future__ import annotations

from model_analysis.pruning_map_compare import compare_model_pruning_maps


def make_map(name: str, opportunity_types: list[str], risks: list[str]) -> dict:
    return {
        "model_name": name,
        "opportunities": [
            {
                "opportunity_id": f"{name}:{kind}:{index}",
                "opportunity_type": kind,
                "risk_level": "medium",
                "executability": "analysis_only",
            }
            for index, kind in enumerate(opportunity_types)
        ],
        "structural_risks": [{"risk_type": risk} for risk in risks],
        "summary": {
            "opportunity_type_counts": {kind: opportunity_types.count(kind) for kind in set(opportunity_types)},
            "risk_level_counts": {"medium": len(opportunity_types)},
            "executability_counts": {"analysis_only": len(opportunity_types)},
        },
    }


def test_compare_model_pruning_maps_builds_matrices():
    comparison = compare_model_pruning_maps(
        [
            make_map("bert", ["mlp_intermediate", "attention_qkv_heads"], ["qkv_consistency"]),
            make_map("gpt2", ["mlp_intermediate", "onnx_matmul_dimension"], ["qkv_consistency"]),
        ]
    )

    assert comparison["num_models"] == 2
    assert comparison["opportunity_type_matrix"]["bert"]["mlp_intermediate"] == 1
    assert comparison["executability_matrix"]["gpt2"]["analysis_only"] == 2
    assert comparison["common_opportunity_types"] == ["mlp_intermediate"]
    assert comparison["common_risks"] == ["qkv_consistency"]
    assert comparison["model_specific_opportunity_types"]["bert"] == ["attention_qkv_heads"]
