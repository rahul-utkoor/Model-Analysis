from __future__ import annotations

from model_analysis.ir_analysis import explain_blocked_regions


def test_blocked_region_explanations_include_mitigation():
    ir = {
        "model_name": "tiny",
        "dimension_variables": [
            {"var_id": "dim::hidden", "prunable": True},
            {"var_id": "dim::peer", "prunable": True},
        ],
        "constraint_equations": [
            {
                "constraint_id": "c1",
                "lhs": "dim::hidden",
                "rhs": "dim::peer",
                "relation": "eq",
                "constraint_type": "residual_hidden_equality",
                "blocking": True,
                "reason": "Residual path",
            }
        ],
        "blocked_dimensions": ["dim::hidden"],
    }

    blocked = explain_blocked_regions(ir)

    assert blocked
    assert any("residual" in item["explanation"].lower() for item in blocked)
    assert all(item["mitigation"] for item in blocked)
