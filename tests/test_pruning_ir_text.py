from __future__ import annotations

from model_analysis.dimension_ir import build_pruning_ir
from model_analysis.pruning_ir_text import pruning_ir_to_text


def synthetic_pruning_map() -> dict:
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "pruning_dimensions": [
            {
                "dim_id": "dim::torch:linear:fc1::intermediate_dim",
                "unit_id": "torch:linear:fc1",
                "unit_name": "fc1",
                "unit_type": "mlp_expansion",
                "dim_name": "intermediate_dim",
                "size": 8,
                "structural_role": "coupled",
                "confidence": "medium",
                "reason": "fc1 intermediate",
            },
            {
                "dim_id": "dim::torch:linear:fc2::intermediate_dim",
                "unit_id": "torch:linear:fc2",
                "unit_name": "fc2",
                "unit_type": "mlp_projection",
                "dim_name": "intermediate_dim",
                "size": 8,
                "structural_role": "coupled",
                "confidence": "medium",
                "reason": "fc2 intermediate",
            },
        ],
        "propagation_constraints": [
            {
                "constraint_id": "constraint_00001",
                "src_dim_id": "dim::torch:linear:fc1::intermediate_dim",
                "dst_dim_id": "dim::torch:linear:fc2::intermediate_dim",
                "constraint_type": "mlp_same_intermediate_indices",
                "direction": "bidirectional",
                "edge_type": "mlp_hidden_coupling",
                "confidence": "medium",
                "evidence": [],
                "reason": "MLP pair",
            },
            {
                "constraint_id": "constraint_00002",
                "src_dim_id": "dim::missing::x",
                "dst_dim_id": "dim::torch:linear:fc2::intermediate_dim",
                "constraint_type": "unknown_mapping",
                "direction": "forward",
                "edge_type": "shape_dependency",
                "confidence": "low",
                "evidence": [],
                "reason": "Unknown mapping",
            },
        ],
        "opportunities": [
            {
                "opportunity_id": "opp::blocked",
                "risk_level": "blocked",
                "executability": "blocked",
                "prunable_dimensions": ["dim::torch:linear:fc2::intermediate_dim"],
            }
        ],
        "blocked_opportunities": ["opp::blocked"],
    }


def test_pruning_ir_text_contains_expected_ops():
    ir = build_pruning_ir(synthetic_pruning_map())

    text = pruning_ir_to_text(ir)

    assert "pruning.module @tiny" in text
    assert "pruning.dim" in text
    assert "pruning.constraint" in text
    assert "pruning.eq_class" in text
    assert "// blocked:" in text
    assert "// unresolved:" in text
