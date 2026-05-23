from __future__ import annotations

from model_analysis.region_ir_analysis import explain_region_blocked_dimensions
from test_region_ir_graph import synthetic_region_ir


def test_blocked_dimension_explanations_include_mitigations() -> None:
    rows = explain_region_blocked_dimensions(synthetic_region_ir())

    residual = next(item for item in rows if item["block_type"] == "residual_hidden_equality")
    axis = next(item for item in rows if item["block_type"] == "axis_transform_mapping")
    assert "keep hidden_dim unchanged" in residual["mitigation"].lower()
    assert "axis mapping" in axis["mitigation"].lower()
