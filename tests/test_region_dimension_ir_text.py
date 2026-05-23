from __future__ import annotations

from model_analysis.region_dimension_ir import build_region_dimension_ir
from model_analysis.region_dimension_ir_text import region_dimension_ir_to_text
from test_region_dimension_ir import synthetic_region_tree


def test_region_dimension_ir_text_contains_dimensions_constraints_and_classes() -> None:
    text = region_dimension_ir_to_text(build_region_dimension_ir(synthetic_region_tree()))

    assert "region_dim.module @tiny-regions" in text
    assert "region_dim %" in text
    assert "region_constraint %" in text
    assert "region_eq_class %" in text
    assert "mlp_intermediate_same_indices" in text
    assert "// blocked:" in text
    assert "// unresolved:" in text
