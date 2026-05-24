from __future__ import annotations

from model_analysis.op_semantics import build_op_semantics_ir, op_semantics_ir_to_dict
from model_analysis.op_semantics_compare import compare_op_semantics, comparison_to_markdown


def report(model_name: str, path: str, op_type: str) -> dict:
    return op_semantics_ir_to_dict(
        build_op_semantics_ir(
            {
                "model_name": model_name,
                "source_frontend": "onnx",
                "ops": [
                    {
                        "op_id": "op_0",
                        "source_node_name": path,
                        "name": path,
                        "op_type": op_type,
                        "source_location": {"node_index": 0},
                    }
                ],
            }
        )
    )


def test_compare_aggregates_semantic_counts() -> None:
    comparison = compare_op_semantics(
        [
            report("a", "/model/bert/encoder/layer.0/attention/self/MatMul", "MatMul"),
            report("b", "/model/bert/encoder/layer.0/intermediate/dense/MatMul", "MatMul"),
        ]
    )

    assert comparison["semantic_kind_matrix"]["attention_score_matmul"]["a"] == 1
    assert comparison["semantic_category_matrix"]["parameterized_projection"]["b"] == 1
    markdown = comparison_to_markdown(comparison)
    assert "Op Semantics Comparison" in markdown
    assert "Semantic Kinds" in markdown

