from __future__ import annotations

from model_analysis.op_semantics import build_op_semantics_ir, op_semantics_ir_to_markdown
from model_analysis.op_semantics_text import op_semantics_to_text


def test_text_dump_contains_semantic_kind_lines() -> None:
    ir = build_op_semantics_ir(
        {
            "model_name": "synthetic",
            "source_frontend": "onnx",
            "ops": [
                {
                    "op_id": "op_0",
                    "source_node_name": "/model/bert/encoder/layer.0/attention/self/MatMul",
                    "name": "/model/bert/encoder/layer.0/attention/self/MatMul",
                    "op_type": "MatMul",
                    "source_location": {"node_index": 0},
                }
            ],
        }
    )

    text = op_semantics_to_text(ir)
    assert "op_semantics @synthetic" in text
    assert "semantic_kind = attention_score_matmul" in text
    assert "category = attention_contraction" in text


def test_markdown_contains_summary_and_sections() -> None:
    ir = build_op_semantics_ir(
        {
            "model_name": "synthetic",
            "source_frontend": "onnx",
            "ops": [
                {
                    "op_id": "op_0",
                    "source_node_name": "/model/bert/encoder/layer.0/intermediate/dense/MatMul",
                    "name": "/model/bert/encoder/layer.0/intermediate/dense/MatMul",
                    "op_type": "MatMul",
                    "source_location": {"node_index": 0},
                }
            ],
        }
    )

    markdown = op_semantics_ir_to_markdown(ir)
    assert "# Op Semantics: synthetic" in markdown
    assert "Semantic Kind Counts" in markdown
    assert "Parameterized Projection Ops" in markdown
    assert "parameterized_linear_matmul" in markdown

