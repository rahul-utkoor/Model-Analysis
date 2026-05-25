from __future__ import annotations

from pathlib import Path

from model_analysis.full_model_analysis_report import (
    layer_summary_from_pack,
    polish_subgraph,
    polished_verdict,
)
from model_analysis.full_model_analysis_report_text import layer_report_to_markdown, model_report_to_markdown


def _subgraph(ordinal: int, name: str, category: str, pruning_class: str) -> dict:
    return {
        "ordinal": ordinal,
        "node_slug": f"{ordinal:02d}_{category}",
        "display_name": name,
        "layer_index": 0,
        "semantic_category": category,
        "source_region_type": "SyntheticRegion",
        "classification": {
            "pruning_class": pruning_class,
            "plan_status": "no_plan_expected",
            "validation_status": "not_applicable",
        },
        "primitive_ops": [{"source_name": "op", "op_type": "MatMul", "topological_index": ordinal}],
        "local_op_semantics": [],
        "local_ranking": [],
        "local_plans": [],
        "local_validations": [],
        "onnx_export": {"status": "skipped"},
    }


def test_polished_attention_verdicts_and_layernorm_names():
    assert "head-axis mapping proof" in polished_verdict(_subgraph(4, "Layer 0 Attention", "attention_skeleton", "blocked"))
    assert "probability normalization" in polished_verdict(_subgraph(7, "Layer 0 Attention Softmax", "attention_softmax", "blocked"))
    assert "output hidden_dim feeds residual" in polished_verdict(
        _subgraph(9, "Layer 0 Attention Output Projection", "attention_output_projection", "constrained")
    )

    first_ln = polish_subgraph(_subgraph(11, "Layer 0 LayerNorm", "layer_norm", "blocked"))
    second_ln = polish_subgraph(_subgraph(17, "Layer 0 LayerNorm", "layer_norm", "blocked"))
    assert first_ln["display_name"] == "Layer 0 Attention Output LayerNorm"
    assert second_ln["display_name"] == "Layer 0 FFN Output LayerNorm"


def test_layer_summary_and_markdown_tables(tmp_path: Path):
    pack = {
        "model_name": "bert-base-uncased",
        "layer_index": 0,
        "subgraphs": [
            polish_subgraph(_subgraph(1, "Layer 0 Query Projection", "query_projection", "constrained")),
            polish_subgraph(_subgraph(12, "Layer 0 Feed Forward", "feed_forward_block", "safe")),
        ],
        "summary": {
            "total_subgraphs": 2,
            "onnx_exported": 0,
            "onnx_skipped": 2,
            "onnx_failed": 0,
            "safe_subgraphs": 1,
            "constrained_subgraphs": 1,
            "blocked_subgraphs": 0,
            "auxiliary_subgraphs": 0,
            "unknown_subgraphs": 0,
            "valid_plan_subgraphs": 1,
        },
    }
    layer_summary = layer_summary_from_pack(
        model_report_root=tmp_path / "reports",
        artifact_root=tmp_path / "artifacts",
        model_name="bert-base-uncased",
        layer_index=0,
        pack=pack,
    )
    assert layer_summary["total_subgraphs"] == 2
    assert layer_summary["safe"] == 1
    markdown = layer_report_to_markdown({"pack": pack, "summary": layer_summary})
    assert "## Ordered subgraph table" in markdown
    assert "Layer 0 Feed Forward" in markdown


def test_model_index_markdown_has_layer_table():
    report = {
        "model_name": "bert-base-uncased",
        "available_artifacts": {},
        "model_summary": {
            "layers_generated": 1,
            "total_subgraphs": 2,
            "safe": 1,
            "constrained": 1,
            "blocked": 0,
            "auxiliary": 0,
            "unknown": 0,
            "ranking": {"safe": 1, "constrained": 1, "blocked": 0, "auxiliary": 0, "unknown": 0},
            "plans": {"ready_symbolic": 1, "incomplete": 0, "blocked": 0, "unknown": 0},
            "plan_validation": {"valid": 1, "warning": 0, "invalid": 0, "unknown": 0},
            "op_semantic_counts": {},
            "region_semantic_counts": {},
        },
        "layers": [{"layer_index": 0, "total_subgraphs": 2, "safe": 1, "constrained": 1}],
        "safe_opportunities": [],
        "constrained_opportunities": [],
        "blocked_structures": [],
        "auxiliary_structures": [],
    }
    markdown = model_report_to_markdown(report)
    assert "## 4. Layer summary table" in markdown
    assert "## 10. Research conclusions" in markdown
