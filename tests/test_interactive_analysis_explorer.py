from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from model_analysis.interactive_analysis_explorer import (
    discover_layers,
    discover_models,
    discover_subgraphs,
    find_onnx_path,
    search_subgraphs,
    summarize_subgraph,
    validation_summary,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_report_tree(tmp_path: Path) -> Path:
    root = tmp_path / "reports" / "model_analysis_reports"
    write_json(root / "bert-base-uncased" / "index.json", {"model_name": "bert-base-uncased", "model_summary": {}})
    write_json(root / "cross_model" / "index.json", {"model_name": "cross"})
    write_json(root / "bert-base-uncased" / "layers" / "layer_0" / "index.json", {"summary": {"layer_index": 0, "total_subgraphs": 1}})
    write_json(root / "bert-base-uncased" / "layers" / "layer_1" / "index.json", {"summary": {"layer_index": 1, "total_subgraphs": 0}})
    subgraph = {
        "ordinal": 1,
        "node_slug": "01_layer_0_feed_forward",
        "display_name": "Layer 0 Feed Forward",
        "semantic_category": "feed_forward_block",
        "source_region_type": "GenericMLPRegion",
        "classification": {"pruning_class": "safe", "plan_status": "valid_plan", "validation_status": "valid"},
        "onnx_export": {"status": "exported"},
        "primitive_ops": [{"source_name": "ffn", "op_type": "MatMul"}],
        "local_ranking": [{}],
        "local_plans": [{}],
        "local_validations": [{}],
        "verdict": "safe; symbolic plan exists and validates",
    }
    write_json(root / "bert-base-uncased" / "layers" / "layer_0" / "subgraphs" / "01_layer_0_feed_forward" / "analysis.json", subgraph)
    (root / "bert-base-uncased" / "layers" / "layer_0" / "subgraphs" / "01_layer_0_feed_forward" / "explanation.md").write_text("Feed Forward explanation", encoding="utf-8")
    return root


def test_discover_models_excludes_cross_model(tmp_path: Path) -> None:
    root = make_report_tree(tmp_path)

    models = discover_models(root)

    assert [model.safe_name for model in models] == ["bert-base-uncased"]


def test_discover_layers_finds_layer_indices(tmp_path: Path) -> None:
    model = discover_models(make_report_tree(tmp_path))[0]

    assert [item["layer_index"] for item in discover_layers(model.model_dir)] == [0, 1]


def test_discover_subgraphs_finds_analysis_json(tmp_path: Path) -> None:
    model = discover_models(make_report_tree(tmp_path))[0]

    subgraphs = discover_subgraphs(model.model_dir, 0)

    assert subgraphs[0]["display_name"] == "Layer 0 Feed Forward"
    assert subgraphs[0]["_explanation_path"].endswith("explanation.md")


def test_onnx_path_lookup_prefers_model_analysis_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "artifacts/model_analysis_subgraphs/bert-base-uncased/layers/layer_0/01_ffn/subgraph.onnx"
    path.parent.mkdir(parents=True)
    path.write_text("onnx", encoding="utf-8")

    found = find_onnx_path("bert-base-uncased", 0, "01_ffn", tmp_path / "artifacts/model_analysis_subgraphs", tmp_path / "artifacts/layer_subgraphs")

    assert found == path


def test_onnx_path_lookup_falls_back_to_layer_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "artifacts/layer_subgraphs/bert-base-uncased/layer_0/01_ffn/subgraph.onnx"
    path.parent.mkdir(parents=True)
    path.write_text("onnx", encoding="utf-8")

    found = find_onnx_path("bert-base-uncased", 0, "01_ffn", tmp_path / "artifacts/model_analysis_subgraphs", tmp_path / "artifacts/layer_subgraphs")

    assert found == path


def test_summarize_subgraph_returns_compact_fields(tmp_path: Path) -> None:
    model = discover_models(make_report_tree(tmp_path))[0]
    record = discover_subgraphs(model.model_dir, 0)[0]

    summary = summarize_subgraph(record)

    assert summary["display_name"] == "Layer 0 Feed Forward"
    assert summary["pruning_class"] == "safe"
    assert summary["plan_status"] == "valid_plan"
    assert summary["validation_status"] == "valid"


def test_search_subgraphs_matches_name_and_category(tmp_path: Path) -> None:
    model = discover_models(make_report_tree(tmp_path))[0]
    subgraphs = discover_subgraphs(model.model_dir, 0)

    assert search_subgraphs(subgraphs, "Feed Forward")
    assert search_subgraphs(subgraphs, "feed_forward_block")


def test_missing_files_return_empty_results(tmp_path: Path) -> None:
    assert discover_models(tmp_path / "missing") == []
    assert discover_layers(tmp_path / "missing-model") == []
    assert discover_subgraphs(tmp_path / "missing-model", 0) == []


def test_validation_summary_reads_canonical_and_compat_fields() -> None:
    assert validation_summary({"total_validations": 2, "valid": 1, "invalid": 1})["valid"] == 1
    assert validation_summary({"total_plans": 2, "valid_plans": 2})["valid"] == 2


def test_cli_help_works() -> None:
    result = subprocess.run([sys.executable, "tools/interactive_analysis_explorer.py", "--help"], text=True, capture_output=True, check=False)

    assert result.returncode == 0
    assert "--model" in result.stdout
