from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_server_module():
    path = Path("tools/analysis_ui_api_server.py").resolve()
    spec = importlib.util.spec_from_file_location("analysis_ui_api_server", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_tree(tmp_path: Path) -> object:
    module = load_server_module()
    root = tmp_path
    report_root = root / "reports/model_analysis_reports"
    artifact_root = root / "artifacts/model_analysis_subgraphs"
    fallback_layer_root = root / "reports/layer_subgraph_validation"
    write_json(
        report_root / "bert-base-uncased/index.json",
        {
            "model_name": "bert-base-uncased",
            "model_summary": {
                "layers_generated": 1,
                "total_subgraphs": 1,
                "ranking": {"safe": 1, "mlp_safe_candidates": 1},
                "plans": {"total_plans": 1},
                "plan_validation": {"valid": 1},
            },
        },
    )
    write_json(report_root / "cross_model/index.json", {"model_name": "cross"})
    write_json(
        report_root / "bert-base-uncased/layers/layer_0/index.json",
        {"summary": {"layer_index": 0, "total_subgraphs": 1, "safe": 1, "constrained": 0, "blocked": 0, "auxiliary": 0, "unknown": 0, "valid_plan_subgraphs": 1}},
    )
    subgraph = {
        "ordinal": 1,
        "node_slug": "01_ffn",
        "display_name": "Layer 0 Feed Forward",
        "semantic_category": "feed_forward_block",
        "classification": {"pruning_class": "safe", "plan_status": "valid_plan", "validation_status": "valid"},
        "onnx_export": {"status": "exported"},
        "explanation": "safe FFN",
    }
    write_json(report_root / "bert-base-uncased/layers/layer_0/subgraphs/01_ffn/analysis.json", subgraph)
    (report_root / "bert-base-uncased/layers/layer_0/subgraphs/01_ffn/explanation.md").write_text("Feed Forward explanation", encoding="utf-8")
    onnx = artifact_root / "bert-base-uncased/layers/layer_0/01_ffn/subgraph.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"onnx")
    write_json(root / "reports/static_coverage_study/index.json", {"models": [{"model_name": "bert-base-uncased", "final_status": "complete", "artifacts": {"ranking": {"safe": 1}, "plans": {"plans": 1}, "validation": {"valid_plans": 1}, "full_model_report": {"layers": 1, "subgraphs": 1}}}]})
    return module.ServerConfig(
        root=root,
        report_root=report_root,
        artifact_root=artifact_root,
        fallback_layer_root=fallback_layer_root,
        fallback_artifact_root=root / "artifacts/layer_subgraphs",
        ui_dist=root / "ui/dist",
        verbose=False,
    )


def test_model_discovery_excludes_cross_model(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    models = module.discover_models(config)

    assert [item["id"] for item in models] == ["bert-base-uncased"]


def test_health_and_models_endpoints(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    assert module.route_api(config, "/api/health", {}) == (module.HTTPStatus.OK, {"ok": True})
    status, models = module.route_api(config, "/api/models", {})
    assert status == module.HTTPStatus.OK
    assert models[0]["display_name"] == "bert-base-uncased"


def test_layers_and_subgraphs_endpoints(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    assert module.route_api(config, "/api/models/bert-base-uncased/layers", {})[1][0]["layer_index"] == 0
    assert module.route_api(config, "/api/models/bert-base-uncased/layers/0/subgraphs", {})[1][0]["display_name"] == "Layer 0 Feed Forward"
    status, detail = module.route_api(config, "/api/models/bert-base-uncased/layers/0/subgraphs/01_ffn", {})
    assert status == module.HTTPStatus.OK
    assert detail["analysis"]["semantic_category"] == "feed_forward_block"
    assert "onnx" in detail["artifact_paths"]


def test_search_finds_subgraph_by_name_and_class(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    assert module.route_api(config, "/api/search", {"q": ["Feed Forward"]})[1]["matches"]
    assert module.route_api(config, "/api/search", {"q": ["safe"]})[1]["matches"]


def test_artifact_serving_rejects_path_traversal(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    assert not module.is_allowed_file(config, Path("/etc/passwd"))


def test_missing_model_returns_404_not_stack_trace(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    status, body = module.route_api(config, "/api/models/missing", {})
    assert status == module.HTTPStatus.NOT_FOUND
    assert body["error"] == "model not found"
