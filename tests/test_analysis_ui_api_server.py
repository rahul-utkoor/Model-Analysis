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
    write_json(root / "reports/deadbranch_propagation/bert-base-uncased.json", {"model_name": "bert-base-uncased", "summary": {"total_pairs": 1}})
    write_json(
        root / "reports/final/static_pruning_propagation_final_summary.json",
        {
            "aggregate": {
                "expected_plans": 108,
                "proven_plans": 108,
                "native_mlir_evidence": 108,
                "high_level_mlir_fallback": 0,
                "unsupported": 0,
                "partial": 0,
                "missing": 0,
                "failed": 0,
            },
            "models": [
                {
                    "model_name": "bert-base-uncased",
                    "layers": 12,
                    "expected_plans": 24,
                    "proven_plans": 24,
                    "ffn_proven": 12,
                    "attention_value_proven": 12,
                    "native_evidence": 24,
                    "fallback_evidence": 0,
                    "final_verdict": "complete_plan_proof",
                }
            ],
        },
    )
    (root / "reports/final/static_pruning_propagation_final_report.md").write_text("# Final report\n", encoding="utf-8")
    write_json(root / "reports/formalization/index.json", {"reports": []})
    write_json(root / "reports/all_model_plan_proof/index.json", {"aggregate": {"total_expected": 108, "total_proven": 108}})
    (root / "reports/bert_24_plan_proof").mkdir(parents=True)
    (root / "reports/bert_24_plan_proof/index.md").write_text("# BERT proof\n", encoding="utf-8")
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


def test_deadbranch_endpoint_returns_optional_report(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    status, report = module.route_api(config, "/api/models/bert-base-uncased/deadbranch", {})
    assert status == module.HTTPStatus.OK
    assert report["summary"]["total_pairs"] == 1


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


def test_teaching_overview_and_proof_summary_endpoints(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    status, overview = module.route_api(config, "/api/overview", {})
    assert status == module.HTTPStatus.OK
    assert overview["title"] == "Static Pruning Propagation Analysis"
    assert overview["final_summary"]["proven_plans"] == 108
    assert any(step["id"] == "dfa" for step in overview["pipeline_steps"])

    status, proof = module.route_api(config, "/api/proof-summary", {})
    assert status == module.HTTPStatus.OK
    assert proof["aggregate"]["native_mlir_evidence"] == 108
    assert proof["models"][0]["attention_value_proven"] == 12


def test_teaching_flow_and_case_studies_endpoints(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    status, flow = module.route_api(config, "/api/teaching-flow", {})
    assert status == module.HTTPStatus.OK
    assert any(section["id"] == "qk" for section in flow["sections"])

    status, studies = module.route_api(config, "/api/case-studies", {})
    assert status == module.HTTPStatus.OK
    bert = next(study for study in studies["case_studies"] if study["id"] == "bert-24-plan")
    assert bert["available"]
    assert bert["key_numbers"]["proven"] == "24 / 24"


def test_report_text_reads_safe_markdown_and_rejects_traversal(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    status, report = module.route_api(
        config,
        "/api/report-text",
        {"path": ["final/static_pruning_propagation_final_report.md"]},
    )
    assert status == module.HTTPStatus.OK
    assert report["text"] == "# Final report\n"

    status, report = module.route_api(config, "/api/report-text", {"path": ["../secret.md"]})
    assert status == module.HTTPStatus.BAD_REQUEST
    assert report["error"] == "invalid report path"
