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
    (onnx.parent / "subgraph.dot").write_text("digraph ffn {}\n", encoding="utf-8")
    (onnx.parent / "subgraph.svg").write_text("<svg></svg>\n", encoding="utf-8")
    annotated = root / "artifacts/annotated_onnx/bert-base-uncased/layer_0/feed_forward.axis_annotated.onnx"
    annotated.parent.mkdir(parents=True)
    annotated.write_bytes(b"annotated")
    (annotated.parent / "feed_forward.axis_annotated.dot").write_text("digraph annotated {}\n", encoding="utf-8")
    (annotated.parent / "feed_forward.axis_annotated.svg").write_text("<svg></svg>\n", encoding="utf-8")
    write_json(
        root / "reports/onnx_axis_semantics/bert-base-uncased_layer0_feed_forward.json",
        {
            "strict_mlir_semantics": True,
            "semantic_counts": {"UNKNOWN": 1},
            "leader_candidate_counts": {"unknown": 1},
            "evidence_tier_counts": {"NONE": 1},
            "blocker_counts": {"mlir_toolchain_missing": 1},
        },
    )
    (root / "reports/onnx_axis_semantics/bert-base-uncased_layer0_feed_forward.leaders.md").write_text("# Leaders\n", encoding="utf-8")
    mlir_root = root / "reports/mlir_evidence_coverage_bert_24_plan/artifacts/bert_layer0_mlp"
    (mlir_root / "mlir_artifacts").mkdir(parents=True)
    (mlir_root / "mlir_artifacts/subgraph_onnx.onnx.mlir").write_text('%0 = "onnx.MatMul"() : () -> tensor<1xf32>\n', encoding="utf-8")
    (mlir_root / "mlir_artifacts/subgraph_lowered.onnx.mlir").write_text("affine.for %j = 0 to 4 {\n  %0 = affine.load %X[%j] : memref<4xf32>\n  affine.store %0, %Y[%j] : memref<4xf32>\n}\n", encoding="utf-8")
    write_json(
        mlir_root / "native/subgraph_lowered.onnx.native_dependence.json",
        {
            "analysis_tool": "native_mlir_pass",
            "relations": [
                {
                    "relation_kind": "preserved",
                    "source_tensor": "X",
                    "target_tensor": "Y",
                }
            ],
        },
    )
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


def test_pipeline_flow_returns_visual_stages_and_examples(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    status, flow = module.route_api(config, "/api/pipeline-flow", {})

    assert status == module.HTTPStatus.OK
    assert flow["title"] == "Static Pruning Propagation Pipeline"
    assert len(flow["stages"]) == 7
    assert flow["aggregate"]["proven_plans"] == 108
    assert flow["stages"][0]["visual"]["type"] == "graph"
    assert flow["examples"]["attention_value"]["relations"][0]["relation"] == "PRESERVED"
    assert "qk_score_contraction_mixes_channels" in flow["examples"]["qk_blocker"]["facts"][-1]


def test_overview_survives_missing_final_report(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)
    (config.root / "reports/final/static_pruning_propagation_final_summary.json").unlink()

    status, data = module.route_api(config, "/api/overview", {})

    assert status == module.HTTPStatus.OK
    assert data["final_summary"]["proven_plans"] == 0
    assert data["warnings"]


def test_evidence_traces_returns_graph_mlir_pattern_and_dfa_examples(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    status, data = module.route_api(config, "/api/evidence-traces", {})

    assert status == module.HTTPStatus.OK
    assert len(data["examples"]) == 3
    ffn = next(example for example in data["examples"] if example["id"] == "ffn_intermediate")
    qk = next(example for example in data["examples"] if example["id"] == "qk_score_blocker")
    assert ffn["graph"]["nodes"]
    assert ffn["mlir"]
    assert ffn["pattern_match"]["after"] == "FFN_INTERMEDIATE_CHAIN"
    assert ffn["dfa_trace"][0]["kind"] == "seed"
    assert qk["verdict"] == "blocked_as_expected"


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


def test_artifact_text_reads_safe_mlir_and_rejects_traversal(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)
    mlir_path = "reports/mlir_evidence_coverage_bert_24_plan/artifacts/bert_layer0_mlp/mlir_artifacts/subgraph_lowered.onnx.mlir"

    status, payload = module.route_api(config, "/api/artifact-text", {"path": [mlir_path]})

    assert status == module.HTTPStatus.OK
    assert payload["language"] == "mlir"
    assert "affine.load" in payload["text"]
    assert payload["truncated"] is False

    status, payload = module.route_api(config, "/api/artifact-text", {"path": ["../secret.mlir"]})
    assert status == module.HTTPStatus.BAD_REQUEST
    assert payload["error"] == "invalid artifact path"


def test_artifact_text_focus_extracts_affine_section(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)
    mlir_path = "reports/mlir_evidence_coverage_bert_24_plan/artifacts/bert_layer0_mlp/mlir_artifacts/subgraph_lowered.onnx.mlir"

    status, payload = module.route_api(
        config,
        "/api/artifact-text",
        {"path": [mlir_path], "focus": ["affine"], "context": ["2"]},
    )

    assert status == module.HTTPStatus.OK
    assert payload["sections"]
    assert {match["kind"] for match in payload["matches"]} >= {"affine.for", "affine.load", "affine.store"}
    assert "affine.for" in payload["text"]
    assert payload["sections"][0]["start_line"] <= payload["matches"][0]["line_no"]


def test_artifact_text_focus_falls_back_to_onnx_gemm(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)
    mlir_path = "reports/mlir_evidence_coverage_bert_24_plan/artifacts/bert_layer0_mlp/mlir_artifacts/subgraph_onnx.onnx.mlir"

    status, payload = module.route_api(
        config,
        "/api/artifact-text",
        {"path": [mlir_path], "focus": ["affine"], "context": ["1"]},
    )

    assert status == module.HTTPStatus.OK
    assert payload["sections"]
    assert payload["matches"][0]["kind"] == "onnx.MatMul"
    assert any("No affine.for" in warning for warning in payload["warnings"])


def test_artifact_bundle_discovers_graph_mlir_and_dependence_files(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    status, payload = module.route_api(
        config,
        "/api/artifact-bundle",
        {"model": ["bert-base-uncased"], "layer": ["0"], "subgraph": ["01_ffn"]},
    )

    assert status == module.HTTPStatus.OK
    assert payload["paths"]["svg"].endswith("subgraph.svg")
    assert payload["paths"]["dot"].endswith("subgraph.dot")
    assert payload["paths"]["annotated_onnx"].endswith("feed_forward.axis_annotated.onnx")
    assert payload["paths"]["annotated_svg"].endswith("feed_forward.axis_annotated.svg")
    assert payload["paths"]["axis_semantics_json"].endswith("bert-base-uncased_layer0_feed_forward.json")
    assert payload["paths"]["axis_leader_report"].endswith("bert-base-uncased_layer0_feed_forward.leaders.md")
    assert payload["mlir"]["available"]
    assert any(artifact["stage"] == "lowered_affine" for artifact in payload["mlir"]["artifacts"])
    assert payload["dependence"]["native_json"].endswith("native_dependence.json")
    assert payload["evidence"]["pattern"] == "FFN_INTERMEDIATE_CHAIN"
    assert payload["evidence"]["evidence_tier"] == "native_mlir_dependence_evidence"
    assert payload["evidence"]["semantic_counts"] == {"UNKNOWN": 1}
    assert payload["evidence"]["leader_candidate_counts"] == {"unknown": 1}
    lowered = next(artifact for artifact in payload["mlir"]["artifacts"] if artifact["stage"] == "lowered_affine")
    assert lowered["interesting_counts"]["affine.for"] == 1
    assert lowered["interesting_counts"]["affine.load"] == 1
    assert lowered["interesting_counts"]["affine.store"] == 1
    assert lowered["first_interesting_line"] == 1
    assert "focus=affine" in lowered["focused_text_url"]


def test_evidence_artifact_map_has_canonical_examples(tmp_path: Path) -> None:
    module = load_server_module()
    config = make_tree(tmp_path)

    status, payload = module.route_api(config, "/api/evidence-artifact-map", {})

    assert status == module.HTTPStatus.OK
    assert payload["ffn_intermediate"]["subgraph"] == "12_layer_0_feed_forward"
    assert payload["attention_value_path"]["model"] == "bert-base-uncased"
    assert payload["qk_score_blocker"]["subgraph"] == "05_layer_0_attention_score_matmul"
