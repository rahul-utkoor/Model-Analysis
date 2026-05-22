from model_analysis.correspondence import (
    build_module_node_correspondence,
    build_parameter_evidence,
    correspondence_report_to_markdown,
    name_similarity_score,
    normalize_name,
)


def fake_torch_summary():
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "linear_layers": [
            {
                "name": "encoder.layer.0.attention.self.query",
                "in_features": 768,
                "out_features": 768,
                "parameters": 590592,
                "weight_name": "encoder.layer.0.attention.self.query.weight",
                "bias_name": "encoder.layer.0.attention.self.query.bias",
                "weight_shape": [768, 768],
                "bias_shape": [768],
            }
        ],
        "embedding_layers": [],
        "normalization_layers": [],
    }


def fake_onnx_summary():
    return {
        "model_name": "tiny",
        "hf_id": "local/tiny",
        "task": "unit-test",
        "initializers": [
            {"name": "encoder.layer.0.attention.self.query.weight", "dims": [768, 768], "data_type": "FLOAT"},
            {"name": "encoder.layer.0.attention.self.query.bias", "dims": [768], "data_type": "FLOAT"},
            {"name": "unmatched.weight", "dims": [10, 10], "data_type": "FLOAT"},
        ],
        "initializer_shapes": {
            "encoder.layer.0.attention.self.query.weight": [768, 768],
            "encoder.layer.0.attention.self.query.bias": [768],
        },
        "tensor_shape_map": {
            "hidden": [1, 16, 768],
            "query_out": [1, 16, 768],
            "encoder.layer.0.attention.self.query.weight": [768, 768],
        },
        "nodes": [
            {
                "name": "encoder.layer.0.attention.self.query/Gemm",
                "op_type": "Gemm",
                "inputs": ["hidden", "encoder.layer.0.attention.self.query.weight", "encoder.layer.0.attention.self.query.bias"],
                "outputs": ["query_out"],
            }
        ],
    }


def test_normalize_name_and_similarity():
    assert normalize_name("model.encoder.layer-0/query.weight") == "encoder_layer_0_query_weight"
    assert name_similarity_score("encoder.layer.0.query.weight", "model_encoder_layer_0_query_weight") == 1.0
    assert name_similarity_score("layer.0.query.weight", "encoder.layer.0.query.weight") >= 0.70


def test_build_parameter_evidence_matches_exact_and_unmatched_initializer_reported():
    evidence = build_parameter_evidence(fake_torch_summary(), fake_onnx_summary())

    assert evidence[0].confidence == "high"
    assert evidence[0].match_type == "exact_name"
    assert evidence[0].onnx_initializer_name == "encoder.layer.0.attention.self.query.weight"


def test_build_module_node_correspondence_finds_gemm():
    parameter_evidence = build_parameter_evidence(fake_torch_summary(), fake_onnx_summary())
    report = build_module_node_correspondence(
        fake_torch_summary(),
        fake_onnx_summary(),
        parameter_evidence,
        {
            "prunable_units": [
                {
                    "unit_id": "torch:linear:encoder.layer.0.attention.self.query",
                    "name": "encoder.layer.0.attention.self.query",
                    "module_or_node_name": "encoder.layer.0.attention.self.query",
                }
            ]
        },
    )

    corr = report.module_node_correspondences[0]
    assert corr.confidence in {"high", "medium"}
    assert corr.onnx_node_names == ["encoder.layer.0.attention.self.query/Gemm"]
    assert corr.torch_unit_id == "torch:linear:encoder.layer.0.attention.self.query"
    assert report.unmatched_onnx_initializers
    assert "# PyTorch-to-ONNX Correspondence: tiny" in correspondence_report_to_markdown(report)
