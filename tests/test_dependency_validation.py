from model_analysis.correspondence import ModuleNodeCorrespondence, ParameterEvidence, CorrespondenceReport
from model_analysis.dependency_validation import dependency_validation_to_markdown, validate_dependency_graph_with_evidence
from model_analysis.shape_evidence import NodeShapeEvidence, ShapeEvidenceReport


def fake_dependency_graph():
    return {
        "model_name": "tiny",
        "prunable_units": [
            {"unit_id": "torch:linear:q", "name": "q", "module_or_node_name": "q"},
            {"unit_id": "torch:linear:k", "name": "k", "module_or_node_name": "k"},
        ],
        "dependency_edges": [
            {
                "src": "torch:linear:q",
                "dst": "torch:linear:k",
                "edge_type": "qkv_coupling",
                "affected_dims": ["hidden_dim"],
                "direction": "bidirectional",
            }
        ],
    }


def fake_correspondence_report():
    q_param = ParameterEvidence("q.weight", "q", [8, 8], "q.weight", [8, 8], "exact_name", "high", "matched")
    k_param = ParameterEvidence("k.weight", "k", [8, 8], "k.weight", [8, 8], "exact_name", "high", "matched")
    return CorrespondenceReport(
        model_name="tiny",
        hf_id="local/tiny",
        task="unit-test",
        module_node_correspondences=[
            ModuleNodeCorrespondence(
                torch_module_name="q",
                torch_module_type="Linear",
                torch_unit_id="torch:linear:q",
                onnx_node_names=["q/Gemm"],
                onnx_op_types=["Gemm"],
                onnx_initializer_names=["q.weight"],
                input_tensors=["x"],
                output_tensors=["q_out"],
                input_shapes={"x": [1, 8]},
                output_shapes={"q_out": [1, 8]},
                parameter_evidence=[q_param],
                confidence="high",
                reason="matched",
            ),
            ModuleNodeCorrespondence(
                torch_module_name="k",
                torch_module_type="Linear",
                torch_unit_id="torch:linear:k",
                onnx_node_names=["k/Gemm"],
                onnx_op_types=["Gemm"],
                onnx_initializer_names=["k.weight"],
                input_tensors=["x"],
                output_tensors=["k_out"],
                input_shapes={"x": [1, 8]},
                output_shapes={"k_out": [1, 8]},
                parameter_evidence=[k_param],
                confidence="high",
                reason="matched",
            ),
        ],
        parameter_evidence=[q_param, k_param],
    )


def fake_shape_report():
    return ShapeEvidenceReport(
        model_name="tiny",
        hf_id="local/tiny",
        task="unit-test",
        node_shapes=[
            NodeShapeEvidence("q/Gemm", "Gemm", {"x": [1, 8]}, {"q_out": [1, 8]}, [], "high", "known"),
            NodeShapeEvidence("k/Gemm", "Gemm", {"x": [1, 8]}, {"k_out": [1, 8]}, [], "high", "known"),
        ],
    )


def test_validate_dependency_graph_with_evidence_supports_qkv_edge():
    validation = validate_dependency_graph_with_evidence(fake_dependency_graph(), fake_correspondence_report(), fake_shape_report())

    assert validation["summary"]["num_validated_units"] == 2
    assert validation["summary"]["num_validated_edges"] == 1
    assert validation["shape_supported_edges"]
    assert validation["correspondence_supported_edges"]


def test_dependency_validation_markdown_has_sections():
    validation = validate_dependency_graph_with_evidence(fake_dependency_graph(), fake_correspondence_report(), fake_shape_report())
    markdown = dependency_validation_to_markdown(validation)

    assert "# Validated Dependency Graph: tiny" in markdown
    assert "## Validated Units" in markdown
    assert "## Shape-Supported Edges" in markdown
    assert "## Manual Review Items" in markdown
