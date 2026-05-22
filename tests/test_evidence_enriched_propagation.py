from model_analysis.correspondence import CorrespondenceReport, ModuleNodeCorrespondence, ParameterEvidence
from model_analysis.dependency_graph import DependencyEdge, DependencyGraph, PrunableUnit
from model_analysis.propagation_engine import simulate_pruning_action
from model_analysis.pruning_action import PruningAction
from model_analysis.shape_evidence import ShapeEvidenceReport


def make_graph():
    return DependencyGraph(
        model_name="tiny",
        hf_id="local/tiny",
        task="unit-test",
        prunable_units=[
            PrunableUnit("torch:linear:q", "q", "torch", "linear", "q", ["out_features"], 72, [8, 8], "medium", "q"),
            PrunableUnit("torch:linear:k", "k", "torch", "linear", "k", ["out_features"], 72, [8, 8], "medium", "k"),
        ],
        dependency_edges=[
            DependencyEdge("torch:linear:q", "torch:linear:k", "qkv_coupling", ["hidden_dim"], "bidirectional", "medium", "qkv")
        ],
    )


def make_action():
    return PruningAction("test", "tiny", "torch:linear:q", None, None, "out_features", [0, 1], 2, None, "manual_indices", None)


def make_evidence():
    param = ParameterEvidence("q.weight", "q", [8, 8], "q.weight", [8, 8], "exact_name", "high", "matched")
    correspondence = CorrespondenceReport(
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
                parameter_evidence=[param],
                confidence="high",
                reason="matched",
            )
        ],
        summary={"num_module_correspondences": 1},
    )
    shape = ShapeEvidenceReport("tiny", "local/tiny", "unit-test", summary={"num_tensor_shapes": 1})
    validation = {
        "summary": {"num_validated_edges": 1},
        "validated_edges": [
            {"src": "torch:linear:q", "dst": "torch:linear:k", "edge_type": "qkv_coupling", "confidence": "medium", "reason": "validated"}
        ],
    }
    return correspondence, shape, validation


def test_simulation_remains_compatible_without_evidence():
    plan = simulate_pruning_action(make_graph(), make_action())

    assert plan.status == "valid_global"
    assert "evidence" not in plan.metadata


def test_simulation_adds_evidence_metadata_and_constraints():
    correspondence, shape, validation = make_evidence()
    plan = simulate_pruning_action(make_graph(), make_action(), correspondence, shape, validation)

    assert "evidence" in plan.metadata
    assert any(constraint.get("type") == "evidence_supported_propagation" for constraint in plan.constraints)
