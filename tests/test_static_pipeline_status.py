from __future__ import annotations

from model_analysis.static_pipeline_status import StageStatus, final_status_from_stages, make_model_status


def test_final_status_complete_for_complete_core_stages():
    stages = [
        StageStatus("tensor_ir", "present_existing"),
        StageStatus("op_semantics", "present_existing"),
        StageStatus("structural_region_tree", "present_existing"),
        StageStatus("region_dimension_ir", "present_existing"),
        StageStatus("region_pruning_semantics", "present_existing"),
        StageStatus("pruning_opportunity_ranking", "present_existing"),
        StageStatus("layer_subgraph_validation", "present_existing"),
        StageStatus("full_model_report", "built"),
    ]
    assert final_status_from_stages(stages) == "complete"


def test_model_status_summary_counts_missing_artifacts():
    status = make_model_status(
        model_name="model",
        configured_model={"name": "model"},
        stages=[
            StageStatus("tensor_ir", "present_existing"),
            StageStatus("op_semantics", "skipped", missing_inputs=["reports/tensor_ir/model.json"]),
        ],
        artifacts={},
    )
    assert status.final_status == "partial"
    assert status.summary["completed_stages"] == 1
    assert status.summary["skipped_stages"] == 1
    assert status.summary["missing_artifacts"] == ["reports/tensor_ir/model.json"]
