from __future__ import annotations

import json
from pathlib import Path

from model_analysis.rule_gap_diagnosis import detect_model_family, diagnose_rule_gaps_for_model


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_detect_model_family_from_ffn_paths():
    assert detect_model_family("facebook/opt-125m", {"ops": [{"source_name": "/model/decoder/layers.0/fc1/Gemm"}]}) == "opt_decoder"
    assert detect_model_family("distilbert-base-uncased", {"ops": [{"source_name": "/model/distilbert/transformer/layer.0/ffn/lin1/MatMul"}]}) == "distilbert_encoder"


def test_diagnosis_identifies_incomplete_plan_evidence(tmp_path: Path):
    model = "facebook/opt-125m"
    safe = "facebook__opt-125m"
    write_json(tmp_path / "reports/static_pipeline_status" / f"{safe}.json", {"final_status": "partial", "stages": []})
    write_json(tmp_path / "reports/op_semantics" / f"{safe}.json", {"ops": [{"source_name": "/model/decoder/layers.0/fc1/Gemm"}], "summary": {"unknown_ops": 1}})
    write_json(tmp_path / "reports/pruning_opportunity_rankings" / f"{safe}.json", {"summary": {"safe_candidates": 12}})
    write_json(tmp_path / "reports/pruning_plans" / f"{safe}.json", {"summary": {"total_plans": 12, "incomplete": 12, "missing_evidence_counts": {"missing activation evidence": 12}}})
    write_json(tmp_path / "reports/pruning_plan_validation" / f"{safe}.json", {"summary": {"total_plans": 12, "invalid_plans": 12, "failed_checks_by_type": {"op_semantics_agree": 12}}})
    diagnosis = diagnose_rule_gaps_for_model(tmp_path, model)
    gap_types = {gap.gap_type for gap in diagnosis.gaps}
    assert "missing_ffn_evidence_binding" in gap_types
    assert "validation_policy_too_bert_specific" in gap_types


def test_diagnosis_identifies_zero_safe_distilbert_with_ffn_ops(tmp_path: Path):
    model = "distilbert-base-uncased"
    safe = "distilbert-base-uncased"
    write_json(tmp_path / "reports/static_pipeline_status" / f"{safe}.json", {"final_status": "complete", "stages": []})
    write_json(
        tmp_path / "reports/op_semantics" / f"{safe}.json",
        {
            "ops": [
                {"source_name": "/model/distilbert/transformer/layer.0/ffn/lin1/MatMul"},
                {"source_name": "/model/distilbert/transformer/layer.0/ffn/lin2/MatMul"},
            ],
            "summary": {"unknown_ops": 0},
        },
    )
    write_json(tmp_path / "reports/pruning_opportunity_rankings" / f"{safe}.json", {"summary": {"safe_candidates": 0}})

    diagnosis = diagnose_rule_gaps_for_model(tmp_path, model)

    assert diagnosis.detected_model_family == "distilbert_encoder"
    assert any(gap.gap_type == "missing_feedforward_fusion" for gap in diagnosis.gaps)


def test_diagnosis_identifies_skipped_layer_grouping(tmp_path: Path):
    model = "gpt2"
    write_json(
        tmp_path / "reports/static_pipeline_status/gpt2.json",
        {
            "final_status": "partial",
            "stages": [{"stage_name": "layer_subgraph_validation", "status": "skipped"}],
        },
    )
    write_json(tmp_path / "reports/op_semantics/gpt2.json", {"ops": [{"source_name": "/model/transformer/h.0/mlp/c_fc/Gemm"}], "summary": {"unknown_ops": 0}})
    write_json(tmp_path / "reports/pruning_opportunity_rankings/gpt2.json", {"summary": {"safe_candidates": 1}})

    diagnosis = diagnose_rule_gaps_for_model(tmp_path, model)

    assert diagnosis.detected_model_family == "gpt2_decoder"
    assert any(gap.gap_type == "missing_layer_grouping" for gap in diagnosis.gaps)
