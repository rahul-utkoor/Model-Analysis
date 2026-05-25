from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_analysis import static_pipeline_orchestrator as orch


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_stage_status_present_existing_when_output_exists(tmp_path: Path):
    write_json(tmp_path / "reports" / "op_semantics" / "bert-base-uncased.json", {})
    stage = orch._status_for_existing_or_missing(
        root=tmp_path,
        model_cli_name="bert-base-uncased",
        safe="bert-base-uncased",
        stage_name="op_semantics",
        build_missing_analysis=False,
        build_layer_packs=False,
        strict=False,
        verbose=False,
    )
    assert stage.status == "present_existing"


def test_stage_status_skipped_when_prerequisites_missing(tmp_path: Path):
    stage = orch._status_for_existing_or_missing(
        root=tmp_path,
        model_cli_name="bert-base-uncased",
        safe="bert-base-uncased",
        stage_name="op_semantics",
        build_missing_analysis=False,
        build_layer_packs=False,
        strict=False,
        verbose=False,
    )
    assert stage.status == "skipped"
    assert "reports/tensor_ir/bert-base-uncased.json" in stage.missing_inputs


def test_stage_status_failed_when_builder_fails_and_not_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    write_json(tmp_path / "reports" / "tensor_ir" / "bert-base-uncased.json", {})

    def fake_run(*args, **kwargs):
        return False, "synthetic failure", 0.1

    monkeypatch.setattr(orch, "_run_stage_script", fake_run)
    stage = orch._status_for_existing_or_missing(
        root=tmp_path,
        model_cli_name="bert-base-uncased",
        safe="bert-base-uncased",
        stage_name="op_semantics",
        build_missing_analysis=True,
        build_layer_packs=False,
        strict=False,
        verbose=False,
    )
    assert stage.status == "failed"
    assert stage.error == "synthetic failure"


def test_validation_not_applicable_when_no_safe_plans(tmp_path: Path):
    write_json(
        tmp_path / "reports" / "pruning_opportunity_rankings" / "toy.json",
        {"summary": {"safe_candidates": 0}},
    )
    stage = orch._status_for_existing_or_missing(
        root=tmp_path,
        model_cli_name="toy",
        safe="toy",
        stage_name="pruning_plan_validation",
        build_missing_analysis=False,
        build_layer_packs=False,
        strict=False,
        verbose=False,
    )
    assert stage.status == "not_applicable"
