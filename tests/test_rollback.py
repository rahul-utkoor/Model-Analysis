from pathlib import Path

from model_analysis.pruning_execution import AppliedPruneRecord, PruningExecutionReport
from model_analysis.rollback import create_rollback_manifest, rollback_manifest_to_markdown


def test_create_rollback_manifest_contains_dirs_and_modules(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")
    report = PruningExecutionReport(
        execution_id="exec1",
        model_name="tiny",
        source_model_dir=str(source),
        output_model_dir=str(output),
        action_id="a1",
        plan_id="p1",
        status="success",
        applied_records=[
            AppliedPruneRecord("fc", "Linear", "out_features", [0], [4, 4], [3, 4], [4], [3], "applied", "ok")
        ],
    )

    manifest = create_rollback_manifest(report, source, output, tmp_path / "rollback.json")

    assert manifest["source_model_dir"] == str(source)
    assert manifest["output_model_dir"] == str(output)
    assert manifest["applied_modules"] == ["fc"]
    assert manifest["files_created"]
    assert "# Rollback Manifest: exec1" in rollback_manifest_to_markdown(manifest)
