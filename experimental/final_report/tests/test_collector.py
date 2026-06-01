from experimental.final_report.collector import collect_final_report_data
from experimental.final_report.tests.helpers import write_all_model_proof


def test_collector_handles_synthetic_all_model_proof(tmp_path) -> None:
    write_all_model_proof(tmp_path)
    data = collect_final_report_data(tmp_path)

    assert data.aggregate_summary.expected_plans == 108
    assert data.aggregate_summary.proven_plans == 108
    assert data.aggregate_summary.native_mlir_evidence == 108
    assert len(data.per_model_summary) == 5


def test_non_strict_collector_handles_missing_optional_files(tmp_path) -> None:
    write_all_model_proof(tmp_path)
    data = collect_final_report_data(tmp_path, strict=False)

    assert data.aggregate_summary.proven_plans == 108
    assert data.warnings
