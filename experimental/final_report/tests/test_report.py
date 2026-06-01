from experimental.final_report.collector import collect_final_report_data
from experimental.final_report.report import render_case_csv, render_claims, render_final_report, write_report_bundle
from experimental.final_report.tests.helpers import write_all_model_proof


def test_report_contains_final_research_story(tmp_path) -> None:
    write_all_model_proof(tmp_path)
    text = render_final_report(collect_final_report_data(tmp_path))

    assert "108/108" in text
    assert "24/24" in text
    assert "Native MLIR evidence" in text
    assert "Sparsity is not the same as deadness" in text
    assert "MLIR is a local evidence generator" in text


def test_claims_file_contains_claims_and_non_claims(tmp_path) -> None:
    write_all_model_proof(tmp_path)
    text = render_claims(collect_final_report_data(tmp_path))

    assert "Evidence-Backed Claims" in text
    assert "Non-Claims / Scope Boundaries" in text
    assert "We do not claim accuracy recovery" in text


def test_csv_contains_total_row(tmp_path) -> None:
    write_all_model_proof(tmp_path)
    text = render_case_csv(collect_final_report_data(tmp_path))

    assert "\nTOTAL," in text


def test_bundle_writes_expected_files(tmp_path) -> None:
    write_all_model_proof(tmp_path)
    output = tmp_path / "reports/final"
    written = write_report_bundle(output, collect_final_report_data(tmp_path))

    assert len(written) == 6
    assert (output / "index.md").is_file()
    assert (output / "static_pruning_propagation_final_summary.json").is_file()
