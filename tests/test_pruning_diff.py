from model_analysis.pruning_diff import compute_structural_diff, pruning_diff_to_markdown


def test_compute_structural_diff_reports_parameter_delta():
    before = {
        "parameter_summary": {"total_parameters": 30},
        "linear_layers": [{"name": "fc", "out_features": 5, "in_features": 4, "parameters": 25}],
    }
    after = {
        "parameter_summary": {"total_parameters": 18},
        "linear_layers": [{"name": "fc", "out_features": 3, "in_features": 4, "parameters": 15}],
    }

    diff = compute_structural_diff(before, after)

    assert diff["parameter_delta"] == -12
    assert diff["changed_linear_layers"][0]["module_name"] == "fc"
    assert "# Pruning Structural Diff" in pruning_diff_to_markdown(diff)
