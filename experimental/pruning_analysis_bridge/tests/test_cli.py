from __future__ import annotations

import subprocess
import sys

from experimental.pruning_analysis_bridge.axis_to_dfa import run_bridge_analysis
from experimental.pruning_analysis_bridge.examples import ffn_from_access_example
from experimental.pruning_analysis_bridge.report import render_markdown


def test_cli_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "experimental.pruning_analysis_bridge.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "--show-axis-evidence" in proc.stdout
    assert "--show-dfa-trace" in proc.stdout
    assert "--show-all" in proc.stdout


def test_markdown_report_contains_key_story() -> None:
    example = ffn_from_access_example()
    result = run_bridge_analysis(example.region_spec, example.seed_policy, example_name=example.example_name, interpretation=example.interpretation)
    markdown = render_markdown(result)

    assert "## Axis-Transfer Evidence" in markdown
    assert "## DFA Propagation Trace" in markdown
    assert "Semantic roles are not assigned directly" in markdown
    assert "FFN_INTERMEDIATE_CHAIN" in markdown
    assert "producer_from_axis_summary.output" in markdown
