from __future__ import annotations

import subprocess
import sys

from experimental.axis_transfer_analysis.access_analysis import analyze_region
from experimental.axis_transfer_analysis.examples import attention_value_path_example, qk_score_example
from experimental.axis_transfer_analysis.pattern_recognition import recognize_patterns
from experimental.axis_transfer_analysis.report import render_markdown


def test_cli_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "experimental.axis_transfer_analysis.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "--show-relations" in proc.stdout
    assert "--show-patterns" in proc.stdout


def test_qk_markdown_explains_reduced_blocked_axis() -> None:
    example = qk_score_example()
    summary = analyze_region(example.region)
    markdown = render_markdown(example, summary, recognize_patterns(example.region, summary))

    assert "REDUCED" in markdown
    assert "qk_score_contraction_mixes_channels" in markdown


def test_attention_value_markdown_contains_preserved_path_pattern() -> None:
    example = attention_value_path_example()
    summary = analyze_region(example.region)
    markdown = render_markdown(example, summary, recognize_patterns(example.region, summary))

    assert "PRESERVED" in markdown
    assert "ATTENTION_VALUE_PATH" in markdown
