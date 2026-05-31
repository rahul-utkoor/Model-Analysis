from __future__ import annotations

import subprocess
import sys

import pytest

from experimental.onnx_axis_bridge.cli import main


def test_cli_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "experimental.onnx_axis_bridge.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "--onnx" in proc.stdout
    assert "--show-all" in proc.stdout


def test_cli_missing_file_fails_gracefully(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", ["onnx-axis-bridge", "--onnx", str(tmp_path / "missing.onnx")])

    assert main() == 2
    assert "ONNX subgraph does not exist" in capsys.readouterr().out


def test_synthetic_ffn_markdown_report(tmp_path) -> None:
    pytest.importorskip("onnx")
    from experimental.onnx_axis_bridge.bridge_runner import analyze_onnx_subgraph
    from experimental.onnx_axis_bridge.report import render_markdown
    from experimental.onnx_axis_bridge.tests.helpers import make_ffn

    markdown = render_markdown(analyze_onnx_subgraph(make_ffn(tmp_path / "ffn.onnx")))

    assert "FFN_LIKE" in markdown
    assert "FFN_INTERMEDIATE_CHAIN" in markdown
    assert "producer_from_axis_summary.output" in markdown
