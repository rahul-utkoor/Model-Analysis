import subprocess
import sys

from experimental.mlir_evidence_coverage.cli import parse_args


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "experimental.mlir_evidence_coverage.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--layers" in completed.stdout
    assert "--patterns" in completed.stdout


def test_cli_defaults_to_layer0() -> None:
    assert parse_args([]).layers == "layer0"
