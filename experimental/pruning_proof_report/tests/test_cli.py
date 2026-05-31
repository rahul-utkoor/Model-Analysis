import subprocess
import sys

from experimental.pruning_proof_report.cli import parse_args


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "experimental.pruning_proof_report.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--native-pass-tool" in completed.stdout
    assert "--no-native-pass" in completed.stdout


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.models == "default"
    assert args.format == "both"
