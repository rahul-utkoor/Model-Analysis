from __future__ import annotations

import subprocess
import sys


def test_cli_help() -> None:
    completed = subprocess.run([sys.executable, "-m", "experimental.mlir_axis_bridge.cli", "--help"], capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    assert "--onnx-mlir" in completed.stdout
    assert "--show-all" in completed.stdout
