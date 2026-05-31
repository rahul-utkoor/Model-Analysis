from __future__ import annotations

import subprocess
import sys

from experimental.dfa_pruning_propagation.examples import attention_qk_example, attention_qk_renamed_example
from experimental.dfa_pruning_propagation.lattice import FactKind
from experimental.dfa_pruning_propagation.worklist import analyze


def by_tensor(result, tensor: str):
    return next(fact for axis, fact in result.state.items() if axis.tensor == tensor)


def test_qk_score_contraction_blocks_simple_propagation() -> None:
    example = attention_qk_example()
    result = analyze(example.graph, example.seed_facts)

    assert by_tensor(result, "q_proj.output").kind == FactKind.BLOCKED
    assert any("qk_score_contraction_mixes_channels" in event.output_fact for event in result.blocked_events)


def test_cli_help_works() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "experimental.dfa_pruning_propagation.cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "--example" in proc.stdout


def test_qk_renamed_blocks_by_semantics() -> None:
    example = attention_qk_renamed_example()
    result = analyze(example.graph, example.seed_facts)

    assert by_tensor(result, "left_branch.output").kind == FactKind.BLOCKED
    assert any("qk_score_contraction_mixes_channels" in event.output_fact for event in result.blocked_events)
