from __future__ import annotations

from model_analysis.pruning_plan_validation_text import pruning_plan_validation_to_text
from test_pruning_plan_validation import validated


def test_text_dump_contains_pass_and_fail_check_lines() -> None:
    data = validated()
    text = pruning_plan_validation_to_text(data)

    assert "pruning_plan_validation @synthetic" in text
    assert "candidate_is_safe pass" in text
    assert "op_semantics_agree pass" in text
