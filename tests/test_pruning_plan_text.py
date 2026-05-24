from __future__ import annotations

from model_analysis.pruning_plan_synthesis import pruning_plan_set_to_markdown
from model_analysis.pruning_plan_text import pruning_plan_set_to_text
from test_pruning_plan_synthesis import build_plan_set


def test_text_dump_contains_action_lines() -> None:
    text = pruning_plan_set_to_text(build_plan_set())

    assert "pruning_plans @synthetic" in text
    assert "prune_producer_output" in text
    assert "prune_bias" in text
    assert "prune_consumer_input" in text


def test_markdown_contains_plan_sections() -> None:
    markdown = pruning_plan_set_to_markdown(build_plan_set())

    assert "# Pruning Plans: synthetic" in markdown
    assert "Ready Symbolic Plans" in markdown
    assert "Validation Checks" in markdown
