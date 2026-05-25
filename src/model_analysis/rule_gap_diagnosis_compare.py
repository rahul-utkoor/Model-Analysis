"""Compatibility wrapper for rule-gap diagnosis comparisons."""

from __future__ import annotations

from typing import Any

from model_analysis.rule_gap_diagnosis import compare_rule_gap_diagnoses
from model_analysis.rule_gap_diagnosis_text import rule_gap_compare_to_markdown


def compare_rule_gaps(diagnoses: list[dict[str, Any]]) -> dict[str, Any]:
    return compare_rule_gap_diagnoses(diagnoses)


def compare_rule_gaps_to_markdown(comparison: dict[str, Any]) -> str:
    return rule_gap_compare_to_markdown(comparison)

