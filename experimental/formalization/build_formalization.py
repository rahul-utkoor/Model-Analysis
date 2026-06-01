"""Build the static pruning propagation formalization report bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experimental.formalization.report import (
    FormalizationInputs,
    render_bert_case_study,
    render_index,
    render_paper_methodology,
    render_static_notes,
    render_teaching_slides,
)


DEFAULT_INPUT_PATHS = {
    "bert_proof": "reports/bert_24_plan_proof/index.json",
    "bert_coverage": "reports/mlir_evidence_coverage_bert_24_plan/index.json",
    "bert_value_paths": "reports/attention_value_path_subgraphs/bert-base-uncased/summary.json",
    "bert_validation": "reports/pruning_plan_validation/bert-base-uncased.json",
}


def _read_optional(path: str | Path, label: str, warnings: list[str]) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        warnings.append(f"{label} input is missing: {source}")
        return {}
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"{label} input could not be read: {source}: {exc}")
        return {}


def load_inputs(paths: dict[str, str | Path] | None = None) -> FormalizationInputs:
    selected = {**DEFAULT_INPUT_PATHS, **(paths or {})}
    warnings: list[str] = []
    return FormalizationInputs(
        _read_optional(selected["bert_proof"], "BERT 24-plan proof", warnings),
        _read_optional(selected["bert_coverage"], "BERT MLIR coverage", warnings),
        _read_optional(selected["bert_value_paths"], "BERT value-path summary", warnings),
        _read_optional(selected["bert_validation"], "BERT plan validation", warnings),
        tuple(warnings),
    )


def build_formalization(output_dir: str | Path = "reports/formalization", paths: dict[str, str | Path] | None = None) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(paths)
    documents = {
        "static_pruning_propagation_notes.md": render_static_notes(inputs),
        "bert_24_plan_case_study.md": render_bert_case_study(inputs),
        "teaching_slide_outline.md": render_teaching_slides(inputs),
        "paper_methodology_outline.md": render_paper_methodology(inputs),
    }
    written: list[Path] = []
    for name, text in documents.items():
        path = output / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    index_md = output / "index.md"
    index_md.write_text(render_index(inputs, list(documents)), encoding="utf-8")
    written.append(index_md)
    index_json = output / "index.json"
    index_json.write_text(
        json.dumps(
            {
                "documents": list(documents),
                "bert_24_plan_summary": inputs.bert_proof.get("summary", {}),
                "input_paths": {key: str(value) for key, value in {**DEFAULT_INPUT_PATHS, **(paths or {})}.items()},
                "warnings": list(inputs.warnings),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(index_json)
    return written
