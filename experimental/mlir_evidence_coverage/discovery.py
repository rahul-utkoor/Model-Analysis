"""Discover selected ONNX subgraphs and build stable coverage cases."""

from __future__ import annotations

import re
from pathlib import Path

from experimental.mlir_evidence_coverage.config import ModelSpec, PatternSpec, model_specs, pattern_specs
from experimental.mlir_evidence_coverage.coverage_case import CoverageCase


LAYER_RE = re.compile(r"layer_(\d+)")


def discover_model_subgraphs(
    model_name: str,
    artifact_root: str | Path = "artifacts/model_analysis_subgraphs",
    fallback_root: str | Path = "artifacts/layer_subgraphs",
    value_path_root: str | Path = "artifacts/attention_value_path_subgraphs",
) -> list[Path]:
    roots = (Path(value_path_root) / model_name / "layers", Path(artifact_root) / model_name / "layers", Path(fallback_root) / model_name)
    return list(dict.fromkeys(path for root in roots if root.is_dir() for path in sorted(root.glob("layer_*/*/subgraph.onnx"))))


def _layer_index(path: Path) -> int:
    match = next((LAYER_RE.fullmatch(part) for part in path.parts if LAYER_RE.fullmatch(part)), None)
    return int(match.group(1)) if match else 0


def _best_match(paths: list[Path], aliases: tuple[str, ...]) -> Path | None:
    candidates: list[tuple[int, int, int, str, Path]] = []
    for position, path in enumerate(paths):
        slug = path.parent.name.lower()
        for index, alias in enumerate(aliases):
            if alias in slug:
                candidates.append((index, len(slug), position, slug, path))
                break
    return min(candidates)[-1] if candidates else None


def _missing_path(spec: ModelSpec, layer_index: int, pattern: PatternSpec, artifact_root: str | Path) -> Path:
    return Path(artifact_root) / spec.artifact_name / "layers" / f"layer_{layer_index}" / pattern.case_suffix / "subgraph.onnx"


def match_cases_for_model(
    model: ModelSpec | str,
    patterns: list[PatternSpec],
    *,
    layers: str = "layer0",
    artifact_root: str | Path = "artifacts/model_analysis_subgraphs",
    fallback_root: str | Path = "artifacts/layer_subgraphs",
    value_path_root: str | Path = "artifacts/attention_value_path_subgraphs",
) -> list[CoverageCase]:
    if isinstance(model, str):
        matches = [spec for spec in model_specs("all") if model in {spec.model_name, spec.artifact_name, spec.short_name}]
        if not matches:
            raise ValueError(f"unknown model: {model}")
        model = matches[0]
    paths = discover_model_subgraphs(model.artifact_name, artifact_root, fallback_root, value_path_root)
    discovered_layers = sorted({_layer_index(path) for path in paths})
    layer_indices = discovered_layers if layers == "all" and discovered_layers else [0]
    cases: list[CoverageCase] = []
    for layer_index in layer_indices:
        layer_paths = [path for path in paths if _layer_index(path) == layer_index]
        for pattern in patterns:
            matched = _best_match(layer_paths, pattern.search_aliases)
            path = matched or _missing_path(model, layer_index, pattern, artifact_root)
            cases.append(
                CoverageCase(
                    f"{model.short_name}_layer{layer_index}_{pattern.case_suffix}",
                    model.model_name,
                    layer_index,
                    pattern.kind,
                    matched.parent.name if matched else pattern.case_suffix,
                    str(path),
                    pattern.expected_pattern,
                    pattern.expected_result,
                    pattern.required_for(model.model_name),
                    pattern.notes,
                )
            )
    return cases


def build_default_coverage_cases(
    models: str = "default",
    layers: str = "layer0",
    patterns: str = "all",
    *,
    artifact_root: str | Path = "artifacts/model_analysis_subgraphs",
    fallback_root: str | Path = "artifacts/layer_subgraphs",
    value_path_root: str | Path = "artifacts/attention_value_path_subgraphs",
) -> list[CoverageCase]:
    if layers not in {"layer0", "all"}:
        raise ValueError(f"unknown layer selector: {layers}")
    selected_patterns = pattern_specs(patterns)
    return [
        case
        for model in model_specs(models)
        for case in match_cases_for_model(model, selected_patterns, layers=layers, artifact_root=artifact_root, fallback_root=fallback_root, value_path_root=value_path_root)
    ]
