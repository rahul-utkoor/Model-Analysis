"""Supported model and pruning-pattern configuration."""

from __future__ import annotations

from dataclasses import dataclass

from experimental.mlir_evidence_coverage.coverage_case import CoveragePatternKind


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    artifact_name: str
    short_name: str
    expected_layers: int


@dataclass(frozen=True)
class PatternSpec:
    kind: CoveragePatternKind
    case_suffix: str
    search_aliases: tuple[str, ...]
    expected_pattern: str
    expected_result: str
    required_models: tuple[str, ...] = ()
    notes: str = ""

    def required_for(self, model_name: str) -> bool:
        return "*" in self.required_models or model_name in self.required_models


SUPPORTED_MODELS = (
    ModelSpec("bert-base-uncased", "bert-base-uncased", "bert", 12),
    ModelSpec("distilbert-base-uncased", "distilbert-base-uncased", "distilbert", 6),
    ModelSpec("facebook/opt-125m", "facebook__opt-125m", "opt", 12),
    ModelSpec("gpt2", "gpt2", "gpt2", 12),
    ModelSpec("google/vit-base-patch16-224", "google__vit-base-patch16-224", "vit", 12),
)

PATTERN_SPECS = (
    PatternSpec(
        CoveragePatternKind.FFN_MLP_INTERMEDIATE,
        "mlp",
        ("mlp_native_core", "mlp_block", "feed_forward", "ffn"),
        "FFN_INTERMEDIATE_CHAIN",
        "producer-output deadness",
        ("*",),
    ),
    PatternSpec(
        CoveragePatternKind.ATTENTION_QK_SCORE,
        "attention_score",
        ("attention_score", "score_matmul"),
        "QK_SCORE_BLOCKER",
        "blocked",
        ("bert-base-uncased",),
    ),
    PatternSpec(
        CoveragePatternKind.ATTENTION_CONTEXT_VALUE_AXIS,
        "attention_context",
        ("attention_context", "context_matmul"),
        "ATTENTION_CONTEXT_LIKE",
        "V.value_dim -> Context.value_context_dim PRESERVED",
        ("bert-base-uncased",),
        "A standalone context subgraph proves a local mapping but may not seed DFA propagation.",
    ),
    PatternSpec(
        CoveragePatternKind.ATTENTION_VALUE_PATH,
        "attention_value_path",
        ("attention_value_path", "value_path", "v_proj", "out_proj"),
        "ATTENTION_VALUE_PATH",
        "value producer-output deadness",
        ("bert-base-uncased", "distilbert-base-uncased", "facebook/opt-125m", "gpt2", "google/vit-base-patch16-224"),
        "A full local value-path artifact must include value projection, context, and output projection.",
    ),
    PatternSpec(
        CoveragePatternKind.RESIDUAL_HIDDEN_PROTECTED,
        "residual",
        ("residual_add", "residual_merge", "residual", "_add"),
        "RESIDUAL_HIDDEN_PROTECTED",
        "protected",
    ),
    PatternSpec(
        CoveragePatternKind.LAYERNORM_HIDDEN_PROTECTED,
        "layernorm",
        ("layernorm", "layer_norm", "normalization"),
        "LAYERNORM_HIDDEN_PROTECTED",
        "protected",
    ),
)


def model_specs(selector: str = "default") -> list[ModelSpec]:
    if selector in {"default", "all"}:
        return list(SUPPORTED_MODELS)
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [spec for spec in SUPPORTED_MODELS if spec.model_name in requested or spec.artifact_name in requested or spec.short_name in requested]
    unknown = requested - {value for spec in selected for value in (spec.model_name, spec.artifact_name, spec.short_name)}
    if unknown:
        raise ValueError(f"unknown models: {', '.join(sorted(unknown))}")
    return selected


def pattern_specs(selector: str = "all") -> list[PatternSpec]:
    if selector == "all":
        return list(PATTERN_SPECS)
    requested = {item.strip().upper() for item in selector.split(",") if item.strip()}
    selected = [spec for spec in PATTERN_SPECS if spec.kind.value in requested]
    unknown = requested - {spec.kind.value for spec in selected}
    if unknown:
        raise ValueError(f"unknown patterns: {', '.join(sorted(unknown))}")
    return selected
