"""Supported model expectations for the all-model propagation proof."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlanFamily(str, Enum):
    FFN_INTERMEDIATE = "ffn_intermediate"
    ATTENTION_VALUE_PATH = "attention_value_path"
    QK_BLOCKER = "qk_blocker"


class AttentionValuePolicy(str, Enum):
    REQUIRED = "required"
    REQUIRED_IF_SEPARABLE = "required_if_separable"
    FUSED_QKV_GAP = "fused_qkv_gap"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ModelPlanExpectation:
    model_name: str
    artifact_name: str
    short_name: str
    layer_count: int
    ffn_expected: int
    attention_value_expected: int
    total_expected: int
    attention_value_policy: AttentionValuePolicy
    qk_policy: str = "blocker_only"
    notes: str = ""


SUPPORTED_MODELS = (
    ModelPlanExpectation(
        "bert-base-uncased",
        "bert-base-uncased",
        "bert",
        12,
        12,
        12,
        24,
        AttentionValuePolicy.REQUIRED,
        notes="QK score contractions are blockers and are excluded from propagation-plan counts.",
    ),
    ModelPlanExpectation(
        "distilbert-base-uncased",
        "distilbert-base-uncased",
        "distilbert",
        6,
        6,
        6,
        12,
        AttentionValuePolicy.REQUIRED_IF_SEPARABLE,
        notes="The separable v_lin -> context -> out_lin value path is required when recoverable.",
    ),
    ModelPlanExpectation(
        "facebook/opt-125m",
        "facebook__opt-125m",
        "opt",
        12,
        12,
        12,
        24,
        AttentionValuePolicy.REQUIRED,
        notes="OPT value-path artifacts were introduced in Milestone 48.",
    ),
    ModelPlanExpectation(
        "gpt2",
        "gpt2",
        "gpt2",
        12,
        12,
        12,
        24,
        AttentionValuePolicy.FUSED_QKV_GAP,
        notes="Attention value-path proof requires fused-QKV value-slice recovery.",
    ),
    ModelPlanExpectation(
        "google/vit-base-patch16-224",
        "google__vit-base-patch16-224",
        "vit",
        12,
        12,
        12,
        24,
        AttentionValuePolicy.FUSED_QKV_GAP,
        notes="Attention value-path proof requires fused-QKV value-slice and output-path recovery.",
    ),
)


def model_expectations(selector: str = "all") -> list[ModelPlanExpectation]:
    if selector in {"all", "default"}:
        return list(SUPPORTED_MODELS)
    requested = {item.strip() for item in selector.split(",") if item.strip()}
    selected = [
        spec
        for spec in SUPPORTED_MODELS
        if requested.intersection({spec.model_name, spec.artifact_name, spec.short_name})
    ]
    known = {value for spec in selected for value in (spec.model_name, spec.artifact_name, spec.short_name)}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown models: {', '.join(sorted(unknown))}")
    return selected
