"""Default selected real subgraphs for the cross-evidence proof report."""

from __future__ import annotations

from pathlib import Path

from experimental.pruning_proof_report.proof_case import ProofCase


ARTIFACT_ROOT = Path("artifacts/model_analysis_subgraphs")
VALUE_PATH_ROOT = Path("artifacts/attention_value_path_subgraphs")


def _case(
    case_id: str,
    model: str,
    layer: int,
    folder: str,
    expected_pattern: str,
    expected_dfa: str,
    notes: str = "",
    *,
    artifact_model: str | None = None,
) -> ProofCase:
    artifact_dir = artifact_model or model
    path = ARTIFACT_ROOT / artifact_dir / "layers" / f"layer_{layer}" / folder / "subgraph.onnx"
    return ProofCase(case_id, model, layer, folder, str(path), expected_pattern, expected_dfa, notes)


def _context_path() -> str:
    preferred = ARTIFACT_ROOT / "bert-base-uncased/layers/layer_0/08_layer_0_attention_context_matmul/subgraph.onnx"
    if preferred.is_file():
        return str(preferred)
    matches = sorted((ARTIFACT_ROOT / "bert-base-uncased/layers/layer_0").glob("*attention_context*/subgraph.onnx"))
    return str(matches[0] if matches else preferred)


def _value_path(model: str, layer: int, fallback_slug: str) -> str:
    root = VALUE_PATH_ROOT / model / "layers" / f"layer_{layer}"
    matches = sorted(root.glob("*/subgraph.onnx"))
    return str(matches[0] if matches else root / fallback_slug / "subgraph.onnx")


def default_proof_cases() -> list[ProofCase]:
    return [
        _case("gpt2_layer0_mlp", "gpt2", 0, "03_gpt_2_block_0_mlp_block", "FFN_INTERMEDIATE_CHAIN", "producer-output deadness"),
        _case("opt_layer0_mlp", "facebook/opt-125m", 0, "06_opt_decoder_block_0_mlp_block", "FFN_INTERMEDIATE_CHAIN", "producer-output deadness", artifact_model="facebook__opt-125m"),
        ProofCase(
            "opt_layer0_attention_value_path",
            "facebook/opt-125m",
            0,
            "attention_value_path",
            _value_path("facebook__opt-125m", 0, "opt_layer_0_attention_value_path"),
            "ATTENTION_VALUE_PATH",
            "value producer output deadness",
            "Complete value projection -> context -> output projection chain when generated.",
        ),
        ProofCase(
            "bert_layer0_attention_value_path",
            "bert-base-uncased",
            0,
            "attention_value_path",
            _value_path("bert-base-uncased", 0, "bert_layer_0_attention_value_path"),
            "ATTENTION_VALUE_PATH",
            "value producer output deadness",
            "Complete BERT self.value -> context -> attention.output.dense chain when generated.",
        ),
        _case("bert_layer0_attention_score", "bert-base-uncased", 0, "05_layer_0_attention_score_matmul", "QK_SCORE_BLOCKER", "blocked"),
        ProofCase(
            "bert_layer0_attention_context",
            "bert-base-uncased",
            0,
            "attention_context_matmul",
            _context_path(),
            "ATTENTION_CONTEXT_LIKE",
            "V.value_dim -> Context.value_context_dim PRESERVED",
            "Standalone context proves a local value-axis mapping and may not have a DFA seed.",
        ),
        _case("distilbert_layer0_mlp", "distilbert-base-uncased", 0, "03_distilbert_layer_0_mlp_block", "FFN_INTERMEDIATE_CHAIN", "producer-output deadness"),
        _case("vit_layer0_mlp", "google/vit-base-patch16-224", 0, "03_vit_layer_0_mlp_block", "FFN_INTERMEDIATE_CHAIN", "producer-output deadness", artifact_model="google__vit-base-patch16-224"),
        _case("bert_layer0_residual", "bert-base-uncased", 0, "10_layer_0_attention_residual_add", "RESIDUAL_HIDDEN_PROTECTED", "protected"),
        _case("bert_layer0_layernorm", "bert-base-uncased", 0, "11_layer_0_layernorm", "LAYERNORM_HIDDEN_PROTECTED", "protected"),
    ]


def proof_cases(selector: str = "default") -> list[ProofCase]:
    if selector not in {"default", "all"}:
        raise ValueError(f"unknown proof-case selector: {selector}")
    return default_proof_cases()


def select_case(cases: list[ProofCase], case_id: str | None) -> list[ProofCase]:
    if case_id is None:
        return cases
    selected = [case for case in cases if case.case_id == case_id]
    if not selected:
        raise ValueError(f"unknown proof case: {case_id}")
    return selected
