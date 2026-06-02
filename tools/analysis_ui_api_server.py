#!/usr/bin/env python3
"""Local stdlib API server for the pruning analysis web UI."""

from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse


ROOT = Path(__file__).resolve().parents[1]
REPORT_TEXT_SUFFIXES = {".md", ".json", ".csv"}
ARTIFACT_TEXT_SUFFIXES = {".mlir", ".dot", ".json", ".md", ".txt", ".csv"}
MAX_ARTIFACT_TEXT_BYTES = 3 * 1024 * 1024
MAX_FOCUSED_CONTEXT_LINE_CHARS = 4000
FOCUS_TERMS = {
    "affine": ["affine.for", "affine.load", "affine.store"],
    "loops": ["affine.for", "scf.for"],
    "loads": ["affine.load", "memref.load"],
    "stores": ["affine.store", "memref.store"],
    "matmul": ["linalg.matmul", "krnl.matmul", "onnx.MatMul", "onnx.Gemm"],
    "all": ["affine.for", "affine.load", "affine.store", "scf.for", "memref.load", "memref.store", "linalg.matmul", "krnl.matmul", "onnx.MatMul", "onnx.Gemm"],
}
FALLBACK_FOCUS_TERMS = ["krnl.", "scf.for", "linalg.", "onnx.MatMul", "onnx.Gemm"]


@dataclass
class ServerConfig:
    root: Path
    report_root: Path
    artifact_root: Path
    fallback_layer_root: Path
    fallback_artifact_root: Path
    ui_dist: Path
    verbose: bool = False


def safe_model_name(value: str) -> str:
    return value.replace("/", "__")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def load_optional_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        warnings.append(f"missing optional report: {path.relative_to(path.parents[2])}")
        return {}
    data = load_json(path)
    if not data:
        warnings.append(f"could not read optional report: {path}")
    return data


def final_summary(config: ServerConfig) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    data = load_optional_json(config.root / "reports" / "final" / "static_pruning_propagation_final_summary.json", warnings)
    aggregate = data.get("aggregate", {})
    return {
        "expected_plans": aggregate.get("expected_plans", 0),
        "proven_plans": aggregate.get("proven_plans", 0),
        "native_mlir_evidence": aggregate.get("native_mlir_evidence", 0),
        "fallback": aggregate.get("high_level_mlir_fallback", aggregate.get("fallback_evidence", 0)),
        "unsupported": aggregate.get("unsupported", 0),
        "partial": aggregate.get("partial", 0),
        "missing": aggregate.get("missing", 0),
        "failed": aggregate.get("failed", 0),
    }, warnings


def pipeline_steps() -> list[dict[str, Any]]:
    return [
        {
            "id": "problem",
            "title": "Problem",
            "summary": "Structural pruning is a graph-transformation legality question.",
            "details": ["If an axis becomes dead or pruned, determine what else must change.", "No model weights are mutated by this analysis."],
        },
        {
            "id": "sparse-vs-structural",
            "title": "Sparse-weight vs structural pruning",
            "summary": "Zeros and dead axes are different compiler facts.",
            "details": ["Sparse-weight pruning preserves tensor shapes.", "Structural pruning removes or makes whole channels dead.", "Sparsity is not the same as deadness."],
        },
        {
            "id": "axis-facts",
            "title": "Axis facts",
            "summary": "Facts track what is known about each tensor axis.",
            "details": ["UNKNOWN, LIVE, DEAD, PRUNED, PROTECTED, and BLOCKED form the analysis vocabulary."],
        },
        {
            "id": "axis-transfer",
            "title": "Axis-transfer evidence",
            "summary": "Loop and access relations show how axes move through operators.",
            "details": ["PRESERVED, REDUCED, MIXED, PROTECTED, BLOCKED, BROADCAST, and PERMUTED summarize local behavior."],
        },
        {
            "id": "onnx-subgraphs",
            "title": "ONNX subgraphs",
            "summary": "Selected local subgraphs are evidence units.",
            "details": ["The analysis lowers focused regions instead of whole models.", "Names are optional labels; graph and shape structure carry the evidence."],
        },
        {
            "id": "mlir-evidence",
            "title": "MLIR/native dependence evidence",
            "summary": "ONNX-MLIR exposes local loop and access facts.",
            "details": ["Native MLIR dependence evidence is preferred.", "MLIR is a local evidence generator, not the pruning framework itself."],
        },
        {
            "id": "patterns",
            "title": "Pattern recognition",
            "summary": "Axis relations induce pruning-amenable patterns.",
            "details": ["FFN_INTERMEDIATE_CHAIN", "ATTENTION_VALUE_PATH", "QK_SCORE_BLOCKER", "RESIDUAL_HIDDEN_PROTECTED", "LAYERNORM_HIDDEN_PROTECTED"],
        },
        {
            "id": "dfa",
            "title": "DFA/worklist propagation",
            "summary": "Transfer rules propagate facts until a fixed point.",
            "details": ["Seed a DEAD or PRUNED fact.", "Apply semantic transfer rules.", "Report DEAD, PROTECTED, or BLOCKED conclusions."],
        },
        {
            "id": "proof",
            "title": "Model proof summaries",
            "summary": "Five supported models reach complete propagation-plan proofs.",
            "details": ["BERT, DistilBERT, OPT, GPT-2, and ViT contribute 108 proven plans in total."],
        },
        {
            "id": "limitations",
            "title": "Limitations",
            "summary": "This is static evidence and proof reporting only.",
            "details": ["The analysis does not choose pruning indices, execute pruning, or evaluate accuracy or speedup."],
        },
    ]


def proof_summary(config: ServerConfig) -> dict[str, Any]:
    final, warnings = final_summary(config)
    final_data = load_optional_json(config.root / "reports" / "final" / "static_pruning_propagation_final_summary.json", warnings)
    models = [
        {
            "model_name": item.get("model_name", ""),
            "layers": item.get("layers", 0),
            "expected_plans": item.get("expected_plans", 0),
            "proven_plans": item.get("proven_plans", 0),
            "ffn_proven": item.get("ffn_proven", 0),
            "attention_value_proven": item.get("attention_value_proven", 0),
            "native_mlir_evidence": item.get("native_evidence", 0),
            "fallback": item.get("fallback_evidence", 0),
            "verdict": item.get("final_verdict", "unknown"),
        }
        for item in final_data.get("models", [])
    ]
    return {"models": models, "aggregate": final, "warnings": sorted(set(warnings))}


def overview(config: ServerConfig) -> dict[str, Any]:
    final, warnings = final_summary(config)
    for path in [
        config.root / "reports" / "final" / "static_pruning_propagation_final_report.md",
        config.root / "reports" / "formalization" / "index.json",
        config.root / "reports" / "all_model_plan_proof" / "index.json",
    ]:
        if not path.exists():
            warnings.append(f"missing optional report: {path}")
    return {
        "title": "Static Pruning Propagation Analysis",
        "subtitle": "From dead axes to compiler-style propagation proofs.",
        "final_summary": final,
        "pipeline_steps": pipeline_steps(),
        "teaching_takeaways": [
            "Names are syntax; evidence comes from graph, shape, loop, and access relations.",
            "FFN propagation follows the produced, preserved, and consumed intermediate axis.",
            "Attention value-path propagation follows the preserved context value axis.",
            "QK score contractions are blockers, not pruning plans.",
            "MLIR supplies local evidence while DFA computes the propagation fixed point.",
        ],
        "warnings": sorted(set(warnings)),
    }


def pipeline_flow(config: ServerConfig) -> dict[str, Any]:
    summary, warnings = final_summary(config)
    examples = {
        "ffn": {
            "title": "FFN / MLP intermediate propagation",
            "pattern": "FFN_INTERMEDIATE_CHAIN",
            "nodes": ["Expansion Projection", "Activation", "Contraction Projection"],
            "dimensions": ["hidden", "intermediate", "intermediate", "hidden"],
            "edges": [["Expansion Projection", "Activation"], ["Activation", "Contraction Projection"]],
            "equations": [
                "Y[b, s, j] = gelu(X[b, s, j])",
                "O[b, s, h] += A[b, s, j] * W[j, h]",
            ],
            "relations": [
                {"source": "X.j", "target": "Y.j", "relation": "PRESERVED"},
                {"source": "A.j", "target": "O.h", "relation": "REDUCED / CONSUMED"},
            ],
            "facts": [
                "seed: contraction.input[j] = DEAD",
                "activation.output[j] = DEAD",
                "activation.input[j] = DEAD",
                "expansion.output[j] = DEAD",
                "fixed point reached",
            ],
        },
        "attention_value": {
            "title": "Attention value path",
            "pattern": "ATTENTION_VALUE_PATH",
            "nodes": ["Value Projection", "Attention Context", "Output Projection"],
            "dimensions": ["value_dim", "context_value_dim", "hidden"],
            "edges": [["Value Projection", "Attention Context"], ["Attention Context", "Output Projection"]],
            "equations": ["C[b, h, q, d] += P[b, h, q, k] * V[b, h, k, d]"],
            "relations": [{"source": "V.d", "target": "C.d", "relation": "PRESERVED"}],
            "facts": [
                "seed: output_projection.input[d] = DEAD",
                "context.value_axis[d] = DEAD",
                "value_projection.output[d] = DEAD",
                "fixed point reached",
            ],
        },
        "qk_blocker": {
            "title": "QK score blocker",
            "pattern": "QK_SCORE_BLOCKER",
            "nodes": ["Q Projection", "Score MatMul", "K Projection"],
            "dimensions": ["head_dim", "score", "head_dim"],
            "edges": [["Q Projection", "Score MatMul"], ["K Projection", "Score MatMul"]],
            "equations": ["S[b, h, q, k] += Q[b, h, q, d] * K[b, h, k, d]"],
            "relations": [{"source": "Q/K.d", "target": "Score", "relation": "REDUCED / MIXED"}],
            "facts": [
                "attempt: Q/K feature axis d",
                "MLIR relation: d is reduced / mixed",
                "BLOCKED: qk_score_contraction_mixes_channels",
            ],
        },
    }
    return {
        "title": "Static Pruning Propagation Pipeline",
        "summary": "A compiler-style evidence pipeline for proving pruning propagation.",
        "aggregate": {
            "expected_plans": summary["expected_plans"],
            "proven_plans": summary["proven_plans"],
            "native_mlir_evidence": summary["native_mlir_evidence"],
            "fallback": summary["fallback"],
        },
        "stages": [
            {
                "id": "onnx_subgraph",
                "title": "ONNX Subgraph",
                "kind": "input",
                "short": "Select a local evidence unit.",
                "example": "BERT layer 0 FFN",
                "visual": {"type": "graph", "nodes": examples["ffn"]["nodes"], "edges": examples["ffn"]["edges"]},
                "proven": "The evidence unit contains the complete local path.",
                "not_claimed": "No model execution or whole-model lowering.",
            },
            {
                "id": "onnx_mlir",
                "title": "ONNX-MLIR Lowering",
                "kind": "lowering",
                "short": "Lower the selected region to inspectable MLIR.",
                "equations": ["onnx.MatMul -> affine.for / affine.load / affine.store"],
                "proven": "The local subgraph reaches an inspectable compiler IR.",
                "not_claimed": "MLIR does not infer pruning rules by itself.",
            },
            {
                "id": "mlir_access",
                "title": "Native MLIR Dependence Evidence",
                "kind": "evidence",
                "short": "Extract loop-IV and indexed access flow.",
                "equations": examples["ffn"]["equations"],
                "proven": "Native access relations identify preserved and consumed axes.",
                "not_claimed": "The pass does not choose channel indices.",
            },
            {
                "id": "axis_transfer",
                "title": "Axis-Transfer Summary",
                "kind": "analysis",
                "short": "Summarize local axis behavior.",
                "relations": examples["ffn"]["relations"],
                "proven": "The intermediate axis is preserved through activation and consumed by contraction.",
            },
            {
                "id": "pattern",
                "title": "Pattern Recognition",
                "kind": "pattern",
                "short": "Select an evidence-backed pruning pattern.",
                "pattern": "FFN_INTERMEDIATE_CHAIN",
                "proven": "The path matches a pruning-amenable FFN intermediate chain.",
            },
            {
                "id": "dfa",
                "title": "DFA Worklist Propagation",
                "kind": "fixed_point",
                "short": "Propagate deadness to a fixed point.",
                "facts": examples["ffn"]["facts"],
                "proven": "Consumer-input deadness reaches expansion-output deadness.",
                "not_claimed": "The DFA does not mutate model weights.",
            },
            {
                "id": "verdict",
                "title": "Proof Verdict",
                "kind": "verdict",
                "short": "Record the supported propagation result.",
                "facts": ["108 / 108 expected propagation plans proven", "108 native MLIR evidence proofs", "0 fallback proofs"],
                "proven": "The supported-model proof matrix is complete.",
                "not_claimed": "No accuracy or speedup result is claimed.",
            },
        ],
        "examples": examples,
        "warnings": warnings,
    }


def evidence_traces(config: ServerConfig) -> dict[str, Any]:
    summary, warnings = final_summary(config)
    return {
        "summary": {
            "title": "Evidence Trace",
            "description": "Step through graph, MLIR, axis-transfer, pattern, and DFA evidence.",
            "plans_proven": summary["proven_plans"],
            "native_mlir_evidence": summary["native_mlir_evidence"],
        },
        "examples": [
            {
                "id": "ffn_intermediate",
                "title": "FFN / MLP intermediate propagation",
                "pattern": "FFN_INTERMEDIATE_CHAIN",
                "verdict": "proven",
                "graph": {
                    "nodes": [
                        {"id": "expansion", "label": "Expansion Projection", "op": "MatMul / Gemm", "axis_role": "produces intermediate j", "shape": "hidden -> intermediate"},
                        {"id": "activation", "label": "Activation", "op": "GELU / ReLU", "axis_role": "preserves intermediate j", "shape": "intermediate -> intermediate"},
                        {"id": "contraction", "label": "Contraction Projection", "op": "MatMul / Gemm", "axis_role": "consumes intermediate j", "shape": "intermediate -> hidden"},
                    ],
                    "edges": [
                        {"source": "expansion", "target": "activation", "axis": "j", "relation": "PRESERVED"},
                        {"source": "activation", "target": "contraction", "axis": "j", "relation": "CONSUMED"},
                    ],
                },
                "mlir": [
                    {"title": "Activation", "code": "Y[b, s, j] = gelu(X[b, s, j])", "relation": "X.j -> Y.j = PRESERVED"},
                    {"title": "Contraction", "code": "O[b, s, h] += A[b, s, j] * W[j, h]", "relation": "A.j is consumed by contraction"},
                ],
                "pattern_match": {
                    "before": ["MatMul / Gemm", "Elementwise", "MatMul / Gemm"],
                    "after": "FFN_INTERMEDIATE_CHAIN",
                    "why": ["first projection produces intermediate axis j", "activation preserves j", "second projection consumes j"],
                },
                "dfa_trace": [
                    {"fact": "contraction.input[j] = DEAD", "kind": "seed", "active_nodes": ["contraction"], "active_edges": []},
                    {"fact": "activation.output[j] = DEAD", "kind": "propagate", "active_nodes": ["activation", "contraction"], "active_edges": [["activation", "contraction"]]},
                    {"fact": "activation.input[j] = DEAD", "kind": "propagate", "active_nodes": ["activation"], "active_edges": []},
                    {"fact": "expansion.output[j] = DEAD", "kind": "fixed_point", "active_nodes": ["expansion", "activation"], "active_edges": [["expansion", "activation"]]},
                ],
                "not_claimed": ["Does not prune hidden output axis.", "Does not choose channel index j.", "Does not mutate weights."],
            },
            {
                "id": "attention_value_path",
                "title": "Attention value-path propagation",
                "pattern": "ATTENTION_VALUE_PATH",
                "verdict": "proven",
                "graph": {
                    "nodes": [
                        {"id": "value_projection", "label": "Value Projection", "op": "MatMul / Gemm or recovered V slice", "axis_role": "produces value axis d"},
                        {"id": "context", "label": "Attention Context", "op": "MatMul", "axis_role": "preserves value axis d"},
                        {"id": "output_projection", "label": "Output Projection", "op": "MatMul / Gemm", "axis_role": "consumes context value axis d"},
                    ],
                    "edges": [
                        {"source": "value_projection", "target": "context", "axis": "d", "relation": "PRESERVED"},
                        {"source": "context", "target": "output_projection", "axis": "d", "relation": "CONSUMED"},
                    ],
                },
                "mlir": [
                    {"title": "Attention context", "code": "C[b, h, q, d] += P[b, h, q, k] * V[b, h, k, d]", "relation": "V.d -> C.d = PRESERVED; k is REDUCED"},
                    {"title": "Output projection", "code": "O[b, q, h] += C[b, q, d] * W[d, h]", "relation": "C.d is consumed by output projection"},
                ],
                "pattern_match": {
                    "before": ["Value Projection", "Context MatMul", "Output Projection"],
                    "after": "ATTENTION_VALUE_PATH",
                    "why": ["value projection produces d", "context preserves d", "output projection consumes d"],
                },
                "dfa_trace": [
                    {"fact": "output_projection.input[d] = DEAD", "kind": "seed", "active_nodes": ["output_projection"], "active_edges": []},
                    {"fact": "context.value_axis[d] = DEAD", "kind": "propagate", "active_nodes": ["context", "output_projection"], "active_edges": [["context", "output_projection"]]},
                    {"fact": "value_projection.output[d] = DEAD", "kind": "fixed_point", "active_nodes": ["value_projection", "context"], "active_edges": [["value_projection", "context"]]},
                ],
                "not_claimed": ["Does not prune Q/K score path.", "Does not choose value channel d.", "Does not evaluate accuracy."],
            },
            {
                "id": "qk_score_blocker",
                "title": "QK score blocker",
                "pattern": "QK_SCORE_BLOCKER",
                "verdict": "blocked_as_expected",
                "graph": {
                    "nodes": [
                        {"id": "q_projection", "label": "Q Projection", "op": "MatMul / Gemm", "axis_role": "produces head_dim d"},
                        {"id": "score_matmul", "label": "Score MatMul", "op": "MatMul", "axis_role": "reduces / mixes d"},
                        {"id": "k_projection", "label": "K Projection", "op": "MatMul / Gemm", "axis_role": "produces head_dim d"},
                    ],
                    "edges": [
                        {"source": "q_projection", "target": "score_matmul", "axis": "d", "relation": "REDUCED / MIXED"},
                        {"source": "k_projection", "target": "score_matmul", "axis": "d", "relation": "REDUCED / MIXED"},
                    ],
                },
                "mlir": [
                    {"title": "QK score contraction", "code": "S[b, h, q, k] += Q[b, h, q, d] * K[b, h, k, d]", "relation": "d is REDUCED / MIXED and disappears from output"},
                ],
                "pattern_match": {
                    "before": ["Q Projection", "Score MatMul", "K Projection"],
                    "after": "QK_SCORE_BLOCKER",
                    "why": ["feature axis d is contracted", "there is no one-to-one output axis for d", "simple propagation is blocked"],
                },
                "dfa_trace": [
                    {"fact": "attempt Q/K feature-axis propagation", "kind": "seed", "active_nodes": ["q_projection", "k_projection"], "active_edges": []},
                    {"fact": "score_matmul reduces / mixes d", "kind": "blocker", "active_nodes": ["score_matmul"], "active_edges": [["q_projection", "score_matmul"], ["k_projection", "score_matmul"]]},
                    {"fact": "BLOCKED: qk_score_contraction_mixes_channels", "kind": "blocked", "active_nodes": ["score_matmul"], "active_edges": []},
                ],
                "not_claimed": ["QK is not counted as a pruning plan.", "This proves non-propagatability for simple axis pruning."],
            },
        ],
        "warnings": warnings,
    }


def teaching_flow(config: ServerConfig) -> dict[str, Any]:
    summary, warnings = final_summary(config)
    return {
        "title": "Pipeline Walkthrough",
        "summary": summary,
        "sections": [
            {"id": "why", "title": "Why pruning propagation?", "summary": "Structural pruning must account for every affected axis.", "points": ["Begin from a local dead or pruned axis.", "Follow legal axis mappings.", "Stop at explicit protection or blocker boundaries."]},
            {"id": "split", "title": "Sparse-weight vs structural pruning", "summary": "Sparsity is not the same as deadness.", "points": ["Fine-grained zeros preserve shapes.", "Dead channels are compiler-visible structural facts."]},
            {"id": "ffn", "title": "MLP/FFN example", "summary": "Consumer-input deadness propagates back to the expansion output.", "points": ["hidden -> intermediate -> intermediate -> hidden", "op3 input[j] DEAD -> op2 output/input[j] DEAD -> op1 output[j] DEAD"]},
            {"id": "attention", "title": "Attention value-path example", "summary": "The preserved context value axis connects output projection input to value projection output.", "points": ["value projection -> attention context -> output projection", "out projection input[d] DEAD -> context value axis[d] DEAD -> value projection output[d] DEAD"]},
            {"id": "qk", "title": "QK blocker example", "summary": "Score[q,k] += Q[q,d] * K[k,d] reduces and mixes d.", "points": ["Simple one-to-one Q/K propagation is BLOCKED.", "QK score contractions are intentionally excluded from pruning-plan counts."]},
            {"id": "evidence", "title": "MLIR evidence hierarchy", "summary": "Use the strongest available local evidence and state fallbacks explicitly.", "points": ["native_mlir_dependence_evidence", "actual_loop_access_evidence", "high_level_mlir_dialect_evidence", "onnx_hint_fallback"]},
            {"id": "dfa", "title": "DFA worklist", "summary": "Seed, transfer, join, and iterate to a fixed point.", "points": ["The DFA consumes semantic roles derived from evidence.", "It reports DEAD, PROTECTED, and BLOCKED facts."]},
            {"id": "proof", "title": "All-model proof", "summary": "The current supported-model set reaches 108 / 108 plans proven.", "points": ["BERT 24/24, DistilBERT 12/12, OPT 24/24, GPT-2 24/24, ViT 24/24."]},
            {"id": "limits", "title": "Limitations", "summary": "The UI visualizes static analysis only.", "points": ["No pruning execution.", "No weight mutation.", "No accuracy or speedup evaluation.", "Ambiguous fused-QKV remains blocked unless branch evidence is recoverable."]},
        ],
        "warnings": warnings,
    }


def case_studies(config: ServerConfig) -> dict[str, Any]:
    studies = [
        ("bert-24-plan", "BERT 24-plan proof", "12 FFN + 12 attention value-path plans are proven.", "bert_24_plan_proof/index.md", {"proven": "24 / 24"}),
        ("all-model", "All-model proof", "Five supported models reach complete propagation-plan proofs.", "all_model_plan_proof/index.md", {"proven": "108 / 108"}),
        ("fused-qkv", "Fused-QKV recovery", "GPT-2 and ViT recover their value branches from fused QKV projections.", "formalization/static_pruning_propagation_notes.md", {"models": "GPT-2, ViT"}),
        ("opt-ffn-native", "OPT FFN native diagnosis", "Narrow fc1 -> activation -> fc2 evidence units upgrade OPT FFN plans to native evidence.", "opt_ffn_native_diagnosis/index.md", {"native": "12 / 12"}),
        ("qk-blocker", "QK blocker", "Q/K feature dimensions are reduced and mixed by score contraction, so they block simple propagation.", "formalization/static_pruning_propagation_notes.md", {"status": "BLOCKED"}),
    ]
    return {
        "case_studies": [
            {
                "id": study_id,
                "title": title,
                "summary": summary,
                "report_path": report_path,
                "report_url": f"/api/report-text?path={quote(report_path)}",
                "key_numbers": numbers,
                "available": (config.root / "reports" / report_path).exists(),
            }
            for study_id, title, summary, report_path, numbers in studies
        ]
    }


def resolve_report_text_path(config: ServerConfig, value: str) -> Path | None:
    decoded = unquote(value).strip()
    relative = Path(decoded)
    if decoded.startswith("reports/"):
        relative = Path(decoded.removeprefix("reports/"))
    if not decoded or relative.is_absolute() or ".." in relative.parts or relative.suffix not in REPORT_TEXT_SUFFIXES:
        return None
    report_root = (config.root / "reports").resolve()
    target = (report_root / relative).resolve()
    if report_root not in target.parents or not target.is_file():
        return None
    return target


def discover_models(config: ServerConfig) -> list[dict[str, Any]]:
    coverage = load_json(config.root / "reports" / "static_coverage_study" / "index.json")
    status_by_model = {item.get("model_name", ""): item for item in coverage.get("models", [])}
    models: list[dict[str, Any]] = []
    if not config.report_root.exists():
        return []
    for model_dir in sorted(config.report_root.iterdir(), key=lambda item: item.name):
        if not model_dir.is_dir() or model_dir.name == "cross_model":
            continue
        index_path = model_dir / "index.json"
        if not index_path.exists():
            continue
        index = load_json(index_path)
        model_name = index.get("model_name") or model_dir.name
        summary = index.get("model_summary", {})
        ranking = summary.get("ranking", {})
        plans = summary.get("plans", {})
        validation = summary.get("plan_validation", {})
        status = status_by_model.get(model_name, {})
        models.append(
            {
                "id": model_dir.name,
                "display_name": model_name,
                "index_path": str(index_path),
                "num_layers": summary.get("layers_generated", 0),
                "num_subgraphs": summary.get("total_subgraphs", 0),
                "safe_candidates": ranking.get("safe", 0),
                "mlp_safe_candidates": ranking.get("mlp_safe_candidates", 0),
                "plans": plans.get("total_plans", plans.get("plans", 0)),
                "valid_plans": validation.get("valid", validation.get("valid_plans", 0)),
                "status": status.get("final_status", "unknown"),
            }
        )
    return models


def resolve_model_dir(config: ServerConfig, model_id: str) -> Path | None:
    decoded = unquote(model_id)
    candidates = [decoded, safe_model_name(decoded)]
    for candidate in candidates:
        path = config.report_root / candidate
        if path.exists() and (path / "index.json").exists():
            return path
    for path in config.report_root.glob("*/index.json"):
        parent = path.parent
        index = load_json(path)
        if decoded in {parent.name, index.get("model_name", "")}:
            return parent
    return None


def model_index(config: ServerConfig, model_id: str) -> dict[str, Any] | None:
    model_dir = resolve_model_dir(config, model_id)
    if not model_dir:
        return None
    data = load_json(model_dir / "index.json")
    data["id"] = model_dir.name
    return data


def layer_dirs(model_dir: Path) -> list[Path]:
    root = model_dir / "layers"
    if not root.exists():
        return []
    return sorted(root.glob("layer_*"), key=lambda item: int(re.findall(r"\d+", item.name)[-1]) if re.findall(r"\d+", item.name) else 10**9)


def layer_summaries(model_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for layer_dir in layer_dirs(model_dir):
        data = load_json(layer_dir / "index.json")
        summary = data.get("summary", {})
        if summary:
            out.append(summary)
    return out


def subgraph_analysis_paths(config: ServerConfig, model_dir: Path, layer: int) -> list[Path]:
    primary = model_dir / "layers" / f"layer_{layer}" / "subgraphs"
    paths = sorted(primary.glob("*/analysis.json"), key=lambda p: p.parent.name)
    if paths:
        return paths
    fallback = config.fallback_layer_root / model_dir.name / f"layer_{layer}"
    return sorted(fallback.glob("*/analysis.json"), key=lambda p: p.parent.name)


def subgraph_summaries(config: ServerConfig, model_dir: Path, layer: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in subgraph_analysis_paths(config, model_dir, layer):
        analysis = load_json(path)
        cls = analysis.get("classification", {})
        out.append(
            {
                "ordinal": analysis.get("ordinal"),
                "node_slug": analysis.get("node_slug") or path.parent.name,
                "display_name": analysis.get("display_name", path.parent.name),
                "semantic_category": analysis.get("semantic_category", ""),
                "pruning_class": cls.get("pruning_class", "unknown"),
                "plan_status": cls.get("plan_status", "unknown"),
                "validation_status": cls.get("validation_status", "unknown"),
                "onnx_status": analysis.get("onnx_export", {}).get("status", "skipped"),
            }
        )
    out.sort(key=lambda item: int(item.get("ordinal") or 10**9))
    return out


def artifact_paths(config: ServerConfig, model_safe: str, layer: int, node: str) -> dict[str, dict[str, str]]:
    candidates = [
        config.artifact_root / model_safe / "layers" / f"layer_{layer}" / node,
        config.root / "artifacts" / "attention_value_path_subgraphs" / model_safe / "layers" / f"layer_{layer}" / node,
        config.root / "artifacts" / "opt_ffn_native_subgraphs" / model_safe / "layers" / f"layer_{layer}" / node,
        config.fallback_artifact_root / model_safe / f"layer_{layer}" / node,
    ]
    out: dict[str, dict[str, str]] = {}
    for root in candidates:
        for ext, key in [(".onnx", "onnx"), (".svg", "svg"), (".dot", "dot")]:
            path = root / f"subgraph{ext}"
            if path.exists() and key not in out:
                out[key] = {"path": str(path), "url": artifact_url(path)}
    return out


def artifact_url(path: Path) -> str:
    return "/artifact/" + quote(str(path.resolve()), safe="")


def indexed_artifact_paths(config: ServerConfig, model_safe: str, layer: int, node: str) -> dict[str, dict[str, str]]:
    index = load_json(config.root / "reports" / "ui_artifact_index" / "index.json")
    for entry in index.get("entries", []):
        if entry.get("model") != model_safe or entry.get("layer") != layer or entry.get("subgraph") != node:
            continue
        return {
            kind: {"path": str(config.root / path), "url": artifact_url(config.root / path)}
            for kind, path in entry.get("paths", {}).items()
            if (config.root / path).exists()
        }
    return {}


def relative_to_root(config: ServerConfig, path: Path) -> str:
    return str(path.resolve().relative_to(config.root.resolve()))


def artifact_text_url(config: ServerConfig, path: Path, focus: str | None = None, context: int | None = None) -> str:
    params: dict[str, Any] = {"path": relative_to_root(config, path)}
    if focus:
        params["focus"] = focus
    if context is not None:
        params["context"] = context
    return "/api/artifact-text?" + urlencode(params)


def resolve_artifact_text_path(config: ServerConfig, value: str) -> Path | None:
    decoded = unquote(value).strip()
    if not decoded:
        return None
    raw = Path(decoded)
    if ".." in raw.parts or raw.suffix not in ARTIFACT_TEXT_SUFFIXES:
        return None
    target = raw.resolve() if raw.is_absolute() else (config.root / raw).resolve()
    repo_root = config.root.resolve()
    if target != repo_root and repo_root not in target.parents:
        return None
    return target if target.is_file() else None


def _detect_code_language(path: Path) -> str:
    return {
        ".mlir": "mlir",
        ".dot": "dot",
        ".json": "json",
        ".md": "markdown",
        ".csv": "csv",
    }.get(path.suffix, "text")


def _read_text_artifact_safe(path: Path, max_bytes: int) -> tuple[str, bool, int, list[str]]:
    size_bytes = path.stat().st_size
    truncated = size_bytes > max_bytes
    text = path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    warnings = [f"Artifact exceeds {max_bytes} bytes; focused search covers the returned prefix only."] if truncated else []
    return text, truncated, size_bytes, warnings


def _read_focused_lines_safe(path: Path) -> tuple[list[str], bool, int, list[str]]:
    lines: list[str] = []
    abbreviated = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if len(line) > MAX_FOCUSED_CONTEXT_LINE_CHARS:
                abbreviated = True
                line = f"{line[:MAX_FOCUSED_CONTEXT_LINE_CHARS]} ... [oversized line abbreviated]"
            lines.append(line)
    warnings = ["Oversized MLIR constant lines were abbreviated so loop/access regions remain visible."] if abbreviated else []
    return lines, abbreviated, path.stat().st_size, warnings


def _find_focus_matches(text: str, focus: str) -> list[dict[str, Any]]:
    terms = FOCUS_TERMS.get(focus, FOCUS_TERMS["all"])
    matches: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for kind in terms:
            if kind in line:
                matches.append({"line_no": line_no, "kind": kind, "text": line})
                break
    return matches


def _extract_context_sections(lines: list[str], matches: list[dict[str, Any]], context: int) -> list[dict[str, Any]]:
    ranges: list[tuple[int, int]] = []
    for match in matches:
        start = max(1, match["line_no"] - context)
        end = min(len(lines), match["line_no"] + context)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    sections: list[dict[str, Any]] = []
    for start, end in ranges:
        section_matches = [match for match in matches if start <= match["line_no"] <= end]
        first = section_matches[0]
        sections.append(
            {
                "title": f"{first['kind']} around line {first['line_no']}",
                "start_line": start,
                "end_line": end,
                "text": "\n".join(lines[start - 1 : end]),
                "match_lines": [match["line_no"] for match in section_matches],
            }
        )
    return sections


def _bounded_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        return min(maximum, max(minimum, int(value or default)))
    except ValueError:
        return default


def artifact_text_payload(config: ServerConfig, value: str, focus: str | None = None, context: int = 20, max_bytes: int = MAX_ARTIFACT_TEXT_BYTES) -> dict[str, Any] | None:
    path = resolve_artifact_text_path(config, value)
    if not path:
        return None
    requested_focus = focus if focus in FOCUS_TERMS else None
    if requested_focus:
        lines, truncated, size_bytes, warnings = _read_focused_lines_safe(path)
        text = "\n".join(lines)
    else:
        text, truncated, size_bytes, warnings = _read_text_artifact_safe(path, max_bytes)
        lines = text.splitlines()
        if truncated:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                line_count = sum(1 for _ in handle)
        else:
            line_count = len(lines)
    matches: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    if requested_focus:
        matches = _find_focus_matches(text, requested_focus)
        if requested_focus == "affine" and not matches:
            warnings.append("No affine.for/affine.load/affine.store found in this artifact.")
            matches = [
                {"line_no": line_no, "kind": kind, "text": line}
                for line_no, line in enumerate(lines, start=1)
                for kind in FALLBACK_FOCUS_TERMS
                if kind in line
            ]
            if matches:
                warnings.append("Showing fallback high-level or alternate loop operations.")
        sections = _extract_context_sections(lines, matches, context)
    return {
        "path": relative_to_root(config, path),
        "language": _detect_code_language(path),
        "text": "\n\n".join(section["text"] for section in sections) if requested_focus else text,
        "truncated": truncated,
        "size_bytes": size_bytes,
        "line_count": len(lines) if requested_focus else line_count,
        "focus": requested_focus or "all",
        "matches": matches,
        "sections": sections,
        "warnings": warnings,
    }


def artifact_kind(node: str) -> str:
    value = node.lower()
    if "attention_value_path" in value or "value_path" in value:
        return "attention_value_path"
    if "attention_score" in value or "score_matmul" in value:
        return "score"
    if "attention_context" in value or "context_matmul" in value:
        return "context"
    if any(token in value for token in ["feed_forward", "mlp", "ffn"]):
        return "mlp"
    return "unknown"


def model_alias(model: str) -> str:
    safe = safe_model_name(model).lower()
    if safe.startswith("distilbert"):
        return "distilbert"
    if safe.startswith("bert"):
        return "bert"
    if "opt" in safe:
        return "opt"
    if safe == "gpt2":
        return "gpt2"
    if "vit" in safe:
        return "vit"
    return re.sub(r"[^a-z0-9]+", "_", safe).strip("_")


def mlir_case_aliases(model: str, layer: int, node: str) -> list[str]:
    prefix = model_alias(model)
    kind = artifact_kind(node)
    if kind == "score":
        return [f"{prefix}_layer{layer}_score", f"{prefix}_layer{layer}_attention_score"]
    if kind == "context":
        return [f"{prefix}_layer{layer}_context", f"{prefix}_layer{layer}_attention_context"]
    if kind == "attention_value_path":
        return [f"{prefix}_layer{layer}_attention_value_path", f"{prefix}_layer{layer}_value_path"]
    if kind == "mlp":
        return [f"{prefix}_layer{layer}_mlp", f"{prefix}_layer{layer}_ffn"]
    return []


def mlir_search_roots(config: ServerConfig) -> list[Path]:
    return [
        config.root / "reports" / "mlir_axis_bridge",
        config.root / "reports" / "mlir_evidence_coverage",
        config.root / "reports" / "mlir_evidence_coverage_bert_24_plan",
        config.root / "reports" / "mlir_evidence_coverage_opt_ffn_native_diagnosis",
        config.root / "reports" / "all_model_plan_proof",
        config.root / "reports" / "opt_ffn_native_diagnosis",
    ]


def discover_mlir_case_dirs(config: ServerConfig, model: str, layer: int, node: str) -> list[Path]:
    aliases = mlir_case_aliases(model, layer, node)
    if not aliases:
        return []
    found: list[Path] = []
    seen: set[Path] = set()
    for root in mlir_search_roots(config):
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_dir():
                continue
            if not any(candidate.name == alias or candidate.name.startswith(f"{alias}_") for alias in aliases):
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                found.append(candidate)
                seen.add(resolved)
    return found


def dialect_hints(path: Path) -> list[str]:
    hints = [
            "onnx.",
            "krnl.",
            "linalg.",
            "scf.for",
            "affine.for",
            "affine.load",
            "affine.store",
            "memref.load",
            "memref.store",
    ]
    found: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            found.update(hint for hint in hints if hint in line)
    return [hint for hint in hints if hint in found]


def mlir_interesting_summary(path: Path) -> dict[str, Any]:
    counts = {term: 0 for term in FOCUS_TERMS["all"]}
    line_count = 0
    first_structural_line: int | None = None
    first_fallback_line: int | None = None
    structural_terms = ["affine.for", "affine.load", "affine.store", "scf.for", "memref.load", "memref.store", "linalg.matmul", "krnl.matmul"]
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_count, line in enumerate(handle, start=1):
            matched = False
            for term in FOCUS_TERMS["all"]:
                occurrences = line.count(term)
                counts[term] += occurrences
                matched = matched or bool(occurrences)
            if matched and first_fallback_line is None:
                first_fallback_line = line_count
            if first_structural_line is None and any(term in line for term in structural_terms):
                first_structural_line = line_count
    return {
        "line_count": line_count,
        "interesting_counts": counts,
        "first_interesting_line": first_structural_line or first_fallback_line,
    }


def mlir_stage(path: Path) -> str:
    if "_onnx.onnx.mlir" in path.name:
        return "onnx_dialect"
    if ".input.mlir" in path.name:
        return "lowered_input"
    return "lowered_affine"


def discover_mlir_bundle(config: ServerConfig, model: str, layer: int, node: str) -> tuple[list[dict[str, Any]], Path | None, Path | None]:
    artifacts: list[dict[str, Any]] = []
    native_json: Path | None = None
    python_json: Path | None = None
    seen_mlir: set[Path] = set()
    for case_dir in discover_mlir_case_dirs(config, model, layer, node):
        for path in sorted(case_dir.rglob("*.mlir")):
            resolved = path.resolve()
            if resolved in seen_mlir:
                continue
            seen_mlir.add(resolved)
            artifacts.append({
                "stage": mlir_stage(path),
                "path": relative_to_root(config, path),
                "text_url": artifact_text_url(config, path),
                "focused_text_url": artifact_text_url(config, path, focus="affine", context=30),
                "dialect_hints": dialect_hints(path),
                **mlir_interesting_summary(path),
            })
        if not native_json:
            native_json = next(iter(sorted(case_dir.rglob("*native_dependence*.json"))), None)
        if not python_json:
            python_json = next(iter(sorted(case_dir.rglob("*python_dependence*.json"))), None)
    artifacts.sort(key=lambda item: ({"onnx_dialect": 0, "lowered_input": 1, "lowered_affine": 2}.get(item["stage"], 9), item["path"]))
    return artifacts, native_json, python_json


def evidence_summary(model: str, layer: int, node: str, native_json: Path | None, mlir_artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    kind = artifact_kind(node)
    evidence_tier = "native_mlir_dependence_evidence" if native_json else ("actual_loop_access_evidence" if mlir_artifacts else "unavailable")
    if kind == "mlp":
        return {
            "pattern": "FFN_INTERMEDIATE_CHAIN",
            "evidence_tier": evidence_tier,
            "axis_relations": [{"source": "activation.j", "target": "contraction.input.j", "relation": "PRESERVED / CONSUMED"}],
            "dfa_verdict": "proven",
        }
    if kind == "attention_value_path":
        return {
            "pattern": "ATTENTION_VALUE_PATH",
            "evidence_tier": evidence_tier,
            "axis_relations": [{"source": "V.d", "target": "Context.d", "relation": "PRESERVED"}],
            "dfa_verdict": "proven",
        }
    if kind == "score":
        return {
            "pattern": "QK_SCORE_BLOCKER",
            "evidence_tier": evidence_tier,
            "axis_relations": [{"source": "Q/K.d", "target": "Score", "relation": "REDUCED / MIXED"}],
            "dfa_verdict": "blocked_as_expected",
        }
    return {"pattern": "unknown", "evidence_tier": evidence_tier, "axis_relations": [], "dfa_verdict": "unknown"}


def subgraph_title(config: ServerConfig, model: str, layer: int, node: str) -> str:
    analysis = load_json(config.report_root / safe_model_name(model) / "layers" / f"layer_{layer}" / "subgraphs" / node / "analysis.json")
    return analysis.get("display_name", node.replace("_", " ").title())


def artifact_bundle(config: ServerConfig, model: str, layer: int, node: str) -> dict[str, Any]:
    model_safe = safe_model_name(model)
    paths = artifact_paths(config, model_safe, layer, node)
    if not paths:
        paths = indexed_artifact_paths(config, model_safe, layer, node)
    mlir_artifacts, native_json, python_json = discover_mlir_bundle(config, model_safe, layer, node)
    warnings: list[str] = []
    if not paths:
        warnings.append("No ONNX, SVG, or DOT artifacts were found for this subgraph.")
    if not mlir_artifacts:
        warnings.append("No pre-generated MLIR artifacts were found. The UI does not lower ONNX on request.")
    if not native_json:
        warnings.append("No pre-generated native dependence JSON was found.")
    dependence_links = {
        key: artifact_text_url(config, value)
        for key, value in [("native_json", native_json), ("python_json", python_json)]
        if value
    }
    return {
        "model": model_safe,
        "layer": layer,
        "subgraph": node,
        "title": subgraph_title(config, model_safe, layer, node),
        "paths": {key: relative_to_root(config, Path(value["path"])) for key, value in paths.items()},
        "links": {key: value["url"] for key, value in paths.items()},
        "mlir": {"available": bool(mlir_artifacts), "artifacts": mlir_artifacts},
        "dependence": {
            "native_json": relative_to_root(config, native_json) if native_json else None,
            "python_json": relative_to_root(config, python_json) if python_json else None,
            "links": dependence_links,
        },
        "evidence": evidence_summary(model_safe, layer, node, native_json, mlir_artifacts),
        "warnings": warnings,
    }


def evidence_artifact_map() -> dict[str, dict[str, Any]]:
    mappings = {
        "ffn_intermediate": ("bert-base-uncased", 0, "12_layer_0_feed_forward"),
        "attention_value_path": ("bert-base-uncased", 0, "bert_layer_0_attention_value_path"),
        "qk_score_blocker": ("bert-base-uncased", 0, "05_layer_0_attention_score_matmul"),
        "gpt2_attention_value_path": ("gpt2", 0, "gpt2_layer_0_attention_value_path"),
        "opt_ffn_native": ("facebook__opt-125m", 0, "opt_layer_0_ffn_native_core"),
    }
    return {
        key: {
            "model": model,
            "layer": layer,
            "subgraph": subgraph,
            "artifact_bundle_url": "/api/artifact-bundle?" + urlencode({"model": model, "layer": layer, "subgraph": subgraph}),
        }
        for key, (model, layer, subgraph) in mappings.items()
    }


def search(config: ServerConfig, query: str, model: str | None = None, layer: int | None = None) -> list[dict[str, Any]]:
    needle = query.lower().strip()
    if not needle:
        return []
    models = [resolve_model_dir(config, model)] if model else [config.report_root / item["id"] for item in discover_models(config)]
    matches: list[dict[str, Any]] = []
    for model_dir in [item for item in models if item]:
        index = load_json(model_dir / "index.json")
        model_name = index.get("model_name", model_dir.name)
        layers = [layer] if layer is not None else [int(re.findall(r"\d+", d.name)[-1]) for d in layer_dirs(model_dir) if re.findall(r"\d+", d.name)]
        for layer_index in layers:
            for path in subgraph_analysis_paths(config, model_dir, int(layer_index)):
                analysis = load_json(path)
                explanation = read_text(path.with_name("explanation.md"))
                cls = analysis.get("classification", {})
                haystack = " ".join(
                    str(value)
                    for value in [
                        model_name,
                        analysis.get("display_name", ""),
                        analysis.get("semantic_category", ""),
                        cls.get("pruning_class", ""),
                        cls.get("plan_status", ""),
                        cls.get("validation_status", ""),
                        analysis.get("verdict", ""),
                        analysis.get("explanation", ""),
                        explanation,
                    ]
                ).lower()
                if needle in haystack:
                    matches.append(
                        {
                            "model": model_name,
                            "model_id": model_dir.name,
                            "layer": int(layer_index),
                            "node_slug": analysis.get("node_slug", path.parent.name),
                            "display_name": analysis.get("display_name", path.parent.name),
                            "semantic_category": analysis.get("semantic_category", ""),
                            "pruning_class": cls.get("pruning_class", "unknown"),
                            "validation_status": cls.get("validation_status", "unknown"),
                        }
                    )
    return matches[:200]


def is_allowed_file(config: ServerConfig, path: Path) -> bool:
    try:
        resolved = path.resolve()
    except Exception:
        return False
    allowed_roots = [
        config.report_root.resolve(),
        config.artifact_root.resolve(),
        config.fallback_layer_root.resolve(),
        config.fallback_artifact_root.resolve(),
        (config.root / "artifacts" / "attention_value_path_subgraphs").resolve(),
        (config.root / "artifacts" / "opt_ffn_native_subgraphs").resolve(),
        (config.root / "reports" / "static_coverage_study").resolve(),
        (config.root / "reports" / "rule_gap_diagnosis_compare").resolve(),
        (config.root / "reports" / "rule_gap_diagnosis").resolve(),
    ]
    if resolved.suffix not in {".onnx", ".svg", ".dot", ".md", ".json"}:
        return False
    return any(resolved == root or root in resolved.parents for root in allowed_roots)


def route_api(config: ServerConfig, path: str, query: dict[str, list[str]]) -> tuple[HTTPStatus, Any]:
    parts = [unquote(item) for item in path.strip("/").split("/")]
    if path == "/api/health":
        return HTTPStatus.OK, {"ok": True}
    if path == "/api/models":
        return HTTPStatus.OK, discover_models(config)
    if path == "/api/coverage":
        coverage = load_json(config.root / "reports" / "static_coverage_study" / "index.json")
        table = [
            {
                "model": item.get("model_name"),
                "status": item.get("final_status"),
                "layers": item.get("artifacts", {}).get("full_model_report", {}).get("layers", 0),
                "subgraphs": item.get("artifacts", {}).get("full_model_report", {}).get("subgraphs", 0),
                "safe": item.get("artifacts", {}).get("ranking", {}).get("safe", 0),
                "plans": item.get("artifacts", {}).get("plans", {}).get("plans", 0),
                "valid_plans": item.get("artifacts", {}).get("validation", {}).get("valid_plans", 0),
            }
            for item in coverage.get("models", [])
        ]
        return HTTPStatus.OK, {"coverage": coverage, "table": table}
    if path == "/api/overview":
        return HTTPStatus.OK, overview(config)
    if path == "/api/proof-summary":
        return HTTPStatus.OK, proof_summary(config)
    if path == "/api/teaching-flow":
        return HTTPStatus.OK, teaching_flow(config)
    if path == "/api/pipeline-flow":
        return HTTPStatus.OK, pipeline_flow(config)
    if path == "/api/evidence-traces":
        return HTTPStatus.OK, evidence_traces(config)
    if path == "/api/evidence-artifact-map":
        return HTTPStatus.OK, evidence_artifact_map()
    if path == "/api/case-studies":
        return HTTPStatus.OK, case_studies(config)
    if path == "/api/artifact-text":
        payload = artifact_text_payload(
            config,
            query.get("path", [""])[0],
            focus=query.get("focus", [None])[0],
            context=_bounded_int(query.get("context", [None])[0], default=20, minimum=0, maximum=200),
            max_bytes=_bounded_int(query.get("max_bytes", [None])[0], default=MAX_ARTIFACT_TEXT_BYTES, minimum=1024, maximum=MAX_ARTIFACT_TEXT_BYTES),
        )
        return (HTTPStatus.OK, payload) if payload else (HTTPStatus.BAD_REQUEST, {"error": "invalid artifact path"})
    if path == "/api/artifact-bundle":
        model = query.get("model", [""])[0]
        layer_value = query.get("layer", [""])[0]
        subgraph = query.get("subgraph", [""])[0]
        if not model or not layer_value.isdigit() or not subgraph or ".." in Path(subgraph).parts:
            return HTTPStatus.BAD_REQUEST, {"error": "invalid artifact bundle request"}
        return HTTPStatus.OK, artifact_bundle(config, model, int(layer_value), subgraph)
    if path == "/api/report-text":
        report_path = resolve_report_text_path(config, query.get("path", [""])[0])
        if not report_path:
            return HTTPStatus.BAD_REQUEST, {"error": "invalid report path"}
        return HTTPStatus.OK, {
            "path": str(report_path.relative_to(config.root)),
            "format": report_path.suffix.removeprefix("."),
            "text": read_text(report_path),
        }
    if path == "/api/search":
        text = query.get("q", [""])[0]
        model = query.get("model", [None])[0]
        layer_value = query.get("layer", [None])[0]
        layer = int(layer_value) if layer_value not in {None, ""} else None
        return HTTPStatus.OK, {"matches": search(config, text, model, layer)}
    if len(parts) >= 3 and parts[1] == "models":
        model_dir = resolve_model_dir(config, parts[2])
        if not model_dir:
            return HTTPStatus.NOT_FOUND, {"error": "model not found"}
        if len(parts) == 3:
            data = load_json(model_dir / "index.json")
            data["id"] = model_dir.name
            return HTTPStatus.OK, data
        if len(parts) == 4 and parts[3] == "layers":
            return HTTPStatus.OK, layer_summaries(model_dir)
        if len(parts) >= 5 and parts[3] == "layers":
            layer = int(parts[4])
            layer_path = model_dir / "layers" / f"layer_{layer}" / "index.json"
            if len(parts) == 5:
                data = load_json(layer_path)
                return (HTTPStatus.OK, data) if data else (HTTPStatus.NOT_FOUND, {"error": "layer not found"})
            if len(parts) == 6 and parts[5] == "subgraphs":
                return HTTPStatus.OK, subgraph_summaries(config, model_dir, layer)
            if len(parts) == 7 and parts[5] == "subgraphs":
                node = parts[6]
                analysis_path = model_dir / "layers" / f"layer_{layer}" / "subgraphs" / node / "analysis.json"
                explanation_path = analysis_path.with_name("explanation.md")
                if not analysis_path.exists():
                    analysis_path = config.fallback_layer_root / model_dir.name / f"layer_{layer}" / node / "analysis.json"
                    explanation_path = analysis_path.with_name("explanation.md")
                if not analysis_path.exists():
                    return HTTPStatus.NOT_FOUND, {"error": "subgraph not found"}
                return HTTPStatus.OK, {
                    "analysis": load_json(analysis_path),
                    "explanation_md": read_text(explanation_path),
                    "artifact_paths": artifact_paths(config, model_dir.name, layer, node),
                }
        if len(parts) == 4 and parts[3] in {"ranking", "plans", "validation", "diagnosis", "status", "deadbranch"}:
            safe = model_dir.name
            mapping = {
                "ranking": config.root / "reports" / "pruning_opportunity_rankings" / f"{safe}.json",
                "plans": config.root / "reports" / "pruning_plans" / f"{safe}.json",
                "validation": config.root / "reports" / "pruning_plan_validation" / f"{safe}.json",
                "diagnosis": config.root / "reports" / "rule_gap_diagnosis" / f"{safe}.json",
                "status": config.root / "reports" / "static_pipeline_status" / f"{safe}.json",
                "deadbranch": config.root / "reports" / "deadbranch_propagation" / f"{safe}.json",
            }
            return HTTPStatus.OK, load_json(mapping[parts[3]])
    return HTTPStatus.NOT_FOUND, {"error": "not found"}


def create_handler(config: ServerConfig):
    class AnalysisUIHandler(SimpleHTTPRequestHandler):
        server_version = "AnalysisUIServer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            if config.verbose:
                super().log_message(fmt, *args)

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self.handle_api(parsed.path, parse_qs(parsed.query))
            elif parsed.path.startswith("/artifact/"):
                self.handle_artifact(parsed.path)
            else:
                self.handle_ui(parsed.path)

        def handle_api(self, path: str, query: dict[str, list[str]]) -> None:
            try:
                status, data = route_api(config, path, query)
                return self.write_json(data, status=status)
            except Exception as exc:
                if config.verbose:
                    raise
                return self.write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def handle_artifact(self, path: str) -> None:
            requested = Path(unquote(path.removeprefix("/artifact/")))
            if not requested.is_absolute():
                requested = config.root / requested
            if not requested.exists() or not is_allowed_file(config, requested):
                self.send_error(HTTPStatus.NOT_FOUND, "artifact not found")
                return
            mime = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(requested.stat().st_size))
            self.end_headers()
            with requested.open("rb") as handle:
                self.wfile.write(handle.read())

        def handle_ui(self, path: str) -> None:
            if config.ui_dist.exists() and (config.ui_dist / "index.html").exists():
                clean = posixpath.normpath(unquote(path)).lstrip("/")
                target = config.ui_dist / clean if clean and clean != "." else config.ui_dist / "index.html"
                if not target.exists() or target.is_dir():
                    target = config.ui_dist / "index.html"
                if config.ui_dist.resolve() not in target.resolve().parents and target.resolve() != config.ui_dist.resolve():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime = mimetypes.guess_type(str(target))[0] or "text/html"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(target.stat().st_size))
                self.end_headers()
                self.wfile.write(target.read_bytes())
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"""<!doctype html><html><body><h1>Pruning Analysis Web UI</h1>
<p>React build not found. Run:</p>
<pre>cd ui/pruning-analysis-explorer
npm install
npm run build</pre>
<p>Then restart this server.</p></body></html>"""
            )

        def write_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return AnalysisUIHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the pruning analysis web UI and local report API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--report-root", default="reports/model_analysis_reports")
    parser.add_argument("--artifact-root", default="artifacts/model_analysis_subgraphs")
    parser.add_argument("--fallback-layer-root", default="reports/layer_subgraph_validation")
    parser.add_argument("--ui-dist", default="ui/pruning-analysis-explorer/dist")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ServerConfig(
        root=ROOT,
        report_root=ROOT / args.report_root,
        artifact_root=ROOT / args.artifact_root,
        fallback_layer_root=ROOT / args.fallback_layer_root,
        fallback_artifact_root=ROOT / "artifacts" / "layer_subgraphs",
        ui_dist=ROOT / args.ui_dist,
        verbose=args.verbose,
    )
    handler = create_handler(config)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[analysis-ui] http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[analysis-ui] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
