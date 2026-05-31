from experimental.pruning_proof_report.aggregate import aggregate_evidence
from experimental.pruning_proof_report.proof_case import AxisRelationRecord, ProofEvidence
from experimental.pruning_proof_report.runner import _verdict
from experimental.pruning_proof_report.proof_case import ProofCase


def _evidence(case_id: str, verdict: str, source: str = "native_mlir_dependence_evidence", found: bool = True) -> ProofEvidence:
    return ProofEvidence(case_id, "model", 0, "subgraph", "subgraph.onnx", found, evidence_source=source, verdict=verdict)


def test_aggregate_counts() -> None:
    aggregate = aggregate_evidence(
        [
            _evidence("a", "proven"),
            _evidence("b", "fallback_proven", "onnx_hint_fallback"),
            _evidence("c", "blocked"),
            _evidence("d", "unknown", "unavailable", False),
        ]
    )
    assert aggregate.cases_total == 4
    assert aggregate.cases_found == 3
    assert aggregate.cases_missing == 1
    assert aggregate.proven == 1
    assert aggregate.fallback_proven == 1
    assert aggregate.blocked == 1


def test_verdict_rules() -> None:
    ffn = ProofCase("ffn", "model", 0, "ffn", "ffn.onnx", "FFN_INTERMEDIATE_CHAIN", "deadness", "")
    qk = ProofCase("qk", "model", 0, "qk", "qk.onnx", "QK_SCORE_BLOCKER", "blocked", "")
    context = ProofCase("context", "model", 0, "context", "context.onnx", "ATTENTION_CONTEXT_LIKE", "preserved", "")
    preserved = [AxisRelationRecord("V.value_dim", "Context.value_context_dim", "PRESERVED", "high", "same free axis")]
    no_dfa = {"ran": False}
    assert _verdict(ffn, "native_mlir_dependence_evidence", ["FFN_INTERMEDIATE_CHAIN"], [], {"ran": True}) == "proven"
    assert _verdict(ffn, "onnx_hint_fallback", ["FFN_INTERMEDIATE_CHAIN"], [], {"ran": True}) == "fallback_proven"
    assert _verdict(qk, "actual_loop_access_evidence", ["QK_SCORE_BLOCKER"], [], {"ran": True}) == "blocked"
    assert _verdict(context, "native_mlir_dependence_evidence", ["ATTENTION_CONTEXT_LIKE"], preserved, no_dfa) == "partial"
