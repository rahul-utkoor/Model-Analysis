from experimental.pruning_proof_report.aggregate import aggregate_evidence
from experimental.pruning_proof_report.proof_case import ProofEvidence
from experimental.pruning_proof_report.report import render_index_markdown


def test_markdown_report_contains_key_teaching_conclusions() -> None:
    evidence = [ProofEvidence("ffn", "model", 0, "mlp", "subgraph.onnx", True, verdict="proven")]
    text = render_index_markdown(evidence, aggregate_evidence(evidence))
    assert "# Cross-Evidence Pruning Proof Report" in text
    assert "Names are syntax; evidence comes from graph/shape/loop/access relations." in text
    assert "FFN propagation is proven by intermediate-axis produced/preserved/consumed structure." in text
    assert "Attention value context is proven by V.value_dim preservation into Context.value_context_dim." in text
    assert "Q/K propagation is blocked because Q/K feature axes are reduced/mixed in the score contraction." in text
    assert "MLIR is used as a selected-subgraph local evidence generator, not as the pruning framework itself." in text
