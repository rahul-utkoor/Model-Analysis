from pathlib import Path

from experimental.pruning_proof_report.config import default_proof_cases, select_case
from experimental.pruning_proof_report.proof_case import ProofCase
from experimental.pruning_proof_report.runner import ProofRunOptions, run_proof_case


def test_default_config_cases() -> None:
    cases = default_proof_cases()
    assert {case.case_id for case in cases} >= {
        "gpt2_layer0_mlp",
        "opt_layer0_mlp",
        "bert_layer0_attention_score",
        "bert_layer0_attention_context",
    }
    assert all(isinstance(case.onnx_path, str) for case in cases)
    assert select_case(cases, "gpt2_layer0_mlp")[0].model_name == "gpt2"


def test_missing_case_not_fatal(tmp_path: Path) -> None:
    case = ProofCase("missing", "test-model", 0, "missing-node", str(tmp_path / "missing.onnx"), None, None, "")
    evidence = run_proof_case(case, ProofRunOptions())
    assert evidence.found is False
    assert evidence.verdict == "unknown"
    assert "missing" in evidence.limitations[0]
