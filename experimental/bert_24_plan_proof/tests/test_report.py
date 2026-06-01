from experimental.bert_24_plan_proof.report import render_markdown
from experimental.bert_24_plan_proof.runner import build_bert_24_plan_proof
from experimental.bert_24_plan_proof.tests.test_bert_24_plan_summary import _coverage, _paths, _plans, _validations


def test_report_contains_teaching_story() -> None:
    text = render_markdown(build_bert_24_plan_proof(_plans(), _validations(), _paths(), _coverage()))
    assert "BERT 24-Plan" in text
    assert "12 x 2 = 24" in text
    assert "FFN_INTERMEDIATE_CHAIN" in text
    assert "ATTENTION_VALUE_PATH" in text
    assert "QK score paths remain blockers" in text
