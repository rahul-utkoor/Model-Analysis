from model_analysis.attention_value_path_subgraph import make_attention_value_path_report
from model_analysis.attention_value_path_subgraph_text import attention_value_path_report_to_markdown
from test_attention_value_path_subgraph import _model, _pair
from model_analysis.attention_value_path_subgraph import bind_path_to_onnx, detect_attention_value_paths


def test_text_report_explains_seedable_value_path() -> None:
    path = detect_attention_value_paths("facebook/opt-125m", {"pairs": [_pair()]})[0]
    bind_path_to_onnx(path, _model())
    text = attention_value_path_report_to_markdown(make_attention_value_path_report("facebook/opt-125m", [path]))
    assert "value projection" in text
    assert "attention context" in text
    assert "output projection" in text
    assert "Seedable: `1`" in text
    assert "V.value_dim -> Context.value_context_dim" in text
