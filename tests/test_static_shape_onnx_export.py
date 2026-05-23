from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_static_shape_onnx.py"
SPEC = importlib.util.spec_from_file_location("export_static_shape_onnx", SCRIPT_PATH)
assert SPEC and SPEC.loader
static_export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(static_export)

TextModelOnnxWrapper = static_export.TextModelOnnxWrapper
export_static_text_model = static_export.export_static_text_model
filter_inputs_for_model = static_export.filter_inputs_for_model


class InputIdsAndMaskModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=input_ids.float() + attention_mask.float())


class BertInputModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, token_type_ids):
        return SimpleNamespace(logits=input_ids.float() + attention_mask.float() + token_type_ids.float())


class KeywordCaptureModel(torch.nn.Module):
    def __init__(self, tuple_output: bool = False) -> None:
        super().__init__()
        self.received = None
        self.tuple_output = tuple_output

    def forward(self, **kwargs):
        self.received = kwargs
        logits = kwargs["input_ids"].float()
        return (logits,) if self.tuple_output else SimpleNamespace(logits=logits)


class DummyTokenizer:
    pad_token = "[PAD]"
    eos_token = None
    unk_token = "[UNK]"

    def __call__(self, texts, **kwargs):
        shape = (len(texts), kwargs["max_length"])
        return {
            "input_ids": torch.ones(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
            "token_type_ids": torch.zeros(shape, dtype=torch.long),
        }


def tokenizer_inputs() -> dict[str, torch.Tensor]:
    tensor = torch.ones((1, 4), dtype=torch.long)
    return {
        "input_ids": tensor,
        "attention_mask": tensor,
        "token_type_ids": tensor,
    }


def test_filter_inputs_drops_token_type_ids_when_forward_does_not_accept_it() -> None:
    filtered, dropped = filter_inputs_for_model(InputIdsAndMaskModel(), tokenizer_inputs())

    assert list(filtered) == ["input_ids", "attention_mask"]
    assert dropped == ["token_type_ids"]


def test_filter_inputs_keeps_token_type_ids_when_forward_accepts_it() -> None:
    filtered, dropped = filter_inputs_for_model(BertInputModel(), tokenizer_inputs())

    assert list(filtered) == ["input_ids", "attention_mask", "token_type_ids"]
    assert dropped == []


def test_text_wrapper_calls_model_with_keyword_arguments_and_returns_logits() -> None:
    model = KeywordCaptureModel()
    wrapper = TextModelOnnxWrapper(model, ["input_ids", "attention_mask"])
    input_ids = torch.ones((1, 2), dtype=torch.long)
    attention_mask = torch.ones((1, 2), dtype=torch.long)

    output = wrapper(input_ids, attention_mask)

    assert set(model.received) == {"input_ids", "attention_mask"}
    assert torch.equal(output, input_ids.float())


def test_text_wrapper_returns_first_tuple_output() -> None:
    wrapper = TextModelOnnxWrapper(KeywordCaptureModel(tuple_output=True), ["input_ids"])
    input_ids = torch.ones((1, 2), dtype=torch.long)

    assert torch.equal(wrapper(input_ids), input_ids.float())


def test_static_text_export_metadata_records_dropped_inputs(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    def fake_export(wrapper, args, output_path, **kwargs):
        observed["input_names"] = kwargs["input_names"]
        wrapper(*args)

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    metadata = export_static_text_model(
        model=InputIdsAndMaskModel(),
        tokenizer=DummyTokenizer(),
        output_path=tmp_path / "dummy.onnx",
        batch_size=1,
        seq_len=4,
        opset=17,
        device=torch.device("cpu"),
    )

    assert metadata["dropped_inputs"] == ["token_type_ids"]
    assert metadata["input_names"] == ["input_ids", "attention_mask"]
    assert observed["input_names"] == ["input_ids", "attention_mask"]
