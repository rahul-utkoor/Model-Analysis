from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from model_analysis.bert_mlp_pruning import (
    detect_bert_mlp_block_targets,
    execute_bert_mlp_pruning,
    get_bert_mlp_block_target,
    make_bert_mlp_prune_spec,
)


class TinyIntermediate(torch.nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8):
        super().__init__()
        self.dense = torch.nn.Linear(hidden, intermediate)


class TinyOutput(torch.nn.Module):
    def __init__(self, intermediate: int = 8, hidden: int = 4):
        super().__init__()
        self.dense = torch.nn.Linear(intermediate, hidden)


class TinyLayer(torch.nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8):
        super().__init__()
        self.intermediate = TinyIntermediate(hidden, intermediate)
        self.output = TinyOutput(intermediate, hidden)

    def forward(self, inputs):
        return self.output.dense(torch.relu(self.intermediate.dense(inputs)))


class TinyEncoder(torch.nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8):
        super().__init__()
        self.layer = torch.nn.ModuleList([TinyLayer(hidden, intermediate)])


class TinyBert(torch.nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8):
        super().__init__()
        self.encoder = TinyEncoder(hidden, intermediate)


class TinyModel(torch.nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8):
        super().__init__()
        self.bert = TinyBert(hidden, intermediate)

    def forward(self, inputs):
        return self.bert.encoder.layer[0](inputs)


class MalformedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(7, 4)


class MalformedLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate = TinyIntermediate()
        self.output = MalformedOutput()


class MalformedEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.ModuleList([MalformedLayer()])


class MalformedBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MalformedEncoder()


class MalformedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = MalformedBert()


def test_detect_bert_mlp_block_targets_finds_layer_zero():
    targets = detect_bert_mlp_block_targets(TinyModel(), "tiny")

    assert len(targets) == 1
    assert targets[0].layer_index == 0
    assert targets[0].hidden_size == 4
    assert targets[0].intermediate_size == 8
    assert targets[0].confidence == "high"


def test_make_spec_with_indices_succeeds():
    target = get_bert_mlp_block_target(TinyModel(), "tiny", 0)

    spec = make_bert_mlp_prune_spec(target, indices=[0, 1], strategy="first_n")

    assert spec.prune_indices == [0, 1]
    assert spec.intermediate_size_after == 6


def test_execute_dry_run_does_not_change_shapes(tmp_path: Path):
    model = TinyModel()
    target = get_bert_mlp_block_target(model, "tiny", 0)
    spec = make_bert_mlp_prune_spec(target, indices=[0, 1])

    report = execute_bert_mlp_pruning(model, "tiny", tmp_path / "source", tmp_path / "out", spec, dry_run=True)

    assert report.status == "dry_run"
    assert list(model.bert.encoder.layer[0].intermediate.dense.weight.shape) == [8, 4]
    assert list(model.bert.encoder.layer[0].output.dense.weight.shape) == [4, 8]


def test_actual_execution_repairs_pair_and_forward_still_works(tmp_path: Path):
    model = TinyModel()
    target = get_bert_mlp_block_target(model, "tiny", 0)
    spec = make_bert_mlp_prune_spec(target, indices=[0, 1])

    report = execute_bert_mlp_pruning(model, "tiny", tmp_path / "source", tmp_path / "out", spec, smoke_test_after=True)

    assert report.status == "success"
    assert list(model.bert.encoder.layer[0].intermediate.dense.weight.shape) == [6, 4]
    assert list(model.bert.encoder.layer[0].output.dense.weight.shape) == [4, 6]
    assert list(model(torch.zeros(1, 4)).shape) == [1, 4]
    assert report.after_forward_smoke["status"] == "passed"


def test_pruning_all_intermediate_features_is_rejected():
    target = get_bert_mlp_block_target(TinyModel(), "tiny", 0)

    with pytest.raises(ValueError):
        make_bert_mlp_prune_spec(target, indices=list(range(8)))


def test_invalid_layer_index_is_rejected():
    with pytest.raises(ValueError):
        get_bert_mlp_block_target(TinyModel(), "tiny", 5)


def test_malformed_pair_is_detected_low_confidence_and_rejected():
    targets = detect_bert_mlp_block_targets(MalformedModel(), "bad")

    assert targets[0].confidence == "low"
    with pytest.raises(ValueError):
        get_bert_mlp_block_target(MalformedModel(), "bad", 0)
