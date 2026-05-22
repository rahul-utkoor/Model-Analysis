from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model_analysis.linear_pruning import get_module_by_name, make_keep_indices, prune_linear_layer, replace_module_by_name


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 6)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()


def test_out_features_pruning_changes_rows_and_bias():
    layer = torch.nn.Linear(4, 6)
    new_layer, metadata = prune_linear_layer(layer, "out_features", [0, 2])

    assert list(new_layer.weight.shape) == [4, 4]
    assert list(new_layer.bias.shape) == [4]
    assert metadata["keep_indices"] == [1, 3, 4, 5]
    assert new_layer.weight.dtype == layer.weight.dtype
    assert new_layer.weight.device == layer.weight.device


def test_in_features_pruning_changes_columns_and_keeps_bias():
    layer = torch.nn.Linear(4, 6)
    old_bias = layer.bias.detach().clone()
    new_layer, metadata = prune_linear_layer(layer, "in_features", [1])

    assert list(new_layer.weight.shape) == [6, 3]
    assert list(new_layer.bias.shape) == [6]
    assert torch.equal(new_layer.bias, old_bias)
    assert metadata["keep_indices"] == [0, 2, 3]


def test_invalid_indices_rejected():
    with pytest.raises(ValueError):
        make_keep_indices(4, [-1])
    with pytest.raises(ValueError):
        make_keep_indices(4, [4])


def test_pruning_all_features_rejected():
    with pytest.raises(ValueError):
        make_keep_indices(2, [0, 1])


def test_nested_module_replacement():
    model = Tiny()
    replacement = torch.nn.Linear(4, 3)

    replace_module_by_name(model, "block.fc1", replacement)

    assert get_module_by_name(model, "block.fc1") is replacement
