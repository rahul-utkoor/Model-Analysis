from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model_analysis.paired_linear_pruning import apply_paired_linear_repair, apply_repair_plan
from model_analysis.repair_plan import RepairPlan, RepairSpec


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 2)

    def forward(self, inputs):
        return self.fc2(torch.relu(self.fc1(inputs)))


def make_spec(indices: list[int] | None = None) -> RepairSpec:
    return RepairSpec(
        repair_id="r1",
        repair_type="mlp_pair",
        source_module="fc1",
        source_prune_dim="out_features",
        target_module="fc2",
        target_prune_dim="in_features",
        indices=indices or [0, 1],
        dependency_edge_type="mlp_hidden_coupling",
        confidence="medium",
        reason="test",
    )


def test_paired_linear_pruning_changes_both_layers_and_forward_works():
    model = TinyMLP()

    record = apply_paired_linear_repair(model, make_spec())

    assert record.status == "applied"
    assert list(model.fc1.weight.shape) == [6, 4]
    assert list(model.fc2.weight.shape) == [2, 6]
    output = model(torch.zeros(1, 4))
    assert list(output.shape) == [1, 2]


def test_invalid_target_indices_do_not_modify_either_layer():
    model = TinyMLP()

    record = apply_paired_linear_repair(model, make_spec([100]))

    assert record.status == "rejected"
    assert list(model.fc1.weight.shape) == [8, 4]
    assert list(model.fc2.weight.shape) == [2, 8]


def test_apply_repair_plan_dry_run_does_not_modify_shapes():
    model = TinyMLP()
    plan = RepairPlan("rp1", "tiny", "a1", "p1", repair_specs=[make_spec()], status="executable")

    records = apply_repair_plan(model, plan, dry_run=True)

    assert records[0].status == "skipped"
    assert list(model.fc1.weight.shape) == [8, 4]
    assert list(model.fc2.weight.shape) == [2, 8]
