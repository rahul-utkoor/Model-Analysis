from experimental.dfa_pruning_propagation.ir import Axis
from experimental.dfa_pruning_propagation.lattice import Fact, FactKind, join


AXIS = Axis("tensor", "channel_j", "intermediate_dim")


def fact(kind: FactKind) -> Fact:
    return Fact(AXIS, kind, kind.value.lower())


def test_unknown_join_dead_is_dead() -> None:
    assert join(fact(FactKind.UNKNOWN), fact(FactKind.DEAD)).kind == FactKind.DEAD


def test_dead_join_pruned_is_pruned() -> None:
    assert join(fact(FactKind.DEAD), fact(FactKind.PRUNED)).kind == FactKind.PRUNED


def test_protected_join_pruned_is_blocked() -> None:
    assert join(fact(FactKind.PROTECTED), fact(FactKind.PRUNED)).kind == FactKind.BLOCKED


def test_live_join_dead_is_blocked() -> None:
    assert join(fact(FactKind.LIVE), fact(FactKind.DEAD)).kind == FactKind.BLOCKED
