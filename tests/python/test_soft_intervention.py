import pytest

from causal_graph_algorithms.directed_bidirected import DAG
from causal_graph_algorithms.errors import GraphValidationError
from causal_graph_algorithms.soft_intervention import (
    controlled_soft_intervention_mag,
    controlled_soft_intervention_markov_equivalent,
    psi_markov_equivalent,
    psi_soft_intervention_mag,
)


def test_kocaoglu_figure_2():
    left = DAG.from_text("X --> W --> Y; Z --> Y; X <-- Lxz --> Z; Z <-- Lzy --> Y")
    right = DAG.from_text("Z --> X --> Y; Z --> Y; X --> W; X <-- Lxy --> Y; W <-- Lwy --> Y")
    observed = ("X", "W", "Y", "Z")
    targets = ((), "X")
    assert controlled_soft_intervention_markov_equivalent(
        left, right, observed=observed, targets=targets,
    )
    for graph in (left, right):
        mag = controlled_soft_intervention_mag(graph, observed=observed, targets=targets)
        (f,) = set(mag.topology.nodes) - set(observed)
        assert "X" in mag.topology.directed_children(f)
        assert "Y" in mag.topology.directed_children(f)


def test_jaber_figure_1():
    d1 = DAG.from_text("Z --> X --> Y; Z --> Y; X --> W; Y --> W; X <-- Lxw --> W")
    d2 = DAG.from_text(
        "X --> Y; Z --> Y; X <-- Lxz --> Z; Z <-- Lzw --> W; "
        "X <-- Lxw --> W; W <-- Lwy --> Y"
    )
    d3 = DAG.from_text(
        "X --> Y; Z --> W; X <-- Lxz --> Z; Z <-- Lzy --> Y; "
        "X <-- Lxw --> W; W <-- Lwy --> Y"
    )
    observed = ("X", "Y", "Z", "W")
    i1 = ({}, {"X": 1})
    i2 = ({"X": 1}, {"W": 1})
    assert psi_markov_equivalent(d1, i1, d2, i2, observed=observed)
    assert not psi_markov_equivalent(d1, i1, d3, i1, observed=observed)
    assert not psi_markov_equivalent(d2, i2, d3, i1, observed=observed)
    mag = psi_soft_intervention_mag(d1, observed=observed, regimes=i1)
    (f,) = set(mag.topology.nodes) - set(observed)
    assert "W" in mag.topology.directed_children(f)


def test_same_targets_can_encode_different_mechanisms():
    graph = DAG.from_text("X --> Y")
    mag = psi_soft_intervention_mag(graph, observed=("X", "Y"), regimes=[{"X": 1}, {"X": 2}])
    (f,) = set(mag.topology.nodes) - {"X", "Y"}
    assert "X" in mag.topology.directed_children(f)
    assert not psi_markov_equivalent(
        graph, ({"X": 1}, {"X": 2}), graph, ({"X": 2}, {"X": 2}), observed=("X", "Y"),
    )


def test_comparison_nodes_are_deduplicated_only_for_controlled_experiments():
    graph = DAG.from_text("X --> Y")
    controlled = controlled_soft_intervention_mag(graph, observed=("X", "Y"), targets=[(), "X", "X"])
    (f,) = set(controlled.topology.nodes) - {"X", "Y"}
    assert "X" in controlled.topology.directed_children(f)
    psi = psi_soft_intervention_mag(graph, observed=("X", "Y"), regimes=[{}, {"X": 1}, {"X": 1}])
    nodes = set(psi.topology.nodes) - {"X", "Y"}
    assert len(nodes) == 3
    assert sum("X" in psi.topology.directed_children(f) for f in nodes) == 2


def test_labeled_regimes_align_by_position_even_with_auxiliary_label_collisions():
    left = DAG.from_text("X --> Y; __F0 <-- X")
    right = DAG.from_text("X --> Y")
    observed = ("X", "Y")
    i1 = [{}, {"X": "new"}, {"Y": "new"}]
    assert psi_markov_equivalent(left, i1, right, i1, observed=observed)
    assert not psi_markov_equivalent(left, i1, right, [i1[0], i1[2], i1[1]], observed=observed)
    with pytest.raises(GraphValidationError):
        psi_soft_intervention_mag(left, observed=observed, regimes=[{"__F0": 1}])
