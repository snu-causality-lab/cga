"""Rendered-paper graph fixtures and population calculations, not fitted simulations."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    DAG, DAGAdjustmentNotApplicable, DAGOptimalAdjustmentSet, DirectedBidirectedGraph,
    dag_adjustment_set_is_valid, dag_forbidden_projection, dag_optimal_adjustment_set,
    prune_dag_adjustment_set,
)
from .dag_adjustment_oracles import avar, subsets, valid

FIXTURES = json.loads((Path(__file__).parents[1] / "fixtures/papers/henckel_witte_adjustment.json").read_text())


def make_graph(row):
    return DAG.from_graph(DirectedBidirectedGraph(nodes=row["nodes"], directed=row["edges"]))


@pytest.mark.parametrize("row", FIXTURES["henckel_figures"], ids=lambda r: "henckel_" + r["figure"])
def test_henckel_graph_examples(row):
    graph = make_graph(row)
    result = dag_optimal_adjustment_set(graph, interventions=row["x"], outcomes=row["y"])
    assert isinstance(result, DAGOptimalAdjustmentSet)
    assert set(result.vertices) == set(row["o_set"])
    nodes, edges = tuple(row["nodes"]), frozenset(map(tuple, row["edges"]))
    assert valid(nodes, edges, {row["x"]}, {row["y"]}, set(row["o_set"]))


@pytest.mark.parametrize("row", FIXTURES["witte_figure2"], ids=lambda r: "witte_2" + r["panel"])
def test_witte_forbidden_projection_panels(row):
    graph = make_graph(row)
    projected = dag_forbidden_projection(graph, interventions=row["x"], outcomes=row["y"]).topology
    assert set(projected.nodes) == set(row["retained"])
    assert set(projected.directed) == set(map(tuple, row["projected_directed"]))
    assert {frozenset(e) for e in projected.bidirected} == {frozenset(e) for e in row["projected_bidirected"]}
    result = dag_optimal_adjustment_set(graph, interventions=row["x"], outcomes=row["y"])
    if "reason" in row:
        assert isinstance(result, DAGAdjustmentNotApplicable)
        assert result.reason == row["reason"]
        for z in subsets(tuple(v for v in row["nodes"] if v not in row["x"] + row["y"])):
            assert not valid(tuple(row["nodes"]), frozenset(map(tuple, row["edges"])), set(row["x"]), set(row["y"]), z)
            assert not dag_adjustment_set_is_valid(graph, interventions=row["x"], outcomes=row["y"], adjustment=z)
    else:
        assert isinstance(result, DAGOptimalAdjustmentSet)
        assert set(result.vertices) == set(row["o_set"])


def test_witte_schizophrenia_graph_and_projection():
    row = FIXTURES["witte_figures1_and3"]
    graph = make_graph(row)
    result = dag_optimal_adjustment_set(graph, interventions="ALN", outcomes="DET")
    assert isinstance(result, DAGOptimalAdjustmentSet)
    assert set(result.vertices) == set(row["o_set"])
    projected = dag_forbidden_projection(graph, interventions="ALN", outcomes="DET").topology
    assert set(projected.nodes) == set(row["retained"])
    assert set(projected.directed) == set(map(tuple, row["projected_directed"]))
    assert not projected.bidirected
    for z in (("AFF", "SAN"), ("AIS", "CDR", "AFF"), ("AFF", "APA", "AIS", "CDR", "SAN")):
        assert dag_adjustment_set_is_valid(graph, interventions="ALN", outcomes="DET", adjustment=z)


def specified_covariance(nodes, edges, weights):
    """Independent exact expansion of the paper's linear SEM with unit noise variances."""
    expansions = {}
    while len(expansions) < len(nodes):
        for v in nodes:
            if v in expansions:
                continue
            parents = [u for u, w in edges if w == v]
            if all(u in expansions for u in parents):
                row = [Fraction(int(v == w)) for w in nodes]
                for u in parents:
                    row = [a + weights.get((u, v), Fraction(1)) * b for a, b in zip(row, expansions[u])]
                expansions[v] = row
    return [[sum((a * b for a, b in zip(expansions[u], expansions[v])), Fraction()) for v in nodes] for u in nodes]


def test_henckel_table2_population_variances_and_pruning_limits():
    row = FIXTURES["henckel_figures"][2]
    graph = make_graph(row)
    nodes = row["nodes"]
    sets = [("C",), ("B", "C"), ("A", "C"), (), ("A", "B"), ("A",)]
    cases = [
        ({("A", "B"): Fraction(1, 2), ("A", "X"): Fraction(1, 4), ("X", "Y"): Fraction(2)}, ["0.48", "0.49", "0.50", "0.97", "0.75", "1.00"]),
        ({("A", "X"): Fraction(7, 10), ("C", "Y"): Fraction(1, 2)}, ["0.40", "0.45", "0.50", "0.50", "0.56", "0.62"]),
    ]
    for weights, expected in cases:
        cov = specified_covariance(nodes, row["edges"], weights)
        actual = [avar(cov, nodes.index("X"), nodes.index("Y"), [nodes.index(v) for v in z]) for z in sets]
        assert [f"{float(v):.2f}" for v in actual] == expected
    # Without C, graphical pruning cannot select the parameter-dependent best subset.
    assert prune_dag_adjustment_set(graph, interventions="X", outcomes="Y", adjustment=("A", "B")) == ("A", "B")
    assert prune_dag_adjustment_set(graph, interventions="X", outcomes="Y", adjustment=nodes[:4]) == ("C",)


def test_henckel_figure2a_pruning_of_every_valid_set():
    graph = make_graph(FIXTURES["henckel_figures"][1])
    for extra in subsets(("A", "C", "D")):
        pruned = prune_dag_adjustment_set(graph, interventions="X", outcomes="Y", adjustment=extra | {"B"})
        assert set(pruned) == ({"B", "C"} if "C" in extra else {"B"})


def test_henckel_figure5_exact_unfaithful_variance_equality():
    row = FIXTURES["henckel_figures"][3]
    nodes = row["nodes"]
    cov = specified_covariance(nodes, row["edges"], {("O", "Y"): Fraction(-1)})
    def variance(z):
        return avar(cov, nodes.index("X"), nodes.index("Y"), [nodes.index(v) for v in z])
    assert variance(("P", "O")) == variance(("P",))
