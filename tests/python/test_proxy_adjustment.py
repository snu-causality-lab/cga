from fractions import Fraction
from itertools import combinations, product

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DirectedBidirectedGraph,
    GraphValidationError,
    find_minimum_cost_proxy_adjustment,
)


def _admg(nodes, directed, bidirected=()):
    return AcyclicDirectedBidirectedGraph(
        DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=bidirected)
    )


def test_elahi_figure_2a_proxy_design():
    graph = _admg(
        ("v1", "v2", "v3", "w", "y"),
        (("v1", "v3"), ("v3", "w"), ("w", "y"), ("v2", "y")),
        (("v1", "v3"), ("v1", "w"), ("v1", "v2")),
    )
    design = find_minimum_cost_proxy_adjustment(
        graph, ("v2", "v3"), "y", allowed="v1", costs={"v1": Fraction(3, 2)}
    )
    assert design is not None
    assert design.intervention == ("v1",)
    assert design.total_cost == Fraction(3, 2)
    assert set(design.minimal_treatments) == {"v2", "v3"}
    assert find_minimum_cost_proxy_adjustment(graph, ("v2", "v3"), "y", allowed=()) is None


def test_protects_outcome_in_confounding_bow():
    graph = _admg(("X", "Y"), (("X", "Y"),), (("X", "Y"),))
    design = find_minimum_cost_proxy_adjustment(graph, "X", "Y", costs={"X": 2, "Y": 1})
    assert design is not None
    assert design.intervention == ("X",)
    assert design.adjustment == ()
    assert design.total_cost == 2
    assert find_minimum_cost_proxy_adjustment(graph, "X", "Y", allowed="Y") is None


def test_removes_irrelevant_treatment_and_can_adjust_without_experiment():
    graph = _admg(("U", "X", "W", "Y", "D"),
                  (("U", "X"), ("X", "Y"), ("Y", "D")), (("U", "Y"),))
    design = find_minimum_cost_proxy_adjustment(graph, ("X", "W"), "Y")
    assert design is not None
    assert design.intervention == ()
    assert design.adjustment == ("U",)
    assert design.minimal_treatments == ("X",)
    empty = find_minimum_cost_proxy_adjustment(graph, "W", "Y")
    assert empty is not None
    assert empty.minimal_treatments == empty.intervention == empty.adjustment == ()


def test_proxy_cost_must_be_nonnegative_and_rational():
    graph = _admg(("X", "Y"), (("X", "Y"),))
    with pytest.raises(GraphValidationError, match="nonnegative"):
        find_minimum_cost_proxy_adjustment(graph, "X", "Y", costs={"X": -1})


def _subsets(vertices):
    return (
        frozenset(subset)
        for size in range(len(vertices) + 1)
        for subset in combinations(vertices, size)
    )


def _moral_separates(topology, treatment, district_ancestors, interventions, adjustment):
    """Definition-level check by constructing the modified moral graph directly."""
    parents = {node: set() for node in topology.nodes}
    for source, target in topology.directed:
        if source != treatment and target not in interventions:
            parents[target].add(source)
    for i, edge in enumerate(topology.bidirected):
        latent = ("latent", i)
        parents[latent] = set()
        for node in edge:
            if node not in interventions:
                parents[node].add(latent)
    neighbors = {node: set() for node in parents if node not in adjustment}
    for node, ancestor_set in parents.items():
        for left, right in combinations(ancestor_set | {node}, 2):
            if left in neighbors and right in neighbors:
                neighbors[left].add(right)
                neighbors[right].add(left)
    seen = {treatment}
    for _ in range(len(neighbors)):
        seen |= set().union(*(neighbors[node] for node in seen))
    return seen.isdisjoint(district_ancestors)


def test_proxy_adjustment_exact_cost_against_independent_cut_enumeration():
    vertices = ("U", "X", "V", "Y")
    extra_arrows = tuple(e for e in combinations(vertices, 2) if e != ("X", "Y"))
    bidirected = tuple(combinations(vertices, 2))
    costs = {"U": 1, "X": 4, "V": 0, "Y": 1}
    for arrows, confounding in product(_subsets(extra_arrows), _subsets(bidirected)):
        graph = _admg(vertices, arrows | {("X", "Y")}, confounding)
        topo = graph.topology
        ancestor_pool = topo.directed_ancestors_reflexive(("X", "Y"))
        subgraph = topo.__class__(
            nodes=(v for v in vertices if v in ancestor_pool),
            directed=(e for e in topo.directed if set(e) <= ancestor_pool),
            bidirected=(e for e in topo.bidirected if set(e) <= ancestor_pool),
        )
        district_ancestors = {"Y"}
        for _ in vertices:
            district_ancestors |= {
                a for a, b in subgraph.directed if a != "X" and b in district_ancestors
            }
        experimental = tuple(v for v in subgraph.nodes if v not in district_ancestors)
        covariates = tuple(v for v in experimental if v != "X")
        feasible = [
            sum(costs[v] for v in intervened)
            for intervened in _subsets(experimental)
            if any(_moral_separates(subgraph, "X", district_ancestors, intervened, adjusted)
                   for adjusted in _subsets(covariates))
        ]
        design = find_minimum_cost_proxy_adjustment(graph, "X", "Y", costs=costs)
        if feasible:
            assert design is not None
            assert design.total_cost == min(feasible)
            assert _moral_separates(
                subgraph, "X", district_ancestors, design.intervention, design.adjustment
            )
        else:
            assert design is None


def test_post_treatment_covariate_cannot_be_used_as_adjustment():
    graph = _admg(
        ("X1", "Y1", "Z", "X2", "Y2"),
        (("X1", "Y1"), ("Y1", "Z"), ("Z", "X2"), ("X2", "Y2")),
    )
    query = (("X1", "X2"), ("Y1", "Y2"))
    assert find_minimum_cost_proxy_adjustment(graph, *query, allowed=()) is None
    design = find_minimum_cost_proxy_adjustment(graph, *query, allowed="Z")
    assert design is not None
    assert design.intervention == ("Z",)
    assert design.adjustment == ()
