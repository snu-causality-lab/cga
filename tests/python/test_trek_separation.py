from itertools import combinations, permutations, product

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
    trek_separated,
)


def test_sullivant_talaska_draisma_figure_1():
    # Example 2.9, https://pure.tue.nl/ws/portalfiles/portal/3002904/Metis230934.pdf#page=7
    graph = DAG.from_text("V1 --> V4; V2 --> V4; V2 --> V5; V3 --> V4 --> V5")
    assert trek_separated(graph, ("V1", "V3"), ("V4", "V5"), right_blocked="V4")
    assert not trek_separated(graph, ("V1", "V3"), ("V4", "V5"), left_blocked="V4")
    bypass = DAG.from_text("V1 --> V4; V2 --> V4; V2 --> V5; V3 --> V4 --> V5; V1 --> V5")
    assert not trek_separated(bypass, ("V1", "V3"), ("V4", "V5"), right_blocked="V4")


def test_overlapping_endpoints_admit_a_zero_edge_trek():
    graph = DAG.from_text("U --> X")
    assert not trek_separated(graph, "X", "X", left_blocked="U", right_blocked="U")
    assert trek_separated(graph, "X", "X", left_blocked="X")
    assert trek_separated(graph, "X", "X", right_blocked="X")
    assert trek_separated(graph, (), "X")


def test_rejects_unknown_vertices_and_non_dag_views():
    graph = DAG.from_text("A --> B")
    with pytest.raises(ValueError):
        trek_separated(graph, (), "B", right_blocked="missing")
    with pytest.raises(ValueError):
        trek_separated(AcyclicDirectedBidirectedGraph.from_text("A <-> B"), "A", "B")


def test_matches_directed_path_pair_oracle():
    nodes = ("c", "a", "b")
    pairs = tuple(combinations(nodes, 2))
    subsets = tuple(
        frozenset(subset)
        for size in range(len(nodes) + 1)
        for subset in combinations(nodes, size)
    )
    for included in product((False, True), repeat=len(pairs)):
        edges = {edge for edge, keep in zip(pairs, included) if keep}
        graph = DAG(DirectedBidirectedGraph(nodes=nodes, directed=edges))
        paths = tuple(
            route
            for size in range(1, len(nodes) + 1)
            for route in permutations(nodes, size)
            if all(edge in edges for edge in zip(route, route[1:]))
        )
        treks = tuple((a, b) for a, b in product(paths, repeat=2) if a[0] == b[0])
        for left, right in product(nodes, repeat=2):
            for left_blocked, right_blocked in product(subsets, repeat=2):
                separated = not any(
                    a[-1] == left
                    and b[-1] == right
                    and left_blocked.isdisjoint(a)
                    and right_blocked.isdisjoint(b)
                    for a, b in treks
                )
                assert trek_separated(
                    graph, left, right, left_blocked=left_blocked, right_blocked=right_blocked
                ) == separated
