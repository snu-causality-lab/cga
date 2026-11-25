from __future__ import annotations

import itertools

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
    GraphValidationError,
)


def _literal_bidirected_component(
    nodes: tuple[str, ...],
    bidirected: frozenset[tuple[str, str]],
    start: str,
) -> frozenset[str]:
    reached = {start}
    pending = [start]
    while pending:
        current = pending.pop()
        for left, right in bidirected:
            if left == current:
                neighbor = right
            elif right == current:
                neighbor = left
            else:
                continue
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return frozenset(reached)


def _literal_districts(
    nodes: tuple[str, ...],
    bidirected: frozenset[tuple[str, str]],
) -> tuple[frozenset[str], ...]:
    unseen = set(nodes)
    components: list[frozenset[str]] = []
    for node in nodes:
        if node not in unseen:
            continue
        component = _literal_bidirected_component(nodes, bidirected, node)
        components.append(component)
        unseen.difference_update(component)
    return tuple(components)


def test_bidirected_neighbors_ignore_directed_edges_and_preserve_bows() -> None:
    graph = DirectedBidirectedGraph(
        nodes=("A", "B", "C", "D"),
        directed=(("A", "B"), ("B", "C"), ("D", "C")),
        bidirected=(("B", "A"), ("B", "C")),
    )

    assert graph.bidirected_neighbors("A") == frozenset({"B"})
    assert graph.bidirected_neighbors("B") == frozenset({"A", "C"})
    assert graph.bidirected_neighbors("C") == frozenset({"B"})
    assert graph.bidirected_neighbors("D") == frozenset()


def test_district_is_maximal_bidirected_connectivity_with_singletons() -> None:
    graph = DirectedBidirectedGraph(
        nodes=("D", "A", "C", "B", "E"),
        directed=(("D", "A"), ("A", "C"), ("C", "B")),
        bidirected=(("A", "B"), ("B", "C")),
    )

    expected_abc = frozenset({"A", "B", "C"})
    assert graph.district("A") == expected_abc
    assert graph.district("B") == expected_abc
    assert graph.district("C") == expected_abc
    assert graph.district("D") == frozenset({"D"})
    assert graph.district("E") == frozenset({"E"})
    assert graph.districts() == (
        frozenset({"D"}),
        expected_abc,
        frozenset({"E"}),
    )


def test_raw_directed_cycles_do_not_change_bidirected_districts() -> None:
    graph = DirectedBidirectedGraph(
        nodes=("A", "B", "C"),
        directed=(("A", "B"), ("B", "A"), ("B", "C")),
        bidirected=(("B", "C"),),
    )

    assert graph.districts() == (
        frozenset({"A"}),
        frozenset({"B", "C"}),
    )


def test_empty_graph_has_no_districts() -> None:
    graph = DirectedBidirectedGraph(nodes=())
    assert graph.districts() == ()


@pytest.mark.parametrize("unknown", ["missing", "", 1, None])
def test_vertex_district_relations_reject_unknown_vertices(
    unknown: object,
) -> None:
    graph = DirectedBidirectedGraph(nodes=("A", "B"), bidirected=(("A", "B"),))
    with pytest.raises(GraphValidationError, match="unknown vertex"):
        graph.bidirected_neighbors(unknown)  # type: ignore[arg-type]
    with pytest.raises(GraphValidationError, match="unknown vertex"):
        graph.district(unknown)  # type: ignore[arg-type]


def test_proof_views_forward_district_relations() -> None:
    topology = DirectedBidirectedGraph(
        nodes=("A", "B", "C"),
        directed=(("A", "B"), ("B", "C")),
        bidirected=(("A", "B"),),
    )
    acyclic = AcyclicDirectedBidirectedGraph.from_graph(topology)

    assert acyclic.bidirected_neighbors("B") == frozenset({"A"})
    assert acyclic.district("A") == frozenset({"A", "B"})
    assert acyclic.districts() == (
        frozenset({"A", "B"}),
        frozenset({"C"}),
    )

    dag = DAG.from_graph(
        DirectedBidirectedGraph(
            nodes=("C", "A", "B"),
            directed=(("A", "B"),),
        )
    )
    assert dag.bidirected_neighbors("A") == frozenset()
    assert dag.district("A") == frozenset({"A"})
    assert dag.districts() == (
        frozenset({"C"}),
        frozenset({"A"}),
        frozenset({"B"}),
    )


def test_districts_match_literal_connectivity_for_every_three_node_mask() -> None:
    nodes = ("A", "B", "C")
    pairs = tuple(itertools.combinations(nodes, 2))
    directed = (("A", "B"), ("B", "C"))

    for mask in range(1 << len(pairs)):
        bidirected = tuple(
            pair for index, pair in enumerate(pairs) if mask & (1 << index)
        )
        graph = DirectedBidirectedGraph(
            nodes=nodes,
            directed=directed,
            bidirected=bidirected,
        )
        expected = _literal_districts(nodes, graph.bidirected)
        assert graph.districts() == expected
        for node in nodes:
            assert graph.district(node) == next(
                component for component in expected if node in component
            )
