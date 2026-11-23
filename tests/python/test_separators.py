"""Constrained separation against literal active-path and subset oracles."""
from itertools import combinations, product

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph, AncestralGraph, DAG,
    DirectedBidirectedGraph, EndpointMarkedGraph,
)
from causal_graph_algorithms.separators import (
    SeparatorDomainError, SeparatorNotFound, SeparatorSet,
    find_m_separator, find_minimal_m_separator, m_separator_is_minimal,
)
from tests.python.oracles import m_separated_by_simple_edge_paths


def subsets(values):
    return (frozenset(c) for k in range(len(values) + 1)
            for c in combinations(values, k))


def atoms(graph):
    raw = graph.topology
    if isinstance(raw, DirectedBidirectedGraph):
        return tuple((a, False, b, True) for a, b in raw.directed) + tuple(
            (a, True, b, True) for a, b in raw.bidirected)
    return tuple((a, x.value == "arrow", b, y.value == "arrow")
                 for a, x, b, y in raw.edges)


def oracle(graph, left, right, z):
    return all(m_separated_by_simple_edge_paths(graph.topology.nodes, atoms(graph),
                                          (x,), (y,), z)
               for x in left for y in right)


def test_empty_separator_and_impossibility_are_distinct():
    graph = DAG.from_text("X --> C <-- Y")
    result = find_m_separator(graph, "X", "Y", allowed=())
    assert isinstance(result, SeparatorSet) and result.vertices == ()
    assert isinstance(find_m_separator(graph, "X", "Y", required="C"),
                      SeparatorNotFound)
    assert not m_separator_is_minimal(graph, "X", "Y", "C")


def test_required_minimality_and_unavailable_ancestors():
    graph = DAG.from_text("X --> A --> B --> Y")
    result = find_minimal_m_separator(graph, "X", "Y", required=("A", "B"))
    assert isinstance(result, SeparatorSet) and result.vertices == ("A", "B")
    assert m_separator_is_minimal(graph, "X", "Y", ("A", "B"),
                                  required=("A", "B"))
    assert not m_separator_is_minimal(graph, "X", "Y", ("A", "B"))
    assert isinstance(find_m_separator(graph, "X", "Y", allowed=()),
                      SeparatorNotFound)
    graph = DAG.from_text("X <-- U --> Y")
    assert isinstance(find_m_separator(graph, "X", "Y", allowed=()),
                      SeparatorNotFound)


def test_domains_constraints_and_multivertex_queries():
    graph = DAG.from_text("A --> M --> Y; B --> M --> Z")
    result = find_minimal_m_separator(graph, ("A", "B"), ("Y", "Z"))
    assert result.vertices == ("M",)
    for kwargs in ({"required": "Y"}, {"allowed": "Y"},
                   {"required": "M", "allowed": ()}, {"allowed": "missing"}):
        with pytest.raises(SeparatorDomainError):
            find_m_separator(graph, "A", "Y", **kwargs)
    with pytest.raises(SeparatorDomainError):
        find_m_separator(EndpointMarkedGraph.from_text("X --- Y"), "X", "Y")
    with pytest.raises(SeparatorDomainError):
        find_m_separator(graph, (), "Y")
    ag = AncestralGraph.from_text("X --- S --- Y; S --> Z")
    assert find_minimal_m_separator(ag, "X", "Y").vertices == ("S",)


@pytest.mark.parametrize("bow", [False, True])
def test_all_three_vertex_acyclic_mixed_queries_against_subsets(bow):
    nodes = ("X", "M", "Y")
    pairs = tuple(combinations(nodes, 2))
    # All ordered DAG parts; all bidirected subsets, including bows.
    for states in product(range(3), repeat=3):
        directed = tuple((a, b) if s == 1 else (b, a)
                         for (a, b), s in zip(pairs, states) if s)
        for flags in product((False, True), repeat=3):
            bidirected = tuple(p for p, f in zip(pairs, flags) if f)
            if not bow and any((a, b) in directed or (b, a) in directed
                               for a, b in bidirected):
                continue
            try:
                graph = AcyclicDirectedBidirectedGraph.from_graph(
                    DirectedBidirectedGraph(nodes=nodes, directed=directed,
                                            bidirected=bidirected))
            except ValueError:
                continue
            for x, y in (("X", "Y"), ("Y", "M"), ("M", "X")):
                candidates = tuple(n for n in nodes if n not in (x, y))
                for allowed in subsets(candidates):
                    for required in subsets(tuple(allowed)):
                        valid = tuple(z for z in subsets(tuple(allowed))
                                      if required <= z and oracle(graph, (x,), (y,), z))
                        minimal = tuple(z for z in valid
                                        if not any(w < z for w in valid))
                        result = find_minimal_m_separator(
                            graph, x, y, required=required, allowed=allowed)
                        assert isinstance(result, SeparatorNotFound) == (not valid)
                        if valid:
                            assert frozenset(result.vertices) in minimal
                        for z in subsets(candidates):
                            expected = required <= z and oracle(graph, (x,), (y,), z)
                            expected = expected and not any(
                                required <= w and w < z and oracle(graph, (x,), (y,), w)
                                for w in subsets(tuple(z)))
                            assert m_separator_is_minimal(graph, x, y, z,
                                                           required=required) == expected
