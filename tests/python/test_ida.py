from collections import Counter
from itertools import combinations, product

import pytest

from causal_graph_algorithms import CPDAG, DAG, DirectedBidirectedGraph, EndpointMarkedGraph, Mark, PDAG
from causal_graph_algorithms.errors import GraphValidationError
from causal_graph_algorithms.ida import ida_parent_sets, joint_ida_parent_counts, joint_ida_parent_sets
from causal_graph_algorithms.pdag import PDAGDomainError


def _cpdag(nodes, *, lines=(), arrows=()):
    return CPDAG.from_graph(EndpointMarkedGraph(
        nodes=nodes,
        edges=[(u, Mark.TAIL, v, Mark.TAIL) for u, v in lines]
        + [(u, Mark.TAIL, v, Mark.ARROW) for u, v in arrows],
    ))


def _full_parent_counts(graph, treatments):
    """Orient all lines and filter DAGs by a directly computed collider set."""
    raw = graph.topology
    skeleton = {frozenset((u, v)) for u, _, v, _ in raw.edges}
    lines = [(u, v) for u, left, v, right in raw.edges if left == right == Mark.TAIL]
    fixed = {(u, v) if right == Mark.ARROW else (v, u)
             for u, left, v, right in raw.edges if left != right}

    def colliders(arrows):
        return {(middle, frozenset((u, v))) for middle in raw.nodes
                for u, v in combinations((p for p, q in arrows if q == middle), 2)
                if frozenset((u, v)) not in skeleton}

    expected = colliders(fixed)
    counts = Counter()
    for orientations in product((False, True), repeat=len(lines)):
        arrows = fixed | {(v, u) if backward else (u, v)
                          for (u, v), backward in zip(lines, orientations, strict=True)}
        if colliders(arrows) == expected:
            try:
                dag = DAG.from_graph(DirectedBidirectedGraph(nodes=raw.nodes, directed=arrows))
            except GraphValidationError:
                continue
            counts[tuple(dag.topology.directed_parents(v) for v in treatments)] += 1
    return counts


def test_maathuis_figure2_local_support_and_multiplicity():
    graph = _cpdag(
        ("X1", "X2", "X3", "X4", "Y"),
        lines=(("X1", "X2"), ("X1", "X4"), ("X2", "X3")),
        arrows=(("X3", "Y"), ("X4", "Y")),
    )
    expected = {(frozenset(),): 1, (frozenset({"X2"}),): 2, (frozenset({"X4"}),): 1}
    assert ida_parent_sets(graph, "X1") == frozenset(parents for (parents,) in expected)
    assert joint_ida_parent_counts(graph, "X1") == expected == _full_parent_counts(graph, ("X1",))
    assert ida_parent_sets(graph, "Y") == {frozenset({"X3", "X4"})}


def test_nandy_example3_local_choices_are_not_all_jointly_valid():
    graph = _cpdag(("1", "2", "3"), lines=(("1", "3"), ("2", "3")))
    empty, three = frozenset(), frozenset({"3"})
    assert ida_parent_sets(graph, "1") == ida_parent_sets(graph, "2") == {empty, three}
    assert joint_ida_parent_sets(graph, ("1", "2")) == {(empty, three), (three, empty), (three, three)}


def test_nandy_example4_component_product_retains_relative_counts():
    graph = _cpdag(
        tuple("12345678"),
        lines=(("1", "4"), ("3", "4"), ("2", "6"), ("5", "8")),
        arrows=(("8", "7"), ("3", "7"), ("7", "2"), ("7", "6")),
    )
    empty, four = frozenset(), frozenset({"4"})
    treatments = ("1", "2", "3")
    counts = joint_ida_parent_counts(graph, treatments)
    assert counts == {(left, middle, right): 1
                      for left, right in ((empty, four), (four, empty), (four, four))
                      for middle in (frozenset({"7"}), frozenset({"6", "7"}))}
    assert _full_parent_counts(graph, treatments) == {key: 2 * count for key, count in counts.items()}
    assert joint_ida_parent_counts(graph, ("3", "1", "2")) == {
        (right, left, middle): count for (left, middle, right), count in counts.items()
    }


def test_triangle_cycles_are_excluded_and_bad_inputs_are_rejected():
    graph = _cpdag(("a", "b", "c"), lines=(("a", "b"), ("a", "c"), ("b", "c")))
    assert sum(joint_ida_parent_counts(graph, ("c", "a")).values()) == 6
    assert joint_ida_parent_counts(graph, ("c", "a")) == _full_parent_counts(graph, ("c", "a"))
    with pytest.raises(PDAGDomainError):
        ida_parent_sets(PDAG.from_graph(graph.topology), "a")
    for treatments in ((), ("a", "a"), ("absent",)):
        with pytest.raises(PDAGDomainError):
            joint_ida_parent_counts(graph, treatments)


def test_all_three_vertex_classes_match_full_dag_enumeration():
    nodes = ("b", "a", "c")
    pairs = tuple(combinations(nodes, 2))
    classes = set()
    for choices in product((0, 1, 2), repeat=len(pairs)):
        arrows = [(u, v) if direction == 1 else (v, u)
                  for (u, v), direction in zip(pairs, choices, strict=True) if direction]
        try:
            dag = DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=arrows))
        except GraphValidationError:
            continue
        classes.add(CPDAG.from_dag(dag))
    for graph in classes:
        for count in range(1, len(nodes) + 1):
            for targets in combinations(nodes, count):
                full = _full_parent_counts(graph, targets)
                local = joint_ida_parent_counts(graph, targets)
                assert local.keys() == full.keys()
                assert all(local[key] * sum(full.values()) == full[key] * sum(local.values())
                           for key in local)
                if count == 1:
                    assert ida_parent_sets(graph, targets[0]) == {key[0] for key in full}
