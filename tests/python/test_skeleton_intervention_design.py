from itertools import combinations, permutations, product

import pytest

from causal_graph_algorithms import (
    CPDAG,
    DAG,
    PDAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    unseparated_skeleton_edges,
)


def test_kocaoglu_figure_1_separating_designs_and_missing_regime():
    graph = PDAG(EndpointMarkedGraph.from_text(
        "U1 --- U2; U1 --- U3; U1 --- U4; U1 --- U5; "
        "U2 --- U3; U2 --- U5; U3 --- U4"
    ))
    graph_design = (("U1", "U3"), ("U2", "U3"))
    color_design = (("U1",), ("U2", "U4"))
    assert unseparated_skeleton_edges(graph, targets=graph_design) == ()
    assert unseparated_skeleton_edges(graph, targets=color_design) == ()
    assert unseparated_skeleton_edges(graph, targets=graph_design[:1]) == (
        ("U1", "U3"), ("U2", "U5"),
    )


def test_only_adjacencies_are_used_and_edges_follow_graph_order():
    graph = DAG.from_text("C; B; A; A --> B <-- C")
    cpdag = CPDAG.from_dag(graph)
    pdag = PDAG(EndpointMarkedGraph.from_text("C; B; A; A --- B --- C"))
    for view in (graph, cpdag, pdag):
        assert unseparated_skeleton_edges(view, targets=("A",)) == (("C", "B"),)
        assert unseparated_skeleton_edges(view, targets=()) == (("C", "B"), ("B", "A"))
        assert unseparated_skeleton_edges(view, targets=((), ("A", "B", "C"))) == (
            ("C", "B"), ("B", "A"),
        )
        assert unseparated_skeleton_edges(view, targets=iter(("B", "B"))) == ()


@pytest.mark.parametrize("targets", ["A", 42, (("missing",),)])
def test_invalid_design_is_rejected_even_when_skeleton_has_no_edges(targets):
    graph = DAG.from_text("A")
    with pytest.raises(GraphValidationError):
        unseparated_skeleton_edges(graph, targets=targets)


def test_no_targets_are_needed_for_an_edgeless_skeleton():
    graph = DAG.from_text("A; B")
    assert unseparated_skeleton_edges(graph, targets=()) == ()


def _subsets(items):
    return tuple(
        choice for size in range(len(items) + 1) for choice in combinations(items, size)
    )


def _acyclic_orientations(nodes, skeleton):
    orientations = set()
    for order in permutations(nodes):
        positions = {node: index for index, node in enumerate(order)}
        orientations.add(frozenset(
            (left, right) if positions[left] < positions[right] else (right, left)
            for left, right in skeleton
        ))
    return orientations


def _cut_signature(nodes, arrows, target):
    directed = frozenset((parent, child) for parent, child in arrows if child not in target)
    skeleton = frozenset(frozenset(edge) for edge in directed)
    colliders = frozenset(
        (frozenset((left, right)), center)
        for center in nodes
        for left, right in combinations(nodes, 2)
        if (left, center) in directed and (right, center) in directed
        and frozenset((left, right)) not in skeleton
    )
    return skeleton, colliders


def test_design_criterion_against_all_three_vertex_acyclic_orientations():
    nodes = ("A", "B", "C")
    possible_targets = _subsets(nodes)
    designs = ((), *((target,) for target in possible_targets),
               *product(possible_targets, repeat=2))
    for skeleton in _subsets(tuple(combinations(nodes, 2))):
        graph = DAG(DirectedBidirectedGraph(nodes=nodes, directed=skeleton))
        orientations = _acyclic_orientations(nodes, skeleton)
        for design in designs:
            signatures = {
                arrows: tuple(_cut_signature(nodes, arrows, target) for target in ((), *design))
                for arrows in orientations
            }
            result = unseparated_skeleton_edges(graph, targets=design)
            assert (result == ()) == (len(set(signatures.values())) == len(orientations))
            ambiguous_reversals = {
                frozenset(edge)
                for first, second in combinations(orientations, 2)
                if signatures[first] == signatures[second] and len(first ^ second) == 2
                for edge in first - second
            }
            assert {frozenset(edge) for edge in result} == ambiguous_reversals
