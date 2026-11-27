from itertools import combinations, pairwise, permutations, product

import pytest

from causal_graph_algorithms import (
    EliminationBag,
    EliminationForest,
    EndpointMarkedGraph,
    GraphValidationError,
    Mark,
    chordal_elimination_forest,
)


def _undirected_graph(nodes, edges):
    return EndpointMarkedGraph(
        nodes=nodes,
        edges=((a, Mark.TAIL, b, Mark.TAIL) for a, b in edges),
    )


def test_chordless_four_cycle_is_rejected():
    graph = _undirected_graph(
        ("a", "b", "c", "d"), (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))
    )
    with pytest.raises(GraphValidationError, match="not chordal"):
        chordal_elimination_forest(graph)
    with pytest.raises(GraphValidationError, match="not a perfect elimination order"):
        chordal_elimination_forest(graph, order=("a", "b", "c", "d"))


def test_chorded_four_cycle_with_supplied_order():
    graph = _undirected_graph(
        ("a", "b", "c", "d"),
        (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")),
    )
    order = ("b", "d", "a", "c")
    assert chordal_elimination_forest(graph, order=iter(order)) == EliminationForest(
        order=order,
        bags=(
            EliminationBag("b", ("b", "a", "c"), "a", ("a", "c")),
            EliminationBag("d", ("d", "a", "c"), "a", ("a", "c")),
            EliminationBag("a", ("a", "c"), "c", ("c",)),
            EliminationBag("c", ("c",), None, ()),
        ),
    )


def test_maximum_cardinality_search_uses_declared_vertex_order_for_ties():
    nodes = ("z", "y", "x", "w")
    edges = (("z", "y"), ("y", "x"), ("x", "w"), ("w", "z"), ("z", "x"))
    forest = chordal_elimination_forest(_undirected_graph(nodes, edges))
    assert forest.order == ("w", "x", "y", "z")
    assert forest == chordal_elimination_forest(_undirected_graph(nodes, reversed(edges)))


def test_complete_graph_keeps_one_bag_per_vertex():
    nodes = ("a", "b", "c")
    graph = _undirected_graph(nodes, combinations(nodes, 2))
    forest = chordal_elimination_forest(graph, order=nodes)
    assert tuple(bag.members for bag in forest.bags) == (("a", "b", "c"), ("b", "c"), ("c",))
    assert tuple(bag.parent for bag in forest.bags) == ("b", "c", None)


def test_components_isolates_and_empty_graph():
    graph = _undirected_graph(("a", "b", "c", "d", "isolated"), (("a", "b"), ("c", "d")))
    forest = chordal_elimination_forest(graph, order=graph.nodes)
    assert tuple(bag.vertex for bag in forest.bags if bag.parent is None) == (
        "b", "d", "isolated"
    )
    assert forest.bags[-1] == EliminationBag("isolated", ("isolated",), None, ())
    assert chordal_elimination_forest(_undirected_graph((), ())) == EliminationForest((), ())


@pytest.mark.parametrize(
    "order",
    (
        ("a", "b"),
        ("a", "b", "c", "a"),
        ("a", "b", "a"),
        ("a", "b", "z"),
        ("a", "b", 3),
        ("b", "a", "c"),
        "abc",
        {"a", "b", "c"},
        {"a": 1, "b": 2, "c": 3},
        42,
    ),
)
def test_rejects_invalid_supplied_orders(order):
    graph = _undirected_graph(("a", "b", "c"), (("a", "b"), ("b", "c")))
    with pytest.raises(GraphValidationError):
        chordal_elimination_forest(graph, order=order)


@pytest.mark.parametrize(
    "marks", ((Mark.TAIL, Mark.ARROW), (Mark.ARROW, Mark.ARROW), (Mark.CIRCLE, Mark.CIRCLE))
)
def test_rejects_nonundirected_topologies(marks):
    graph = EndpointMarkedGraph(nodes=("a", "b"), edges=(("a", marks[0], "b", marks[1]),))
    with pytest.raises(GraphValidationError, match="undirected"):
        chordal_elimination_forest(graph)


def _has_induced_cycle(nodes, edges):
    for size in range(4, len(nodes) + 1):
        for vertices in combinations(nodes, size):
            selected = set(vertices)
            induced = {edge for edge in edges if edge <= selected}
            if len(induced) != size:
                continue
            for rest in permutations(vertices[1:]):
                cycle = (vertices[0], *rest, vertices[0])
                if all(frozenset(edge) in induced for edge in pairwise(cycle)):
                    return True
    return False


def _check_forest(forest, nodes, edges):
    assert set(forest.order) == set(nodes)
    assert tuple(bag.vertex for bag in forest.bags) == forest.order
    bags = {bag.vertex: bag for bag in forest.bags}
    adjacent = {vertex: set() for vertex in nodes}
    for index, bag in enumerate(forest.bags):
        later = tuple(
            vertex
            for vertex in forest.order[index + 1 :]
            if frozenset((bag.vertex, vertex)) in edges
        )
        assert bag.members == (bag.vertex, *later)
        assert all(frozenset(edge) in edges for edge in combinations(bag.members, 2))
        if later:
            assert bag.parent == later[0]
            assert set(bag.separator) == set(bag.members) & set(bags[bag.parent].members)
            adjacent[bag.vertex].add(bag.parent)
            adjacent[bag.parent].add(bag.vertex)
        else:
            assert bag.parent is None
            assert not bag.separator
    for vertex in nodes:
        containing = {bag.vertex for bag in forest.bags if vertex in bag.members}
        reached = {vertex}
        pending = [vertex]
        while pending:
            new = (adjacent[pending.pop()] & containing) - reached
            reached |= new
            pending.extend(new)
        assert reached == containing


def test_matches_induced_cycle_and_running_intersection_oracles():
    for size in range(6):
        nodes = tuple(str(index) for index in range(size))
        all_edges = tuple(combinations(nodes, 2))
        for keep in product((False, True), repeat=len(all_edges)):
            chosen = tuple(edge for edge, present in zip(all_edges, keep) if present)
            edges = {frozenset(edge) for edge in chosen}
            graph = _undirected_graph(nodes, chosen)
            if _has_induced_cycle(nodes, edges):
                with pytest.raises(GraphValidationError, match="not chordal"):
                    chordal_elimination_forest(graph)
            else:
                _check_forest(chordal_elimination_forest(graph), nodes, edges)
