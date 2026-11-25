from __future__ import annotations

import copy
import itertools
import pickle

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    Mark,
)

from .oracles import (
    has_topological_order,
    possible_ancestors_by_simple_paths,
    possible_descendants_by_simple_paths,
    strict_ancestors_by_simple_paths,
    strict_descendants_by_simple_paths,
)


def test_empty_graph_and_declared_node_order_are_values() -> None:
    assert DirectedBidirectedGraph(nodes=[]).nodes == ()
    graph = DirectedBidirectedGraph(nodes=["B", "α", "A"])
    assert graph.nodes == ("B", "α", "A")


def test_bow_is_lossless_in_directed_bidirected_family() -> None:
    graph = DirectedBidirectedGraph(
        nodes=["A", "B"],
        directed=[("A", "B")],
        bidirected=[("B", "A")],
    )
    assert graph.has_directed("A", "B")
    assert graph.has_bidirected("A", "B")
    assert graph.directed == frozenset({("A", "B")})
    assert graph.bidirected == frozenset({("A", "B")})


@pytest.mark.parametrize(
    ("nodes", "directed", "bidirected"),
    [
        (["A", "A"], [], []),
        ([""], [], []),
        (["\ud800"], [], []),
        (["A"], [("A", "A")], []),
        (["A"], [], [("A", "A")]),
        (["A"], [("A", "missing")], []),
        (["A", "B"], [("A", "B"), ("A", "B")], []),
        (["A", "B"], [], [("A", "B"), ("B", "A")]),
    ],
)
def test_directed_bidirected_constructor_fails_closed(
    nodes: list[str],
    directed: list[tuple[str, str]],
    bidirected: list[tuple[str, str]],
) -> None:
    with pytest.raises(GraphValidationError):
        DirectedBidirectedGraph(
            nodes=nodes, directed=directed, bidirected=bidirected
        )


def test_constructor_copies_mutable_inputs() -> None:
    nodes = ["A", "B"]
    directed = [("A", "B")]
    graph = DirectedBidirectedGraph(nodes=nodes, directed=directed)
    nodes.append("C")
    directed.clear()
    assert graph.nodes == ("A", "B")
    assert graph.directed == frozenset({("A", "B")})
    with pytest.raises(AttributeError):
        graph.nodes = ("changed",)  # type: ignore[misc]


def test_generated_state_hooks_cannot_bypass_immutability_or_validation() -> None:
    raw = DirectedBidirectedGraph(nodes=["A", "B"], directed=[("A", "B")])
    raw_before = DirectedBidirectedGraph(nodes=["A", "B"], directed=[("A", "B")])
    with pytest.raises((AttributeError, TypeError)):
        raw.__setstate__(  # type: ignore[attr-defined]
            [("X",), frozenset(), frozenset({("X", "X")})]
        )
    assert raw == raw_before

    dag = DAG.from_graph(raw)
    cyclic = DirectedBidirectedGraph(
        nodes=["A", "B"], directed=[("A", "B"), ("B", "A")]
    )
    with pytest.raises((AttributeError, TypeError)):
        dag.__setstate__([cyclic])  # type: ignore[attr-defined]
    assert dag.topology == raw

    acyclic = AcyclicDirectedBidirectedGraph.from_graph(raw)
    with pytest.raises((AttributeError, TypeError)):
        acyclic.__setstate__([cyclic])  # type: ignore[attr-defined]
    assert acyclic.topology == raw

    endpoint = EndpointMarkedGraph(
        nodes=["A", "B"], edges=[("A", Mark.TAIL, "B", Mark.ARROW)]
    )
    with pytest.raises((AttributeError, TypeError)):
        endpoint.__setstate__([("X",)])  # type: ignore[attr-defined]
    assert endpoint.nodes == ("A", "B")
    assert endpoint.edge("A", "B") == (Mark.TAIL, Mark.ARROW)

    for value in (raw, dag, acyclic, endpoint):
        assert not hasattr(value, "__dict__")


@pytest.mark.parametrize(
    "base",
    [
        DirectedBidirectedGraph,
        AcyclicDirectedBidirectedGraph,
        DAG,
        EndpointMarkedGraph,
    ],
)
def test_semantic_value_classes_are_sealed(base: type[object]) -> None:
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("MutableSemanticSubclass", (base,), {})


def test_copy_is_identity_and_pickle_is_explicitly_unsupported() -> None:
    raw = DirectedBidirectedGraph(nodes=["A", "B"], directed=[("A", "B")])
    values = (
        raw,
        AcyclicDirectedBidirectedGraph.from_graph(raw),
        DAG.from_graph(raw),
        EndpointMarkedGraph(
            nodes=["A", "B"], edges=[("A", Mark.TAIL, "B", Mark.ARROW)]
        ),
    )
    for value in values:
        assert copy.copy(value) is value
        assert copy.deepcopy(value) is value
        with pytest.raises(TypeError, match="canonical JSON"):
            pickle.dumps(value)


def test_acyclic_and_dag_views_have_distinct_requirements() -> None:
    bow = DirectedBidirectedGraph(
        nodes=["A", "B"],
        directed=[("A", "B")],
        bidirected=[("A", "B")],
    )
    acyclic = AcyclicDirectedBidirectedGraph.from_graph(bow)
    assert acyclic.topology == bow
    with pytest.raises(GraphValidationError):
        DAG.from_graph(bow)

    cycle = DirectedBidirectedGraph(
        nodes=["A", "B"], directed=[("A", "B"), ("B", "A")]
    )
    with pytest.raises(GraphValidationError):
        AcyclicDirectedBidirectedGraph.from_graph(cycle)

    dag_topology = DirectedBidirectedGraph(
        nodes=["A", "B", "C"], directed=[("A", "B"), ("B", "C")]
    )
    assert DAG.from_graph(dag_topology).topology == dag_topology
    with pytest.raises(GraphValidationError):
        DAG.from_graph(cycle)


def test_acyclic_validator_matches_permutation_oracle_exhaustively_at_n3() -> None:
    nodes = ("A", "B", "C")
    possible_edges = tuple(
        (left, right) for left in nodes for right in nodes if left != right
    )
    for mask in range(1 << len(possible_edges)):
        directed = frozenset(
            edge for index, edge in enumerate(possible_edges) if mask & (1 << index)
        )
        raw = DirectedBidirectedGraph(nodes=nodes, directed=directed)
        expected = has_topological_order(nodes, directed)
        if expected:
            assert AcyclicDirectedBidirectedGraph.from_graph(raw).topology == raw
            assert DAG.from_graph(raw).topology == raw
        else:
            with pytest.raises(GraphValidationError):
                AcyclicDirectedBidirectedGraph.from_graph(raw)
            with pytest.raises(GraphValidationError):
                DAG.from_graph(raw)


def test_strict_reflexive_and_external_set_semantics() -> None:
    graph = DirectedBidirectedGraph(
        nodes=["W", "X", "Y", "Z"],
        directed=[("W", "X"), ("X", "Y"), ("Z", "Y")],
    )
    seeds = {"X", "Y"}
    assert graph.directed_parents("Y") == frozenset({"X", "Z"})
    assert graph.directed_children("W") == frozenset({"X"})
    assert graph.external_directed_parents(seeds) == frozenset({"W", "Z"})
    assert graph.directed_ancestors_strict(seeds) == frozenset({"W", "Z"})
    assert graph.directed_ancestors_reflexive(seeds) == frozenset(
        {"W", "X", "Y", "Z"}
    )
    assert graph.directed_descendants_strict({"W", "X"}) == frozenset({"Y"})
    assert graph.directed_descendants_reflexive({"W", "X"}) == frozenset(
        {"W", "X", "Y"}
    )


def test_directed_relations_match_simple_path_oracle() -> None:
    nodes = ("A", "B", "C", "D")
    directed = frozenset(
        {("A", "B"), ("B", "C"), ("A", "D"), ("D", "C"), ("C", "A")}
    )
    graph = DirectedBidirectedGraph(nodes=nodes, directed=directed)
    for size in range(1, len(nodes) + 1):
        for seeds in itertools.combinations(nodes, size):
            assert graph.directed_ancestors_strict(seeds) == (
                strict_ancestors_by_simple_paths(nodes, directed, seeds)
            )
            assert graph.directed_descendants_strict(seeds) == (
                strict_descendants_by_simple_paths(nodes, directed, seeds)
            )


def test_directed_relations_match_oracle_exhaustively_at_n3() -> None:
    nodes = ("A", "B", "C")
    candidates = tuple(
        (left, right) for left in nodes for right in nodes if left != right
    )
    seed_sets = [
        seeds
        for size in range(1, len(nodes) + 1)
        for seeds in itertools.combinations(nodes, size)
    ]
    for mask in range(1 << len(candidates)):
        directed = frozenset(
            edge for index, edge in enumerate(candidates) if mask & (1 << index)
        )
        graph = DirectedBidirectedGraph(nodes=nodes, directed=directed)
        for seeds in seed_sets:
            assert graph.directed_ancestors_strict(seeds) == (
                strict_ancestors_by_simple_paths(nodes, directed, seeds)
            )
            assert graph.directed_descendants_strict(seeds) == (
                strict_descendants_by_simple_paths(nodes, directed, seeds)
            )


def test_endpoint_marked_edges_and_possible_path_departing_endpoint() -> None:
    graph = EndpointMarkedGraph(
        nodes=["A", "B", "C", "D"],
        edges=[
            ("A", Mark.TAIL, "B", Mark.ARROW),
            ("B", Mark.TAIL, "C", Mark.ARROW),
            ("D", Mark.ARROW, "B", Mark.CIRCLE),
        ],
    )
    assert graph.edge("B", "A") == (Mark.ARROW, Mark.TAIL)
    assert graph.edge("A", "B") == (Mark.TAIL, Mark.ARROW)
    assert graph.edge("A", "D") is None
    assert graph.edge("A", "A") is None
    assert graph.possible_descendants_strict("A") == frozenset({"B", "C", "D"})
    assert graph.possible_descendants_reflexive("A") == frozenset(
        {"A", "B", "C", "D"}
    )
    assert "D" in graph.possible_descendants_strict("B")
    assert "B" not in graph.possible_descendants_strict("D")

    oracle_edges = tuple(
        (left, left_mark.value, right, right_mark.value)
        for left, left_mark, right, right_mark in graph.edges
    )
    assert graph.possible_descendants_strict("A") == (
        possible_descendants_by_simple_paths(graph.nodes, oracle_edges, ["A"])
    )
    assert graph.possible_ancestors_strict("D") == frozenset({"A", "B"})
    assert graph.possible_ancestors_strict("B") == frozenset({"A"})
    assert graph.possible_ancestors_strict({"C", "D"}) == frozenset({"A", "B"})
    assert graph.possible_ancestors_reflexive({"C", "D"}) == frozenset(
        {"A", "B", "C", "D"}
    )
    assert graph.possible_ancestors_strict({"C", "D"}) == (
        possible_ancestors_by_simple_paths(graph.nodes, oracle_edges, ["C", "D"])
    )


def test_endpoint_directed_parent_and_child_relations_in_both_storage_orders() -> None:
    graph = EndpointMarkedGraph(
        nodes=["B", "A", "C", "D"],
        edges=[
            ("B", Mark.TAIL, "A", Mark.ARROW),
            ("C", Mark.TAIL, "A", Mark.ARROW),
            ("A", Mark.ARROW, "D", Mark.ARROW),
        ],
    )
    assert graph.directed_parents("A") == frozenset({"B", "C"})
    assert graph.directed_children("B") == frozenset({"A"})
    assert graph.directed_children("C") == frozenset({"A"})
    assert graph.directed_parents("B") == frozenset()
    assert graph.directed_children("A") == frozenset()


def test_possible_relations_match_oracle_exhaustively_at_n3() -> None:
    nodes = ("A", "B", "C")
    pairs = (("A", "B"), ("A", "C"), ("B", "C"))
    states: tuple[None | tuple[Mark, Mark], ...] = (None,) + tuple(
        itertools.product(tuple(Mark), repeat=2)
    )
    seed_sets = [
        seeds
        for size in range(1, len(nodes) + 1)
        for seeds in itertools.combinations(nodes, size)
    ]
    for choices in itertools.product(states, repeat=len(pairs)):
        edges = [
            (left, marks[0], right, marks[1])
            for (left, right), marks in zip(pairs, choices, strict=True)
            if marks is not None
        ]
        graph = EndpointMarkedGraph(nodes=nodes, edges=edges)
        oracle_edges = tuple(
            (left, left_mark.value, right, right_mark.value)
            for left, left_mark, right, right_mark in graph.edges
        )
        for seeds in seed_sets:
            assert graph.possible_descendants_strict(seeds) == (
                possible_descendants_by_simple_paths(nodes, oracle_edges, seeds)
            )
            assert graph.possible_ancestors_strict(seeds) == (
                possible_ancestors_by_simple_paths(nodes, oracle_edges, seeds)
            )


def test_endpoint_marked_duplicate_pair_and_bad_marks_fail_closed() -> None:
    with pytest.raises(GraphValidationError):
        EndpointMarkedGraph(
            nodes=["A", "B"],
            edges=[
                ("A", Mark.TAIL, "B", Mark.ARROW),
                ("B", Mark.ARROW, "A", Mark.TAIL),
            ],
        )
    with pytest.raises(GraphValidationError):
        EndpointMarkedGraph(
            nodes=["A", "B"], edges=[("A", "none", "B", "arrow")]
        )


@pytest.mark.parametrize(
    ("nodes", "edges"),
    [
        (["A", "A"], []),
        ([""], []),
        (["\ud800"], []),
        (["A"], [("A", Mark.TAIL, "A", Mark.ARROW)]),
        (["A"], [("A", Mark.TAIL, "missing", Mark.ARROW)]),
    ],
)
def test_endpoint_marked_constructor_shared_validation(
    nodes: list[str], edges: list[tuple[object, ...]]
) -> None:
    with pytest.raises(GraphValidationError):
        EndpointMarkedGraph(nodes=nodes, edges=edges)


def test_endpoint_marked_is_unbounded_and_accepts_empty_graph() -> None:
    assert EndpointMarkedGraph(nodes=[]).nodes == ()
    nodes = tuple(f"V{index}" for index in range(40))
    graph = EndpointMarkedGraph(
        nodes=nodes,
        edges=[
            (nodes[index], Mark.CIRCLE, nodes[index + 1], Mark.CIRCLE)
            for index in range(39)
        ],
    )
    assert graph.possible_descendants_strict("V0") == frozenset(nodes[1:])


def test_python_semantics_are_not_limited_to_sixteen_vertices() -> None:
    nodes = tuple(f"V{index}" for index in range(40))
    directed = tuple((nodes[index], nodes[index + 1]) for index in range(39))
    graph = DirectedBidirectedGraph(nodes=nodes, directed=directed)
    assert graph.directed_descendants_strict("V0") == frozenset(nodes[1:])
    assert graph.directed_ancestors_strict("V39") == frozenset(nodes[:-1])
