from __future__ import annotations

import itertools

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    AncestralGraph,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    Mark,
    graph_to_canonical_json,
    induced_subgraph,
)


def test_directed_bidirected_induced_subgraph_preserves_raw_edge_atoms() -> None:
    graph = DirectedBidirectedGraph(
        nodes=["D", "A", "C", "B"],
        directed=[("D", "A"), ("A", "C"), ("C", "A"), ("C", "B")],
        bidirected=[("A", "C"), ("D", "B")],
    )

    result = induced_subgraph(graph, ["C", "A"])

    assert type(result) is DirectedBidirectedGraph
    assert result.nodes == ("A", "C")
    assert result.directed == frozenset({("A", "C"), ("C", "A")})
    assert result.bidirected == frozenset({("A", "C")})
    assert graph.nodes == ("D", "A", "C", "B")


def test_endpoint_marked_induced_subgraph_preserves_whole_edges() -> None:
    graph = EndpointMarkedGraph(
        nodes=["D", "A", "C", "B"],
        edges=[
            ("D", Mark.CIRCLE, "A", Mark.ARROW),
            ("A", Mark.TAIL, "C", Mark.TAIL),
            ("C", Mark.ARROW, "B", Mark.CIRCLE),
        ],
    )

    result = induced_subgraph(graph, ["C", "A", "C"])

    assert type(result) is EndpointMarkedGraph
    assert result.nodes == ("A", "C")
    assert result.edges == (("A", Mark.TAIL, "C", Mark.TAIL),)
    assert graph.edge("D", "A") == (Mark.CIRCLE, Mark.ARROW)


def test_induced_subgraph_preserves_existing_proof_views() -> None:
    dag = DAG.from_graph(
        DirectedBidirectedGraph(
            nodes=["A", "B", "C"],
            directed=[("A", "B"), ("B", "C")],
        )
    )
    acyclic = AcyclicDirectedBidirectedGraph.from_graph(
        DirectedBidirectedGraph(
            nodes=["A", "B", "C"],
            directed=[("A", "B"), ("B", "C")],
            bidirected=[("A", "C")],
        )
    )
    ancestral = AncestralGraph.from_graph(
        EndpointMarkedGraph(
            nodes=["A", "B", "C"],
            edges=[
                ("A", Mark.TAIL, "B", Mark.TAIL),
                ("B", Mark.TAIL, "C", Mark.ARROW),
            ],
        )
    )

    dag_result = induced_subgraph(dag, ["C", "B"])
    acyclic_result = induced_subgraph(acyclic, ["C", "A"])
    ancestral_result = induced_subgraph(ancestral, ["C", "B"])

    assert type(dag_result) is DAG
    assert dag_result.topology == DirectedBidirectedGraph(
        nodes=["B", "C"], directed=[("B", "C")]
    )
    assert type(acyclic_result) is AcyclicDirectedBidirectedGraph
    assert acyclic_result.topology == DirectedBidirectedGraph(
        nodes=["A", "C"], bidirected=[("A", "C")]
    )
    assert type(ancestral_result) is AncestralGraph
    assert ancestral_result.topology == EndpointMarkedGraph(
        nodes=["B", "C"],
        edges=[("B", Mark.TAIL, "C", Mark.ARROW)],
    )


@pytest.mark.parametrize(
    "graph",
    [
        DirectedBidirectedGraph(nodes=["A", "B"]),
        EndpointMarkedGraph(nodes=["A", "B"]),
        DAG.from_graph(DirectedBidirectedGraph(nodes=["A", "B"])),
        AcyclicDirectedBidirectedGraph.from_graph(
            DirectedBidirectedGraph(nodes=["A", "B"])
        ),
        AncestralGraph.from_graph(EndpointMarkedGraph(nodes=["A", "B"])),
    ],
)
def test_empty_full_and_string_vertex_selectors(graph: object) -> None:
    empty = induced_subgraph(graph, [])
    one = induced_subgraph(graph, "B")
    full = induced_subgraph(graph, ["B", "A"])

    empty_topology = getattr(empty, "topology", empty)
    one_topology = getattr(one, "topology", one)
    full_topology = getattr(full, "topology", full)
    assert empty_topology.nodes == ()
    assert one_topology.nodes == ("B",)
    assert full_topology.nodes == ("A", "B")


def test_induced_subgraph_pins_declared_order_in_canonical_bytes() -> None:
    graph = DirectedBidirectedGraph(
        nodes=["Z", "A", "M"],
        directed=[("Z", "M"), ("A", "M")],
        bidirected=[("Z", "A")],
    )

    result = induced_subgraph(graph, ["M", "Z", "A"])

    assert graph_to_canonical_json(result) == (
        '{"schema":"cga.graph/1","family":"directed-bidirected",'
        '"nodes":["Z","A","M"],"directed":[["Z","M"],["A","M"]],'
        '"bidirected":[["Z","A"]]}'
    )


def test_unknown_vertices_and_unsupported_inputs_fail_closed() -> None:
    graph = DirectedBidirectedGraph(nodes=["A", "B"])

    with pytest.raises(GraphValidationError, match="unknown vertex"):
        induced_subgraph(graph, ["A", "missing"])
    with pytest.raises(TypeError, match="supported graph"):
        induced_subgraph(object(), ["A"])


def test_view_closure_over_small_exhaustive_universes() -> None:
    nodes = ("A", "B", "C")
    directed_pairs = tuple(
        (left, right) for left in nodes for right in nodes if left != right
    )
    unordered_pairs = tuple(itertools.combinations(nodes, 2))

    for directed_bits in range(1 << len(directed_pairs)):
        topology = DirectedBidirectedGraph(
            nodes=nodes,
            directed=[
                edge
                for index, edge in enumerate(directed_pairs)
                if directed_bits & (1 << index)
            ],
        )
        try:
            dag = DAG.from_graph(topology)
        except GraphValidationError:
            continue
        for retained_bits in range(1 << len(nodes)):
            retained = [
                node
                for index, node in enumerate(nodes)
                if retained_bits & (1 << index)
            ]
            assert type(induced_subgraph(dag, retained)) is DAG

    states: tuple[None | tuple[Mark, Mark], ...] = (
        None,
        (Mark.TAIL, Mark.TAIL),
        (Mark.TAIL, Mark.ARROW),
        (Mark.ARROW, Mark.TAIL),
        (Mark.ARROW, Mark.ARROW),
    )
    for choices in itertools.product(range(len(states)), repeat=len(unordered_pairs)):
        topology = EndpointMarkedGraph(
            nodes=nodes,
            edges=[
                (left, state[0], right, state[1])
                for (left, right), choice in zip(
                    unordered_pairs, choices, strict=True
                )
                if (state := states[choice]) is not None
            ],
        )
        try:
            ancestral = AncestralGraph.from_graph(topology)
        except GraphValidationError:
            continue
        for retained_bits in range(1 << len(nodes)):
            retained = [
                node
                for index, node in enumerate(nodes)
                if retained_bits & (1 << index)
            ]
            assert type(induced_subgraph(ancestral, retained)) is AncestralGraph
