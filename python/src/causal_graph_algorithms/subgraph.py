from __future__ import annotations

from collections.abc import Iterable
from typing import overload

from ._validation import node_positions, normalize_seeds
from .ancestral import AncestralGraph
from .directed_bidirected import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
)
from .endpoint_marked import EndpointMarkedGraph
from .maximality import MaximalAncestralGraph


def _directed_bidirected_induced_subgraph(
    graph: DirectedBidirectedGraph, retained: str | Iterable[str]
) -> DirectedBidirectedGraph:
    retained_set = normalize_seeds(node_positions(graph.nodes), retained)
    nodes = tuple(node for node in graph.nodes if node in retained_set)
    return DirectedBidirectedGraph(
        nodes=nodes,
        directed=(
            (source, target)
            for source, target in graph.directed
            if source in retained_set and target in retained_set
        ),
        bidirected=(
            (left, right)
            for left, right in graph.bidirected
            if left in retained_set and right in retained_set
        ),
    )


def _endpoint_marked_induced_subgraph(
    graph: EndpointMarkedGraph, retained: str | Iterable[str]
) -> EndpointMarkedGraph:
    retained_set = normalize_seeds(node_positions(graph.nodes), retained)
    nodes = tuple(node for node in graph.nodes if node in retained_set)
    return EndpointMarkedGraph(
        nodes=nodes,
        edges=(
            edge
            for edge in graph.edges
            if edge[0] in retained_set and edge[2] in retained_set
        ),
    )


@overload
def induced_subgraph(
    graph: DirectedBidirectedGraph, retained: str | Iterable[str]
) -> DirectedBidirectedGraph: ...


@overload
def induced_subgraph(
    graph: AcyclicDirectedBidirectedGraph, retained: str | Iterable[str]
) -> AcyclicDirectedBidirectedGraph: ...


@overload
def induced_subgraph(
    graph: DAG, retained: str | Iterable[str]
) -> DAG: ...


@overload
def induced_subgraph(
    graph: EndpointMarkedGraph, retained: str | Iterable[str]
) -> EndpointMarkedGraph: ...


@overload
def induced_subgraph(
    graph: AncestralGraph, retained: str | Iterable[str]
) -> AncestralGraph: ...


@overload
def induced_subgraph(
    graph: MaximalAncestralGraph, retained: str | Iterable[str]
) -> MaximalAncestralGraph: ...


def induced_subgraph(
    graph: DirectedBidirectedGraph
    | AcyclicDirectedBidirectedGraph
    | DAG
    | EndpointMarkedGraph
    | AncestralGraph
    | MaximalAncestralGraph,
    retained: str | Iterable[str],
) -> (
    DirectedBidirectedGraph
    | AcyclicDirectedBidirectedGraph
    | DAG
    | EndpointMarkedGraph
    | AncestralGraph
    | MaximalAncestralGraph
):
    """Return the vertex-induced subgraph, preserving an existing proof view."""

    if isinstance(graph, DirectedBidirectedGraph):
        return _directed_bidirected_induced_subgraph(graph, retained)
    if isinstance(graph, AcyclicDirectedBidirectedGraph):
        return AcyclicDirectedBidirectedGraph.from_graph(
            _directed_bidirected_induced_subgraph(graph.topology, retained)
        )
    if isinstance(graph, DAG):
        return DAG.from_graph(
            _directed_bidirected_induced_subgraph(graph.topology, retained)
        )
    if isinstance(graph, EndpointMarkedGraph):
        return _endpoint_marked_induced_subgraph(graph, retained)
    if isinstance(graph, AncestralGraph):
        return AncestralGraph.from_graph(
            _endpoint_marked_induced_subgraph(graph.topology, retained)
        )
    if isinstance(graph, MaximalAncestralGraph):
        return MaximalAncestralGraph.from_graph(
            _endpoint_marked_induced_subgraph(graph.topology, retained)
        )
    raise TypeError("induced_subgraph requires a supported graph or validated view")
