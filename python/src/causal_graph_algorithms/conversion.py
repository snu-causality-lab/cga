from __future__ import annotations

from .directed_bidirected import DirectedBidirectedGraph
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import LossyConversionError


def directed_bidirected_to_endpoint_marked(
    graph: DirectedBidirectedGraph,
) -> EndpointMarkedGraph:
    if not isinstance(graph, DirectedBidirectedGraph):
        raise TypeError("expected DirectedBidirectedGraph")

    positions = {node: index for index, node in enumerate(graph.nodes)}
    by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for source, target in graph.directed:
        pair = (
            (source, target)
            if positions[source] < positions[target]
            else (target, source)
        )
        by_pair.setdefault(pair, []).append(("directed", source))
    for left, right in graph.bidirected:
        by_pair.setdefault((left, right), []).append(("bidirected", left))

    edges: list[tuple[str, Mark, str, Mark]] = []
    for pair, kinds in by_pair.items():
        if len(kinds) != 1:
            raise LossyConversionError(
                f"endpoint-marked topology cannot preserve pair {pair!r}"
            )
        kind, source = kinds[0]
        left, right = pair
        if kind == "bidirected":
            edges.append((left, Mark.ARROW, right, Mark.ARROW))
        elif source == left:
            edges.append((left, Mark.TAIL, right, Mark.ARROW))
        else:
            edges.append((left, Mark.ARROW, right, Mark.TAIL))
    return EndpointMarkedGraph(nodes=graph.nodes, edges=edges)


def endpoint_marked_to_directed_bidirected(
    graph: EndpointMarkedGraph,
) -> DirectedBidirectedGraph:
    if not isinstance(graph, EndpointMarkedGraph):
        raise TypeError("expected EndpointMarkedGraph")

    directed: list[tuple[str, str]] = []
    bidirected: list[tuple[str, str]] = []
    for left, left_mark, right, right_mark in graph.edges:
        marks = (left_mark, right_mark)
        if marks == (Mark.TAIL, Mark.ARROW):
            directed.append((left, right))
        elif marks == (Mark.ARROW, Mark.TAIL):
            directed.append((right, left))
        elif marks == (Mark.ARROW, Mark.ARROW):
            bidirected.append((left, right))
        else:
            raise LossyConversionError(
                f"directed-bidirected topology cannot preserve marks {marks!r}"
            )
    return DirectedBidirectedGraph(
        nodes=graph.nodes, directed=directed, bidirected=bidirected
    )
