"""Perfect elimination and elimination forests for undirected graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

from ._validation import node_positions, normalize_nodes
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import GraphValidationError


@dataclass(frozen=True, slots=True)
class EliminationBag:
    """A vertex and its later neighbors in a perfect elimination order."""

    vertex: str
    members: tuple[str, ...]
    parent: str | None
    separator: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EliminationForest:
    """A perfect elimination order and its running-intersection bags."""

    order: tuple[str, ...]
    bags: tuple[EliminationBag, ...]


def _maximum_cardinality_order(neighbors: dict[str, set[str]]) -> tuple[str, ...]:
    positions = node_positions(tuple(neighbors))
    scores = dict.fromkeys(neighbors, 0)
    queue = [(0, position, vertex) for vertex, position in positions.items()]
    heapify(queue)
    visited: set[str] = set()
    order: list[str] = []
    while len(order) < len(neighbors):
        _, _, vertex = heappop(queue)
        if vertex in visited:
            continue
        visited.add(vertex)
        order.append(vertex)
        for neighbor in neighbors[vertex] - visited:
            scores[neighbor] += 1
            heappush(queue, (-scores[neighbor], positions[neighbor], neighbor))
    return tuple(reversed(order))


def chordal_elimination_forest(
    graph: EndpointMarkedGraph, *, order: Iterable[str] | None = None
) -> EliminationForest:
    """Construct an elimination bag for each vertex of a chordal graph.

    Only tail–tail edges are permitted. A supplied order lists every vertex
    exactly once, from first to last eliminated. By default, reverse maximum
    cardinality search gives the candidate order, using input vertex order to
    break ties. The candidate is checked before returning any bags.
    """
    if not isinstance(graph, EndpointMarkedGraph):
        raise GraphValidationError("chordal elimination requires an undirected endpoint-marked graph")
    neighbors: dict[str, set[str]] = {vertex: set() for vertex in graph.nodes}
    for left, left_mark, right, right_mark in graph.edges:
        if left_mark is not Mark.TAIL or right_mark is not Mark.TAIL:
            raise GraphValidationError("chordal elimination requires undirected edges")
        neighbors[left].add(right)
        neighbors[right].add(left)

    if order is None:
        elimination = _maximum_cardinality_order(neighbors)
    else:
        if isinstance(order, (str, bytes, bytearray, Mapping, Set)):
            raise GraphValidationError("order must be an ordered iterable of vertex labels")
        elimination = normalize_nodes(order)
        if set(elimination) != neighbors.keys():
            raise GraphValidationError("order must list every graph vertex exactly once")

    positions = node_positions(elimination)
    later: dict[str, list[str]] = {vertex: [] for vertex in elimination}
    for vertex in elimination:
        for neighbor in neighbors[vertex]:
            if positions[neighbor] < positions[vertex]:
                later[neighbor].append(vertex)

    bags: list[EliminationBag] = []
    for vertex in elimination:
        separator = tuple(later[vertex])
        parent = separator[0] if separator else None
        # If the remaining later neighbors are adjacent to the first, they
        # are later neighbors of the parent. Reverse induction gives cliques.
        if parent is not None and any(
            neighbor not in neighbors[parent] for neighbor in separator[1:]
        ):
            if order is None:
                raise GraphValidationError("input graph is not chordal")
            raise GraphValidationError("supplied order is not a perfect elimination order")
        bags.append(EliminationBag(vertex, (vertex, *separator), parent, separator))
    return EliminationForest(elimination, tuple(bags))
