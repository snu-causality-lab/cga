from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NoReturn, SupportsIndex

from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import GraphValidationError


def _directed_edges(
    graph: EndpointMarkedGraph,
) -> frozenset[tuple[str, str]]:
    directed: set[tuple[str, str]] = set()
    for left, left_mark, right, right_mark in graph.edges:
        if left_mark is Mark.TAIL and right_mark is Mark.ARROW:
            directed.add((left, right))
        elif left_mark is Mark.ARROW and right_mark is Mark.TAIL:
            directed.add((right, left))
    return frozenset(directed)


def _directed_part_is_acyclic(
    nodes: tuple[str, ...], directed: frozenset[tuple[str, str]]
) -> bool:
    children: dict[str, set[str]] = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for parent, child in directed:
        children[parent].add(child)
        indegree[child] += 1
    pending = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    return visited == len(nodes)


def _directed_reachable(
    nodes: tuple[str, ...],
    directed: frozenset[tuple[str, str]],
    source: str,
    target: str,
) -> bool:
    children: dict[str, set[str]] = {node: set() for node in nodes}
    for parent, child in directed:
        children[parent].add(child)
    reached = {source}
    pending = [source]
    while pending:
        current = pending.pop()
        for child in children[current]:
            if child == target:
                return True
            if child not in reached:
                reached.add(child)
                pending.append(child)
    return False


def _require_ancestral(graph: EndpointMarkedGraph) -> None:
    directed = _directed_edges(graph)
    undirected_endpoints: set[str] = set()
    arrowhead_endpoints: set[str] = set()
    bidirected: list[tuple[str, str]] = []

    for left, left_mark, right, right_mark in graph.edges:
        if left_mark is Mark.CIRCLE or right_mark is Mark.CIRCLE:
            raise GraphValidationError("ancestral graph cannot contain circle marks")
        if left_mark is Mark.TAIL and right_mark is Mark.TAIL:
            undirected_endpoints.update((left, right))
        if left_mark is Mark.ARROW:
            arrowhead_endpoints.add(left)
        if right_mark is Mark.ARROW:
            arrowhead_endpoints.add(right)
        if left_mark is Mark.ARROW and right_mark is Mark.ARROW:
            bidirected.append((left, right))

    violating_undirected = undirected_endpoints & arrowhead_endpoints
    if violating_undirected:
        vertex = min(violating_undirected, key=graph.nodes.index)
        raise GraphValidationError(
            f"undirected-edge endpoint has an arrowhead: {vertex!r}"
        )
    if not _directed_part_is_acyclic(graph.nodes, directed):
        raise GraphValidationError("ancestral graph contains a directed cycle")
    for left, right in bidirected:
        if _directed_reachable(graph.nodes, directed, left, right) or (
            _directed_reachable(graph.nodes, directed, right, left)
        ):
            raise GraphValidationError(
                f"bidirected edge lies on an almost-directed cycle: {(left, right)!r}"
            )


@dataclass(frozen=True, init=False)
class AncestralGraph:
    """Validated Richardson--Spirtes ancestral endpoint-marked graph."""

    __slots__ = ("topology",)

    topology: EndpointMarkedGraph

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AncestralGraph cannot be subclassed")

    def __copy__(self) -> AncestralGraph:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> AncestralGraph:
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("pickle is unsupported; use canonical JSON")

    def __init__(self, topology: EndpointMarkedGraph) -> None:
        if not isinstance(topology, EndpointMarkedGraph):
            raise GraphValidationError(
                "ancestral view requires endpoint-marked topology"
            )
        _require_ancestral(topology)
        object.__setattr__(self, "topology", topology)

    @classmethod
    def from_graph(cls, graph: EndpointMarkedGraph) -> AncestralGraph:
        return cls(graph)

    @classmethod
    def from_text(cls, text: str) -> AncestralGraph:
        """Parse endpoint text and validate an ancestral graph."""

        from .text_io import _endpoint_marked_graph_from_text

        return cls.from_graph(_endpoint_marked_graph_from_text(text))

    def directed_ancestors_strict(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        return self.topology.directed_ancestors_strict(seeds)

    def directed_ancestors_reflexive(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        return self.topology.directed_ancestors_reflexive(seeds)

    def directed_descendants_strict(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        return self.topology.directed_descendants_strict(seeds)

    def directed_descendants_reflexive(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        return self.topology.directed_descendants_reflexive(seeds)

    def anterior_strict(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        return self.topology.anterior_strict(seeds)

    def anterior_reflexive(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        return self.topology.anterior_reflexive(seeds)

    def bidirected_spouses(self, vertex: str) -> frozenset[str]:
        return self.topology.bidirected_spouses(vertex)

    def undirected_neighbors(self, vertex: str) -> frozenset[str]:
        return self.topology.undirected_neighbors(vertex)
