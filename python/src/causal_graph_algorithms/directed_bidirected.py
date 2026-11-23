from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NoReturn, SupportsIndex

from ._validation import (
    node_positions,
    normalize_nodes,
    normalize_pair,
    normalize_seeds,
    require_vertex,
)
from .errors import GraphValidationError


@dataclass(frozen=True, init=False)
class DirectedBidirectedGraph:
    """Immutable raw directed-plus-bidirected topology; bows are permitted."""

    __slots__ = ("nodes", "directed", "bidirected")

    nodes: tuple[str, ...]
    directed: frozenset[tuple[str, str]]
    bidirected: frozenset[tuple[str, str]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("DirectedBidirectedGraph cannot be subclassed")

    def __copy__(self) -> DirectedBidirectedGraph:
        return self

    def __deepcopy__(
        self, memo: dict[int, object]
    ) -> DirectedBidirectedGraph:
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("pickle is unsupported; use canonical JSON")

    def __init__(
        self,
        *,
        nodes: Iterable[str],
        directed: Iterable[tuple[str, str]] = (),
        bidirected: Iterable[tuple[str, str]] = (),
    ) -> None:
        normalized_nodes = normalize_nodes(nodes)
        positions = node_positions(normalized_nodes)

        normalized_directed: set[tuple[str, str]] = set()
        for raw_edge in directed:
            edge = normalize_pair(raw_edge, positions, relation="directed")
            if edge in normalized_directed:
                raise GraphValidationError(f"duplicate directed edge: {edge!r}")
            normalized_directed.add(edge)

        normalized_bidirected: set[tuple[str, str]] = set()
        for raw_edge in bidirected:
            left, right = normalize_pair(raw_edge, positions, relation="bidirected")
            edge = (
                (left, right)
                if positions[left] < positions[right]
                else (right, left)
            )
            if edge in normalized_bidirected:
                raise GraphValidationError(f"duplicate bidirected edge: {edge!r}")
            normalized_bidirected.add(edge)

        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "directed", frozenset(normalized_directed))
        object.__setattr__(self, "bidirected", frozenset(normalized_bidirected))

    @classmethod
    def from_text(cls, text: str) -> DirectedBidirectedGraph:
        """Parse human-readable directed/bidirected graph text."""

        from .text_io import _directed_bidirected_graph_from_text

        parsed = _directed_bidirected_graph_from_text(text)
        return cls(
            nodes=parsed.nodes,
            directed=parsed.directed,
            bidirected=parsed.bidirected,
        )

    def _positions(self) -> dict[str, int]:
        return node_positions(self.nodes)

    def has_directed(self, source: str, target: str) -> bool:
        positions = self._positions()
        require_vertex(positions, source)
        require_vertex(positions, target)
        return (source, target) in self.directed

    def has_bidirected(self, left: str, right: str) -> bool:
        positions = self._positions()
        require_vertex(positions, left)
        require_vertex(positions, right)
        if left == right:
            return False
        edge = (
            (left, right)
            if positions[left] < positions[right]
            else (right, left)
        )
        return edge in self.bidirected

    def directed_parents(self, vertex: str) -> frozenset[str]:
        positions = self._positions()
        require_vertex(positions, vertex)
        return frozenset(left for left, right in self.directed if right == vertex)

    def directed_children(self, vertex: str) -> frozenset[str]:
        positions = self._positions()
        require_vertex(positions, vertex)
        return frozenset(right for left, right in self.directed if left == vertex)

    def bidirected_neighbors(self, vertex: str) -> frozenset[str]:
        """Return the vertices joined to ``vertex`` by a bidirected edge."""

        positions = self._positions()
        required = require_vertex(positions, vertex)
        neighbors: set[str] = set()
        for left, right in self.bidirected:
            if left == required:
                neighbors.add(right)
            elif right == required:
                neighbors.add(left)
        return frozenset(neighbors)

    def district(self, vertex: str) -> frozenset[str]:
        """Return the maximal bidirected-connected set containing ``vertex``."""

        positions = self._positions()
        required = require_vertex(positions, vertex)
        adjacency: dict[str, set[str]] = {node: set() for node in self.nodes}
        for left, right in self.bidirected:
            adjacency[left].add(right)
            adjacency[right].add(left)

        reached = {required}
        pending = [required]
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        return frozenset(reached)

    def districts(self) -> tuple[frozenset[str], ...]:
        """Return districts in the graph's declared vertex order."""

        unseen = set(self.nodes)
        result: list[frozenset[str]] = []
        for node in self.nodes:
            if node not in unseen:
                continue
            component = self.district(node)
            result.append(component)
            unseen.difference_update(component)
        return tuple(result)

    def external_directed_parents(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        positions = self._positions()
        seed_set = normalize_seeds(positions, seeds)
        parents = {
            left for left, right in self.directed if right in seed_set
        }
        return frozenset(parents - seed_set)

    def _directed_closure(
        self, seeds: str | Iterable[str], *, reverse: bool
    ) -> tuple[frozenset[str], frozenset[str]]:
        positions = self._positions()
        seed_set = normalize_seeds(positions, seeds)
        adjacency: dict[str, set[str]] = {node: set() for node in self.nodes}
        for left, right in self.directed:
            if reverse:
                adjacency[right].add(left)
            else:
                adjacency[left].add(right)

        reached = set(seed_set)
        pending = list(seed_set)
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        return seed_set, frozenset(reached)

    def directed_ancestors_strict(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        seed_set, reached = self._directed_closure(seeds, reverse=True)
        return reached - seed_set

    def directed_ancestors_reflexive(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        _, reached = self._directed_closure(seeds, reverse=True)
        return reached

    def directed_descendants_strict(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        seed_set, reached = self._directed_closure(seeds, reverse=False)
        return reached - seed_set

    def directed_descendants_reflexive(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        _, reached = self._directed_closure(seeds, reverse=False)
        return reached


def _directed_part_is_acyclic(graph: DirectedBidirectedGraph) -> bool:
    outgoing: dict[str, set[str]] = {node: set() for node in graph.nodes}
    indegree = {node: 0 for node in graph.nodes}
    for left, right in graph.directed:
        outgoing[left].add(right)
        indegree[right] += 1

    pending = [node for node in graph.nodes if indegree[node] == 0]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        for child in outgoing[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    return visited == len(graph.nodes)


@dataclass(frozen=True, init=False)
class AcyclicDirectedBidirectedGraph:
    __slots__ = ("topology",)

    topology: DirectedBidirectedGraph

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AcyclicDirectedBidirectedGraph cannot be subclassed")

    def __copy__(self) -> AcyclicDirectedBidirectedGraph:
        return self

    def __deepcopy__(
        self, memo: dict[int, object]
    ) -> AcyclicDirectedBidirectedGraph:
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("pickle is unsupported; use canonical JSON")

    def __init__(self, topology: DirectedBidirectedGraph) -> None:
        if not isinstance(topology, DirectedBidirectedGraph):
            raise GraphValidationError("acyclic view requires directed-bidirected topology")
        if not _directed_part_is_acyclic(topology):
            raise GraphValidationError("directed part contains a cycle")
        object.__setattr__(self, "topology", topology)

    @classmethod
    def from_graph(
        cls, graph: DirectedBidirectedGraph
    ) -> AcyclicDirectedBidirectedGraph:
        return cls(graph)

    @classmethod
    def from_text(cls, text: str) -> AcyclicDirectedBidirectedGraph:
        """Parse graph text and validate an acyclic directed part."""

        from .text_io import _directed_bidirected_graph_from_text

        return cls.from_graph(_directed_bidirected_graph_from_text(text))

    def bidirected_neighbors(self, vertex: str) -> frozenset[str]:
        return self.topology.bidirected_neighbors(vertex)

    def district(self, vertex: str) -> frozenset[str]:
        return self.topology.district(vertex)

    def districts(self) -> tuple[frozenset[str], ...]:
        return self.topology.districts()


@dataclass(frozen=True, init=False)
class DAG:
    __slots__ = ("topology",)

    topology: DirectedBidirectedGraph

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("DAG cannot be subclassed")

    def __copy__(self) -> DAG:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> DAG:
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("pickle is unsupported; use canonical JSON")

    def __init__(self, topology: DirectedBidirectedGraph) -> None:
        if not isinstance(topology, DirectedBidirectedGraph):
            raise GraphValidationError("DAG view requires directed-bidirected topology")
        if topology.bidirected:
            raise GraphValidationError("DAG cannot contain bidirected edges")
        if not _directed_part_is_acyclic(topology):
            raise GraphValidationError("DAG directed part contains a cycle")
        object.__setattr__(self, "topology", topology)

    @classmethod
    def from_graph(cls, graph: DirectedBidirectedGraph) -> DAG:
        return cls(graph)

    @classmethod
    def from_text(cls, text: str) -> DAG:
        """Parse graph text and validate a DAG."""

        from .text_io import _directed_bidirected_graph_from_text

        return cls.from_graph(_directed_bidirected_graph_from_text(text))

    def bidirected_neighbors(self, vertex: str) -> frozenset[str]:
        return self.topology.bidirected_neighbors(vertex)

    def district(self, vertex: str) -> frozenset[str]:
        return self.topology.district(vertex)

    def districts(self) -> tuple[frozenset[str], ...]:
        return self.topology.districts()
