from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, SupportsIndex, TypeAlias

from ._validation import (
    node_positions,
    normalize_nodes,
    normalize_seeds,
    require_vertex,
)
from .errors import GraphValidationError


class Mark(str, Enum):
    TAIL = "tail"
    ARROW = "arrow"
    CIRCLE = "circle"


EndpointEdge: TypeAlias = tuple[str, Mark, str, Mark]


def _coerce_mark(value: object) -> Mark:
    try:
        return Mark(value)
    except (TypeError, ValueError) as error:
        raise GraphValidationError(f"invalid endpoint mark: {value!r}") from error


@dataclass(frozen=True, init=False)
class EndpointMarkedGraph:
    """Immutable raw one-edge-per-pair endpoint-marked topology."""

    __slots__ = ("nodes", "edges")

    nodes: tuple[str, ...]
    edges: tuple[EndpointEdge, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("EndpointMarkedGraph cannot be subclassed")

    def __copy__(self) -> EndpointMarkedGraph:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> EndpointMarkedGraph:
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("pickle is unsupported; use canonical JSON")

    def __init__(
        self,
        *,
        nodes: Iterable[str],
        edges: Iterable[tuple[object, object, object, object]] = (),
    ) -> None:
        normalized_nodes = normalize_nodes(nodes)
        positions = node_positions(normalized_nodes)
        normalized_edges: list[EndpointEdge] = []
        seen_pairs: set[tuple[str, str]] = set()

        for raw_edge in edges:
            if not isinstance(raw_edge, (tuple, list)) or len(raw_edge) != 4:
                raise GraphValidationError(
                    "endpoint-marked edges must have two endpoints and two marks"
                )
            raw_left, raw_left_mark, raw_right, raw_right_mark = raw_edge
            left = require_vertex(positions, raw_left)
            right = require_vertex(positions, raw_right)
            if left == right:
                raise GraphValidationError("endpoint-marked self-loops are not allowed")
            left_mark = _coerce_mark(raw_left_mark)
            right_mark = _coerce_mark(raw_right_mark)

            if positions[left] < positions[right]:
                edge = (left, left_mark, right, right_mark)
                pair = (left, right)
            else:
                edge = (right, right_mark, left, left_mark)
                pair = (right, left)
            if pair in seen_pairs:
                raise GraphValidationError(
                    f"multiple endpoint-marked edges for pair: {pair!r}"
                )
            seen_pairs.add(pair)
            normalized_edges.append(edge)

        normalized_edges.sort(
            key=lambda edge: (positions[edge[0]], positions[edge[2]])
        )
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "edges", tuple(normalized_edges))

    @classmethod
    def from_text(cls, text: str) -> EndpointMarkedGraph:
        """Parse human-readable endpoint-marked graph text."""

        from .text_io import _endpoint_marked_graph_from_text

        parsed = _endpoint_marked_graph_from_text(text)
        return cls(nodes=parsed.nodes, edges=parsed.edges)

    def _positions(self) -> dict[str, int]:
        return node_positions(self.nodes)

    def edge(self, left: str, right: str) -> tuple[Mark, Mark] | None:
        positions = self._positions()
        require_vertex(positions, left)
        require_vertex(positions, right)
        if left == right:
            return None
        for first, first_mark, second, second_mark in self.edges:
            if first == left and second == right:
                return first_mark, second_mark
            if first == right and second == left:
                return second_mark, first_mark
        return None

    def directed_parents(self, vertex: str) -> frozenset[str]:
        positions = self._positions()
        require_vertex(positions, vertex)
        parents: set[str] = set()
        for left, left_mark, right, right_mark in self.edges:
            if right == vertex and left_mark is Mark.TAIL and right_mark is Mark.ARROW:
                parents.add(left)
            if left == vertex and left_mark is Mark.ARROW and right_mark is Mark.TAIL:
                parents.add(right)
        return frozenset(parents)

    def directed_children(self, vertex: str) -> frozenset[str]:
        positions = self._positions()
        require_vertex(positions, vertex)
        children: set[str] = set()
        for left, left_mark, right, right_mark in self.edges:
            if left == vertex and left_mark is Mark.TAIL and right_mark is Mark.ARROW:
                children.add(right)
            if right == vertex and left_mark is Mark.ARROW and right_mark is Mark.TAIL:
                children.add(left)
        return frozenset(children)

    def _directed_closure(
        self, seeds: str | Iterable[str], *, reverse: bool
    ) -> tuple[frozenset[str], frozenset[str]]:
        positions = self._positions()
        seed_set = normalize_seeds(positions, seeds)
        adjacency: dict[str, set[str]] = {node: set() for node in self.nodes}
        for left, left_mark, right, right_mark in self.edges:
            if left_mark is Mark.TAIL and right_mark is Mark.ARROW:
                source, target = left, right
            elif left_mark is Mark.ARROW and right_mark is Mark.TAIL:
                source, target = right, left
            else:
                continue
            if reverse:
                adjacency[target].add(source)
            else:
                adjacency[source].add(target)

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

    def _anterior_closure(
        self, seeds: str | Iterable[str]
    ) -> tuple[frozenset[str], frozenset[str]]:
        positions = self._positions()
        seed_set = normalize_seeds(positions, seeds)
        adjacency: dict[str, list[tuple[str, Mark, Mark]]] = {
            node: [] for node in self.nodes
        }
        for left, left_mark, right, right_mark in self.edges:
            adjacency[left].append((right, left_mark, right_mark))
            adjacency[right].append((left, right_mark, left_mark))

        reached = set(seed_set)
        pending = list(seed_set)
        while pending:
            current = pending.pop()
            for neighbor, current_mark, neighbor_mark in adjacency[current]:
                if neighbor_mark is not Mark.TAIL:
                    continue
                if current_mark not in (Mark.TAIL, Mark.ARROW):
                    continue
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        return seed_set, frozenset(reached)

    def anterior_strict(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        seed_set, reached = self._anterior_closure(seeds)
        return reached - seed_set

    def anterior_reflexive(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        _, reached = self._anterior_closure(seeds)
        return reached

    def bidirected_spouses(self, vertex: str) -> frozenset[str]:
        positions = self._positions()
        require_vertex(positions, vertex)
        spouses: set[str] = set()
        for left, left_mark, right, right_mark in self.edges:
            if left_mark is not Mark.ARROW or right_mark is not Mark.ARROW:
                continue
            if left == vertex:
                spouses.add(right)
            elif right == vertex:
                spouses.add(left)
        return frozenset(spouses)

    def undirected_neighbors(self, vertex: str) -> frozenset[str]:
        positions = self._positions()
        require_vertex(positions, vertex)
        neighbors: set[str] = set()
        for left, left_mark, right, right_mark in self.edges:
            if left_mark is not Mark.TAIL or right_mark is not Mark.TAIL:
                continue
            if left == vertex:
                neighbors.add(right)
            elif right == vertex:
                neighbors.add(left)
        return frozenset(neighbors)

    def _possible_closure(
        self, seeds: str | Iterable[str], *, reverse: bool
    ) -> tuple[frozenset[str], frozenset[str]]:
        positions = self._positions()
        seed_set = normalize_seeds(positions, seeds)
        adjacency: dict[str, list[tuple[str, Mark, Mark]]] = {
            node: [] for node in self.nodes
        }
        for left, left_mark, right, right_mark in self.edges:
            adjacency[left].append((right, left_mark, right_mark))
            adjacency[right].append((left, right_mark, left_mark))

        reached = set(seed_set)
        pending = list(seed_set)
        while pending:
            current = pending.pop()
            for neighbor, current_mark, neighbor_mark in adjacency[current]:
                admissible_mark = neighbor_mark if reverse else current_mark
                if admissible_mark is Mark.ARROW or neighbor in reached:
                    continue
                reached.add(neighbor)
                pending.append(neighbor)
        return seed_set, frozenset(reached)

    def possible_descendants_strict(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        seed_set, reached = self._possible_closure(seeds, reverse=False)
        return reached - seed_set

    def possible_descendants_reflexive(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        _, reached = self._possible_closure(seeds, reverse=False)
        return reached

    def possible_ancestors_strict(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        seed_set, reached = self._possible_closure(seeds, reverse=True)
        return reached - seed_set

    def possible_ancestors_reflexive(
        self, seeds: str | Iterable[str]
    ) -> frozenset[str]:
        _, reached = self._possible_closure(seeds, reverse=True)
        return reached
