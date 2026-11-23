from __future__ import annotations

import itertools
import typing
from dataclasses import dataclass
from typing import NoReturn, Self, SupportsIndex

from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import PAGDomainError
from .maximality import MaximalAncestralGraph


_Conclusion = tuple[str, str, Mark]


class _PAGState:
    """Mutable private state for Zhang's monotone orientation rules."""

    nodes: tuple[str, ...]
    positions: dict[str, int]
    neighbors: dict[str, set[str]]
    marks: dict[tuple[str, str], Mark]
    separators: dict[frozenset[str], frozenset[str]]

    def __init__(self, graph: MaximalAncestralGraph) -> None:
        topology = graph.topology
        circle_topology = EndpointMarkedGraph(
            nodes=topology.nodes,
            edges=(
                (left, Mark.CIRCLE, right, Mark.CIRCLE)
                for left, _, right, _ in topology.edges
            ),
        )
        initialized = type(self).from_topology(
            circle_topology, separators={}
        )
        self.__dict__.update(initialized.__dict__)

        for left, right in itertools.combinations(self.nodes, 2):
            if not self.adjacent(left, right):
                self.separators[frozenset((left, right))] = (
                    graph.anterior_strict((left, right))
                )

    @classmethod
    def from_topology(
        cls,
        topology: EndpointMarkedGraph,
        *,
        separators: dict[frozenset[str], frozenset[str]],
    ) -> Self:
        """Build orientation state from exact endpoint marks."""
        value = object.__new__(cls)
        value.nodes = topology.nodes
        value.positions = {
            node: position for position, node in enumerate(value.nodes)
        }
        value.neighbors = {node: set() for node in value.nodes}
        value.marks = {}
        for left, left_mark, right, right_mark in topology.edges:
            value.neighbors[left].add(right)
            value.neighbors[right].add(left)
            value.marks[(left, right)] = left_mark
            value.marks[(right, left)] = right_mark
        value.separators = dict(separators)
        return value

    def clone(self) -> Self:
        value = object.__new__(type(self))
        value.nodes = self.nodes
        value.positions = self.positions
        value.neighbors = self.neighbors
        value.marks = self.marks.copy()
        value.separators = self.separators
        return value

    def adjacent(self, left: str, right: str) -> bool:
        return right in self.neighbors[left]

    def ordered_neighbors(self, node: str) -> tuple[str, ...]:
        return tuple(
            sorted(self.neighbors[node], key=self.positions.__getitem__)
        )

    def mark(self, at: str, other: str) -> Mark | None:
        return self.marks.get((at, other))

    def separator(self, left: str, right: str) -> frozenset[str]:
        return self.separators[frozenset((left, right))]

    def apply(
        self, rule: int, conclusions: list[_Conclusion]
    ) -> bool:
        desired: dict[tuple[str, str], Mark] = {}
        for at, other, wanted in conclusions:
            key = (at, other)
            previous = desired.get(key)
            if previous is not None and previous is not wanted:
                raise RuntimeError(
                    "PAG orientation invariant failed in "
                    f"R{rule}: conflicting conclusions at {key!r}"
                )
            desired[key] = wanted

        ordered = sorted(
            desired.items(),
            key=lambda item: (
                self.positions[item[0][0]],
                self.positions[item[0][1]],
            ),
        )
        for (at, other), wanted in ordered:
            current = self.mark(at, other)
            if current is None:
                raise RuntimeError(
                    "PAG orientation invariant failed in "
                    f"R{rule}: missing adjacency {(at, other)!r}"
                )
            if current is not Mark.CIRCLE and current is not wanted:
                raise RuntimeError(
                    "PAG orientation invariant failed in "
                    f"R{rule}: endpoint {(at, other)!r} is "
                    f"{current.value}, wanted {wanted.value}"
                )

        changed = False
        for key, wanted in ordered:
            if self.marks[key] is Mark.CIRCLE and wanted is not Mark.CIRCLE:
                self.marks[key] = wanted
                changed = True
        return changed

    def topology(self) -> EndpointMarkedGraph:
        edges = (
            (
                left,
                self.marks[(left, right)],
                right,
                self.marks[(right, left)],
            )
            for left, right in itertools.combinations(self.nodes, 2)
            if self.adjacent(left, right)
        )
        return EndpointMarkedGraph(nodes=self.nodes, edges=edges)


def _r0(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for middle in state.nodes:
        neighbors = state.ordered_neighbors(middle)
        for first, third in itertools.combinations(neighbors, 2):
            if state.adjacent(first, third):
                continue
            if middle not in state.separator(first, third):
                conclusions.append((middle, first, Mark.ARROW))
                conclusions.append((middle, third, Mark.ARROW))
    return conclusions


def _r1(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for middle in state.nodes:
        neighbors = state.ordered_neighbors(middle)
        for first in neighbors:
            if state.mark(middle, first) is not Mark.ARROW:
                continue
            for third in neighbors:
                if (
                    third == first
                    or state.mark(middle, third) is not Mark.CIRCLE
                    or state.adjacent(first, third)
                ):
                    continue
                conclusions.extend(
                    (
                        (middle, third, Mark.TAIL),
                        (third, middle, Mark.ARROW),
                    )
                )
    return conclusions


def _r2(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for first in state.nodes:
        for third in state.ordered_neighbors(first):
            if state.mark(third, first) is not Mark.CIRCLE:
                continue
            common = state.neighbors[first] & state.neighbors[third]
            for middle in sorted(common, key=state.positions.__getitem__):
                if state.mark(third, middle) is not Mark.ARROW:
                    continue
                first_to_middle = (
                    state.mark(first, middle) is Mark.TAIL
                    and state.mark(middle, first) is Mark.ARROW
                )
                first_into_middle = (
                    state.mark(middle, first) is Mark.ARROW
                    and state.mark(middle, third) is Mark.TAIL
                )
                if first_to_middle or first_into_middle:
                    conclusions.append((third, first, Mark.ARROW))
                    break
    return conclusions


def _r3(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for middle in state.nodes:
        for fourth in state.ordered_neighbors(middle):
            if state.mark(middle, fourth) is not Mark.CIRCLE:
                continue
            common = tuple(
                sorted(
                    state.neighbors[middle] & state.neighbors[fourth],
                    key=state.positions.__getitem__,
                )
            )
            for first, third in itertools.combinations(common, 2):
                if state.adjacent(first, third):
                    continue
                if (
                    state.mark(middle, first) is Mark.ARROW
                    and state.mark(fourth, first) is Mark.CIRCLE
                    and state.mark(middle, third) is Mark.ARROW
                    and state.mark(fourth, third) is Mark.CIRCLE
                ):
                    conclusions.append((middle, fourth, Mark.ARROW))
    return conclusions


def _directed(state: _PAGState, source: str, target: str) -> bool:
    return state.mark(source, target) is Mark.TAIL and (
        state.mark(target, source) is Mark.ARROW
    )


def _discriminating_witnesses(
    state: _PAGState, middle: str, endpoint: str
) -> tuple[tuple[str, str], ...]:
    """Return (outside endpoint, last collider) witnesses for R4."""

    witnesses: set[tuple[str, str]] = set()

    for last_collider in state.ordered_neighbors(middle):
        if last_collider == endpoint:
            continue
        if state.mark(last_collider, middle) is not Mark.ARROW:
            continue
        if not _directed(state, last_collider, endpoint):
            continue
        reached = {last_collider}
        pending = [last_collider]
        while pending:
            head = pending.pop()
            for outside in state.ordered_neighbors(head):
                if outside in reached or outside in (middle, endpoint):
                    continue
                if state.mark(head, outside) is not Mark.ARROW:
                    continue
                if not state.adjacent(outside, endpoint):
                    witnesses.add((outside, last_collider))

            for previous in reversed(state.ordered_neighbors(head)):
                if previous in reached or previous in (middle, endpoint):
                    continue
                if not (
                    state.mark(head, previous) is Mark.ARROW
                    and state.mark(previous, head) is Mark.ARROW
                    and _directed(state, previous, endpoint)
                ):
                    continue
                reached.add(previous)
                pending.append(previous)

    return tuple(
        sorted(
            witnesses,
            key=lambda pair: (
                state.positions[pair[0]],
                state.positions[pair[1]],
            ),
        )
    )


def _r4(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for middle in state.nodes:
        for endpoint in state.ordered_neighbors(middle):
            if state.mark(middle, endpoint) is not Mark.CIRCLE:
                continue
            for outside, last_collider in _discriminating_witnesses(
                state, middle, endpoint
            ):
                if middle in state.separator(outside, endpoint):
                    conclusions.extend(
                        (
                            (middle, endpoint, Mark.TAIL),
                            (endpoint, middle, Mark.ARROW),
                        )
                    )
                else:
                    conclusions.extend(
                        (
                            (last_collider, middle, Mark.ARROW),
                            (middle, last_collider, Mark.ARROW),
                            (middle, endpoint, Mark.ARROW),
                            (endpoint, middle, Mark.ARROW),
                        )
                    )
    return conclusions


def _r5_circle_paths(
    state: _PAGState, first: str, endpoint: str
) -> tuple[tuple[str, ...], ...]:
    paths: list[tuple[str, ...]] = []
    pending: list[tuple[str, ...]] = [
        (first, second)
        for second in reversed(state.ordered_neighbors(first))
        if second != endpoint
        and state.mark(first, second) is Mark.CIRCLE
        and state.mark(second, first) is Mark.CIRCLE
    ]
    while pending:
        path = pending.pop()
        current = path[-1]
        if (
            len(path) >= 3
            and state.mark(current, endpoint) is Mark.CIRCLE
            and state.mark(endpoint, current) is Mark.CIRCLE
            and not state.adjacent(first, current)
            and not state.adjacent(path[1], endpoint)
            and not state.adjacent(path[-2], endpoint)
        ):
            paths.append(path + (endpoint,))

        for following in reversed(state.ordered_neighbors(current)):
            if following == endpoint or following in path:
                continue
            if not (
                state.mark(current, following) is Mark.CIRCLE
                and state.mark(following, current) is Mark.CIRCLE
            ):
                continue
            if state.adjacent(path[-2], following):
                continue
            pending.append(path + (following,))
    return tuple(paths)


def _r5(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for first, endpoint in itertools.combinations(state.nodes, 2):
        if not (
            state.mark(first, endpoint) is Mark.CIRCLE
            and state.mark(endpoint, first) is Mark.CIRCLE
        ):
            continue
        paths = _r5_circle_paths(state, first, endpoint)
        if not paths:
            continue
        conclusions.extend(
            (
                (first, endpoint, Mark.TAIL),
                (endpoint, first, Mark.TAIL),
            )
        )
        for path in paths:
            for left, right in itertools.pairwise(path):
                conclusions.extend(
                    (
                        (left, right, Mark.TAIL),
                        (right, left, Mark.TAIL),
                    )
                )
    return conclusions


def _r6(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for middle in state.nodes:
        has_undirected_edge = any(
            state.mark(middle, neighbor) is Mark.TAIL
            and state.mark(neighbor, middle) is Mark.TAIL
            for neighbor in state.ordered_neighbors(middle)
        )
        if not has_undirected_edge:
            continue
        for endpoint in state.ordered_neighbors(middle):
            if state.mark(middle, endpoint) is Mark.CIRCLE:
                conclusions.append((middle, endpoint, Mark.TAIL))
    return conclusions


def _r7(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for middle in state.nodes:
        neighbors = state.ordered_neighbors(middle)
        for first in neighbors:
            if not (
                state.mark(first, middle) is Mark.TAIL
                and state.mark(middle, first) is Mark.CIRCLE
            ):
                continue
            for endpoint in neighbors:
                if (
                    endpoint != first
                    and state.mark(middle, endpoint) is Mark.CIRCLE
                    and not state.adjacent(first, endpoint)
                ):
                    conclusions.append((middle, endpoint, Mark.TAIL))
    return conclusions


def _r8(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for first in state.nodes:
        for endpoint in state.ordered_neighbors(first):
            if not (
                state.mark(first, endpoint) is Mark.CIRCLE
                and state.mark(endpoint, first) is Mark.ARROW
            ):
                continue
            common = state.neighbors[first] & state.neighbors[endpoint]
            for middle in sorted(common, key=state.positions.__getitem__):
                if not _directed(state, middle, endpoint):
                    continue
                if state.mark(first, middle) is Mark.TAIL and state.mark(
                    middle, first
                ) in (Mark.ARROW, Mark.CIRCLE):
                    conclusions.append((first, endpoint, Mark.TAIL))
                    break
    return conclusions


def _zhang_pd_step(state: _PAGState, source: str, target: str) -> bool:
    """Return Zhang's ordered potentially-directed edge predicate."""
    return state.adjacent(source, target) and (
        state.mark(source, target) is not Mark.ARROW
        and state.mark(target, source) is not Mark.TAIL
    )


def _uncovered_pd_reachable(
    state: _PAGState, first: str, second: str
) -> frozenset[str]:
    if not _zhang_pd_step(state, first, second):
        return frozenset()

    reached: set[str] = set()
    pending: list[tuple[str, ...]] = [(first, second)]
    while pending:
        path = pending.pop()
        current = path[-1]
        reached.add(current)
        for following in reversed(state.ordered_neighbors(current)):
            if following in path:
                continue
            if not _zhang_pd_step(state, current, following):
                continue
            if state.adjacent(path[-2], following):
                continue
            pending.append(path + (following,))
    return frozenset(reached)


def _r9(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    reach: dict[tuple[str, str], frozenset[str]] = {}
    for first in state.nodes:
        for endpoint in state.ordered_neighbors(first):
            if not (
                state.mark(first, endpoint) is Mark.CIRCLE
                and state.mark(endpoint, first) is Mark.ARROW
            ):
                continue
            for second in state.ordered_neighbors(first):
                if second == endpoint or state.adjacent(second, endpoint):
                    continue
                key = (first, second)
                if key not in reach:
                    reach[key] = _uncovered_pd_reachable(
                        state, first, second
                    )
                if endpoint in reach[key]:
                    conclusions.append((first, endpoint, Mark.TAIL))
                    break
    return conclusions


def _r10(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    reach: dict[tuple[str, str], frozenset[str]] = {}
    for first in state.nodes:
        for endpoint in state.ordered_neighbors(first):
            if not (
                state.mark(first, endpoint) is Mark.CIRCLE
                and state.mark(endpoint, first) is Mark.ARROW
            ):
                continue
            parents = frozenset(
                node
                for node in state.ordered_neighbors(endpoint)
                if _directed(state, node, endpoint)
            )
            if len(parents) < 2:
                continue
            seconds = state.ordered_neighbors(first)
            parent_reach: dict[str, frozenset[str]] = {}
            for second in seconds:
                key = (first, second)
                if key not in reach:
                    reach[key] = _uncovered_pd_reachable(
                        state, first, second
                    )
                parent_reach[second] = reach[key] & parents

            fired = False
            for left_index, left_second in enumerate(seconds):
                if not parent_reach[left_second]:
                    continue
                for right_second in seconds[left_index + 1 :]:
                    if (
                        state.adjacent(left_second, right_second)
                        or not parent_reach[right_second]
                    ):
                        continue
                    if any(
                        left_parent != right_parent
                        for left_parent in parent_reach[left_second]
                        for right_parent in parent_reach[right_second]
                    ):
                        conclusions.append((first, endpoint, Mark.TAIL))
                        fired = True
                        break
                if fired:
                    break
    return conclusions


_RULES = (_r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _r10)

_CAUSAL_RULES = (
    (1, _r1),
    (2, _r2),
    (3, _r3),
    (4, _r4),
    (8, _r8),
    (9, _r9),
    (10, _r10),
)


def _closed_pag_topology(
    graph: MaximalAncestralGraph,
) -> EndpointMarkedGraph:
    state = _PAGState(graph)
    state.apply(0, _r0(state))

    maximum_changing_passes = len(state.marks) + 1
    for _ in range(maximum_changing_passes):
        changed = False
        for rule_number, rule in enumerate(_RULES, start=1):
            changed = state.apply(rule_number, rule(state)) or changed
        if not changed:
            return state.topology()
    raise RuntimeError("closed PAG orientation failed to reach a fixed point")


def _causal_pag_topology(
    graph: MaximalAncestralGraph,
) -> EndpointMarkedGraph:
    state = _PAGState(graph)
    state.apply(0, _r0(state))

    maximum_changing_passes = len(state.marks) + 1
    for _ in range(maximum_changing_passes):
        changed = False
        for rule_number, rule in _CAUSAL_RULES:
            changed = state.apply(rule_number, rule(state)) or changed
        if not changed:
            return state.topology()
    raise RuntimeError("causal PAG orientation failed to reach a fixed point")


@typing.final
@dataclass(frozen=True, init=False)
class CausalPAG:
    """No-selection causal PAG proof view for one MAG equivalence class."""

    __slots__ = ("topology",)

    topology: EndpointMarkedGraph

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CausalPAG cannot be subclassed")

    def __copy__(self) -> CausalPAG:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> CausalPAG:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("pickle is unsupported; use the source MAG")

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("CausalPAG values are factory-only; use from_mag")

    @classmethod
    def from_mag(cls, graph: MaximalAncestralGraph) -> CausalPAG:
        if not isinstance(graph, MaximalAncestralGraph):
            raise PAGDomainError(
                "causal PAG construction requires a MAG proof view"
            )
        if any(
            left_mark is Mark.TAIL and right_mark is Mark.TAIL
            for _, left_mark, _, right_mark in graph.topology.edges
        ):
            raise PAGDomainError(
                "causal PAG construction requires a MAG without undirected edges"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "topology", _causal_pag_topology(graph))
        return value


@dataclass(frozen=True, init=False)
class ClosedPAG:
    """Zhang's maximally informative PAG for one FULL MAG MEC."""

    __slots__ = ("topology",)

    topology: EndpointMarkedGraph

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ClosedPAG cannot be subclassed")

    def __copy__(self) -> ClosedPAG:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> ClosedPAG:
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("pickle is unsupported; use the source MAG")

    def __init__(self) -> None:
        raise TypeError("ClosedPAG values are factory-only; use from_mag")

    @classmethod
    def from_mag(cls, graph: MaximalAncestralGraph) -> ClosedPAG:
        if not isinstance(graph, MaximalAncestralGraph):
            raise PAGDomainError(
                "closed PAG construction requires a MAG proof view"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "topology", _closed_pag_topology(graph))
        return value
