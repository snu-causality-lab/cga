from __future__ import annotations

import typing
from collections import deque
from dataclasses import InitVar, dataclass
from typing import NoReturn, SupportsIndex

from ._immutable import _ImmutableValue
from ._validation import node_positions, require_vertex
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import PAGDomainError
from .pag import CausalPAG
from .maximality import MaximalAncestralGraph
from .errors import AncestralDomainError


_VISIBILITY_FACTORY_TOKEN = object()


@typing.final
@dataclass(frozen=True, init=False)
class DirectedEdgeVisibilityWitness:
    """A canonical path witnessing visibility of one directed PAG edge."""

    __slots__ = ("graph", "source", "target", "path")

    graph: CausalPAG
    source: str
    target: str
    path: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("DirectedEdgeVisibilityWitness cannot be subclassed")

    def __copy__(self) -> DirectedEdgeVisibilityWitness:
        return self

    def __deepcopy__(
        self, memo: dict[int, object]
    ) -> DirectedEdgeVisibilityWitness:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError(
            "pickle is unsupported for directed-edge visibility witnesses"
        )

    def __init__(
        self,
        factory_token: object,
        *,
        graph: CausalPAG,
        source: str,
        target: str,
        path: tuple[str, ...],
    ) -> None:
        if factory_token is not _VISIBILITY_FACTORY_TOKEN:
            raise TypeError(
                "DirectedEdgeVisibilityWitness values are factory-created"
            )
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "path", path)


def _directed_edge_visibility_path(
    topology: EndpointMarkedGraph,
    source: str,
    target: str,
) -> tuple[str, ...] | None:
    positions = node_positions(topology.nodes)
    neighbor_lists: dict[str, list[str]] = {
        node: [] for node in topology.nodes
    }
    marks: dict[tuple[str, str], Mark] = {}
    for left, left_mark, right, right_mark in topology.edges:
        neighbor_lists[left].append(right)
        neighbor_lists[right].append(left)
        marks[(left, right)] = left_mark
        marks[(right, left)] = right_mark
    neighbors: dict[str, tuple[str, ...]] = {
        node: tuple(sorted(values, key=positions.__getitem__))
        for node, values in neighbor_lists.items()
    }

    parents = topology.directed_parents(target)
    distances = {source: 0}
    pending: deque[str] = deque((source,))
    while pending:
        current = pending.popleft()
        for neighbor in neighbors[current]:
            if neighbor in distances or neighbor not in parents:
                continue
            if not (
                marks[(current, neighbor)] is Mark.ARROW
                and marks[(neighbor, current)] is Mark.ARROW
            ):
                continue
            distances[neighbor] = distances[current] + 1
            pending.append(neighbor)

    suffixes: dict[str, tuple[str, ...]] = {source: (source,)}
    ordered_reached = sorted(
        distances,
        key=lambda node: (distances[node], positions[node]),
    )
    for current in ordered_reached:
        if current == source:
            continue
        predecessors = tuple(
            neighbor
            for neighbor in neighbors[current]
            if distances.get(neighbor) == distances[current] - 1
            and marks[(current, neighbor)] is Mark.ARROW
            and marks[(neighbor, current)] is Mark.ARROW
        )
        if not predecessors:
            raise RuntimeError(
                "directed-edge visibility search lost a shortest-path predecessor"
            )
        predecessor = min(predecessors, key=positions.__getitem__)
        suffixes[current] = (current, *suffixes[predecessor])

    paths: list[tuple[str, ...]] = []
    for external in topology.nodes:
        if external in (source, target) or target in neighbors[external]:
            continue
        for attachment in neighbors[external]:
            suffix = suffixes.get(attachment)
            if suffix is None or marks[(attachment, external)] is not Mark.ARROW:
                continue
            paths.append((external, *suffix))

    if not paths:
        return None
    return min(
        paths,
        key=lambda path: (
            len(path),
            tuple(positions[node] for node in path),
        ),
    )


def find_directed_edge_visibility_witness(
    pag: CausalPAG,
    source: str,
    target: str,
) -> DirectedEdgeVisibilityWitness | None:
    if not isinstance(pag, CausalPAG):
        raise PAGDomainError(
            "directed-edge visibility requires a causal PAG proof view"
        )

    topology = pag.topology
    positions = node_positions(topology.nodes)
    source = require_vertex(positions, source)
    target = require_vertex(positions, target)
    if source == target:
        raise PAGDomainError(
            "directed-edge visibility endpoints must be distinct"
        )
    if topology.edge(source, target) != (Mark.TAIL, Mark.ARROW):
        raise PAGDomainError(
            "directed-edge visibility requires a directed edge from source to target"
        )

    path = _directed_edge_visibility_path(topology, source, target)
    if path is None:
        return None
    return DirectedEdgeVisibilityWitness(
        _VISIBILITY_FACTORY_TOKEN,
        graph=pag,
        source=source,
        target=target,
        path=path,
    )


@typing.final
@dataclass(frozen=True, slots=True)
class MAGDirectedEdgeVisibilityWitness(_ImmutableValue):
    """A no-selection MAG-bound visibility path, distinct from a PAG witness."""
    graph: MaximalAncestralGraph
    source: str
    target: str
    path: tuple[str, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _VISIBILITY_FACTORY_TOKEN:
            raise TypeError("MAG visibility witnesses are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MAGDirectedEdgeVisibilityWitness cannot be subclassed")


def find_mag_directed_edge_visibility_witness(
    mag: MaximalAncestralGraph, source: str, target: str,
) -> MAGDirectedEdgeVisibilityWitness | None:
    """Return a visibility witness in a validated MAG without selection edges."""
    if not isinstance(mag, MaximalAncestralGraph):
        raise AncestralDomainError("MAG visibility requires a MAG proof view")
    raw = mag.topology
    if any(a is Mark.TAIL and b is Mark.TAIL for _, a, _, b in raw.edges):
        raise AncestralDomainError("MAG visibility requires a no-selection MAG")
    try:
        source = require_vertex(node_positions(raw.nodes), source)
        target = require_vertex(node_positions(raw.nodes), target)
    except ValueError as error:
        raise AncestralDomainError(str(error)) from error
    if raw.edge(source, target) != (Mark.TAIL, Mark.ARROW):
        raise AncestralDomainError("visibility requires a directed edge from source to target")
    path = _directed_edge_visibility_path(raw, source, target)
    return (None if path is None else MAGDirectedEdgeVisibilityWitness(
        mag, source, target, path, _token=_VISIBILITY_FACTORY_TOKEN))
