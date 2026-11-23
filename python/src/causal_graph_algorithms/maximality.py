from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NoReturn, SupportsIndex, TypeAlias

from ._validation import node_positions, require_vertex
from .ancestral import AncestralGraph, _require_ancestral
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import AncestralDomainError, GraphValidationError


@dataclass(frozen=True, init=False)
class NonMaximalityWitness:
    """Deterministic diagnostic path; this value is not a proof certificate."""

    __slots__ = ("endpoints", "path")

    endpoints: tuple[str, str]
    path: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("NonMaximalityWitness cannot be subclassed")

    def __copy__(self) -> NonMaximalityWitness:
        return self

    def __deepcopy__(
        self, memo: dict[int, object]
    ) -> NonMaximalityWitness:
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("pickle is unsupported; this witness is diagnostic only")

    def __init__(
        self, *, endpoints: tuple[str, str], path: tuple[str, ...]
    ) -> None:
        if isinstance(endpoints, str) or isinstance(path, str):
            raise ValueError("invalid nonmaximality witness structure")
        normalized_endpoints = tuple(endpoints)
        normalized_path = tuple(path)
        all_vertices = (*normalized_endpoints, *normalized_path)
        if (
            len(normalized_endpoints) != 2
            or normalized_endpoints[0] == normalized_endpoints[1]
            or len(normalized_path) < 2
            or normalized_path[0] != normalized_endpoints[0]
            or normalized_path[-1] != normalized_endpoints[1]
            or any(
                not isinstance(vertex, str) or not vertex
                for vertex in all_vertices
            )
            or len(set(normalized_path)) != len(normalized_path)
        ):
            raise ValueError("invalid nonmaximality witness structure")
        object.__setattr__(self, "endpoints", normalized_endpoints)
        object.__setattr__(self, "path", normalized_path)


def _primitive_inducing_path(
    topology: EndpointMarkedGraph, left: str, right: str
) -> tuple[str, ...] | None:
    """Return the canonical path for a validated ancestral topology."""
    if topology.edge(left, right) is not None:
        return left, right

    positions = node_positions(topology.nodes)
    allowed = topology.directed_ancestors_reflexive((left, right))
    adjacency: dict[str, set[str]] = {node: set() for node in allowed}
    for first, first_mark, second, second_mark in topology.edges:
        if (
            first in allowed
            and second in allowed
            and first_mark is Mark.ARROW
            and second_mark is Mark.ARROW
        ):
            adjacency[first].add(second)
            adjacency[second].add(first)

    pending: deque[tuple[str, ...]] = deque(((left,),))
    reached = {left}
    while pending:
        path = pending.popleft()
        current = path[-1]
        for neighbor in sorted(adjacency[current], key=positions.__getitem__):
            if neighbor in reached:
                continue
            next_path = path + (neighbor,)
            if neighbor == right:
                return next_path
            reached.add(neighbor)
            pending.append(next_path)
    return None


def _first_nonmaximality_witness(
    topology: EndpointMarkedGraph,
) -> NonMaximalityWitness | None:
    for left_index, left in enumerate(topology.nodes):
        for right in topology.nodes[left_index + 1 :]:
            if topology.edge(left, right) is not None:
                continue
            path = _primitive_inducing_path(topology, left, right)
            if path is not None:
                return NonMaximalityWitness(
                    endpoints=(left, right), path=path
                )
    return None


@dataclass(frozen=True, init=False)
class MaximalAncestralGraph:
    """Validated Richardson--Spirtes maximal ancestral graph."""

    __slots__ = ("topology",)

    topology: EndpointMarkedGraph

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MaximalAncestralGraph cannot be subclassed")

    def __copy__(self) -> MaximalAncestralGraph:
        return self

    def __deepcopy__(
        self, memo: dict[int, object]
    ) -> MaximalAncestralGraph:
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("pickle is unsupported; use canonical JSON")

    def __init__(self, topology: EndpointMarkedGraph) -> None:
        if not isinstance(topology, EndpointMarkedGraph):
            raise GraphValidationError(
                "maximal ancestral view requires endpoint-marked topology"
            )
        _require_ancestral(topology)
        witness = _first_nonmaximality_witness(topology)
        if witness is not None:
            raise GraphValidationError(
                "ancestral graph is not maximal; primitive inducing path "
                f"{witness.path!r} joins nonadjacent endpoints "
                f"{witness.endpoints!r}"
            )
        object.__setattr__(self, "topology", topology)

    @classmethod
    def from_graph(
        cls, graph: EndpointMarkedGraph
    ) -> MaximalAncestralGraph:
        return cls(graph)

    @classmethod
    def from_text(cls, text: str) -> MaximalAncestralGraph:
        """Parse endpoint text and validate a maximal ancestral graph."""

        from .text_io import _endpoint_marked_graph_from_text

        return cls.from_graph(_endpoint_marked_graph_from_text(text))

    @classmethod
    def from_ancestral(
        cls, graph: AncestralGraph
    ) -> MaximalAncestralGraph:
        if not isinstance(graph, AncestralGraph):
            raise AncestralDomainError(
                "maximal ancestral view requires an ancestral proof view"
            )
        return cls(graph.topology)

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


AncestralView: TypeAlias = AncestralGraph | MaximalAncestralGraph


def _view_topology(graph: AncestralView) -> EndpointMarkedGraph:
    if isinstance(graph, (AncestralGraph, MaximalAncestralGraph)):
        return graph.topology
    raise AncestralDomainError(
        "maximality operations require an ancestral proof view"
    )


def find_primitive_inducing_path(
    graph: AncestralView, left: str, right: str
) -> tuple[str, ...] | None:
    """Find the canonical primitive inducing path between distinct vertices.

    An existing edge is the two-vertex trivial inducing path.  Only paths
    between nonadjacent vertices diagnose nonmaximality.
    """
    topology = _view_topology(graph)
    positions = node_positions(topology.nodes)
    require_vertex(positions, left)
    require_vertex(positions, right)
    if left == right:
        raise AncestralDomainError(
            "primitive inducing path endpoints must be distinct"
        )
    return _primitive_inducing_path(topology, left, right)


def find_nonmaximality_witness(
    graph: AncestralView,
) -> NonMaximalityWitness | None:
    """Return the first canonical witness between nonadjacent vertices."""
    return _first_nonmaximality_witness(_view_topology(graph))


def maximal_ancestral_extension(
    graph: AncestralView,
) -> MaximalAncestralGraph:
    """Return the unique Richardson--Spirtes maximal ancestral extension."""
    if isinstance(graph, MaximalAncestralGraph):
        return graph
    if not isinstance(graph, AncestralGraph):
        raise AncestralDomainError(
            "maximal ancestral extension requires an ancestral proof view"
        )

    topology = graph.topology
    added: list[tuple[str, Mark, str, Mark]] = []
    for left_index, left in enumerate(topology.nodes):
        for right in topology.nodes[left_index + 1 :]:
            if topology.edge(left, right) is None and (
                _primitive_inducing_path(topology, left, right) is not None
            ):
                added.append((left, Mark.ARROW, right, Mark.ARROW))
    extension = EndpointMarkedGraph(
        nodes=topology.nodes,
        edges=(*topology.edges, *added),
    )
    return MaximalAncestralGraph.from_graph(extension)
