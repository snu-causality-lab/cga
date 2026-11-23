"""Distinct proof views for partially directed graphs and DAG classes."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from itertools import combinations
from typing import Literal, TypeAlias, final

from ._immutable import _ImmutableValue
from ._validation import normalize_nodes
from .ancestral import _directed_edges, _directed_part_is_acyclic
from .directed_bidirected import DAG, DirectedBidirectedGraph
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import GraphValidationError


class PDAGDomainError(ValueError):
    """A PDAG operation received invalid topology or an invalid proof view."""


def _knowledge_edges(values: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    try:
        edges = tuple(values)
        result = set()
        for edge in edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise PDAGDomainError("background edges must be endpoint pairs")
            labels = normalize_nodes(edge)
            result.add((labels[0], labels[1]))
        return tuple(sorted(result))
    except (TypeError, GraphValidationError) as error:
        raise PDAGDomainError("invalid background edge labels") from error


@final
@dataclass(frozen=True, slots=True, init=False)
class BackgroundKnowledge(_ImmutableValue):
    """Required and forbidden directed edges, interpreted on the original DAG class."""
    required: tuple[tuple[str, str], ...]
    forbidden: tuple[tuple[str, str], ...]

    def __init__(self, *, required: Iterable[tuple[str, str]] = (),
                 forbidden: Iterable[tuple[str, str]] = ()) -> None:
        object.__setattr__(self, "required", _knowledge_edges(required))
        object.__setattr__(self, "forbidden", _knowledge_edges(forbidden))

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("BackgroundKnowledge cannot be subclassed")


_TOKEN = object()
_Collider: TypeAlias = tuple[frozenset[str], str]


def _skeleton(raw: EndpointMarkedGraph) -> frozenset[frozenset[str]]:
    return frozenset(frozenset((a, b)) for a, _, b, _ in raw.edges)


def _colliders(raw: EndpointMarkedGraph) -> frozenset[_Collider]:
    skeleton = _skeleton(raw)
    parents = {v: raw.directed_parents(v) for v in raw.nodes}
    return frozenset((frozenset((a, c)), b) for b in raw.nodes
                     for a, c in combinations(parents[b], 2)
                     if frozenset((a, c)) not in skeleton)


@final
@dataclass(frozen=True, slots=True)
class PDAG(_ImmutableValue):
    """Acyclic directed part with directed/undirected edges; may be unextendable."""
    topology: EndpointMarkedGraph

    def __post_init__(self) -> None:
        raw = self.topology
        if not isinstance(raw, EndpointMarkedGraph):
            raise PDAGDomainError("PDAG requires endpoint-marked topology")
        for _, a, _, b in raw.edges:
            if Mark.CIRCLE in (a, b) or (a is Mark.ARROW and b is Mark.ARROW):
                raise PDAGDomainError("PDAG permits only directed and undirected edges")
        if not _directed_part_is_acyclic(raw.nodes, _directed_edges(raw)):
            raise PDAGDomainError("PDAG contains a directed cycle")

    @classmethod
    def from_graph(cls, graph: EndpointMarkedGraph) -> PDAG:
        return cls(graph)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("PDAG cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class CPDAG(_ImmutableValue):
    """The full essential graph of a causally sufficient DAG Markov class."""
    topology: EndpointMarkedGraph
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("CPDAG values are factory-created")

    @classmethod
    def from_dag(cls, graph: DAG) -> CPDAG:
        if not isinstance(graph, DAG):
            raise PDAGDomainError("CPDAG construction requires a DAG view")
        raw = graph.topology
        skeleton = frozenset(frozenset(e) for e in raw.directed)
        compelled: set[tuple[str, str]] = set()
        for b in raw.nodes:
            for a, c in combinations(raw.directed_parents(b), 2):
                if frozenset((a, c)) not in skeleton:
                    compelled.update(((a, b), (c, b)))
        edges = []
        for a, b in combinations(raw.nodes, 2):
            if frozenset((a, b)) in skeleton:
                edges.append((a, Mark.ARROW if (b, a) in compelled else Mark.TAIL,
                              b, Mark.ARROW if (a, b) in compelled else Mark.TAIL))
        from .meek import _close_meek_topology
        closed = _close_meek_topology(EndpointMarkedGraph(nodes=raw.nodes, edges=edges))
        if closed is None:
            raise RuntimeError("DAG pattern produced contradictory Meek orientations")
        return cls(closed, _token=_TOKEN)

    @classmethod
    def from_graph(cls, graph: EndpointMarkedGraph) -> CPDAG:
        pdag = PDAG.from_graph(graph)
        extension = find_consistent_dag_extension(pdag)
        if isinstance(extension, NoConsistentExtension):
            raise PDAGDomainError("CPDAG topology has no consistent DAG extension")
        result = cls.from_dag(extension)
        if result.topology != graph:
            raise PDAGDomainError("topology is not the full essential graph of its DAG class")
        return result

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CPDAG cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class MPDAG(_ImmutableValue):
    """Maximal invariant orientations for the stored consistent extension set."""
    topology: EndpointMarkedGraph
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("MPDAG values are factory-created")

    @classmethod
    def from_graph(cls, graph: EndpointMarkedGraph) -> MPDAG:
        from .meek import meek_closure
        result = meek_closure(PDAG.from_graph(graph))
        if isinstance(result, NoConsistentExtension) or result.topology != graph:
            raise PDAGDomainError("topology is not an extendable maximally oriented PDAG")
        return result

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MPDAG cannot be subclassed")


PDAGView: TypeAlias = PDAG | CPDAG | MPDAG


@final
@dataclass(frozen=True, slots=True)
class NoConsistentExtension(_ImmutableValue):
    """Empty extension set, distinct from malformed input or operational failure."""
    graph: PDAGView
    reason: Literal["input_unextendable", "background_inconsistent"]
    knowledge: BackgroundKnowledge | None = None
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("extension results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("NoConsistentExtension cannot be subclassed")


def _consistent_extension(
    raw: EndpointMarkedGraph, *, original_colliders: frozenset[_Collider] | None = None,
) -> DAG | None:
    """Dor--Tarsi potential-sink elimination, preserving all original v-structures."""
    expected = _colliders(raw) if original_colliders is None else original_colliders
    if not _colliders(raw) <= expected:
        return None
    directed = set(_directed_edges(raw))
    undirected = {frozenset((a, b)) for a, x, b, y in raw.edges
                  if x is Mark.TAIL and y is Mark.TAIL}
    remaining = set(raw.nodes)
    while remaining:
        selected = None
        for v in raw.nodes:
            if v not in remaining or any(a == v and b in remaining for a, b in directed):
                continue
            neighbors = {next(iter(e - {v})) for e in undirected if v in e and e <= remaining}
            parents = {a for a, b in directed if b == v and a in remaining}
            if all(frozenset((u, w)) in undirected or (u, w) in directed or (w, u) in directed
                   for u in neighbors for w in (neighbors | parents) - {u}):
                selected = v
                directed.update((u, v) for u in neighbors)
                break
        if selected is None:
            return None
        remaining.remove(selected)
    result = DAG.from_graph(DirectedBidirectedGraph(nodes=raw.nodes, directed=directed))
    endpoint = EndpointMarkedGraph(nodes=raw.nodes,
        edges=((a, Mark.TAIL, b, Mark.ARROW) for a, b in directed))
    if _colliders(endpoint) != expected:
        return None
    return result


def find_consistent_dag_extension(graph: PDAGView) -> DAG | NoConsistentExtension:
    """Return one DAG with the same skeleton and unshielded colliders, if one exists."""
    if not isinstance(graph, (PDAG, CPDAG, MPDAG)):
        raise PDAGDomainError("consistent extension requires a PDAG, CPDAG, or MPDAG view")
    result = _consistent_extension(graph.topology)
    return (result if result is not None else
            NoConsistentExtension(graph, "input_unextendable", _token=_TOKEN))
