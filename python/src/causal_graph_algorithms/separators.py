"""Constrained m-separators in validated fully specified graphs.

The ancestral reduction and two boundary searches implement the FindMinSep
construction of van der Zander and Liśkiewicz (UAI 2019, Proposition 5.2).
This implementation explicitly constructs the augmented graph; it does not claim
the paper's linear-time bound. Bow-capable ADMGs use an explicit latent DAG.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from typing import TypeAlias, final

from ._immutable import _ImmutableValue
from ._validation import node_positions, normalize_seeds
from .ancestral import AncestralGraph
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .endpoint_marked import EndpointMarkedGraph
from .errors import GraphValidationError, SeparationDomainError
from .maximality import MaximalAncestralGraph
from .separation import SeparationView, _edge_atoms, _view_topology, m_separated_sets


class SeparatorDomainError(ValueError):
    """Malformed query or graph without the required separation semantics."""


_TOKEN = object()


@final
@dataclass(frozen=True, slots=True)
class SeparatorSet(_ImmutableValue):
    """A total separating set satisfying the stored inclusion constraints."""
    graph: SeparationView
    left: tuple[str, ...]
    right: tuple[str, ...]
    required: tuple[str, ...]
    allowed: tuple[str, ...]
    vertices: tuple[str, ...]
    minimal: bool
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("separator results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("SeparatorSet cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class SeparatorNotFound(_ImmutableValue):
    """No m-separator exists between required and allowed sets."""
    graph: SeparationView
    left: tuple[str, ...]
    right: tuple[str, ...]
    required: tuple[str, ...]
    allowed: tuple[str, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("separator results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("SeparatorNotFound cannot be subclassed")


SeparatorResult: TypeAlias = SeparatorSet | SeparatorNotFound


def _ordered(nodes: tuple[str, ...], values: Iterable[str]) -> tuple[str, ...]:
    selected = frozenset(values)
    return tuple(n for n in nodes if n in selected)


def _sets(
    graph: SeparationView, left: str | Iterable[str], right: str | Iterable[str],
    required: str | Iterable[str], allowed: str | Iterable[str] | None,
) -> tuple[tuple[str, ...], frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    try:
        nodes = _view_topology(graph).nodes
        positions = node_positions(nodes)
        x, y, i = (normalize_seeds(positions, v) for v in (left, right, required))
        r = frozenset(nodes) - x - y if allowed is None else normalize_seeds(positions, allowed)
    except (GraphValidationError, SeparationDomainError) as error:
        raise SeparatorDomainError(str(error)) from error
    if not x or not y or x & y:
        raise SeparatorDomainError("left and right must be nonempty and disjoint")
    if (i | r) & (x | y):
        raise SeparatorDomainError("required and allowed sets must exclude query endpoints")
    if not i <= r:
        raise SeparatorDomainError("required must be a subset of allowed")
    return nodes, x, y, i, r


def _latent_dag(graph: AcyclicDirectedBidirectedGraph) -> DAG:
    """One independent latent parent per bidirected edge, retaining all vertices."""
    raw = graph.topology
    nodes = list(raw.nodes)
    directed = list(raw.directed)
    for a, b in sorted(raw.bidirected, key=lambda e: (raw.nodes.index(e[0]), raw.nodes.index(e[1]))):
        label = "__latent_" + str(len(nodes))
        while label in nodes:
            label += "_"
        nodes.append(label)
        directed.extend(((label, a), (label, b)))
    return DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=directed))


def _augmented_reduction(
    graph: SeparationView, endpoints: frozenset[str], required: frozenset[str],
) -> tuple[frozenset[str], dict[str, set[str]]]:
    if isinstance(graph, AcyclicDirectedBidirectedGraph):
        graph = _latent_dag(graph)
    raw = _view_topology(graph)
    seeds = endpoints | required
    relevant = (raw.anterior_reflexive(seeds) if isinstance(raw, EndpointMarkedGraph)
                else raw.directed_ancestors_reflexive(seeds))
    adjacent: dict[str, list[tuple[str, bool, bool]]] = {v: [] for v in relevant}
    for a, arrow_a, b, arrow_b in _edge_atoms(raw):
        if a in relevant and b in relevant:
            adjacent[a].append((b, arrow_a, arrow_b))
            adjacent[b].append((a, arrow_b, arrow_a))
    augmented: dict[str, set[str]] = {v: set() for v in relevant}
    for start in relevant:
        pending = [(v, arrow_v) for v, _, arrow_v in adjacent[start]]
        seen = set(pending)
        while pending:
            v, incoming = pending.pop()
            if v != start:
                augmented[start].add(v)
                augmented[v].add(start)
            if not incoming:
                continue
            for w, at_v, at_w in adjacent[v]:
                state = (w, at_w)
                if at_v and state not in seen:
                    seen.add(state)
                    pending.append(state)
    return relevant, augmented


def _boundary_reachable(
    adjacency: dict[str, set[str]], seeds: frozenset[str], blocked: frozenset[str],
) -> frozenset[str]:
    reached = set(seeds)
    pending = list(seeds)
    while pending:
        v = pending.pop()
        for w in adjacency[v]:
            if w not in reached:
                reached.add(w)
                if w not in blocked:
                    pending.append(w)
    return frozenset(reached)


def _find(
    graph: SeparationView, left: str | Iterable[str], right: str | Iterable[str],
    required: str | Iterable[str], allowed: str | Iterable[str] | None, *, minimal: bool,
) -> SeparatorResult:
    nodes, x, y, i, r = _sets(graph, left, right, required, allowed)
    relevant, adjacency = _augmented_reduction(graph, x | y, i)
    z = r & relevant
    reached_x = _boundary_reachable(adjacency, x, z)
    metadata = (graph, _ordered(nodes, x), _ordered(nodes, y),
                _ordered(nodes, i), _ordered(nodes, r))
    if reached_x & y:
        return SeparatorNotFound(*metadata, _token=_TOKEN)
    if minimal:
        z = (z & reached_x) | i
        z = (z & _boundary_reachable(adjacency, y, z)) | i
    return SeparatorSet(*metadata, vertices=_ordered(nodes, z), minimal=minimal, _token=_TOKEN)


def find_m_separator(
    graph: SeparationView, left: str | Iterable[str], right: str | Iterable[str], *,
    required: str | Iterable[str] = (), allowed: str | Iterable[str] | None = None,
) -> SeparatorResult:
    """Find a total m-separator I ⊆ Z ⊆ R, or prove none exists."""
    return _find(graph, left, right, required, allowed, minimal=False)


def find_minimal_m_separator(
    graph: SeparationView, left: str | Iterable[str], right: str | Iterable[str], *,
    required: str | Iterable[str] = (), allowed: str | Iterable[str] | None = None,
) -> SeparatorResult:
    """Find an I-minimal separator; mandatory vertices need not be indispensable."""
    return _find(graph, left, right, required, allowed, minimal=True)


def m_separator_is_minimal(
    graph: SeparationView, left: str | Iterable[str], right: str | Iterable[str],
    given: str | Iterable[str], *, required: str | Iterable[str] = (),
) -> bool:
    nodes, x, y, i, _ = _sets(graph, left, right, required, None)
    try:
        z = normalize_seeds(node_positions(nodes), given)
    except GraphValidationError as error:
        raise SeparatorDomainError(str(error)) from error
    if z & (x | y):
        raise SeparatorDomainError("conditioning vertices must exclude query endpoints")
    if not i <= z or not m_separated_sets(graph, x, y, z):
        return False
    relevant, adjacency = _augmented_reduction(graph, x | y, i)
    if not z <= relevant:
        return False
    reached_x = _boundary_reachable(adjacency, x, z)
    reached_y = _boundary_reachable(adjacency, y, z)
    return (z - i) <= (reached_x & reached_y)
