"""Graphical validity of Chickering's GES operators on essential graphs."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

from ._validation import node_positions, normalize_seeds, require_vertex
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import GraphValidationError
from .pdag import CPDAG, PDAGDomainError, _skeleton
from .subgraph import induced_subgraph


def _operator_inputs(
    graph: CPDAG, source: str, target: str, selected: str | Iterable[str],
) -> tuple[EndpointMarkedGraph, frozenset[str]]:
    if not isinstance(graph, CPDAG):
        raise PDAGDomainError("GES operators require a validated CPDAG")
    raw = graph.topology
    positions = node_positions(raw.nodes)
    try:
        require_vertex(positions, source)
        require_vertex(positions, target)
        return raw, normalize_seeds(positions, selected)
    except GraphValidationError as error:
        raise PDAGDomainError(str(error)) from error


def _is_clique(
    vertices: frozenset[str], skeleton: frozenset[frozenset[str]],
) -> bool:
    return all(frozenset(pair) in skeleton for pair in combinations(vertices, 2))


def ges_insert_is_valid(
    graph: CPDAG, source: str, target: str, *, tails: str | Iterable[str] = (),
) -> bool:
    """Check Insert admissibility and graphical validity (Chickering, Theorem 15).

    The new arrow goes from source to target. Selected tails must be undirected
    neighbors of the target that are nonadjacent to the source.
    """
    raw, selected = _operator_inputs(graph, source, target, tails)
    if source == target:
        return False
    skeleton = _skeleton(raw)
    if frozenset((source, target)) in skeleton:
        return False

    neighbors = raw.undirected_neighbors(target)
    common = frozenset(v for v in neighbors if frozenset((source, v)) in skeleton)
    if not selected <= neighbors - common:
        return False
    blocked = common | selected
    if not _is_clique(blocked, skeleton):
        return False
    remaining = induced_subgraph(raw, (v for v in raw.nodes if v not in blocked))
    # In a CPDAG an anterior path follows arrows forward or crosses undirected edges.
    return target not in remaining.anterior_reflexive(source)


def ges_delete_is_valid(
    graph: CPDAG, source: str, target: str, *, heads: str | Iterable[str] = (),
) -> bool:
    """Check Delete admissibility and graphical validity (Chickering, Theorem 17).

    The removed edge must be undirected or point from source to target.
    Selected heads must be undirected neighbors of the target adjacent to the source.
    """
    raw, selected = _operator_inputs(graph, source, target, heads)
    if raw.edge(source, target) not in ((Mark.TAIL, Mark.TAIL), (Mark.TAIL, Mark.ARROW)):
        return False

    skeleton = _skeleton(raw)
    common = frozenset(
        v for v in raw.undirected_neighbors(target) if frozenset((source, v)) in skeleton
    )
    return selected <= common and _is_clique(common - selected, skeleton)
