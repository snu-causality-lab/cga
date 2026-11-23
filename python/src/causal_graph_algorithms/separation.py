from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TypeAlias

from ._validation import node_positions, normalize_seeds, require_vertex
from .ancestral import AncestralGraph
from .directed_bidirected import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
)
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import SeparationDomainError
from .maximality import MaximalAncestralGraph


SeparationView: TypeAlias = (
    AcyclicDirectedBidirectedGraph | DAG | AncestralGraph | MaximalAncestralGraph
)
EdgeAtom: TypeAlias = tuple[str, bool, str, bool]


def _view_topology(
    graph: SeparationView,
) -> DirectedBidirectedGraph | EndpointMarkedGraph:
    if isinstance(
        graph,
        (
            AcyclicDirectedBidirectedGraph,
            DAG,
            AncestralGraph,
            MaximalAncestralGraph,
        ),
    ):
        return graph.topology
    raise SeparationDomainError(
        "m-separation requires an acyclic directed-bidirected, DAG, "
        "ancestral, or maximal ancestral view"
    )


def _normalize_query_sets(
    nodes: tuple[str, ...],
    left: str | Iterable[str],
    right: str | Iterable[str],
    given: str | Iterable[str],
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    positions = node_positions(nodes)
    left_set = normalize_seeds(positions, left)
    right_set = normalize_seeds(positions, right)
    given_set = normalize_seeds(positions, given)
    if not left_set or not right_set:
        raise SeparationDomainError("left and right vertex sets must be nonempty")
    if left_set & right_set or left_set & given_set or right_set & given_set:
        raise SeparationDomainError(
            "left, right, and conditioning vertex sets must be pairwise disjoint"
        )
    return left_set, right_set, given_set


def _edge_atoms(
    topology: DirectedBidirectedGraph | EndpointMarkedGraph,
) -> tuple[EdgeAtom, ...]:
    if isinstance(topology, DirectedBidirectedGraph):
        atoms: list[EdgeAtom] = [
            (source, False, target, True)
            for source, target in topology.directed
        ]
        atoms.extend(
            (left, True, right, True) for left, right in topology.bidirected
        )
        return tuple(atoms)
    return tuple(
        (
            left,
            left_mark is Mark.ARROW,
            right,
            right_mark is Mark.ARROW,
        )
        for left, left_mark, right, right_mark in topology.edges
    )


def _directed_ancestors_reflexive(
    nodes: tuple[str, ...],
    atoms: tuple[EdgeAtom, ...],
    seeds: frozenset[str],
) -> frozenset[str]:
    parents: dict[str, set[str]] = {node: set() for node in nodes}
    for left, arrow_left, right, arrow_right in atoms:
        if not arrow_left and arrow_right:
            parents[right].add(left)
        elif arrow_left and not arrow_right:
            parents[left].add(right)
    reached = set(seeds)
    pending = list(seeds)
    while pending:
        current = pending.pop()
        for parent in parents[current]:
            if parent not in reached:
                reached.add(parent)
                pending.append(parent)
    return frozenset(reached)


def _m_separated_sets_unchecked(
    nodes: tuple[str, ...],
    atoms: tuple[EdgeAtom, ...],
    left: frozenset[str],
    right: frozenset[str],
    given: frozenset[str],
) -> bool:
    ancestors = _directed_ancestors_reflexive(nodes, atoms, given)
    adjacency: dict[str, list[tuple[str, bool, bool]]] = {
        node: [] for node in nodes
    }
    for first, arrow_first, second, arrow_second in atoms:
        adjacency[first].append((second, arrow_first, arrow_second))
        adjacency[second].append((first, arrow_second, arrow_first))

    seen: set[tuple[str, bool]] = set()
    pending: list[tuple[str, bool]] = []
    for source in left:
        for neighbor, _, arrow_neighbor in adjacency[source]:
            if neighbor in right:
                return False
            state = (neighbor, arrow_neighbor)
            if state not in seen:
                seen.add(state)
                pending.append(state)

    while pending:
        current, incoming_arrow = pending.pop()
        for neighbor, arrow_current, arrow_neighbor in adjacency[current]:
            collider = incoming_arrow and arrow_current
            if collider:
                if current not in ancestors:
                    continue
            elif current in given:
                continue
            if neighbor in right:
                return False
            state = (neighbor, arrow_neighbor)
            if state not in seen:
                seen.add(state)
                pending.append(state)
    return True


def m_separated_sets(
    graph: SeparationView,
    left: str | Iterable[str],
    right: str | Iterable[str],
    given: str | Iterable[str] = (),
) -> bool:
    topology = _view_topology(graph)
    left_set, right_set, given_set = _normalize_query_sets(
        topology.nodes, left, right, given
    )
    return _m_separated_sets_unchecked(
        topology.nodes,
        _edge_atoms(topology),
        left_set,
        right_set,
        given_set,
    )


def m_separated(
    graph: SeparationView,
    left: str,
    right: str,
    given: str | Iterable[str] = (),
) -> bool:
    topology = _view_topology(graph)
    positions = node_positions(topology.nodes)
    require_vertex(positions, left)
    require_vertex(positions, right)
    return m_separated_sets(graph, left, right, given)


def _dag_moral_separated(
    graph: DAG,
    left: frozenset[str],
    right: frozenset[str],
    given: frozenset[str],
) -> bool:
    topology = graph.topology
    relevant = topology.directed_ancestors_reflexive(
        left | right | given
    )
    parents = {
        node: topology.directed_parents(node) & relevant for node in relevant
    }
    moral: dict[str, set[str]] = {node: set() for node in relevant}
    for child, child_parents in parents.items():
        for parent in child_parents:
            moral[parent].add(child)
            moral[child].add(parent)
        for first, second in itertools.combinations(child_parents, 2):
            moral[first].add(second)
            moral[second].add(first)

    reached = set(left)
    pending = list(left)
    while pending:
        current = pending.pop()
        for neighbor in moral[current]:
            if neighbor in given or neighbor in reached:
                continue
            if neighbor in right:
                return False
            reached.add(neighbor)
            pending.append(neighbor)
    return right.isdisjoint(reached)


def d_separated_sets(
    graph: DAG,
    left: str | Iterable[str],
    right: str | Iterable[str],
    given: str | Iterable[str] = (),
) -> bool:
    if not isinstance(graph, DAG):
        raise SeparationDomainError("d-separation requires a DAG view")
    left_set, right_set, given_set = _normalize_query_sets(
        graph.topology.nodes, left, right, given
    )
    return _dag_moral_separated(graph, left_set, right_set, given_set)


def d_separated(
    graph: DAG,
    left: str,
    right: str,
    given: str | Iterable[str] = (),
) -> bool:
    if not isinstance(graph, DAG):
        raise SeparationDomainError("d-separation requires a DAG view")
    positions = node_positions(graph.topology.nodes)
    require_vertex(positions, left)
    require_vertex(positions, right)
    return d_separated_sets(graph, left, right, given)
