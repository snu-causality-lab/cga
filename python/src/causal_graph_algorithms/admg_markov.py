"""Graphical Markov properties of acyclic directed mixed graphs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

from ._validation import node_positions, normalize_seeds, require_vertex
from .conditional_independence import CIStatement
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .errors import GraphValidationError
from .subgraph import induced_subgraph


@dataclass(frozen=True, slots=True)
class ADMGHeadTail:
    """A head and its conditioning tail in graph vertex order."""

    head: tuple[str, ...]
    tail: tuple[str, ...]


def _topology(graph: DAG | AcyclicDirectedBidirectedGraph) -> DirectedBidirectedGraph:
    if not isinstance(graph, (DAG, AcyclicDirectedBidirectedGraph)):
        raise GraphValidationError("ADMG Markov operations require a validated DAG or ADMG")
    return graph.topology


def _ancestral_set(
    topology: DirectedBidirectedGraph, vertices: str | Iterable[str]
) -> frozenset[str]:
    selected = normalize_seeds(node_positions(topology.nodes), vertices)
    if topology.external_directed_parents(selected):
        raise GraphValidationError("ancestral_vertices must be ancestor-closed")
    return selected


def admg_head_tail_partition(
    graph: DAG | AcyclicDirectedBidirectedGraph,
    *,
    ancestral_vertices: str | Iterable[str] | None = None,
) -> tuple[ADMGHeadTail, ...]:
    """Return Richardson's head and tail pairs for one ancestral margin.

    The full vertex set is the default. Heads are returned in partition rounds
    and then by their district's first vertex in graph order. Within each pair
    the labels follow graph order.
    """
    topology = _topology(graph)
    positions = node_positions(topology.nodes)
    remaining = (
        frozenset(topology.nodes)
        if ancestral_vertices is None
        else _ancestral_set(topology, ancestral_vertices)
    )
    descendants = cache(topology.directed_descendants_strict)
    pairs: list[ADMGHeadTail] = []
    while remaining:
        districts = induced_subgraph(topology, remaining).districts()
        for district in districts:
            head = frozenset(
                vertex for vertex in district
                if len(district) == 1 or descendants(vertex).isdisjoint(district)
            )
            # In an ancestral margin, this round's district agrees with the
            # head's district in its ancestral closure (Richardson 2009, Lemma 15
            # and Corollary 17).
            tail = (district - head) | topology.external_directed_parents(district)
            pairs.append(
                ADMGHeadTail(
                    tuple(sorted(head, key=positions.__getitem__)),
                    tuple(sorted(tail, key=positions.__getitem__)),
                )
            )
            remaining -= head
    return tuple(pairs)


def admg_ordered_local_ci(
    graph: DAG | AcyclicDirectedBidirectedGraph,
    vertex: str,
    *,
    ancestral_vertices: str | Iterable[str],
) -> CIStatement:
    """Return the ordered-local Markov statement for a sink in an ancestral set."""
    topology = _topology(graph)
    vertex = require_vertex(node_positions(topology.nodes), vertex)
    selected = _ancestral_set(topology, ancestral_vertices)
    if vertex not in selected:
        raise GraphValidationError("vertex must belong to ancestral_vertices")
    if not topology.directed_children(vertex).isdisjoint(selected):
        raise GraphValidationError("vertex must have no directed child in ancestral_vertices")
    district = induced_subgraph(topology, selected).district(vertex)
    blanket = (district - {vertex}) | topology.external_directed_parents(district)
    return CIStatement(vertex, selected - blanket - {vertex}, blanket)
