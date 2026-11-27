"""Trek separation in a directed acyclic graph."""

from __future__ import annotations

from collections.abc import Iterable

from ._validation import node_positions, normalize_seeds
from .directed_bidirected import DAG, DirectedBidirectedGraph
from .errors import SeparationDomainError
from .subgraph import induced_subgraph


def _unblocked_tops(
    graph: DirectedBidirectedGraph,
    endpoints: frozenset[str],
    blocked: frozenset[str],
) -> frozenset[str]:
    surviving = endpoints - blocked
    if not surviving:
        return frozenset()
    retained = induced_subgraph(graph, frozenset(graph.nodes) - blocked) if blocked else graph
    return retained.directed_ancestors_reflexive(surviving)


def trek_separated(
    graph: DAG,
    left: str | Iterable[str],
    right: str | Iterable[str],
    *,
    left_blocked: str | Iterable[str] = (),
    right_blocked: str | Iterable[str] = (),
) -> bool:
    """Test Sullivant–Talaska–Draisma (2010), Definition 2.7, in a DAG.

    A trek is a pair of directed paths with a common top. The blockers
    act on their respective sides, including the top and the endpoint.
    Paths of length zero and overlapping query sets are allowed.
    """
    if not isinstance(graph, DAG):
        raise SeparationDomainError("trek separation requires a DAG view")
    topology = graph.topology
    positions = node_positions(topology.nodes)
    left_endpoints = normalize_seeds(positions, left)
    right_endpoints = normalize_seeds(positions, right)
    left_cut = normalize_seeds(positions, left_blocked)
    right_cut = normalize_seeds(positions, right_blocked)
    return _unblocked_tops(topology, left_endpoints, left_cut).isdisjoint(
        _unblocked_tops(topology, right_endpoints, right_cut)
    )
