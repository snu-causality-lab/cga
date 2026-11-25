from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TypeAlias

from ._validation import node_positions, normalize_seeds
from .directed_bidirected import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
)
from .errors import ProjectionDomainError


ProjectionView: TypeAlias = AcyclicDirectedBidirectedGraph | DAG


def latent_projection(
    graph: ProjectionView,
    retained: str | Iterable[str],
) -> AcyclicDirectedBidirectedGraph:
    """Project an acyclic directed-bidirected graph onto retained vertices."""

    if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
        raise ProjectionDomainError(
            "latent projection requires an acyclic directed-bidirected or DAG view"
        )

    topology = graph.topology
    positions = node_positions(topology.nodes)
    retained_set = normalize_seeds(positions, retained)
    directed = set(topology.directed)
    bidirected = {frozenset(edge) for edge in topology.bidirected}

    for latent in (node for node in topology.nodes if node not in retained_set):
        parents = {source for source, target in directed if target == latent}
        children = {target for source, target in directed if source == latent}
        spouses = {
            next(iter(edge - {latent}))
            for edge in bidirected
            if latent in edge
        }

        directed = {
            (source, target)
            for source, target in directed
            if source != latent and target != latent
        }
        bidirected = {edge for edge in bidirected if latent not in edge}

        for parent in parents:
            for child in children:
                if parent != child:
                    directed.add((parent, child))
        for left, right in itertools.combinations(children, 2):
            bidirected.add(frozenset((left, right)))
        for spouse in spouses:
            for child in children:
                if spouse != child:
                    bidirected.add(frozenset((spouse, child)))

    nodes = tuple(node for node in topology.nodes if node in retained_set)
    ordered_bidirected: list[tuple[str, str]] = []
    for edge in bidirected:
        if edge <= retained_set:
            left, right = sorted(edge, key=positions.__getitem__)
            ordered_bidirected.append((left, right))
    projected = DirectedBidirectedGraph(
        nodes=nodes,
        directed=(
            (source, target)
            for source, target in directed
            if source in retained_set and target in retained_set
        ),
        bidirected=ordered_bidirected,
    )
    return AcyclicDirectedBidirectedGraph.from_graph(projected)
