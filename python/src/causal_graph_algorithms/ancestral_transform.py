from __future__ import annotations

import itertools
from collections.abc import Iterable

from ._validation import node_positions, normalize_seeds
from .ancestral import AncestralGraph
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import AncestralDomainError
from .maximality import MaximalAncestralGraph
from .separation import m_separated


def ancestral_marginal_condition_transform(
    graph: AncestralGraph | MaximalAncestralGraph,
    *,
    conditioned: str | Iterable[str] = (),
    marginalized: str | Iterable[str] = (),
) -> MaximalAncestralGraph:
    """Return the Richardson--Spirtes ``G[S,L]`` transform.

    ``conditioned`` is the paper's selection set ``S`` and ``marginalized``
    is its latent set ``L``.  The result is the maximal ancestral graph on
    the remaining vertices.
    """
    if not isinstance(graph, (AncestralGraph, MaximalAncestralGraph)):
        raise AncestralDomainError(
            "ancestral marginal-condition transform requires an ancestral "
            "proof view"
        )

    topology = graph.topology
    positions = node_positions(topology.nodes)
    conditioned_set = normalize_seeds(positions, conditioned)
    marginalized_set = normalize_seeds(positions, marginalized)
    if conditioned_set & marginalized_set:
        raise AncestralDomainError(
            "conditioned and marginalized vertex sets must be disjoint"
        )
    if (
        isinstance(graph, MaximalAncestralGraph)
        and not conditioned_set
        and not marginalized_set
    ):
        return graph

    removed = conditioned_set | marginalized_set
    retained = tuple(node for node in topology.nodes if node not in removed)
    transformed_edges: list[tuple[str, Mark, str, Mark]] = []

    for left, right in itertools.combinations(retained, 2):
        separating_set = topology.anterior_reflexive(
            conditioned_set | frozenset((left, right))
        ) - (marginalized_set | frozenset((left, right)))
        if m_separated(graph, left, right, separating_set):
            continue

        left_mark = (
            Mark.TAIL
            if left
            in topology.anterior_reflexive(
                conditioned_set | frozenset((right,))
            )
            else Mark.ARROW
        )
        right_mark = (
            Mark.TAIL
            if right
            in topology.anterior_reflexive(
                conditioned_set | frozenset((left,))
            )
            else Mark.ARROW
        )
        transformed_edges.append((left, left_mark, right, right_mark))

    transformed = EndpointMarkedGraph(
        nodes=retained,
        edges=transformed_edges,
    )
    return MaximalAncestralGraph.from_graph(transformed)
