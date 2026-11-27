"""Source selection for one supplied consistent counterfactual factor."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass

from ._validation import node_positions, normalize_seeds
from .counterfactual import CounterfactualDomainError
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .errors import GraphValidationError
from .fixing import CADMG, ValidFixingSequence, find_fixing_sequence
from .subgraph import induced_subgraph


@dataclass(frozen=True, slots=True)
class CTFIDUSource:
    """A supplied experiment and a graph fixing sequence from its free vertices."""

    source_index: int
    interventions: tuple[str, ...]
    fixing_sequence: tuple[str, ...]


def find_ctfidu_source(
    graph: DAG | AcyclicDirectedBidirectedGraph,
    *,
    district: str | Iterable[str],
    experiment_interventions: Iterable[str | Iterable[str]],
) -> CTFIDUSource | None:
    """Find the first supplied source reaching a consistent factor's district.

    Each regime specifies the vertices set by hard intervention. Use an
    empty set of vertices for an observational source. The factor and its
    consistent value assignments are caller premises. A negative result
    means that these sources did not reach the requested district.
    """
    if not isinstance(graph, (DAG, AcyclicDirectedBidirectedGraph)):
        raise CounterfactualDomainError("CTFIDU source selection requires a validated DAG or ADMG")
    if isinstance(experiment_interventions, (str, bytes, bytearray, Mapping, Set)):
        raise CounterfactualDomainError("supply an ordered collection of intervention regimes")

    topology = graph.topology
    positions = node_positions(topology.nodes)
    try:
        target = normalize_seeds(positions, district)
        regimes = tuple(normalize_seeds(positions, regime) for regime in experiment_interventions)
    except (GraphValidationError, TypeError) as error:
        raise CounterfactualDomainError(str(error)) from error
    if not target or induced_subgraph(topology, target).district(next(iter(target))) != target:
        raise CounterfactualDomainError("the target must be nonempty and bidirected-connected")

    for index, interventions in enumerate(regimes):
        if not target.isdisjoint(interventions):
            continue
        # Remove incoming directed edges and incident bidirected edges at
        # the initial hard targets. Their outgoing arrows to free nodes remain.
        source = graph
        if interventions:
            source = AcyclicDirectedBidirectedGraph.from_graph(
                DirectedBidirectedGraph(
                    nodes=topology.nodes,
                    directed=((tail, head) for tail, head in topology.directed if head not in interventions),
                    bidirected=((a, b) for a, b in topology.bidirected if a not in interventions and b not in interventions),
                )
            )
        conditional = CADMG.from_graph(source, random_vertices=positions.keys() - interventions)
        reached = find_fixing_sequence(conditional, random_vertices=target)
        if isinstance(reached, ValidFixingSequence):
            return CTFIDUSource(
                source_index=index,
                interventions=tuple(vertex for vertex in topology.nodes if vertex in interventions),
                fixing_sequence=reached.sequence,
            )
    return None
