"""Elahi et al. (2024): weighted proxy experiments for graphical adjustment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from ._validation import node_positions, normalize_seeds, require_vertex
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .errors import GraphValidationError
from .separation import m_separated_sets
from .separator_search import _Budget, _minimum_cut
from .subgraph import induced_subgraph


@dataclass(frozen=True, slots=True)
class ProxyAdjustmentDesign:
    intervention: tuple[str, ...]
    adjustment: tuple[str, ...]
    minimal_treatments: tuple[str, ...]
    total_cost: Fraction


def _remove_irrelevant_treatments(
    topology: DirectedBidirectedGraph, x: frozenset[str], y: frozenset[str],
) -> frozenset[str]:
    while x:
        cut = AcyclicDirectedBidirectedGraph(DirectedBidirectedGraph(
            nodes=topology.nodes,
            directed=((a, b) for a, b in topology.directed if b not in x),
            bidirected=((a, b) for a, b in topology.bidirected if not ({a, b} & x)),
        ))
        for vertex in topology.nodes:
            if vertex in x and m_separated_sets(cut, vertex, y, x - {vertex}):
                x = x - {vertex}
                break
        else:
            break
    return x


def find_minimum_cost_proxy_adjustment(
    graph: DAG | AcyclicDirectedBidirectedGraph,
    interventions: str | Iterable[str],
    outcomes: str | Iterable[str],
    *,
    costs: Mapping[str, int | Fraction] | None = None,
    allowed: str | Iterable[str] | None = None,
    max_states: int = 100_000,
) -> ProxyAdjustmentDesign | None:
    """Find a minimum-cost design for the protected Lemma 2 adjustment criterion.

    Elahi et al. (2024), Section 4, with intervention excluded from the
    outcome ancestors in the graph without the query treatments. Adjustment
    excludes descendants of query treatments before the proxy experiment. Costs are
    additive and nonnegative. None denotes infeasibility under this criterion.
    """
    if not isinstance(graph, (DAG, AcyclicDirectedBidirectedGraph)):
        raise GraphValidationError("proxy adjustment requires a validated DAG or ADMG")
    topology = graph.topology
    positions = node_positions(topology.nodes)
    x = normalize_seeds(positions, interventions)
    y = normalize_seeds(positions, outcomes)
    if not y or x & y:
        raise GraphValidationError("proxy adjustment requires disjoint treatments and nonempty outcomes")
    permitted = frozenset(topology.nodes) if allowed is None else normalize_seeds(positions, allowed)
    weights = {node: Fraction(1) for node in topology.nodes}
    if costs is not None:
        for node, cost in costs.items():
            require_vertex(positions, node)
            if isinstance(cost, bool) or not isinstance(cost, (int, Fraction)) or cost < 0:
                raise GraphValidationError("intervention costs must be nonnegative integers or Fractions")
            weights[node] = Fraction(cost)

    budget = _Budget(max_states)
    x = _remove_irrelevant_treatments(topology, x, y)
    relevant = induced_subgraph(topology, topology.directed_ancestors_reflexive(x | y))
    surviving = induced_subgraph(relevant, frozenset(relevant.nodes) - x)
    target = surviving.directed_ancestors_reflexive(y)
    parents = relevant.external_directed_parents(target)
    if not parents:
        return ProxyAdjustmentDesign((), (), (), Fraction())

    # Distinct internal labels for observed vertices and one latent root
    # per bidirected edge. Each receives its two copies in the cut graph.
    latent = tuple(sorted(relevant.bidirected))
    observed = {node: idx for idx, node in enumerate(relevant.nodes)}
    size = len(observed) + len(latent)
    first = tuple(str(2 * idx) for idx in range(size))
    second = tuple(str(2 * idx + 1) for idx in range(size))
    network = {node: set() for node in first + second}

    def connect(a: str, b: str) -> None:
        network[a].add(b)
        network[b].add(a)

    for a, b in zip(first, second, strict=True):
        connect(a, b)
    for a, b in relevant.directed:
        if a not in parents:
            connect(first[observed[a]], second[observed[b]])
    for idx, (a, b) in enumerate(latent, start=len(observed)):
        connect(first[idx], second[observed[a]])
        connect(first[idx], second[observed[b]])

    # Never separate a terminal by adjusting for it. Intervention on a
    # member of S erases an outcome mechanism needed by the formula.
    treatment_descendants = relevant.directed_descendants_reflexive(parents)
    adjustable = relevant.directed_ancestors_reflexive(target) - target - treatment_descendants
    cut_costs = {first[observed[v]]: Fraction() for v in adjustable}
    cut_costs.update({
        second[idx]: weights[v] for v, idx in observed.items() if v in permitted - target
    })
    cut = _minimum_cut(
        network, tuple(network),
        frozenset(first[observed[v]] for v in target),
        frozenset(first[observed[v]] for v in parents),
        frozenset(), frozenset(cut_costs), cut_costs, budget,
    )
    if cut is None:
        return None
    intervention = tuple(v for v, idx in observed.items() if second[idx] in cut)
    adjustment = tuple(v for v, idx in observed.items() if first[idx] in cut)
    return ProxyAdjustmentDesign(
        intervention, adjustment, tuple(v for v in relevant.nodes if v in x),
        sum((weights[v] for v in intervention), Fraction()),
    )
