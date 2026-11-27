"""Exact graphical checks for published experimental design criteria."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from ._validation import node_positions, normalize_seeds, require_vertex
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .equivalence import dag_markov_equivalent
from .errors import GraphValidationError, MarkovEquivalenceDomainError
from .pdag import CPDAG, PDAG
from .subgraph import induced_subgraph


def dag_intervention_markov_equivalent(
    left: DAG,
    right: DAG,
    targets: Iterable[str | Iterable[str]],
) -> bool:
    """Test Hauser–Bühlmann's criterion for known perfect intervention targets.

    Each vertex must be left untouched by at least one supplied regime. Include
    an empty target set to represent an observational regime, when available.
    """
    if not isinstance(left, DAG) or not isinstance(right, DAG):
        raise MarkovEquivalenceDomainError("interventional equivalence requires two DAG views")
    positions = node_positions(left.topology.nodes)
    if positions.keys() != set(right.topology.nodes):
        return False
    if isinstance(targets, str):
        raise MarkovEquivalenceDomainError("supply a collection of intervention target sets")
    regimes = tuple(normalize_seeds(positions, target) for target in targets)
    if not regimes or frozenset.intersection(*regimes):
        raise MarkovEquivalenceDomainError("the intervention target collection must be conservative")

    def cut(dag: DAG, target: frozenset[str]) -> DAG:
        return DAG(
            DirectedBidirectedGraph(
                nodes=dag.topology.nodes,
                directed=((a, b) for a, b in dag.topology.directed if b not in target),
            )
        )

    return all(dag_markov_equivalent(cut(left, target), cut(right, target)) for target in regimes)


def unseparated_skeleton_edges(
    graph: DAG | PDAG | CPDAG,
    *,
    targets: Iterable[str | Iterable[str]],
) -> tuple[tuple[str, str], ...]:
    """Return edges whose endpoints have identical intervention memberships.

    Only the skeleton is used. An empty result means the fixed known perfect
    interventions distinguish every DAG with that skeleton, the criterion of
    Kocaoglu, Dimakis and Vishwanath (2017), Theorem 1.
    Endpoints and edges follow the declared vertex order.
    """
    adjacencies: Iterable[tuple[str, str]]
    if isinstance(graph, DAG):
        adjacencies = graph.topology.directed
    elif isinstance(graph, (PDAG, CPDAG)):
        adjacencies = (
            (left, right) for left, _, right, _ in graph.topology.edges
        )
    else:
        raise GraphValidationError("skeleton design requires a DAG, PDAG or CPDAG")

    positions = node_positions(graph.topology.nodes)
    if isinstance(targets, (str, bytes)):
        raise GraphValidationError("supply a collection of intervention target sets")
    try:
        experiments = iter(targets)
    except TypeError as error:
        raise GraphValidationError("intervention target sets must be iterable") from error

    membership: dict[str, set[int]] = {node: set() for node in positions}
    for index, target in enumerate(experiments):
        for node in normalize_seeds(positions, target):
            membership[node].add(index)

    unseparated = [
        (left, right) if positions[left] < positions[right] else (right, left)
        for left, right in adjacencies
        if membership[left] == membership[right]
    ]
    unseparated.sort(key=lambda edge: (positions[edge[0]], positions[edge[1]]))
    return tuple(unseparated)


def _single_district(
    graph: AcyclicDirectedBidirectedGraph, target: str | Iterable[str]
) -> frozenset[str]:
    if not isinstance(graph, AcyclicDirectedBidirectedGraph):
        raise GraphValidationError("hedge design requires an acyclic directed-bidirected graph view")
    subset = normalize_seeds(node_positions(graph.topology.nodes), target)
    if not subset or len(induced_subgraph(graph.topology, subset).districts()) != 1:
        raise GraphValidationError("the target must induce a nonempty district")
    return subset


def _hedge_hull(
    topology: DirectedBidirectedGraph,
    target: frozenset[str],
    retained: frozenset[str],
) -> frozenset[str]:
    seed = next(iter(target))
    while True:
        ancestors = induced_subgraph(topology, retained).directed_ancestors_reflexive(target)
        component = induced_subgraph(topology, ancestors).district(seed)
        if component == retained:
            return component
        retained = component


def hedge_hull(
    graph: AcyclicDirectedBidirectedGraph, target: str | Iterable[str]
) -> tuple[str, ...]:
    """Return the target plus the union of its proper hedges, in vertex order.

    The empty-hedge case retains the target, as in Algorithm 1 of
    Akbari–Etesami–Kiyavash (2025).
    """
    subset = _single_district(graph, target)
    topology = graph.topology
    hull = _hedge_hull(topology, subset, frozenset(topology.nodes))
    return tuple(node for node in topology.nodes if node in hull)


@dataclass(frozen=True)
class MinimumCostIntervention:
    vertices: tuple[str, ...]
    total_cost: Fraction


def find_minimum_cost_intervention(
    graph: AcyclicDirectedBidirectedGraph,
    target: str | Iterable[str],
    *,
    costs: Mapping[str, int | Fraction] | None = None,
    allowed: str | Iterable[str] | None = None,
) -> MinimumCostIntervention | None:
    """Find the cheapest single experiment identifying the district kernel Q[S].

    Uses the exact hedge criterion of Akbari–Etesami–Kiyavash (2025),
    Lemma 3. Omitted vertex costs are one. Costs are additive and
    nonnegative. Return None if the allowed vertices cannot hit every hedge.
    Ties are settled by size and then the declared vertex order.
    """
    subset = _single_district(graph, target)
    topology = graph.topology
    positions = node_positions(topology.nodes)
    universe = frozenset(topology.nodes)
    eligible = universe - subset if allowed is None else normalize_seeds(positions, allowed)
    if eligible & subset:
        raise GraphValidationError("allowed intervention vertices must be outside the target")

    weights = {node: Fraction(1) for node in topology.nodes}
    if costs is not None:
        for node, cost in costs.items():
            require_vertex(positions, node)
            if isinstance(cost, bool) or not isinstance(cost, (int, Fraction)) or cost < 0:
                raise GraphValidationError("intervention costs must be nonnegative integers or Fractions")
            weights[node] = Fraction(cost)

    best: tuple[Fraction, int, tuple[int, ...]] | None = None
    seen: set[frozenset[str]] = set()
    pending: list[tuple[frozenset[str], Fraction]] = [(frozenset(), Fraction())]
    while pending:
        chosen, cost = pending.pop()
        if chosen in seen or (best is not None and (cost, len(chosen)) > best[:2]):
            continue
        seen.add(chosen)

        hull = _hedge_hull(topology, subset, universe - chosen)
        if hull == subset:
            key = (cost, len(chosen), tuple(positions[v] for v in topology.nodes if v in chosen))
            if best is None or key < best:
                best = key
            continue

        # A strict fixed point is itself a hedge, so every feasible extension
        # must add at least one of its vertices outside the target.
        branches = (hull - subset) & eligible
        for node in reversed(topology.nodes):
            if node in branches:
                pending.append((chosen | {node}, cost + weights[node]))

    if best is None:
        return None
    return MinimumCostIntervention(tuple(topology.nodes[index] for index in best[2]), best[0])
