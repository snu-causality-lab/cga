"""Perković (UAI 2020) identification and Guo--Perković (AISTATS 2021) IDGraphs.

Inputs are validated, causally sufficient DAG/CPDAG/MPDAG proof views. Results
concern an observational density compatible with the graph and its usual positive
density/query-support semantics; the planner has no data to verify those premises.
An offending proper possibly causal path proves nonidentification. IDGraphs splits
the first edge of a globally shortest offending path and closes Meek rules; it does
not enumerate DAGs or merge formulas by sampled numerical agreement.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from typing import TypeAlias, final

from ._immutable import _ImmutableValue
from ._published_paths import _PathSearch, _positive_limit
from ._validation import node_positions, normalize_seeds
from .directed_bidirected import DAG
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import GraphValidationError
from .meek import meek_closure
from .pdag import BackgroundKnowledge, CPDAG, MPDAG, NoConsistentExtension
from .probability import Probability, ProbabilityExpression, Product, Summation


class MPDAGIdentificationDomainError(ValueError):
    """Unsupported proof view, malformed query, or invalid resource budget."""


class MPDAGIdentificationResourceLimitError(RuntimeError):
    """Path or complete-output budget exhausted; no partial partition is returned."""


MPDAGIdentificationGraph: TypeAlias = DAG | CPDAG | MPDAG
_TOKEN = object()


@final
@dataclass(frozen=True, slots=True)
class IdentifiedMPDAGEffect(_ImmutableValue):
    """An observational functional valid for every DAG in this graph's class."""
    graph: MPDAGIdentificationGraph
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    formula: ProbabilityExpression
    buckets: tuple[tuple[str, ...], ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("MPDAG identification results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("IdentifiedMPDAGEffect cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class NonIdentifiedMPDAGEffect(_ImmutableValue):
    """A shortest proper possibly causal path whose first edge remains undirected."""
    graph: MPDAGIdentificationGraph
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    path: tuple[str, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("MPDAG identification results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("NonIdentifiedMPDAGEffect cannot be subclassed")


MPDAGIdentificationResult: TypeAlias = IdentifiedMPDAGEffect | NonIdentifiedMPDAGEffect


def _raw(graph: MPDAGIdentificationGraph) -> EndpointMarkedGraph:
    if not isinstance(graph, (DAG, CPDAG, MPDAG)):
        raise MPDAGIdentificationDomainError("requires a validated causally sufficient DAG, CPDAG, or MPDAG")
    if isinstance(graph, DAG):
        return EndpointMarkedGraph(nodes=graph.topology.nodes, edges=(
            (a, Mark.TAIL, b, Mark.ARROW) for a, b in graph.topology.directed))
    return graph.topology


def _query(graph: MPDAGIdentificationGraph, interventions: str | Iterable[str],
           outcomes: str | Iterable[str], max_path_states: int,
           ) -> tuple[EndpointMarkedGraph, frozenset[str], frozenset[str]]:
    raw = _raw(graph)
    _positive_limit(max_path_states, "max_path_states", MPDAGIdentificationDomainError)
    try:
        positions = node_positions(raw.nodes)
        x, y = normalize_seeds(positions, interventions), normalize_seeds(positions, outcomes)
    except GraphValidationError as error:
        raise MPDAGIdentificationDomainError(str(error)) from error
    if not y or x & y:
        raise MPDAGIdentificationDomainError("outcomes must be nonempty and disjoint from interventions")
    return raw, x, y


def _ordered(raw: EndpointMarkedGraph, selected: frozenset[str]) -> tuple[str, ...]:
    return tuple(v for v in raw.nodes if v in selected)


def _shortest_offending(search: _PathSearch, x: frozenset[str], y: frozenset[str]) -> tuple[str, ...] | None:
    for path in search.paths(x, excluded=x, first_undirected=True):
        if len(path) > 1 and path[-1] in y:
            return path
    return None


def _partial_order(raw: EndpointMarkedGraph, selected: frozenset[str]) -> tuple[tuple[str, ...], ...]:
    """Literal PCO Algorithm 1: remove full-graph sink buckets, prepend intersections."""
    remaining = set(raw.nodes)
    components: list[frozenset[str]] = []
    for start in raw.nodes:
        if start not in remaining:
            continue
        component = {start}
        pending = [start]
        while pending:
            current = pending.pop()
            for other in raw.nodes:
                if other not in component and raw.edge(current, other) == (Mark.TAIL, Mark.TAIL):
                    component.add(other)
                    pending.append(other)
        components.append(frozenset(component))
        remaining.difference_update(component)
    result: list[tuple[str, ...]] = []
    while components:
        sink = next((component for component in components if not any(
            raw.edge(a, b) == (Mark.TAIL, Mark.ARROW)
            for a in component for other in components if other is not component for b in other
        )), None)
        if sink is None:
            raise RuntimeError("validated MPDAG has no sink bucket in its component quotient")
        components.remove(sink)
        intersection = _ordered(raw, sink & selected)
        if intersection:
            result.insert(0, intersection)
    return tuple(result)


def _formula(raw: EndpointMarkedGraph, x: frozenset[str], y: frozenset[str],
             ) -> tuple[ProbabilityExpression, tuple[tuple[str, ...], ...]]:
    # Directed ancestors in the graph with all treatment vertices removed.
    selected = set(y)
    pending = list(y)
    while pending:
        current = pending.pop()
        for parent in raw.directed_parents(current):
            if parent not in x and parent not in selected:
                selected.add(parent)
                pending.append(parent)
    ancestors = frozenset(selected)
    buckets = _partial_order(raw, ancestors)
    factors = []
    for bucket in buckets:
        parents = frozenset(parent for v in bucket for parent in raw.directed_parents(v)) - frozenset(bucket)
        factors.append(Probability(bucket, conditioned_on=_ordered(raw, parents)))
    formula: ProbabilityExpression = factors[0] if len(factors) == 1 else Product(tuple(factors))
    summed = _ordered(raw, ancestors - y)
    if summed:
        formula = Summation(summed, formula)
    if not y <= formula.free_variables <= x | y:
        raise RuntimeError("MPDAG bucket functional has variables outside the original query")
    return formula, buckets


def identify_mpdag_causal_effect(
    graph: MPDAGIdentificationGraph, *, outcomes: str | Iterable[str],
    interventions: str | Iterable[str] = (), max_path_states: int = 100_000,
) -> MPDAGIdentificationResult:
    """Decide P(Y | do(X)) by Proposition 3.2/Theorem 3.6, for possibly joint X,Y.

    A negative path proves graph-class nonidentification. The positive formula is
    the observational bucket factorization, not an estimator. Literal search may
    take exponential work; its explicit path budget raises separately.
    """
    raw, x, y = _query(graph, interventions, outcomes, max_path_states)
    search = _PathSearch(raw, max_path_states, MPDAGIdentificationResourceLimitError)
    path = _shortest_offending(search, x, y)
    if path is not None:
        return NonIdentifiedMPDAGEffect(graph, _ordered(raw, x), _ordered(raw, y), path, _token=_TOKEN)
    formula, buckets = _formula(raw, x, y)
    return IdentifiedMPDAGEffect(graph, _ordered(raw, x), _ordered(raw, y), formula, buckets, _token=_TOKEN)


def enumerate_mpdag_effects(
    graph: MPDAGIdentificationGraph, *, outcomes: str | Iterable[str],
    interventions: str | Iterable[str] = (), max_results: int,
    max_path_states: int = 100_000,
) -> tuple[IdentifiedMPDAGEffect, ...]:
    """Return IDGraphs' complete minimal partition, or raise without partial output.

    Each result is a subclass MPDAG and its identified functional. Different
    subclasses have different functionals over compatible observational densities;
    this does not require numerically different estimates at every distribution.
    The output can be exponential. max_results bounds completed subclasses and
    max_path_states bounds cumulative literal search states across the split tree.
    Neither parameter is a wall-time or whole-process memory bound.
    """
    raw, x, y = _query(graph, interventions, outcomes, max_path_states)
    _positive_limit(max_results, "max_results", MPDAGIdentificationDomainError)
    initial = graph if isinstance(graph, MPDAG) else MPDAG.from_graph(raw)
    pending = [initial]
    results: list[IdentifiedMPDAGEffect] = []
    visited = 0
    while pending:
        current = pending.pop()
        search = _PathSearch(current.topology, max_path_states, MPDAGIdentificationResourceLimitError, visited)
        path = _shortest_offending(search, x, y)
        visited = search.visited
        if path is None:
            if len(results) == max_results:
                raise MPDAGIdentificationResourceLimitError("complete IDGraphs partition exceeds max_results; no partial result returned")
            formula, buckets = _formula(current.topology, x, y)
            results.append(IdentifiedMPDAGEffect(current, _ordered(raw, x), _ordered(raw, y), formula, buckets, _token=_TOKEN))
            continue
        a, b = path[:2]
        forward = meek_closure(current, knowledge=BackgroundKnowledge(required=((a, b),)))
        reverse = meek_closure(current, knowledge=BackgroundKnowledge(required=((b, a),)))
        if isinstance(forward, NoConsistentExtension) or isinstance(reverse, NoConsistentExtension):
            raise RuntimeError("an undirected MPDAG edge did not admit both orientations")
        pending.extend((reverse, forward))
    return tuple(results)
