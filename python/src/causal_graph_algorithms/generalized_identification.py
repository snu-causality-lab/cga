from __future__ import annotations

import heapq
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import NoReturn, SupportsIndex, TypeAlias

from ._validation import node_positions, normalize_seeds
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG
from .errors import IdentificationDomainError
from .identification import (
    HedgeWitness,
    IdentifiedEffect,
    IdentificationResult,
    NonIdentifiedEffect,
    identify_causal_effect,
)
from .probability import (
    Probability,
    ProbabilityExpression,
    Product,
    Ratio,
    Summation,
)
from .subgraph import induced_subgraph


_DOMAIN_MESSAGE = (
    "generalized causal-effect identification requires an acyclic "
    "directed-bidirected or DAG view"
)
_EXPERIMENT_COLLECTION_MESSAGE = (
    "experiment interventions must be an iterable of intervention sets"
)
_OVERLAP_MESSAGE = "outcomes and interventions must be disjoint"
_FACTORY_TOKEN = object()


class _ImmutableGeneralizedIdentificationValue:
    __slots__ = ()

    def __copy__(self) -> typing.Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> typing.Self:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("pickle is unsupported for generalized identification results")


@typing.final
@dataclass(frozen=True, init=False)
class IdentifiedGeneralizedEffect(_ImmutableGeneralizedIdentificationValue):
    """An identified causal query over an exact collection of experiments."""

    __slots__ = (
        "graph",
        "outcomes",
        "interventions",
        "experiment_interventions",
        "formula",
    )

    graph: AcyclicDirectedBidirectedGraph | DAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    experiment_interventions: tuple[tuple[str, ...], ...]
    formula: ProbabilityExpression

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("IdentifiedGeneralizedEffect cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: AcyclicDirectedBidirectedGraph | DAG,
        outcomes: tuple[str, ...],
        interventions: tuple[str, ...],
        experiment_interventions: tuple[tuple[str, ...], ...],
        formula: ProbabilityExpression,
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("IdentifiedGeneralizedEffect values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(
            self,
            "experiment_interventions",
            experiment_interventions,
        )
        object.__setattr__(self, "formula", formula)


@typing.final
@dataclass(frozen=True, init=False)
class GeneralizedNonIdentificationWitness(_ImmutableGeneralizedIdentificationValue):
    """The first unresolved target district and its exact local hedges."""

    __slots__ = (
        "graph",
        "outcomes",
        "interventions",
        "experiment_interventions",
        "target_vertices",
        "unresolved_district",
        "source_hedges",
    )

    graph: AcyclicDirectedBidirectedGraph | DAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    experiment_interventions: tuple[tuple[str, ...], ...]
    target_vertices: tuple[str, ...]
    unresolved_district: tuple[str, ...]
    source_hedges: tuple[HedgeWitness, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("GeneralizedNonIdentificationWitness cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: AcyclicDirectedBidirectedGraph | DAG,
        outcomes: tuple[str, ...],
        interventions: tuple[str, ...],
        experiment_interventions: tuple[tuple[str, ...], ...],
        target_vertices: tuple[str, ...],
        unresolved_district: tuple[str, ...],
        source_hedges: tuple[HedgeWitness, ...],
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError(
                "GeneralizedNonIdentificationWitness values are factory-created"
            )
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(
            self,
            "experiment_interventions",
            experiment_interventions,
        )
        object.__setattr__(self, "target_vertices", target_vertices)
        object.__setattr__(self, "unresolved_district", unresolved_district)
        object.__setattr__(self, "source_hedges", source_hedges)


@typing.final
@dataclass(frozen=True, init=False)
class NonIdentifiedGeneralizedEffect(_ImmutableGeneralizedIdentificationValue):
    """A generalized query that fails in at least one target district."""

    __slots__ = ("witness",)

    witness: GeneralizedNonIdentificationWitness

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("NonIdentifiedGeneralizedEffect cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        witness: GeneralizedNonIdentificationWitness,
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("NonIdentifiedGeneralizedEffect values are factory-created")
        object.__setattr__(self, "witness", witness)


GeneralizedIdentificationResult: TypeAlias = (
    IdentifiedGeneralizedEffect | NonIdentifiedGeneralizedEffect
)


def _ordered_subset(
    source_order: tuple[str, ...], vertices: frozenset[str]
) -> tuple[str, ...]:
    return tuple(vertex for vertex in source_order if vertex in vertices)


@typing.final
class _GraphIdentity:
    """Hash a graph by identity so cached values retain their owning view."""

    __slots__ = ("graph",)

    graph: AcyclicDirectedBidirectedGraph | DAG

    def __init__(self, graph: AcyclicDirectedBidirectedGraph | DAG) -> None:
        self.graph = graph

    def __hash__(self) -> int:
        return id(self.graph)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _GraphIdentity) and self.graph is other.graph


@lru_cache(maxsize=1_024)
def _cached_topological_order(graph_identity: _GraphIdentity) -> tuple[str, ...]:
    graph = graph_identity.graph
    topology = graph.topology
    positions = node_positions(topology.nodes)
    outgoing: dict[str, list[str]] = {node: [] for node in topology.nodes}
    indegree = {node: 0 for node in topology.nodes}
    for parent, child in topology.directed:
        outgoing[parent].append(child)
        indegree[child] += 1
    for children in outgoing.values():
        children.sort(key=positions.__getitem__)

    pending = [positions[node] for node in topology.nodes if indegree[node] == 0]
    heapq.heapify(pending)
    ordered: list[str] = []
    while pending:
        current = topology.nodes[heapq.heappop(pending)]
        ordered.append(current)
        for child in outgoing[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(pending, positions[child])
    if len(ordered) != len(topology.nodes):
        raise AssertionError("validated acyclic graph has no topological order")
    return tuple(ordered)


def _topological_order(
    graph: AcyclicDirectedBidirectedGraph | DAG,
) -> tuple[str, ...]:
    return _cached_topological_order(_GraphIdentity(graph))


@lru_cache(maxsize=8_192)
def _cached_induced_view(
    graph_identity: _GraphIdentity,
    retained: tuple[str, ...],
) -> AcyclicDirectedBidirectedGraph | DAG:
    graph = graph_identity.graph
    if isinstance(graph, DAG):
        return induced_subgraph(graph, retained)
    return induced_subgraph(graph, retained)


def _induced_view(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    retained: tuple[str, ...],
) -> AcyclicDirectedBidirectedGraph | DAG:
    return _cached_induced_view(_GraphIdentity(graph), retained)


@lru_cache(maxsize=8_192)
def _cached_ordinary_identification(
    graph_identity: _GraphIdentity,
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
) -> IdentificationResult:
    return identify_causal_effect(
        graph_identity.graph,
        outcomes=outcomes,
        interventions=interventions,
    )


def _ordinary_identification(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
) -> IdentificationResult:
    return _cached_ordinary_identification(
        _GraphIdentity(graph),
        outcomes,
        interventions,
    )


@dataclass(frozen=True, slots=True)
class _QueryStructure:
    target_vertices: tuple[str, ...]
    districts: tuple[tuple[str, ...], ...]


@lru_cache(maxsize=4_096)
def _cached_query_structure(
    graph_identity: _GraphIdentity,
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
) -> _QueryStructure:
    owning_graph = graph_identity.graph
    topology = owning_graph.topology
    intervention_set = frozenset(interventions)
    post_intervention_vertices = tuple(
        vertex for vertex in topology.nodes if vertex not in intervention_set
    )
    post_intervention_graph = _induced_view(
        owning_graph,
        post_intervention_vertices,
    )
    target_set = post_intervention_graph.topology.directed_ancestors_reflexive(outcomes)
    target_vertices = _ordered_subset(topology.nodes, target_set)
    target_graph = _induced_view(owning_graph, target_vertices)
    districts = tuple(
        _ordered_subset(topology.nodes, district)
        for district in target_graph.districts()
    )
    return _QueryStructure(target_vertices, districts)


def _query_structure(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
) -> _QueryStructure:
    return _cached_query_structure(
        _GraphIdentity(graph),
        outcomes,
        interventions,
    )


def _product(
    factors: tuple[ProbabilityExpression, ...],
) -> ProbabilityExpression:
    if len(factors) == 1:
        return factors[0]
    return Product(factors)


def _summation(
    variables: tuple[str, ...],
    body: ProbabilityExpression,
) -> ProbabilityExpression:
    if not variables:
        return body
    return Summation(variables, body)


def _stamp_source(
    expression: ProbabilityExpression,
    source: tuple[str, ...],
) -> ProbabilityExpression:
    """Clone an observational formula DAG and stamp every leaf with ``source``."""

    def children(
        current: ProbabilityExpression,
    ) -> tuple[ProbabilityExpression, ...]:
        if isinstance(current, Probability):
            return ()
        if isinstance(current, Product):
            return current.factors
        if isinstance(current, Summation):
            return (current.body,)
        if isinstance(current, Ratio):
            return (current.numerator, current.denominator)
        raise AssertionError("unknown local probability-expression node")

    replacements: dict[int, ProbabilityExpression] = {}
    root_children = children(expression)
    pending: list[
        tuple[ProbabilityExpression, tuple[ProbabilityExpression, ...], int]
    ] = [(expression, root_children, 0)]
    active = {id(expression)}
    while pending:
        current, current_children, next_child = pending[-1]
        identity = id(current)
        if identity in replacements:
            active.discard(identity)
            pending.pop()
            continue
        if next_child < len(current_children):
            child = current_children[next_child]
            pending[-1] = (current, current_children, next_child + 1)
            child_identity = id(child)
            if child_identity in replacements:
                continue
            if child_identity in active:
                raise AssertionError("probability-expression graph contains a cycle")
            child_children = children(child)
            active.add(child_identity)
            pending.append((child, child_children, 0))
            continue

        pending.pop()
        active.remove(identity)
        if isinstance(current, Probability):
            if current.interventions:
                raise AssertionError(
                    "local ID produced a nonobservational probability leaf"
                )
            replacements[identity] = Probability(
                current.variables,
                conditioned_on=current.conditioned_on,
                interventions=source,
            )
        elif isinstance(current, Product):
            replacements[identity] = Product(
                tuple(replacements[id(factor)] for factor in current.factors)
            )
        elif isinstance(current, Summation):
            replacements[identity] = Summation(
                current.variables,
                replacements[id(current.body)],
            )
        elif isinstance(current, Ratio):
            replacements[identity] = Ratio(
                replacements[id(current.numerator)],
                replacements[id(current.denominator)],
            )
        else:
            raise AssertionError("unknown local probability-expression node")
    return replacements[id(expression)]


@dataclass(frozen=True, slots=True)
class _LocalIdentification:
    formula: ProbabilityExpression | None
    witness: HedgeWitness | None


@lru_cache(maxsize=4_096)
def _cached_local_identification(
    graph_identity: _GraphIdentity,
    district: tuple[str, ...],
    source: tuple[str, ...],
) -> _LocalIdentification:
    owning_graph = graph_identity.graph
    topology = owning_graph.topology
    district_set = frozenset(district)
    source_set = frozenset(source)
    available = tuple(vertex for vertex in topology.nodes if vertex not in source_set)
    if not district_set <= frozenset(available):
        raise AssertionError("a local source does not cover its target district")
    source_graph = _induced_view(owning_graph, available)
    local_interventions = tuple(
        vertex for vertex in available if vertex not in district_set
    )
    result = _ordinary_identification(
        source_graph,
        district,
        local_interventions,
    )
    if isinstance(result, IdentifiedEffect):
        return _LocalIdentification(
            _stamp_source(result.formula, source),
            None,
        )
    if isinstance(result, NonIdentifiedEffect):
        return _LocalIdentification(None, result.witness)
    raise AssertionError("ordinary ID returned an unknown result value")


def _local_identification(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    district: tuple[str, ...],
    source: tuple[str, ...],
) -> _LocalIdentification:
    return _cached_local_identification(
        _GraphIdentity(graph),
        district,
        source,
    )


def _bind_nuisance_variables(
    formula: ProbabilityExpression,
    *,
    graph: AcyclicDirectedBidirectedGraph | DAG,
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
    experiment_interventions: tuple[tuple[str, ...], ...],
) -> ProbabilityExpression:
    topological_order = _topological_order(graph)
    outcome_set = frozenset(outcomes)
    intervention_set = frozenset(interventions)
    requested = outcome_set.union(intervention_set)
    nuisance_set = formula.free_variables - requested
    nuisance = _ordered_subset(topological_order, nuisance_set)
    if not nuisance:
        return formula

    tier_one_source = next(
        (
            source
            for source in experiment_interventions
            if frozenset(source) <= requested
        ),
        None,
    )
    if tier_one_source is not None:
        source_set = frozenset(tier_one_source)
        conditioned = _ordered_subset(
            topological_order,
            intervention_set - source_set,
        )
        tier_one_weight = Probability(
            nuisance,
            conditioned_on=conditioned,
            interventions=tier_one_source,
        )
        return Summation(nuisance, Product((formula, tier_one_weight)))

    tier_two_source = next(
        (
            source
            for source in experiment_interventions
            if len(source) < len(graph.topology.nodes)
        ),
        None,
    )
    if tier_two_source is None:
        raise AssertionError("identified formula has no source for exact-query binding")

    source_set = frozenset(tier_two_source)
    available_set = frozenset(graph.topology.nodes) - source_set
    available = _ordered_subset(topological_order, available_set)
    retained = nuisance_set.intersection(available_set)
    source_joint = Probability(available, interventions=tier_two_source)
    score = _summation(
        _ordered_subset(topological_order, available_set - retained),
        source_joint,
    )
    nonquery = score.free_variables - nuisance_set - requested
    shared_normalizer = _summation(
        _ordered_subset(topological_order, nonquery),
        score,
    )
    tier_two_weight = Ratio(
        shared_normalizer,
        Summation(nuisance, shared_normalizer),
    )
    return Summation(nuisance, Product((formula, tier_two_weight)))


def _canonical_experiments(
    topology_nodes: tuple[str, ...],
    experiment_interventions: Iterable[str | Iterable[str]],
) -> tuple[tuple[str, ...], ...]:
    if isinstance(experiment_interventions, (str, bytes, bytearray)):
        raise IdentificationDomainError(_EXPERIMENT_COLLECTION_MESSAGE)
    try:
        iterator = iter(experiment_interventions)
    except TypeError as error:
        raise IdentificationDomainError(_EXPERIMENT_COLLECTION_MESSAGE) from error

    positions = node_positions(topology_nodes)
    normalized: set[tuple[str, ...]] = set()
    for raw_target in iterator:
        target_set = normalize_seeds(positions, raw_target)
        normalized.add(_ordered_subset(topology_nodes, target_set))
    return tuple(
        sorted(
            normalized,
            key=lambda target: (
                len(target),
                tuple(positions[vertex] for vertex in target),
            ),
        )
    )


def identify_generalized_causal_effect(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    *,
    outcomes: str | Iterable[str],
    experiment_interventions: Iterable[str | Iterable[str]],
    interventions: str | Iterable[str] = (),
) -> GeneralizedIdentificationResult:
    """Identify a causal effect from exact strict-positive experiment sources."""

    if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
        raise IdentificationDomainError(_DOMAIN_MESSAGE)

    topology = graph.topology
    positions = node_positions(topology.nodes)
    outcome_set = normalize_seeds(positions, outcomes)
    intervention_set = normalize_seeds(positions, interventions)
    normalized_experiments = _canonical_experiments(
        topology.nodes,
        experiment_interventions,
    )
    if outcome_set.intersection(intervention_set):
        raise IdentificationDomainError(_OVERLAP_MESSAGE)

    normalized_outcomes = _ordered_subset(topology.nodes, outcome_set)
    normalized_interventions = _ordered_subset(topology.nodes, intervention_set)
    if not normalized_outcomes:
        return IdentifiedGeneralizedEffect(
            _FACTORY_TOKEN,
            graph=graph,
            outcomes=normalized_outcomes,
            interventions=normalized_interventions,
            experiment_interventions=normalized_experiments,
            formula=Product(()),
        )

    if () in normalized_experiments:
        observational = _ordinary_identification(
            graph,
            normalized_outcomes,
            normalized_interventions,
        )
        if isinstance(observational, IdentifiedEffect):
            return IdentifiedGeneralizedEffect(
                _FACTORY_TOKEN,
                graph=graph,
                outcomes=normalized_outcomes,
                interventions=normalized_interventions,
                experiment_interventions=normalized_experiments,
                formula=observational.formula,
            )

    structure = _query_structure(
        graph,
        normalized_outcomes,
        normalized_interventions,
    )
    node_set = frozenset(topology.nodes)
    selected_formulas: list[ProbabilityExpression] = []
    for district in structure.districts:
        district_set = frozenset(district)
        failed_hedges: list[HedgeWitness] = []
        selected_formula: ProbabilityExpression | None = None
        for source in normalized_experiments:
            if not district_set <= node_set - frozenset(source):
                continue
            local = _local_identification(graph, district, source)
            if local.formula is not None:
                if local.witness is not None:
                    raise AssertionError("successful local ID also returned a hedge")
                selected_formula = local.formula
                break
            if local.witness is None:
                raise AssertionError("failed local ID returned no exact-query hedge")
            failed_hedges.append(local.witness)

        if selected_formula is None:
            witness = GeneralizedNonIdentificationWitness(
                _FACTORY_TOKEN,
                graph=graph,
                outcomes=normalized_outcomes,
                interventions=normalized_interventions,
                experiment_interventions=normalized_experiments,
                target_vertices=structure.target_vertices,
                unresolved_district=district,
                source_hedges=tuple(failed_hedges),
            )
            return NonIdentifiedGeneralizedEffect(
                _FACTORY_TOKEN,
                witness=witness,
            )
        selected_formulas.append(selected_formula)

    topological_order = _topological_order(graph)
    target_set = frozenset(structure.target_vertices)
    pointwise = _summation(
        _ordered_subset(topological_order, target_set - outcome_set),
        _product(tuple(reversed(selected_formulas))),
    )
    formula = _bind_nuisance_variables(
        pointwise,
        graph=graph,
        outcomes=normalized_outcomes,
        interventions=normalized_interventions,
        experiment_interventions=normalized_experiments,
    )
    requested = outcome_set.union(intervention_set)
    if not outcome_set <= formula.free_variables <= requested:
        raise AssertionError(
            "generalized ID formula is not bound to the exact requested query"
        )
    return IdentifiedGeneralizedEffect(
        _FACTORY_TOKEN,
        graph=graph,
        outcomes=normalized_outcomes,
        interventions=normalized_interventions,
        experiment_interventions=normalized_experiments,
        formula=formula,
    )
