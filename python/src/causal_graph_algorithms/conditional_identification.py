from __future__ import annotations

import typing
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

from ._validation import node_positions, normalize_seeds
from .directed_bidirected import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
)
from .errors import IdentificationDomainError
from .identification import (
    _FACTORY_TOKEN,
    _ImmutableIdentificationValue,
    HedgeWitness,
    IdentifiedEffect,
    NonIdentifiedEffect,
    identify_causal_effect,
)
from .probability import ProbabilityExpression, Product, Ratio, Summation
from .separation import EdgeAtom, _m_separated_sets_unchecked


_DOMAIN_MESSAGE = (
    "conditional causal-effect identification requires an acyclic "
    "directed-bidirected or DAG view"
)
_OUTCOME_INTERVENTION_OVERLAP = "outcomes and interventions must be disjoint"
_OUTCOME_CONDITIONING_OVERLAP = "outcomes and conditioning variables must be disjoint"
_INTERVENTION_CONDITIONING_OVERLAP = (
    "interventions and conditioning variables must be disjoint"
)


@typing.final
@dataclass(frozen=True, init=False)
class IdentifiedConditionalEffect(_ImmutableIdentificationValue):
    """An identified conditional causal query and observational formula."""

    __slots__ = (
        "graph",
        "outcomes",
        "interventions",
        "conditioned_on",
        "rule_two_interventions",
        "formula",
    )

    graph: AcyclicDirectedBidirectedGraph | DAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    rule_two_interventions: tuple[str, ...]
    formula: ProbabilityExpression

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("IdentifiedConditionalEffect cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: AcyclicDirectedBidirectedGraph | DAG,
        outcomes: tuple[str, ...],
        interventions: tuple[str, ...],
        conditioned_on: tuple[str, ...],
        rule_two_interventions: tuple[str, ...],
        formula: ProbabilityExpression,
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("IdentifiedConditionalEffect values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "conditioned_on", conditioned_on)
        object.__setattr__(
            self,
            "rule_two_interventions",
            rule_two_interventions,
        )
        object.__setattr__(self, "formula", formula)


@typing.final
@dataclass(frozen=True, init=False)
class NonIdentifiedConditionalEffect(_ImmutableIdentificationValue):
    """A nonidentified conditional query and its reduced-query hedge."""

    __slots__ = (
        "graph",
        "outcomes",
        "interventions",
        "conditioned_on",
        "rule_two_interventions",
        "witness",
    )

    graph: AcyclicDirectedBidirectedGraph
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    rule_two_interventions: tuple[str, ...]
    witness: HedgeWitness

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("NonIdentifiedConditionalEffect cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: AcyclicDirectedBidirectedGraph,
        outcomes: tuple[str, ...],
        interventions: tuple[str, ...],
        conditioned_on: tuple[str, ...],
        rule_two_interventions: tuple[str, ...],
        witness: HedgeWitness,
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("NonIdentifiedConditionalEffect values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "conditioned_on", conditioned_on)
        object.__setattr__(
            self,
            "rule_two_interventions",
            rule_two_interventions,
        )
        object.__setattr__(self, "witness", witness)


ConditionalIdentificationResult: TypeAlias = (
    IdentifiedConditionalEffect | NonIdentifiedConditionalEffect
)


def _ordered_subset(
    source_order: tuple[str, ...], vertices: frozenset[str] | set[str]
) -> tuple[str, ...]:
    return tuple(vertex for vertex in source_order if vertex in vertices)


def _rule_two_edge_atoms(
    topology: DirectedBidirectedGraph,
    *,
    actions: set[str],
    candidate: str,
) -> tuple[EdgeAtom, ...]:
    atoms: list[EdgeAtom] = []
    for source, target in topology.directed:
        if target in actions or source == candidate:
            continue
        atoms.append((source, False, target, True))
    for left, right in topology.bidirected:
        if left in actions or right in actions:
            continue
        atoms.append((left, True, right, True))
    return tuple(atoms)


def _rule_two_applies(
    topology: DirectedBidirectedGraph,
    *,
    outcomes: frozenset[str],
    actions: set[str],
    residual_conditions: set[str],
    candidate: str,
) -> bool:
    given = frozenset(actions | (residual_conditions - {candidate}))
    return _m_separated_sets_unchecked(
        topology.nodes,
        _rule_two_edge_atoms(
            topology,
            actions=actions,
            candidate=candidate,
        ),
        outcomes,
        frozenset({candidate}),
        given,
    )


def _assert_reduced_query_binding(
    result: IdentifiedEffect | NonIdentifiedEffect,
    *,
    graph: AcyclicDirectedBidirectedGraph | DAG,
    expected_outcomes: tuple[str, ...],
    expected_interventions: tuple[str, ...],
) -> None:
    if isinstance(result, IdentifiedEffect):
        if (
            result.graph is not graph
            or result.outcomes != expected_outcomes
            or result.interventions != expected_interventions
        ):
            raise AssertionError("IDC received a misbound identified ID result")
        return

    witness = result.witness
    if (
        witness.graph is not graph
        or witness.outcomes != expected_outcomes
        or witness.interventions != expected_interventions
    ):
        raise AssertionError("IDC received a misbound exact-query hedge")


def identify_conditional_causal_effect(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    *,
    outcomes: str | Iterable[str],
    interventions: str | Iterable[str] = (),
    conditioned_on: str | Iterable[str] = (),
) -> ConditionalIdentificationResult:
    """Identify ``P(outcomes | do(interventions), conditioned_on)``."""

    if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
        raise IdentificationDomainError(_DOMAIN_MESSAGE)

    topology = graph.topology
    positions = node_positions(topology.nodes)
    outcome_set = normalize_seeds(positions, outcomes)
    intervention_set = normalize_seeds(positions, interventions)
    conditioning_set = normalize_seeds(positions, conditioned_on)

    if outcome_set.intersection(intervention_set):
        raise IdentificationDomainError(_OUTCOME_INTERVENTION_OVERLAP)
    if outcome_set.intersection(conditioning_set):
        raise IdentificationDomainError(_OUTCOME_CONDITIONING_OVERLAP)
    if intervention_set.intersection(conditioning_set):
        raise IdentificationDomainError(_INTERVENTION_CONDITIONING_OVERLAP)

    normalized_outcomes = _ordered_subset(topology.nodes, outcome_set)
    normalized_interventions = _ordered_subset(
        topology.nodes,
        intervention_set,
    )
    normalized_conditioning = _ordered_subset(topology.nodes, conditioning_set)

    if not normalized_outcomes:
        return IdentifiedConditionalEffect(
            _FACTORY_TOKEN,
            graph=graph,
            outcomes=normalized_outcomes,
            interventions=normalized_interventions,
            conditioned_on=normalized_conditioning,
            rule_two_interventions=(),
            formula=Product(()),
        )

    current_actions = set(intervention_set)
    residual_conditions = set(conditioning_set)
    moved_conditions: set[str] = set()
    while True:
        changed = False
        for candidate in topology.nodes:
            if candidate not in residual_conditions:
                continue
            if not _rule_two_applies(
                topology,
                outcomes=outcome_set,
                actions=current_actions,
                residual_conditions=residual_conditions,
                candidate=candidate,
            ):
                continue
            residual_conditions.remove(candidate)
            current_actions.add(candidate)
            moved_conditions.add(candidate)
            changed = True
        if not changed:
            break

    expected_joint_outcomes = _ordered_subset(
        topology.nodes,
        outcome_set | residual_conditions,
    )
    expected_joint_interventions = _ordered_subset(
        topology.nodes,
        current_actions,
    )
    reduced = identify_causal_effect(
        graph,
        outcomes=outcome_set | residual_conditions,
        interventions=current_actions,
    )
    _assert_reduced_query_binding(
        reduced,
        graph=graph,
        expected_outcomes=expected_joint_outcomes,
        expected_interventions=expected_joint_interventions,
    )

    normalized_moved = _ordered_subset(topology.nodes, moved_conditions)
    if isinstance(reduced, NonIdentifiedEffect):
        if not isinstance(graph, AcyclicDirectedBidirectedGraph):
            raise AssertionError("a DAG IDC reduction cannot be nonidentified")
        return NonIdentifiedConditionalEffect(
            _FACTORY_TOKEN,
            graph=graph,
            outcomes=normalized_outcomes,
            interventions=normalized_interventions,
            conditioned_on=normalized_conditioning,
            rule_two_interventions=normalized_moved,
            witness=reduced.witness,
        )

    joint_formula = reduced.formula
    formula = Ratio(
        joint_formula,
        Summation(normalized_outcomes, joint_formula),
    )
    remaining_conditioning = conditioning_set - moved_conditions
    if not (
        outcome_set | remaining_conditioning
        <= formula.free_variables
        <= outcome_set | intervention_set | conditioning_set
    ):
        raise AssertionError("IDC formula is not bound to the exact requested query")
    return IdentifiedConditionalEffect(
        _FACTORY_TOKEN,
        graph=graph,
        outcomes=normalized_outcomes,
        interventions=normalized_interventions,
        conditioned_on=normalized_conditioning,
        rule_two_interventions=normalized_moved,
        formula=formula,
    )
