"""Positive finite-discrete conditional gID (Kivva et al., UAI 2023).

Theorem 3 reduces the query to one exact gID call after maximal Rule-2
exchange. The printed MaxBI line 5 is adjudicated in the feature guide.
"""
from __future__ import annotations

import typing
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

from ._validation import node_positions, normalize_seeds
from .conditional_identification import _rule_two_applies
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG
from .errors import IdentificationDomainError
from .generalized_identification import (
    GeneralizedNonIdentificationWitness, IdentifiedGeneralizedEffect,
    NonIdentifiedGeneralizedEffect, _canonical_experiments,
    identify_generalized_causal_effect,
)
from .identification import _ImmutableIdentificationValue
from .probability import ProbabilityExpression, Product, Ratio, Summation

_TOKEN = object()


class ConditionalGeneralizedBudgetExceeded(RuntimeError):
    """The explicit Rule-2 check limit was exhausted; no decision was made."""


@typing.final
@dataclass(frozen=True, init=False)
class IdentifiedConditionalGeneralizedEffect(_ImmutableIdentificationValue):
    """Original query, exact supplied sources, maximal exchange, and formula."""
    __slots__ = ("graph", "outcomes", "interventions", "conditioned_on",
                 "experiment_interventions", "rule_two_interventions", "formula")
    graph: AcyclicDirectedBidirectedGraph | DAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    experiment_interventions: tuple[tuple[str, ...], ...]
    rule_two_interventions: tuple[str, ...]
    formula: ProbabilityExpression

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("conditional generalized identification values cannot be subclassed")

    def __init__(self, token: object, *, graph: AcyclicDirectedBidirectedGraph | DAG,
                 outcomes: tuple[str, ...], interventions: tuple[str, ...],
                 conditioned_on: tuple[str, ...], experiment_interventions: tuple[tuple[str, ...], ...],
                 rule_two_interventions: tuple[str, ...], formula: ProbabilityExpression) -> None:
        if token is not _TOKEN:
            raise TypeError("conditional generalized identification values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "conditioned_on", conditioned_on)
        object.__setattr__(self, "experiment_interventions", experiment_interventions)
        object.__setattr__(self, "rule_two_interventions", rule_two_interventions)
        object.__setattr__(self, "formula", formula)


@typing.final
@dataclass(frozen=True, init=False)
class NonIdentifiedConditionalGeneralizedEffect(_ImmutableIdentificationValue):
    """Original conditional query and exact *reduced joint query* source hedges."""
    __slots__ = ("graph", "outcomes", "interventions", "conditioned_on",
                 "experiment_interventions", "rule_two_interventions", "witness")
    graph: AcyclicDirectedBidirectedGraph | DAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    experiment_interventions: tuple[tuple[str, ...], ...]
    rule_two_interventions: tuple[str, ...]
    witness: GeneralizedNonIdentificationWitness

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("conditional generalized identification values cannot be subclassed")

    def __init__(self, token: object, *, graph: AcyclicDirectedBidirectedGraph | DAG,
                 outcomes: tuple[str, ...], interventions: tuple[str, ...],
                 conditioned_on: tuple[str, ...], experiment_interventions: tuple[tuple[str, ...], ...],
                 rule_two_interventions: tuple[str, ...], witness: GeneralizedNonIdentificationWitness) -> None:
        if token is not _TOKEN:
            raise TypeError("conditional generalized identification values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "conditioned_on", conditioned_on)
        object.__setattr__(self, "experiment_interventions", experiment_interventions)
        object.__setattr__(self, "rule_two_interventions", rule_two_interventions)
        object.__setattr__(self, "witness", witness)


ConditionalGeneralizedIdentificationResult: TypeAlias = (
    IdentifiedConditionalGeneralizedEffect | NonIdentifiedConditionalGeneralizedEffect
)


class _QueryFields(typing.TypedDict):
    graph: AcyclicDirectedBidirectedGraph | DAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    experiment_interventions: tuple[tuple[str, ...], ...]
    rule_two_interventions: tuple[str, ...]


def identify_conditional_generalized_causal_effect(
    graph: AcyclicDirectedBidirectedGraph | DAG, *, outcomes: str | Iterable[str],
    experiment_interventions: Iterable[str | Iterable[str]],
    interventions: str | Iterable[str] = (), conditioned_on: str | Iterable[str] = (),
    max_rule_two_checks: int | None = None,
) -> ConditionalGeneralizedIdentificationResult:
    """Identify ``P_x(Y | Z)`` from exact complete hard-experiment kernels.

    Known acyclic directed/bidirected topology and disjoint query roles are
    checked. Finite domains, strict positivity of the observational joint even
    if unavailable, compatible sources, and complete source tables for every
    intervention assignment are caller premises. An empty collection supplies
    no data; ``()`` as one member explicitly supplies observation. No source
    closure or weak-positivity extension is inferred. Theorem 3 of Kivva et
    al. (2023) supports the positive-finite decision, with Eq. (6) providing
    normalization. A negative retains exact reduced-query source hedges.

    ``max_rule_two_checks`` limits only separation checks, not subsequent gID
    work, memory, formula expansion or rendering. Exhaustion raises separately
    from nonidentification. Every evidence assignment is supported under the
    stated strict-positive premise; this symbolic function does not inspect
    numerical tables or issue undefined-evidence decisions.
    """
    if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
        raise IdentificationDomainError("conditional generalized identification requires a known acyclic DAG/ADMG")
    if max_rule_two_checks is not None and (
        isinstance(max_rule_two_checks, bool) or not isinstance(max_rule_two_checks, int)
        or max_rule_two_checks <= 0
    ):
        raise IdentificationDomainError("max_rule_two_checks must be a positive integer or None")
    nodes = graph.topology.nodes
    positions = node_positions(nodes)
    y = normalize_seeds(positions, outcomes)
    x = normalize_seeds(positions, interventions)
    z = normalize_seeds(positions, conditioned_on)
    if y & x or y & z or x & z:
        raise IdentificationDomainError("outcomes, interventions and conditioning variables must be pairwise disjoint")
    sources = _canonical_experiments(nodes, experiment_interventions)
    ordered_y = tuple(v for v in nodes if v in y)
    ordered_x = tuple(v for v in nodes if v in x)
    ordered_z = tuple(v for v in nodes if v in z)
    moved: set[str] = set()
    if y:
        for checks, candidate in enumerate(ordered_z):
            if max_rule_two_checks is not None and checks >= max_rule_two_checks:
                raise ConditionalGeneralizedBudgetExceeded("Rule-2 separation check budget exhausted")
            if _rule_two_applies(graph.topology, outcomes=y, actions=set(x),
                                 residual_conditions=set(z), candidate=candidate):
                moved.add(candidate)
    ordered_moved = tuple(v for v in nodes if v in moved)
    common: _QueryFields = dict(graph=graph, outcomes=ordered_y, interventions=ordered_x,
                  conditioned_on=ordered_z, experiment_interventions=sources,
                  rule_two_interventions=ordered_moved)
    if not y:
        return IdentifiedConditionalGeneralizedEffect(_TOKEN, **common, formula=Product(()))
    reduced_y = tuple(v for v in nodes if v in y | (z-moved))
    reduced_x = tuple(v for v in nodes if v in x | moved)
    reduced = identify_generalized_causal_effect(graph, outcomes=reduced_y,
        interventions=reduced_x, experiment_interventions=sources)
    if isinstance(reduced, NonIdentifiedGeneralizedEffect):
        return NonIdentifiedConditionalGeneralizedEffect(_TOKEN, **common, witness=reduced.witness)
    if not isinstance(reduced, IdentifiedGeneralizedEffect):
        raise AssertionError("gID returned an unsupported result")
    formula = Ratio(reduced.formula, Summation(ordered_y, reduced.formula))
    return IdentifiedConditionalGeneralizedEffect(_TOKEN, **common, formula=formula)
