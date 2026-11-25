"""Conservative known-graph identification of concrete counterfactual events.

Admitted routes are logical zero/unit and successful single-world ID/gID/IDC
translations. Cross-world and nested routes remain unresolved. In particular,
failed intermediate factors never establish original-query nonidentification.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from typing import Literal, TypeAlias, final

from ._immutable import _ImmutableValue
from ._validation import node_positions, normalize_nodes, normalize_seeds
from .conditional_identification import IdentifiedConditionalEffect, identify_conditional_causal_effect
from .counterfactual import (CounterfactualAtom, CounterfactualDomainError,
                             CounterfactualEvent, CounterfactualTerm, StateSpaces, ValueLiteral)
from .counterfactual_probability import (ValueConstant, ValueExpression,
                                         _UnboundValue, _bind_probability_formula)
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG
from .errors import GraphValidationError
from .generalized_identification import IdentifiedGeneralizedEffect, identify_generalized_causal_effect
from .probability import Ratio


CounterfactualGraph: TypeAlias = DAG | AcyclicDirectedBidirectedGraph
_TOKEN = object()
_Route = Literal["logical", "single_world_gid", "single_world_idc", "single_world_ratio"]


@final
@dataclass(frozen=True, slots=True)
class CounterfactualDefinedness(_ImmutableValue):
    """Semantic premises for a returned formula; they are not empirically verified."""
    positive_source_kernels: bool
    positive_condition: CounterfactualEvent | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CounterfactualDefinedness cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class IdentifiedCounterfactual(_ImmutableValue):
    graph: CounterfactualGraph
    query: CounterfactualEvent
    state_spaces: StateSpaces
    experiment_interventions: tuple[tuple[str, ...], ...]
    formula: ValueExpression
    definedness: CounterfactualDefinedness
    route: _Route
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("counterfactual identification results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("IdentifiedCounterfactual cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class IdentifiedConditionalCounterfactual(_ImmutableValue):
    graph: CounterfactualGraph
    query: CounterfactualEvent
    conditioned_on: CounterfactualEvent
    state_spaces: StateSpaces
    experiment_interventions: tuple[tuple[str, ...], ...]
    formula: ValueExpression
    definedness: CounterfactualDefinedness
    route: _Route
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("counterfactual identification results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("IdentifiedConditionalCounterfactual cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class UnresolvedCounterfactual(_ImmutableValue):
    """No admitted route succeeded; this value makes no nonidentification claim."""
    graph: CounterfactualGraph
    query: CounterfactualEvent
    conditioned_on: CounterfactualEvent | None
    state_spaces: StateSpaces
    experiment_interventions: tuple[tuple[str, ...], ...]
    reason: str
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("counterfactual identification results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("UnresolvedCounterfactual cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class UndefinedCounterfactualCondition(_ImmutableValue):
    """The evidence is logically impossible in every compatible SCM."""
    graph: CounterfactualGraph
    query: CounterfactualEvent
    conditioned_on: CounterfactualEvent
    state_spaces: StateSpaces
    experiment_interventions: tuple[tuple[str, ...], ...]
    reason: Literal["logically_impossible_condition"]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("counterfactual identification results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("UndefinedCounterfactualCondition cannot be subclassed")


CounterfactualIdentificationResult: TypeAlias = IdentifiedCounterfactual | UnresolvedCounterfactual
ConditionalCounterfactualIdentificationResult: TypeAlias = (
    IdentifiedConditionalCounterfactual | UnresolvedCounterfactual | UndefinedCounterfactualCondition)


def _arguments(
    graph: CounterfactualGraph, events: tuple[CounterfactualEvent, ...],
    state_spaces: Mapping[str, Iterable[str]], experiments: Iterable[str | Iterable[str]],
) -> tuple[StateSpaces, tuple[tuple[str, ...], ...]]:
    if not isinstance(graph, (DAG, AcyclicDirectedBidirectedGraph)):
        raise CounterfactualDomainError("counterfactual identification requires a known DAG or ADMG")
    nodes = graph.topology.nodes
    if not isinstance(state_spaces, Mapping) or set(state_spaces) != set(nodes):
        raise CounterfactualDomainError("state spaces must specify exactly every endogenous vertex")
    spaces: dict[str, tuple[str, ...]] = {}
    try:
        for vertex in nodes:
            raw_states = state_spaces[vertex]
            if isinstance(raw_states, str):
                raise CounterfactualDomainError("state spaces must be ordered categorical alphabets")
            states = normalize_nodes(raw_states)
            if not states:
                raise CounterfactualDomainError("categorical alphabets must be nonempty")
            spaces[vertex] = states
    except (GraphValidationError, TypeError) as error:
        raise CounterfactualDomainError(str(error)) from error

    def validate_term(term: CounterfactualTerm) -> None:
        if term.variable not in spaces:
            raise CounterfactualDomainError("counterfactual term contains an unknown variable")
        for assignment in term.interventions:
            value = assignment.value
            if assignment.variable not in spaces:
                raise CounterfactualDomainError("intervention contains an unknown variable")
            if isinstance(value, ValueLiteral):
                if value.state not in spaces[value.variable]:
                    raise CounterfactualDomainError("intervention state is outside its declared alphabet")
            else:
                validate_term(value)

    for event in events:
        if not isinstance(event, CounterfactualEvent):
            raise CounterfactualDomainError("query and evidence must be CounterfactualEvent values")
        for atom in event.atoms:
            validate_term(atom.term)
            if atom.value.state not in spaces[atom.value.variable]:
                raise CounterfactualDomainError("query state is outside its declared alphabet")
    if isinstance(experiments, (str, bytes)):
        raise CounterfactualDomainError("experiment sources must be a collection of intervention sets")
    try:
        positions = node_positions(nodes)
        source_sets = {normalize_seeds(positions, source) for source in experiments}
    except (GraphValidationError, TypeError) as error:
        raise CounterfactualDomainError(str(error)) from error
    sources = tuple(sorted((tuple(v for v in nodes if v in source) for source in source_sets),
                           key=lambda s: (len(s), tuple(positions[v] for v in s))))
    return tuple(spaces.items()), sources


def _normalize_event(event: CounterfactualEvent) -> tuple[tuple[CounterfactualAtom, ...], bool]:
    # Only pointwise effectiveness and literal equality are used here.
    values: dict[CounterfactualTerm, ValueLiteral] = {}
    result = []
    for atom in event.atoms:
        if atom.term in values and values[atom.term] != atom.value:
            return (), True
        values[atom.term] = atom.value
        self_value = next((a.value for a in atom.term.interventions
                           if a.variable == atom.term.variable), None)
        if isinstance(self_value, ValueLiteral):
            if self_value != atom.value:
                return (), True
            continue
        if atom not in result:
            result.append(atom)
    return tuple(result), False


class _UnadmittedRoute(Exception):
    pass


def _world(atoms: tuple[CounterfactualAtom, ...]
           ) -> tuple[tuple[str, ...], dict[str, ValueLiteral], tuple[str, ...]]:
    if any(not isinstance(a.value, ValueLiteral) for atom in atoms for a in atom.term.interventions):
        raise _UnadmittedRoute("nested_route_not_admitted")
    worlds = {atom.term.interventions for atom in atoms}
    if len(worlds) != 1:
        raise _UnadmittedRoute("cross_world_route_not_admitted")
    interventions = next(iter(worlds))
    bindings = {a.variable: a.value for a in interventions if isinstance(a.value, ValueLiteral)}
    bindings.update((a.term.variable, a.value) for a in atoms)
    return tuple(a.variable for a in interventions), bindings, tuple(a.term.variable for a in atoms)


def identify_counterfactual(
    graph: CounterfactualGraph, *, query: CounterfactualEvent,
    state_spaces: Mapping[str, Iterable[str]], experiment_interventions: Iterable[str | Iterable[str]],
) -> CounterfactualIdentificationResult:
    """Identify an admitted concrete event, retaining exact sources and alphabets.

    Positive single-world results use the existing finite-discrete strict-positive
    gID contract. No failure is promoted to fixed-alphabet, pointwise nonidentification.
    """
    spaces, sources = _arguments(graph, (query,), state_spaces, experiment_interventions)
    atoms, impossible = _normalize_event(query)
    if impossible or not atoms:
        return IdentifiedCounterfactual(graph, query, spaces, sources, ValueConstant(0 if impossible else 1),
            CounterfactualDefinedness(False), "logical", _token=_TOKEN)
    try:
        interventions, values, outcomes = _world(atoms)
    except _UnadmittedRoute as error:
        return UnresolvedCounterfactual(graph, query, None, spaces, sources, str(error), _token=_TOKEN)
    result = identify_generalized_causal_effect(graph, outcomes=outcomes,
        interventions=interventions, experiment_interventions=sources)
    if not isinstance(result, IdentifiedGeneralizedEffect):
        return UnresolvedCounterfactual(graph, query, None, spaces, sources,
            "single_world_not_identified_by_admitted_route", _token=_TOKEN)
    try:
        formula = _bind_probability_formula(result.formula, values, dict(spaces))
    except _UnboundValue:
        return UnresolvedCounterfactual(graph, query, None, spaces, sources,
            "source_value_binding_not_admitted", _token=_TOKEN)
    return IdentifiedCounterfactual(graph, query, spaces, sources, formula,
        CounterfactualDefinedness(True), "single_world_gid", _token=_TOKEN)


def identify_conditional_counterfactual(
    graph: CounterfactualGraph, *, query: CounterfactualEvent, conditioned_on: CounterfactualEvent,
    state_spaces: Mapping[str, Iterable[str]], experiment_interventions: Iterable[str | Iterable[str]],
) -> ConditionalCounterfactualIdentificationResult:
    """Identify admitted conditional events, explicitly requiring positive evidence.

    Logical impossibility is an undefined condition. Otherwise a returned formula
    is asserted only where P(conditioned_on)>0. This is not inferred from positivity
    of experimental kernels for cross-world evidence. The generalized conditional
    route is sufficient, not complete.
    """
    spaces, sources = _arguments(graph, (query, conditioned_on), state_spaces, experiment_interventions)
    query_atoms, query_impossible = _normalize_event(query)
    condition_atoms, condition_impossible = _normalize_event(conditioned_on)
    if condition_impossible:
        return UndefinedCounterfactualCondition(graph, query, conditioned_on, spaces, sources,
            "logically_impossible_condition", _token=_TOKEN)
    positive_condition = conditioned_on if condition_atoms else None
    joint_atoms, joint_impossible = _normalize_event(CounterfactualEvent(query_atoms + condition_atoms))
    query_atoms = tuple(a for a in query_atoms if a not in condition_atoms)
    if query_impossible or joint_impossible or not query_atoms:
        value = 0 if query_impossible or joint_impossible else 1
        return IdentifiedConditionalCounterfactual(graph, query, conditioned_on, spaces, sources,
            ValueConstant(value), CounterfactualDefinedness(False, positive_condition), "logical", _token=_TOKEN)
    if not condition_atoms:
        result = identify_counterfactual(graph, query=query, state_spaces=dict(spaces),
                                         experiment_interventions=sources)
        if isinstance(result, UnresolvedCounterfactual):
            return UnresolvedCounterfactual(graph, query, conditioned_on, spaces, sources,
                result.reason, _token=_TOKEN)
        return IdentifiedConditionalCounterfactual(graph, query, conditioned_on, spaces, sources,
            result.formula, result.definedness, result.route, _token=_TOKEN)
    try:
        interventions, values, _ = _world(joint_atoms)
    except _UnadmittedRoute as error:
        return UnresolvedCounterfactual(graph, query, conditioned_on, spaces, sources,
            str(error), _token=_TOKEN)
    outcomes = tuple(a.term.variable for a in query_atoms)
    conditions = tuple(a.term.variable for a in condition_atoms)
    if () in sources:
        ordinary = identify_conditional_causal_effect(graph, outcomes=outcomes,
            interventions=interventions, conditioned_on=conditions)
        if isinstance(ordinary, IdentifiedConditionalEffect):
            try:
                formula = _bind_probability_formula(ordinary.formula, values, dict(spaces))
            except _UnboundValue:
                pass
            else:
                return IdentifiedConditionalCounterfactual(graph, query, conditioned_on, spaces, sources,
                    formula, CounterfactualDefinedness(True, positive_condition), "single_world_idc", _token=_TOKEN)
    joint = identify_generalized_causal_effect(graph, outcomes=outcomes + conditions,
        interventions=interventions, experiment_interventions=sources)
    evidence = identify_generalized_causal_effect(graph, outcomes=conditions,
        interventions=interventions, experiment_interventions=sources)
    if isinstance(joint, IdentifiedGeneralizedEffect) and isinstance(evidence, IdentifiedGeneralizedEffect):
        try:
            formula = _bind_probability_formula(Ratio(joint.formula, evidence.formula), values, dict(spaces))
        except _UnboundValue:
            pass
        else:
            return IdentifiedConditionalCounterfactual(graph, query, conditioned_on, spaces, sources,
                formula, CounterfactualDefinedness(True, positive_condition), "single_world_ratio", _token=_TOKEN)
    return UnresolvedCounterfactual(graph, query, conditioned_on, spaces, sources,
        "single_world_conditional_route_unresolved", _token=_TOKEN)
