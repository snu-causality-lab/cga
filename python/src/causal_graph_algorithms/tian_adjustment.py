from __future__ import annotations

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn, SupportsIndex, TypeAlias

from ._validation import node_positions, require_vertex
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG
from .errors import TianAdjustmentDomainError


_DOMAIN_MESSAGE = "Tian-to-UCA requires an acyclic directed-bidirected or DAG view"
_DISTINCT_MESSAGE = "treatment and outcome must be distinct"
_PERMUTATION_MESSAGE = "variable_order must contain every graph vertex exactly once"
_TOPOLOGICAL_MESSAGE = "variable_order must be topological for the graph"
_OUTCOME_LAST_MESSAGE = "outcome must be last in variable_order"
_FACTORY_TOKEN = object()


class _ImmutableTianAdjustmentValue:
    __slots__ = ()

    def __copy__(self) -> typing.Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> typing.Self:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("pickle is unsupported for Tian-to-UCA results")


@typing.final
@dataclass(frozen=True, init=False)
class TianUCAFactor(_ImmutableTianAdjustmentValue):
    """One consecutive factor in the corrected Tian-to-UCA representation."""

    __slots__ = ("variables", "conditioned_on", "fixed_conditioning")

    variables: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    fixed_conditioning: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("TianUCAFactor cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        variables: tuple[str, ...],
        conditioned_on: tuple[str, ...],
        fixed_conditioning: tuple[str, ...],
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("TianUCAFactor values are factory-created")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "conditioned_on", conditioned_on)
        object.__setattr__(self, "fixed_conditioning", fixed_conditioning)


@typing.final
@dataclass(frozen=True, init=False)
class TianUCARepresentation(_ImmutableTianAdjustmentValue):
    """A corrected Tian representation grouped into UCA factors."""

    __slots__ = ("graph", "treatment", "outcome", "variable_order", "factors")

    graph: AcyclicDirectedBidirectedGraph | DAG
    treatment: str
    outcome: str
    variable_order: tuple[str, ...]
    factors: tuple[TianUCAFactor, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("TianUCARepresentation cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: AcyclicDirectedBidirectedGraph | DAG,
        treatment: str,
        outcome: str,
        variable_order: tuple[str, ...],
        factors: tuple[TianUCAFactor, ...],
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("TianUCARepresentation values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "variable_order", variable_order)
        object.__setattr__(self, "factors", factors)


@typing.final
@dataclass(frozen=True, init=False)
class TianAdjustmentNotApplicable(_ImmutableTianAdjustmentValue):
    """The ordered children that violate Tian's sufficient condition."""

    __slots__ = (
        "graph",
        "treatment",
        "outcome",
        "variable_order",
        "children_in_treatment_district",
    )

    graph: AcyclicDirectedBidirectedGraph | DAG
    treatment: str
    outcome: str
    variable_order: tuple[str, ...]
    children_in_treatment_district: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("TianAdjustmentNotApplicable cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: AcyclicDirectedBidirectedGraph | DAG,
        treatment: str,
        outcome: str,
        variable_order: tuple[str, ...],
        children_in_treatment_district: tuple[str, ...],
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("TianAdjustmentNotApplicable values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "variable_order", variable_order)
        object.__setattr__(
            self,
            "children_in_treatment_district",
            children_in_treatment_district,
        )


TianToUCAResult: TypeAlias = TianUCARepresentation | TianAdjustmentNotApplicable


def _normalize_variable_order(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    variable_order: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(variable_order, (str, bytes, bytearray)) or not isinstance(
        variable_order, Sequence
    ):
        raise TianAdjustmentDomainError(_PERMUTATION_MESSAGE)

    normalized = tuple(variable_order)
    nodes = graph.topology.nodes
    if (
        len(normalized) != len(nodes)
        or any(not isinstance(vertex, str) for vertex in normalized)
        or len(set(normalized)) != len(normalized)
        or set(normalized) != set(nodes)
    ):
        raise TianAdjustmentDomainError(_PERMUTATION_MESSAGE)
    return normalized


def _require_topological_order(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    variable_order: tuple[str, ...],
) -> None:
    positions = node_positions(variable_order)
    if any(
        positions[parent] >= positions[child]
        for parent, child in graph.topology.directed
    ):
        raise TianAdjustmentDomainError(_TOPOLOGICAL_MESSAGE)


def _factor(
    variables: tuple[str, ...],
    *,
    preceding: tuple[str, ...],
    treatment: str,
    inside_treatment_district: bool,
    first: bool = False,
) -> TianUCAFactor:
    if first:
        conditioned_on: tuple[str, ...] = ()
        fixed_conditioning: tuple[str, ...] = ()
    elif inside_treatment_district:
        conditioned_on = preceding
        fixed_conditioning = ()
    else:
        conditioned_on = tuple(
            vertex for vertex in preceding if vertex != treatment
        )
        fixed_conditioning = (treatment,)
    return TianUCAFactor(
        _FACTORY_TOKEN,
        variables=variables,
        conditioned_on=conditioned_on,
        fixed_conditioning=fixed_conditioning,
    )


def _corrected_factors(
    *,
    treatment: str,
    outcome: str,
    variable_order: tuple[str, ...],
    treatment_district: frozenset[str],
) -> tuple[TianUCAFactor, ...]:
    pre_outcome = variable_order[:-1]
    treatment_index = pre_outcome.index(treatment)

    first_stop = treatment_index + 1
    while (
        first_stop < len(pre_outcome)
        and pre_outcome[first_stop] in treatment_district
    ):
        first_stop += 1

    factors = [
        _factor(
            pre_outcome[:first_stop],
            preceding=(),
            treatment=treatment,
            inside_treatment_district=True,
            first=True,
        )
    ]
    cursor = first_stop
    while cursor < len(pre_outcome):
        run_inside = pre_outcome[cursor] in treatment_district
        stop = cursor + 1
        while (
            stop < len(pre_outcome)
            and (pre_outcome[stop] in treatment_district) == run_inside
        ):
            stop += 1
        factors.append(
            _factor(
                pre_outcome[cursor:stop],
                preceding=pre_outcome[:cursor],
                treatment=treatment,
                inside_treatment_district=run_inside,
            )
        )
        cursor = stop

    factors.append(
        _factor(
            (outcome,),
            preceding=pre_outcome,
            treatment=treatment,
            inside_treatment_district=outcome in treatment_district,
        )
    )
    return tuple(factors)


def tian_adjustment_to_uca(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    *,
    treatment: str,
    outcome: str,
    variable_order: Sequence[str],
) -> TianToUCAResult:
    """Compile Tian's singleton adjustment criterion into corrected UCA blocks."""

    if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
        raise TianAdjustmentDomainError(_DOMAIN_MESSAGE)

    topology = graph.topology
    positions = node_positions(topology.nodes)
    normalized_treatment = require_vertex(positions, treatment)
    normalized_outcome = require_vertex(positions, outcome)
    if normalized_treatment == normalized_outcome:
        raise TianAdjustmentDomainError(_DISTINCT_MESSAGE)

    normalized_order = _normalize_variable_order(graph, variable_order)
    _require_topological_order(graph, normalized_order)
    if normalized_order[-1] != normalized_outcome:
        raise TianAdjustmentDomainError(_OUTCOME_LAST_MESSAGE)

    treatment_district = graph.district(normalized_treatment)
    treatment_children = topology.directed_children(normalized_treatment)
    violating_children = tuple(
        vertex
        for vertex in normalized_order
        if vertex in treatment_district and vertex in treatment_children
    )
    if violating_children:
        return TianAdjustmentNotApplicable(
            _FACTORY_TOKEN,
            graph=graph,
            treatment=normalized_treatment,
            outcome=normalized_outcome,
            variable_order=normalized_order,
            children_in_treatment_district=violating_children,
        )

    factors = _corrected_factors(
        treatment=normalized_treatment,
        outcome=normalized_outcome,
        variable_order=normalized_order,
        treatment_district=treatment_district,
    )
    return TianUCARepresentation(
        _FACTORY_TOKEN,
        graph=graph,
        treatment=normalized_treatment,
        outcome=normalized_outcome,
        variable_order=normalized_order,
        factors=factors,
    )
