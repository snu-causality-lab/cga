"""Immutable finite categorical counterfactual events, including nested terms."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias, final

from ._immutable import _ImmutableValue
from ._validation import normalize_nodes
from .errors import GraphValidationError


class CounterfactualDomainError(ValueError):
    """Malformed counterfactual syntax, alphabet, source collection, or graph domain."""


def _label(value: str) -> str:
    try:
        return normalize_nodes((value,))[0]
    except GraphValidationError as error:
        raise CounterfactualDomainError(str(error)) from error


@final
@dataclass(frozen=True, slots=True)
class ValueLiteral(_ImmutableValue):
    """A concrete categorical state attached to its endogenous variable."""
    variable: str
    state: str

    def __post_init__(self) -> None:
        _label(self.variable)
        _label(self.state)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ValueLiteral cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class InterventionAssignment(_ImmutableValue):
    """A concrete clamp or the natural response of the same variable."""
    variable: str
    value: ValueLiteral | CounterfactualTerm

    def __post_init__(self) -> None:
        _label(self.variable)
        if not isinstance(self.value, (ValueLiteral, CounterfactualTerm)):
            raise CounterfactualDomainError("intervention value must be a literal or a counterfactual term")
        if self.variable != self.value.variable:
            raise CounterfactualDomainError("a nested assignment must use the same variable and alphabet")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("InterventionAssignment cannot be subclassed")


@final
@dataclass(frozen=True, slots=True, init=False)
class CounterfactualTerm(_ImmutableValue):
    """The response of one variable under a canonical finite intervention vector."""
    variable: str
    interventions: tuple[InterventionAssignment, ...]

    def __init__(self, variable: str,
                 interventions: Iterable[InterventionAssignment] = ()) -> None:
        _label(variable)
        try:
            assignments = tuple(interventions)
        except TypeError as error:
            raise CounterfactualDomainError("interventions must be assignment values") from error
        values: dict[str, InterventionAssignment] = {}
        for assignment in assignments:
            if not isinstance(assignment, InterventionAssignment):
                raise CounterfactualDomainError("interventions must be assignment values")
            if assignment.variable in values and values[assignment.variable] != assignment:
                raise CounterfactualDomainError("conflicting interventions on the same variable")
            values[assignment.variable] = assignment
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "interventions", tuple(values[v] for v in sorted(values)))

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CounterfactualTerm cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class CounterfactualAtom(_ImmutableValue):
    term: CounterfactualTerm
    value: ValueLiteral

    def __post_init__(self) -> None:
        if not isinstance(self.term, CounterfactualTerm) or not isinstance(self.value, ValueLiteral):
            raise CounterfactualDomainError("an atom requires a counterfactual term and a literal")
        if self.term.variable != self.value.variable:
            raise CounterfactualDomainError("atom value must belong to the response variable")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CounterfactualAtom cannot be subclassed")


@final
@dataclass(frozen=True, slots=True, init=False)
class CounterfactualEvent(_ImmutableValue):
    """Finite conjunction; contradictions are valid syntax and denote the empty event."""
    atoms: tuple[CounterfactualAtom, ...]

    def __init__(self, atoms: Iterable[CounterfactualAtom] = ()) -> None:
        try:
            values = tuple(atoms)
        except TypeError as error:
            raise CounterfactualDomainError("event atoms must be iterable") from error
        if not all(isinstance(v, CounterfactualAtom) for v in values):
            raise CounterfactualDomainError("event atoms must be CounterfactualAtom values")
        object.__setattr__(self, "atoms", tuple(dict.fromkeys(values)))

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CounterfactualEvent cannot be subclassed")


StateSpaces: TypeAlias = tuple[tuple[str, tuple[str, ...]], ...]


def counterfactual_event_to_tex(event: CounterfactualEvent) -> str:
    """Render the original event as a TeX fragment, without logical rewriting."""
    from .probability import _tex_label
    if not isinstance(event, CounterfactualEvent):
        raise CounterfactualDomainError("expected a CounterfactualEvent")
    def term_tex(term: CounterfactualTerm) -> str:
        variable = _tex_label(term.variable, 0)
        if not term.interventions:
            return variable
        assignments = []
        for assignment in term.interventions:
            value = (_tex_label(assignment.value.state, 0) if isinstance(assignment.value, ValueLiteral)
                     else term_tex(assignment.value))
            assignments.append(_tex_label(assignment.variable, 0) + r"\leftarrow " + value)
        return variable + "_{" + ",".join(assignments) + "}"
    if not event.atoms:
        return r"\Omega"
    return r"\left\{" + ", ".join(term_tex(a.term) + "=" + _tex_label(a.value.state, 0)
                                   for a in event.atoms) + r"\right\}"
