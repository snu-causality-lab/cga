from __future__ import annotations

import unicodedata
import typing
from collections.abc import Mapping, Sequence
from typing import NoReturn, Self, SupportsIndex, TypeAlias

from .errors import ProbabilityExpressionValidationError


def _normalize_labels(
    values: str | Sequence[str],
    *,
    role: str,
    require_nonempty: bool,
) -> tuple[str, ...]:
    if isinstance(values, str):
        normalized: tuple[object, ...] = (values,)
    elif isinstance(values, Sequence):
        normalized = tuple(values)
    else:
        raise ProbabilityExpressionValidationError(
            f"{role} must be an ordered sequence of labels or one string label"
        )

    if require_nonempty and not normalized:
        raise ProbabilityExpressionValidationError(
            f"{role} must contain at least one label"
        )

    result: list[str] = []
    seen: set[str] = set()
    for label in normalized:
        if not isinstance(label, str):
            raise ProbabilityExpressionValidationError(
                f"every {role} label must be a string"
            )
        if not label:
            raise ProbabilityExpressionValidationError(
                f"{role} labels must be non-empty"
            )
        if any(0xD800 <= ord(character) <= 0xDFFF for character in label):
            raise ProbabilityExpressionValidationError(
                f"{role} labels cannot contain unpaired surrogates"
            )
        if label in seen:
            raise ProbabilityExpressionValidationError(
                f"duplicate {role} label: {label!r}"
            )
        seen.add(label)
        result.append(label)
    return tuple(result)


class _ImmutableProbabilityExpression:
    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise AttributeError("probability expressions are immutable")

    def __delattr__(self, name: str) -> NoReturn:
        del name
        raise AttributeError("probability expressions are immutable")

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError(
            "pickle is unsupported for probability expressions"
        )


@typing.final
class Probability(_ImmutableProbabilityExpression):
    """An ordered probability term, optionally at fixed conditioning values.

    Fixed values describe an ordinary conditioning event, not an intervention.
    They are constants and are excluded from ``free_variables``.
    """

    __slots__ = ("variables", "conditioned_on", "interventions", "conditioned_values")

    variables: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    interventions: tuple[str, ...]
    conditioned_values: tuple[tuple[str, str | int], ...]

    def __init__(
        self,
        variables: str | Sequence[str],
        conditioned_on: str | Sequence[str] = (),
        interventions: str | Sequence[str] = (),
        *,
        conditioned_values: Mapping[str, str | int] | None = None,
    ) -> None:
        normalized_variables = _normalize_labels(
            variables,
            role="probability variables",
            require_nonempty=True,
        )
        normalized_conditioned = _normalize_labels(
            conditioned_on,
            role="conditioning variables",
            require_nonempty=False,
        )
        normalized_interventions = _normalize_labels(
            interventions,
            role="intervention variables",
            require_nonempty=False,
        )

        variable_set = set(normalized_variables)
        conditioned_set = set(normalized_conditioned)
        intervention_set = set(normalized_interventions)
        if variable_set & conditioned_set:
            raise ProbabilityExpressionValidationError(
                "probability and conditioning roles must be disjoint"
            )
        if variable_set & intervention_set:
            raise ProbabilityExpressionValidationError(
                "probability and intervention roles must be disjoint"
            )
        if conditioned_set & intervention_set:
            raise ProbabilityExpressionValidationError(
                "conditioning and intervention roles must be disjoint"
            )

        fixed: dict[str, str | int] = {}
        if conditioned_values is not None:
            if not isinstance(conditioned_values, Mapping):
                raise ProbabilityExpressionValidationError(
                    "conditioned_values must be a mapping of conditioning labels to values"
                )
            for label, value in conditioned_values.items():
                if label not in conditioned_set:
                    raise ProbabilityExpressionValidationError(
                        "fixed values must belong to conditioning variables"
                    )
                if type(value) not in (str, int):
                    raise ProbabilityExpressionValidationError(
                        "fixed conditioning values must be strings or integers (not booleans)"
                    )
                if isinstance(value, str) and (
                    not value or any(0xD800 <= ord(c) <= 0xDFFF for c in value)
                ):
                    raise ProbabilityExpressionValidationError(
                        "fixed string values must be non-empty and cannot contain unpaired surrogates"
                    )
                fixed[label] = value

        object.__setattr__(self, "variables", normalized_variables)
        object.__setattr__(self, "conditioned_on", normalized_conditioned)
        object.__setattr__(self, "interventions", normalized_interventions)
        object.__setattr__(self, "conditioned_values", tuple(
            (label, fixed[label]) for label in normalized_conditioned if label in fixed
        ))

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("Probability cannot be subclassed")

    @property
    def free_variables(self) -> frozenset[str]:
        return _free_variables(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Probability) and _expressions_equal(
            self, other
        )

    def __hash__(self) -> int:
        return _expression_hash(self)


@typing.final
class Product(_ImmutableProbabilityExpression):
    """An ordered, unsimplified product of probability expressions."""

    __slots__ = ("factors",)

    factors: tuple[ProbabilityExpression, ...]

    def __init__(self, factors: Sequence[ProbabilityExpression]) -> None:
        if isinstance(factors, (str, bytes, bytearray)) or not isinstance(
            factors, Sequence
        ):
            raise ProbabilityExpressionValidationError(
                "product factors must be an ordered sequence"
            )
        normalized = tuple(
            _require_expression(factor, role="product factor")
            for factor in factors
        )
        object.__setattr__(self, "factors", normalized)

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("Product cannot be subclassed")

    @property
    def free_variables(self) -> frozenset[str]:
        return _free_variables(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Product) and _expressions_equal(self, other)

    def __hash__(self) -> int:
        return _expression_hash(self)


@typing.final
class Summation(_ImmutableProbabilityExpression):
    """A lexical summation binder over an expression body."""

    __slots__ = ("variables", "body")

    variables: tuple[str, ...]
    body: ProbabilityExpression

    def __init__(
        self,
        variables: str | Sequence[str],
        body: ProbabilityExpression,
    ) -> None:
        normalized_variables = _normalize_labels(
            variables,
            role="summation variables",
            require_nonempty=True,
        )
        normalized_body = _require_expression(body, role="summation body")
        if not set(normalized_variables) <= _free_variables(normalized_body):
            raise ProbabilityExpressionValidationError(
                "every summation variable must be free in its body"
            )
        object.__setattr__(self, "variables", normalized_variables)
        object.__setattr__(self, "body", normalized_body)

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("Summation cannot be subclassed")

    @property
    def free_variables(self) -> frozenset[str]:
        return _free_variables(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Summation) and _expressions_equal(self, other)

    def __hash__(self) -> int:
        return _expression_hash(self)


@typing.final
class Ratio(_ImmutableProbabilityExpression):
    """An unsimplified ratio of two probability expressions."""

    __slots__ = ("numerator", "denominator")

    numerator: ProbabilityExpression
    denominator: ProbabilityExpression

    def __init__(
        self,
        numerator: ProbabilityExpression,
        denominator: ProbabilityExpression,
    ) -> None:
        normalized_numerator = _require_expression(
            numerator, role="ratio numerator"
        )
        normalized_denominator = _require_expression(
            denominator, role="ratio denominator"
        )
        object.__setattr__(self, "numerator", normalized_numerator)
        object.__setattr__(self, "denominator", normalized_denominator)

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("Ratio cannot be subclassed")

    @property
    def free_variables(self) -> frozenset[str]:
        return _free_variables(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ratio) and _expressions_equal(self, other)

    def __hash__(self) -> int:
        return _expression_hash(self)


ProbabilityExpression: TypeAlias = Probability | Product | Summation | Ratio


def _expressions_equal(
    left: ProbabilityExpression,
    right: ProbabilityExpression,
) -> bool:
    stack = [(left, right)]
    compared: set[tuple[int, int]] = set()
    while stack:
        current_left, current_right = stack.pop()
        if current_left is current_right:
            continue
        identity_pair = (id(current_left), id(current_right))
        if identity_pair in compared:
            continue
        compared.add(identity_pair)

        if isinstance(current_left, Probability):
            if not isinstance(current_right, Probability):
                return False
            if (
                current_left.variables != current_right.variables
                or current_left.conditioned_on != current_right.conditioned_on
                or current_left.interventions != current_right.interventions
                or current_left.conditioned_values != current_right.conditioned_values
            ):
                return False
        elif isinstance(current_left, Product):
            if not isinstance(current_right, Product) or len(
                current_left.factors
            ) != len(current_right.factors):
                return False
            stack.extend(zip(current_left.factors, current_right.factors))
        elif isinstance(current_left, Summation):
            if (
                not isinstance(current_right, Summation)
                or current_left.variables != current_right.variables
            ):
                return False
            stack.append((current_left.body, current_right.body))
        else:
            if not isinstance(current_right, Ratio):
                return False
            stack.append((current_left.numerator, current_right.numerator))
            stack.append((current_left.denominator, current_right.denominator))
    return True


def _expression_hash(expression: ProbabilityExpression) -> int:
    hashes: dict[int, int] = {}
    stack: list[tuple[ProbabilityExpression, bool]] = [(expression, False)]
    while stack:
        current, expanded = stack.pop()
        identity = id(current)
        if identity in hashes:
            continue
        if isinstance(current, Probability):
            hashes[identity] = hash(
                (
                    Probability,
                    current.variables,
                    current.conditioned_on,
                    current.interventions,
                    current.conditioned_values,
                )
            )
        elif expanded:
            if isinstance(current, Product):
                child_hashes = tuple(hashes[id(factor)] for factor in current.factors)
                hashes[identity] = hash((Product, child_hashes))
            elif isinstance(current, Summation):
                hashes[identity] = hash(
                    (Summation, current.variables, hashes[id(current.body)])
                )
            else:
                hashes[identity] = hash(
                    (
                        Ratio,
                        hashes[id(current.numerator)],
                        hashes[id(current.denominator)],
                    )
                )
        else:
            stack.append((current, True))
            if isinstance(current, Product):
                stack.extend((factor, False) for factor in current.factors)
            elif isinstance(current, Summation):
                stack.append((current.body, False))
            else:
                stack.append((current.numerator, False))
                stack.append((current.denominator, False))
    return hashes[id(expression)]


def _require_expression(
    value: object,
    *,
    role: str,
) -> ProbabilityExpression:
    if not isinstance(value, (Probability, Product, Summation, Ratio)):
        raise ProbabilityExpressionValidationError(
            f"{role} must be a probability expression"
        )
    return value


def _free_variables(expression: ProbabilityExpression) -> frozenset[str]:
    local_free: dict[int, frozenset[str]] = {}
    stack: list[tuple[ProbabilityExpression, bool]] = [(expression, False)]
    while stack:
        current, expanded = stack.pop()
        identity = id(current)
        if identity in local_free:
            continue
        if isinstance(current, Probability):
            local_free[identity] = frozenset(
                (*current.variables, *current.conditioned_on, *current.interventions)
            ).difference(label for label, _ in current.conditioned_values)
        elif expanded:
            if isinstance(current, Product):
                combined: set[str] = set()
                for factor in current.factors:
                    combined.update(local_free[id(factor)])
                local_free[identity] = frozenset(combined)
            elif isinstance(current, Summation):
                local_free[identity] = local_free[id(current.body)].difference(
                    current.variables
                )
            else:
                local_free[identity] = local_free[
                    id(current.numerator)
                ].union(local_free[id(current.denominator)])
        else:
            stack.append((current, True))
            if isinstance(current, Product):
                stack.extend((factor, False) for factor in current.factors)
            elif isinstance(current, Summation):
                stack.append((current.body, False))
            else:
                stack.append((current.numerator, False))
                stack.append((current.denominator, False))
    return local_free[id(expression)]


def _escape_tex_label(label: str) -> str:
    escaped: list[str] = []
    special = frozenset("{}#$%&_")
    for character in label:
        if character == "\\":
            escaped.append(r"\textbackslash{}")
        elif character in special:
            escaped.append("\\" + character)
        elif character == "^":
            escaped.append(r"\textasciicircum{}")
        elif character == "~":
            escaped.append(r"\textasciitilde{}")
        elif character == " ":
            escaped.append(r"\ ")
        elif character.isspace() or unicodedata.category(character).startswith("C"):
            escaped.append(f"\\texttt{{U+{ord(character):04X}}}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _tex_label(label: str, prime_count: int) -> str:
    rendered = f"\\textnormal{{{_escape_tex_label(label)}}}"
    if prime_count:
        rendered += "^{" + (r"\prime" * prime_count) + "}"
    return rendered


_RenderTask: TypeAlias = str | tuple[ProbabilityExpression, dict[str, int]]


def probability_expression_to_tex(expression: ProbabilityExpression) -> str:
    """Render an expression as a TeX math-mode fragment without delimiters."""

    if not isinstance(expression, (Probability, Product, Summation, Ratio)):
        raise TypeError("TeX rendering requires a probability expression")

    root_environment = {label: 0 for label in _free_variables(expression)}
    pieces: list[str] = []
    stack: list[_RenderTask] = [(expression, root_environment)]

    while stack:
        task = stack.pop()
        if isinstance(task, str):
            pieces.append(task)
            continue

        current, environment = task
        if isinstance(current, Probability):
            variables = r",\,".join(
                _tex_label(label, environment.get(label, 0))
                for label in current.variables
            )
            if current.conditioned_on:
                fixed_values = dict(current.conditioned_values)
                conditioned = r",\,".join(
                    (_tex_label(label, 0) + "=" + (
                        str(fixed_values[label]) if isinstance(fixed_values[label], int)
                        else _tex_label(str(fixed_values[label]), 0)
                    )) if label in fixed_values
                    else _tex_label(label, environment.get(label, 0))
                    for label in current.conditioned_on
                )
                interior = f"{variables} \\mid {conditioned}"
            else:
                interior = variables
            if current.interventions:
                interventions = r",\,".join(
                    _tex_label(label, environment.get(label, 0))
                    for label in current.interventions
                )
                pieces.append(
                    f"P_{{{interventions}}}\\!\\left({interior}\\right)"
                )
            else:
                pieces.append(f"P\\!\\left({interior}\\right)")
        elif isinstance(current, Product):
            if not current.factors:
                pieces.append("1")
                continue
            for index in range(len(current.factors) - 1, -1, -1):
                stack.append((current.factors[index], environment))
                if index:
                    stack.append(r"\,")
        elif isinstance(current, Summation):
            body_environment = environment.copy()
            rendered_binders: list[str] = []
            for label in current.variables:
                prime_count = body_environment.get(label, -1) + 1
                body_environment[label] = prime_count
                rendered_binders.append(_tex_label(label, prime_count))
            prefix = r"\sum_{" + r",\,".join(rendered_binders) + r"}\left["
            stack.append(r"\right]")
            stack.append((current.body, body_environment))
            stack.append(prefix)
        else:
            stack.append("}")
            stack.append((current.denominator, environment))
            stack.append("}{")
            stack.append((current.numerator, environment))
            stack.append(r"\frac{")

    return "".join(pieces)
