"""Value-aware formula syntax kept separate from the existing probability IR."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, final

from ._immutable import _ImmutableValue
from ._validation import normalize_nodes
from .counterfactual import CounterfactualDomainError, ValueLiteral, _label
from .errors import GraphValidationError
from .probability import Probability, ProbabilityExpression, Product, Ratio, Summation


class _ValueExpression(_ImmutableValue):
    __slots__ = ()

    @property
    def free_indices(self) -> frozenset[ValueIndex]:
        return _free_indices(self)


@final
@dataclass(frozen=True, slots=True)
class ValueIndex(_ImmutableValue):
    """A lexical categorical summation index with a fresh identity and alphabet."""
    variable: str
    states: tuple[str, ...]
    identity: int

    def __post_init__(self) -> None:
        _label(self.variable)
        try:
            if isinstance(self.states, str):
                raise CounterfactualDomainError("an index needs a nonempty ordered alphabet")
            states = normalize_nodes(self.states)
            if not states:
                raise CounterfactualDomainError("an index needs a nonempty ordered alphabet")
            object.__setattr__(self, "states", states)
        except GraphValidationError as error:
            raise CounterfactualDomainError(str(error)) from error
        if type(self.identity) is not int or self.identity < 0:
            raise CounterfactualDomainError("index identities must be nonnegative integers")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ValueIndex cannot be subclassed")


ValueBinding: TypeAlias = ValueLiteral | ValueIndex


@final
@dataclass(frozen=True, slots=True)
class ValueProbability(_ValueExpression):
    """One source kernel, with a separate value binding for each variable role."""
    kernel: Probability
    bindings: tuple[tuple[str, ValueBinding], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kernel, Probability):
            raise CounterfactualDomainError("a value probability requires a Probability leaf")
        if self.kernel.conditioned_values:
            raise CounterfactualDomainError(
                "value-bound counterfactual sources do not admit fixed conditioning events"
            )
        roles = self.kernel.variables + self.kernel.conditioned_on + self.kernel.interventions
        try:
            raw = tuple(self.bindings)
            values = dict(raw)
        except (TypeError, ValueError) as error:
            raise CounterfactualDomainError("bindings must be variable/value pairs") from error
        if len(values) != len(raw) or set(values) != set(roles):
            raise CounterfactualDomainError("bind every kernel role exactly once")
        if any(not isinstance(v, (ValueLiteral, ValueIndex)) or v.variable != n
               for n, v in values.items()):
            raise CounterfactualDomainError("kernel bindings must use the correct variable")
        object.__setattr__(self, "bindings", tuple((v, values[v]) for v in roles))

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ValueProbability cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class ValueConstant(_ValueExpression):
    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value not in (0, 1):
            raise CounterfactualDomainError("logical constants are zero or one")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ValueConstant cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class ValueProduct(_ValueExpression):
    factors: tuple[ValueExpression, ...]

    def __post_init__(self) -> None:
        try:
            factors = tuple(self.factors)
        except TypeError as error:
            raise CounterfactualDomainError("product factors must be iterable") from error
        for factor in factors:
            _require(factor)
        object.__setattr__(self, "factors", factors)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ValueProduct cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class ValueSummation(_ValueExpression):
    indices: tuple[ValueIndex, ...]
    body: ValueExpression

    def __post_init__(self) -> None:
        _require(self.body)
        try:
            indices = tuple(self.indices)
        except TypeError as error:
            raise CounterfactualDomainError("summation indices must be iterable") from error
        if not indices or not all(isinstance(v, ValueIndex) for v in indices):
            raise CounterfactualDomainError("a summation needs nonempty typed indices")
        if len({v.identity for v in indices}) != len(indices):
            raise CounterfactualDomainError("summation identities must be distinct")
        if not set(indices) <= self.body.free_indices:
            raise CounterfactualDomainError("summation indices must occur free in their body")
        object.__setattr__(self, "indices", indices)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ValueSummation cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class ValueRatio(_ValueExpression):
    numerator: ValueExpression
    denominator: ValueExpression

    def __post_init__(self) -> None:
        _require(self.numerator)
        _require(self.denominator)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ValueRatio cannot be subclassed")


ValueExpression: TypeAlias = ValueProbability | ValueConstant | ValueProduct | ValueSummation | ValueRatio


def _require(value: object) -> ValueExpression:
    if not isinstance(value, (ValueProbability, ValueConstant, ValueProduct, ValueSummation, ValueRatio)):
        raise CounterfactualDomainError("expected a value-aware formula node")
    return value


def _free_indices(value: object) -> frozenset[ValueIndex]:
    expression = _require(value)
    known: dict[int, frozenset[ValueIndex]] = {}
    pending: list[tuple[ValueExpression, bool]] = [(expression, False)]
    while pending:
        node, ready = pending.pop()
        if id(node) in known:
            continue
        if isinstance(node, ValueProbability):
            known[id(node)] = frozenset(v for _, v in node.bindings if isinstance(v, ValueIndex))
        elif isinstance(node, ValueConstant):
            known[id(node)] = frozenset()
        elif ready:
            if isinstance(node, ValueProduct):
                known[id(node)] = frozenset().union(*(known[id(v)] for v in node.factors))
            elif isinstance(node, ValueSummation):
                known[id(node)] = known[id(node.body)] - frozenset(node.indices)
            else:
                known[id(node)] = known[id(node.numerator)] | known[id(node.denominator)]
        else:
            pending.append((node, True))
            if isinstance(node, ValueProduct):
                pending.extend((v, False) for v in node.factors)
            elif isinstance(node, ValueSummation):
                pending.append((node.body, False))
            else:
                pending.extend(((node.numerator, False), (node.denominator, False)))
    return known[id(expression)]


class _UnboundValue(Exception):
    pass


def _bind_probability_formula(
    formula: ProbabilityExpression, values: Mapping[str, ValueBinding],
    state_spaces: Mapping[str, tuple[str, ...]],
    *, reserved_indices: tuple[ValueIndex, ...] = (),
    allowed_free_indices: frozenset[ValueIndex] = frozenset(),
) -> ValueExpression:
    """Bind a formula DAG, reserving outer indices and preserving lexical scope.

    Existing callers require a closed expression. A district composition may
    explicitly retain supplied outer indices until its global summation binds them.
    """
    occupied = reserved_indices + tuple(v for v in values.values() if isinstance(v, ValueIndex))
    next_identity = max((v.identity for v in occupied), default=-1) + 1
    memo: dict[tuple[int, tuple[tuple[str, ValueBinding], ...]], ValueExpression] = {}

    def visit(expression: ProbabilityExpression, environment: dict[str, ValueBinding]) -> ValueExpression:
        nonlocal next_identity
        key = (id(expression), tuple(sorted(environment.items())))
        if key in memo:
            return memo[key]
        result: ValueExpression
        if isinstance(expression, Probability):
            if expression.conditioned_values:
                raise CounterfactualDomainError(
                    "counterfactual formula binding does not admit fixed conditioning events"
                )
            labels = expression.variables + expression.conditioned_on + expression.interventions
            if not set(labels) <= set(environment):
                raise _UnboundValue()
            result = ValueProbability(expression, tuple((v, environment[v]) for v in labels))
        elif isinstance(expression, Product):
            result = ValueProduct(tuple(visit(v, environment) for v in expression.factors))
        elif isinstance(expression, Ratio):
            result = ValueRatio(visit(expression.numerator, environment), visit(expression.denominator, environment))
        elif isinstance(expression, Summation):
            local = dict(environment)
            indices = []
            for variable in expression.variables:
                index = ValueIndex(variable, state_spaces[variable], next_identity)
                next_identity += 1
                indices.append(index)
                local[variable] = index
            result = ValueSummation(tuple(indices), visit(expression.body, local))
        else:
            raise CounterfactualDomainError("unsupported probability syntax")
        memo[key] = result
        return result

    result = visit(formula, dict(values))
    if result.free_indices - allowed_free_indices:
        raise RuntimeError("formula translation left a free summation index")
    return result


def value_expression_to_tex(expression: ValueExpression) -> str:
    """Render explicit leaf values and lexical indices as a TeX math fragment."""
    from .probability import _tex_label
    _require(expression)
    def index_tex(index: ValueIndex) -> str:
        return "i_{" + str(index.identity) + "}"
    def binding_tex(variable: str, value: ValueBinding) -> str:
        return _tex_label(variable, 0) + "=" + (
            _tex_label(value.state, 0) if isinstance(value, ValueLiteral) else index_tex(value))
    def visit(value: ValueExpression) -> str:
        if isinstance(value, ValueConstant):
            return str(value.value)
        if isinstance(value, ValueProbability):
            bindings = dict(value.bindings)
            kernel = value.kernel
            arguments = ", ".join(binding_tex(v, bindings[v]) for v in kernel.variables)
            conditions = [binding_tex(v, bindings[v]) for v in kernel.conditioned_on]
            if kernel.interventions:
                conditions.append(r"\operatorname{do}\left(" + ", ".join(
                    binding_tex(v, bindings[v]) for v in kernel.interventions) + r"\right)")
            if conditions:
                arguments += r"\mid " + ", ".join(conditions)
            return r"P\left(" + arguments + r"\right)"
        if isinstance(value, ValueProduct):
            return (r"\,".join(r"\left(" + visit(f) + r"\right)" if isinstance(f, (ValueSummation, ValueProduct))
                               else visit(f) for f in value.factors) if value.factors else "1")
        if isinstance(value, ValueSummation):
            indices = ", ".join(index_tex(i) + r"\in\left\{" + ",".join(_tex_label(s, 0) for s in i.states)
                                + r"\right\}" for i in value.indices)
            return r"\sum_{" + indices + "}" + r"\left(" + visit(value.body) + r"\right)"
        return r"\frac{" + visit(value.numerator) + "}{" + visit(value.denominator) + "}"
    return visit(expression)
