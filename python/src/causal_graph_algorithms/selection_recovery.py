"""Population recovery from an ordinary conditional data distribution.

The admitted input is a full observed joint, either unconditional or conditional
on one fixed value of an ordinary sink vertex. The known DAG/ADMG describes the
population before conditioning. Positive nonparametric semi-Markovian models are
assumed; no sampling probability or transportability information is supplied.

Implements Bareinboim--Tian--Pearl (2014) Theorem 1, the constructive adjustment
pair criterion of Correa--Tian--Bareinboim (2018) Definition 9, and the selected-only
RC procedure of Bareinboim--Tian (2015). See docs/reference/PUBLISHED_IDENTIFICATION.md for precise
failure meanings and corrections to the printed adjustment pseudocode.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import InitVar, dataclass
from typing import Literal, TypeAlias, final

from ._immutable import _ImmutableValue
from ._validation import node_positions, normalize_seeds
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .errors import GraphValidationError
from .identification import (
    IdentifiedEffect, _Kernel, _ancestors_within, _descendants_within,
    _districts_within, _kernel_chain_factors, _marginal_kernel, _ordered_subset,
    _product, _summation, _topological_order, identify_causal_effect,
)
from .probability import Probability, ProbabilityExpression, Ratio
from .separation import m_separated_sets
from .separators import SeparatorNotFound, find_m_separator


RecoveryGraph: TypeAlias = DAG | AcyclicDirectedBidirectedGraph
_TOKEN = object()


class RecoveryDomainError(ValueError):
    """Invalid query or input outside the admitted theorem domain."""


class RecoveryBudgetExceeded(RuntimeError):
    """The state limit was reached; this is not a negative recovery result."""


@final
@dataclass(frozen=True, slots=True)
class RecoveredDistribution(_ImmutableValue):
    graph: RecoveryGraph
    data: Probability
    outcomes: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    interventions: tuple[str, ...]
    formula: ProbabilityExpression
    method: Literal["conditional", "rc", "id"]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("recovery results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("RecoveredDistribution cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class RecoveryNotEstablished(_ImmutableValue):
    """The selected procedure did not produce a recovery formula.

    In particular this does not rule out recovery using additional data, another
    estimand, or an algorithm outside the admitted input domain.
    """
    graph: RecoveryGraph
    data: Probability
    outcomes: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    interventions: tuple[str, ...]
    reason: Literal["conditional_separation_failed", "rc_failed", "ordinary_id_failed"]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("recovery results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("RecoveryNotEstablished cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class SelectionAdjustmentPair(_ImmutableValue):
    """An adjustment pair and its formula, relative to the declared data sources."""
    graph: RecoveryGraph
    data: Probability
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    adjustment: tuple[str, ...]
    population_adjustment: tuple[str, ...]
    population_data: Probability | None
    formula: ProbabilityExpression
    forbidden_vertices: tuple[str, ...]
    required: tuple[str, ...]
    allowed: tuple[str, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("adjustment-pair results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("SelectionAdjustmentPair cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class NoSelectionAdjustmentPair(_ImmutableValue):
    """No pair of the supported formula form exists under these constraints."""
    graph: RecoveryGraph
    data: Probability
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    population_data: Probability | None
    forbidden_vertices: tuple[str, ...]
    required: tuple[str, ...]
    allowed: tuple[str, ...]
    reason: Literal["required_forbidden", "no_separator"]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("adjustment-pair results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("NoSelectionAdjustmentPair cannot be subclassed")


RecoveryResult: TypeAlias = RecoveredDistribution | RecoveryNotEstablished
SelectionAdjustmentResult: TypeAlias = SelectionAdjustmentPair | NoSelectionAdjustmentPair


def _source(graph: RecoveryGraph, data: Probability) -> frozenset[str]:
    if not isinstance(graph, (DAG, AcyclicDirectedBidirectedGraph)):
        raise RecoveryDomainError("recovery requires a known DAG or acyclic directed-bidirected graph")
    if not isinstance(data, Probability) or data.interventions:
        raise RecoveryDomainError("data must be an observational Probability input")
    nodes = frozenset(graph.topology.nodes)
    if frozenset((*data.variables, *data.conditioned_on)) != nodes:
        raise RecoveryDomainError("data must contain every graph vertex, as an outcome or a condition")
    if len(data.conditioned_on) > 1:
        raise RecoveryDomainError("this implementation admits at most one conditioned vertex")
    context = frozenset(data.conditioned_on)
    if frozenset(label for label, _ in data.conditioned_values) != context:
        raise RecoveryDomainError("the conditioned input must specify a fixed event value")
    if any(a in context for a, _ in graph.topology.directed):
        raise RecoveryDomainError("the conditioned vertex must be a directed sink")
    return context


def _query(graph: RecoveryGraph, data: Probability, x: str | Iterable[str],
           y: str | Iterable[str], *, require_x: bool = False
           ) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    context = _source(graph, data)
    positions = node_positions(graph.topology.nodes)
    try:
        xs, ys = normalize_seeds(positions, x), normalize_seeds(positions, y)
    except GraphValidationError as error:
        raise RecoveryDomainError(str(error)) from error
    if not ys or (require_x and not xs) or xs & ys or (xs | ys) & context:
        raise RecoveryDomainError("query sets must be disjoint observed sets with nonempty outcomes")
    return context, xs, ys


def _subset(graph: RecoveryGraph, labels: str | Iterable[str]) -> frozenset[str]:
    try:
        return normalize_seeds(node_positions(graph.topology.nodes), labels)
    except GraphValidationError as error:
        raise RecoveryDomainError(str(error)) from error


def _population(data: Probability, population_data: Probability | None) -> frozenset[str]:
    if population_data is None:
        return frozenset()
    if (not isinstance(population_data, Probability) or population_data.conditioned_on
            or population_data.interventions
            or not set(population_data.variables) <= set(data.variables)):
        raise RecoveryDomainError("population_data must be an unconditional joint over input variables")
    return frozenset(population_data.variables)


def _from_data(data: Probability, variables: tuple[str, ...],
               given: tuple[str, ...] = ()) -> ProbabilityExpression:
    if not variables:
        return _product(())
    return Probability(variables, conditioned_on=given + data.conditioned_on,
                       conditioned_values=dict(data.conditioned_values))


def recover_conditional_distribution(
    graph: RecoveryGraph, data: Probability, *, outcomes: str | Iterable[str],
    conditioned_on: str | Iterable[str] = (),
) -> RecoveryResult:
    """Recover population P(Y|X) by the 2014 conditional-independence criterion."""
    context, x, y = _query(graph, data, conditioned_on, outcomes)
    order = graph.topology.nodes
    xo, yo = _ordered_subset(order, x), _ordered_subset(order, y)
    if context and not m_separated_sets(graph, y, context, x):
        return RecoveryNotEstablished(graph, data, yo, xo, (),
                                      "conditional_separation_failed", _token=_TOKEN)
    return RecoveredDistribution(graph, data, yo, xo, (), _from_data(data, yo, xo),
                                 "conditional", _token=_TOKEN)


def _proper_backdoor(graph: RecoveryGraph, x: frozenset[str], y: frozenset[str]
                     ) -> tuple[RecoveryGraph, frozenset[str]]:
    raw = graph.topology
    non_x = frozenset(raw.nodes) - x
    ancestors_y = _ancestors_within(raw, y, non_x)
    first = frozenset((a, b) for a, b in raw.directed if a in x and b in ancestors_y)
    causal_nodes = frozenset(v for _, b in first
                            for v in _descendants_within(raw, b, non_x)) & ancestors_y
    # The incoming-X cut prevents the forbidden descendant search from traversing X.
    forbidden = frozenset(v for seed in causal_nodes
                          for v in _descendants_within(raw, seed, non_x))
    pbd = DirectedBidirectedGraph(nodes=raw.nodes,
                                  directed=tuple(e for e in raw.directed if e not in first),
                                  bidirected=raw.bidirected)
    view = (DAG.from_graph(pbd) if isinstance(graph, DAG)
            else AcyclicDirectedBidirectedGraph.from_graph(pbd))
    return view, forbidden


def selection_adjustment_pair_is_valid(
    graph: RecoveryGraph, data: Probability, *, interventions: str | Iterable[str],
    outcomes: str | Iterable[str], adjustment: str | Iterable[str],
    population_adjustment: str | Iterable[str] = (), population_data: Probability | None = None,
) -> bool:
    """Check 2018 Definition 9 and availability of its population weight."""
    context, x, y = _query(graph, data, interventions, outcomes, require_x=True)
    z, zt = _subset(graph, adjustment), _subset(graph, population_adjustment)
    t = _population(data, population_data)
    if z & (x | y | context) or not zt <= z & t:
        return False
    pbd, forbidden = _proper_backdoor(graph, x, y)
    return (not z & forbidden and m_separated_sets(pbd, x, y, z | context)
            and (not context or m_separated_sets(pbd, y, context, zt)))


@dataclass(frozen=True, slots=True)
class _PairSearch:
    graph: RecoveryGraph
    data: Probability
    x: frozenset[str]
    y: frozenset[str]
    context: frozenset[str]
    population_data: Probability | None
    t: frozenset[str]
    pbd: RecoveryGraph
    forbidden: frozenset[str]
    required: frozenset[str]
    allowed: frozenset[str]
    candidates: frozenset[str]


def _search(graph: RecoveryGraph, data: Probability, x: str | Iterable[str],
            y: str | Iterable[str], population_data: Probability | None,
            required: str | Iterable[str], allowed: str | Iterable[str] | None) -> _PairSearch:
    context, xs, ys = _query(graph, data, x, y, require_x=True)
    t = _population(data, population_data)
    i = _subset(graph, required)
    r = frozenset(data.variables) - xs - ys if allowed is None else _subset(graph, allowed)
    if (i | r) & (xs | ys | context) or not i <= r:
        raise RecoveryDomainError("required must be within allowed covariates, excluding query and fixed vertices")
    pbd, forbidden = _proper_backdoor(graph, xs, ys)
    return _PairSearch(graph, data, xs, ys, context, population_data, t, pbd,
                       forbidden, i, r, r - forbidden)


def _pair(search: _PairSearch, z: frozenset[str], zt: frozenset[str]) -> SelectionAdjustmentPair:
    order = search.graph.topology.nodes
    x, y, zo, zto = (_ordered_subset(order, s) for s in (search.x, search.y, z, zt))
    response = _from_data(search.data, y, _ordered_subset(order, search.x | z))
    selected_weight = _from_data(search.data, _ordered_subset(order, z - zt), zto)
    population_weight = Probability(zto) if zto else _product(())
    formula = _summation(zo, _product((response, selected_weight, population_weight)))
    return SelectionAdjustmentPair(
        search.graph, search.data, x, y, zo, zto, search.population_data, formula,
        _ordered_subset(order, search.forbidden), _ordered_subset(order, search.required),
        _ordered_subset(order, search.allowed), _token=_TOKEN)


def _no_pair(search: _PairSearch,
             reason: Literal["required_forbidden", "no_separator"]) -> NoSelectionAdjustmentPair:
    order = search.graph.topology.nodes
    return NoSelectionAdjustmentPair(
        search.graph, search.data, _ordered_subset(order, search.x), _ordered_subset(order, search.y),
        search.population_data, _ordered_subset(order, search.forbidden),
        _ordered_subset(order, search.required), _ordered_subset(order, search.allowed),
        reason, _token=_TOKEN)


def _population_separator(search: _PairSearch, required: frozenset[str],
                          allowed: frozenset[str]) -> frozenset[str] | None:
    if not search.context:
        return required
    sep = find_m_separator(search.pbd, search.context, search.y,
                           required=required, allowed=allowed)
    return None if isinstance(sep, SeparatorNotFound) else frozenset(sep.vertices)


def find_selection_adjustment_pair(
    graph: RecoveryGraph, data: Probability, *, interventions: str | Iterable[str],
    outcomes: str | Iterable[str], population_data: Probability | None = None,
    required: str | Iterable[str] = (), allowed: str | Iterable[str] | None = None,
) -> SelectionAdjustmentResult:
    """Construct a pair or return absence within the adjustment-pair formula class."""
    search = _search(graph, data, interventions, outcomes, population_data, required, allowed)
    if search.required & search.forbidden:
        return _no_pair(search, "required_forbidden")
    sep = find_m_separator(search.pbd, search.x, search.y,
                           required=search.required | search.context,
                           allowed=search.candidates | search.context)
    zt = _population_separator(search, frozenset(), search.candidates & search.t)
    if isinstance(sep, SeparatorNotFound) or zt is None:
        return _no_pair(search, "no_separator")
    z = frozenset(sep.vertices) - search.context
    # Both constructors return the allowed ancestral candidate. The second is
    # contained in the first because the fixed context is mandatory in the first.
    if not zt <= z:
        raise AssertionError("the canonical population separator must be contained in the adjustment set")
    return _pair(search, z, zt)


@dataclass(slots=True)
class _Budget:
    limit: int
    states: int = 0

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 1:
            raise RecoveryDomainError("max_states must be a positive integer")

    def take(self) -> None:
        if self.states >= self.limit:
            raise RecoveryBudgetExceeded("recovery state budget exhausted")
        self.states += 1


def iter_selection_adjustment_pairs(
    graph: RecoveryGraph, data: Probability, *, interventions: str | Iterable[str],
    outcomes: str | Iterable[str], population_data: Probability | None = None,
    required: str | Iterable[str] = (), allowed: str | Iterable[str] | None = None,
    max_states: int = 100_000,
) -> Iterator[SelectionAdjustmentPair]:
    """List all supported pairs deterministically; exhaustion raises separately.

    The corrected two-stage 2018 search is iterative, including the fixed input
    condition in every first-stage separator call. Separator materialization has
    polynomial cost; this implementation does not claim the paper's sharper bound.
    """
    search = _search(graph, data, interventions, outcomes, population_data, required, allowed)
    budget = _Budget(max_states)
    if search.required & search.forbidden:
        return
    order = graph.topology.nodes
    # stage, mandatory, possible, completed first-stage set
    stack: list[tuple[int, frozenset[str], frozenset[str], frozenset[str]]] = [
        (0, search.required | search.context, search.candidates | search.context, frozenset())]
    while stack:
        stage, i, r, z = stack.pop()
        budget.take()
        if stage == 0:
            sep = find_m_separator(search.pbd, search.x, search.y, required=i, allowed=r)
            if isinstance(sep, SeparatorNotFound) or _population_separator(
                    search, frozenset(), r & search.t) is None:
                continue
        elif _population_separator(search, i, r) is None:
            continue
        if i == r:
            if stage == 0:
                z = i - search.context
                stack.append((1, frozenset(), z & search.t, z))
            else:
                yield _pair(search, z, i)
            continue
        v = next(v for v in order if v in r - i)
        stack.append((stage, i, r - {v}, z))
        stack.append((stage, i | {v}, r, z))


def _extract(kernel: _Kernel, component: frozenset[str]) -> _Kernel:
    factors = _kernel_chain_factors(kernel)
    selected = tuple(f for v, f in zip(kernel.nodes, factors) if v in component)
    return _Kernel(tuple(v for v in kernel.nodes if v in component), _product(selected), None)


def _tian_identify(target: frozenset[str], kernel: _Kernel,
                   raw: DirectedBidirectedGraph, budget: _Budget) -> ProbabilityExpression | None:
    while True:
        budget.take()
        component = frozenset(kernel.nodes)
        ancestors = _ancestors_within(raw, target, component)
        if ancestors == target:
            return _summation(tuple(v for v in kernel.nodes if v not in target), kernel.expression)
        if ancestors == component:
            return None
        kernel = _marginal_kernel(kernel, ancestors)
        district = next(d for d in _districts_within(raw, ancestors) if target <= d)
        kernel = _extract(kernel, district)


def _rc(target: frozenset[str], initial: _Kernel, raw: DirectedBidirectedGraph,
        context: frozenset[str], budget: _Budget) -> ProbabilityExpression | None:
    kernel = initial
    vertices = frozenset(raw.nodes)
    while True:
        budget.take()
        keep = _ancestors_within(raw, target | context, vertices)
        if keep != vertices:
            kernel = _marginal_kernel(kernel, keep - context)
            vertices = keep
            continue
        ancestors_context = _ancestors_within(raw, context, vertices)
        eligible = tuple(d for d in _districts_within(raw, vertices) if not d & ancestors_context)
        if not eligible:
            return None
        factors = tuple((d, _extract(kernel, d)) for d in eligible)
        for d, factor in factors:
            if target <= d:
                return _tian_identify(target, factor, raw, budget)
        removed = frozenset().union(*eligible)
        # Removed variables remain fixed kernel parameters until the outer query
        # binds them. Dividing is not marginalization.
        kernel = _Kernel(tuple(v for v in kernel.nodes if v not in removed),
                         Ratio(kernel.expression, _product(tuple(k.expression for _, k in factors))), None)
        vertices = vertices - removed


def recover_causal_effect(
    graph: RecoveryGraph, data: Probability, *, outcomes: str | Iterable[str],
    interventions: str | Iterable[str] = (), max_states: int = 100_000,
) -> RecoveryResult:
    """Recover a population marginal causal effect using selected-only RC or ID.

    RC does not consume an external population table. Failure is returned as
    ``RecoveryNotEstablished``; budget exhaustion and unsupported inputs raise.
    """
    context, x, y = _query(graph, data, interventions, outcomes)
    budget = _Budget(max_states)
    raw, order = graph.topology, graph.topology.nodes
    xo, yo = _ordered_subset(order, x), _ordered_subset(order, y)
    if not context:
        identified = identify_causal_effect(graph, outcomes=yo, interventions=xo)
        if not isinstance(identified, IdentifiedEffect):
            return RecoveryNotEstablished(graph, data, yo, (), xo, "ordinary_id_failed", _token=_TOKEN)
        return RecoveredDistribution(graph, data, yo, (), xo, identified.formula, "id", _token=_TOKEN)
    ancestors_context = _ancestors_within(raw, context, frozenset(order))
    topological = _topological_order(raw)
    kernel_order = tuple(v for v in topological if v in ancestors_context - context) + tuple(
        v for v in topological if v not in ancestors_context)
    chain = tuple(_from_data(data, (v,), kernel_order[:i]) for i, v in enumerate(kernel_order))
    initial = _Kernel(kernel_order, _from_data(data, kernel_order), chain)
    d = _ancestors_within(raw, y, frozenset(data.variables) - x)
    recovered: list[ProbabilityExpression] = []
    for district in _districts_within(raw, d):
        budget.take()
        factor = _rc(district, initial, raw, context, budget)
        if factor is None:
            return RecoveryNotEstablished(graph, data, yo, (), xo, "rc_failed", _token=_TOKEN)
        recovered.append(factor)
    formula = _summation(_ordered_subset(order, d - y), _product(tuple(recovered)))
    # RC kernels can retain irrelevant non-query parameters. Since their product
    # equals the target for every parameter assignment, average them with an
    # available, normalized input marginal rather than leaving them free.
    extras = _ordered_subset(order, formula.free_variables - x - y)
    if extras:
        formula = _summation(extras, _product((formula, _from_data(data, extras))))
    if not y <= formula.free_variables <= x | y:
        raise AssertionError("RC formula is not bound to the requested query")
    return RecoveredDistribution(graph, data, yo, (), xo, formula, "rc", _token=_TOKEN)
