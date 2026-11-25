"""Selection recovery with an explicitly supplied same-population margin.

Correa--Tian--Bareinboim (AAAI 2019), R-38 Algorithms 2 (RCE) and 3
(IDSB), using Theorem 3 and its Appendix C construction. Positive outputs have
the soundness basis of Theorem 4/Appendix D; failure is always unresolved.
The paper's separate selected-only completeness claim is not used.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from typing import Literal, TypeAlias, final

from ._immutable import _ImmutableValue
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DirectedBidirectedGraph
from .identification import (
    _Kernel, _ancestors_within, _descendants_within, _districts_within,
    _kernel_chain_factors, _marginal_kernel, _ordered_subset, _product,
    _summation, _topological_order,
)
from .probability import Probability, ProbabilityExpression, Ratio
from .projection import latent_projection
from .selection_recovery import (
    RecoveryDomainError, RecoveryGraph, _Budget, _extract, _population, _query,
    _tian_identify,
)
from .separation import m_separated_sets
from .subgraph import induced_subgraph

_TOKEN = object()


@final
@dataclass(frozen=True, slots=True)
class RecoveredExternalEffect(_ImmutableValue):
    """A population effect expressed using exactly the two declared sources."""
    graph: RecoveryGraph
    data: Probability
    population_data: Probability | None
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    formula: ProbabilityExpression
    population_variables: tuple[str, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("external recovery results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("RecoveredExternalEffect cannot be subclassed")

    @property
    def method(self) -> Literal["idsb"]:
        return "idsb"


@final
@dataclass(frozen=True, slots=True)
class ExternalRecoveryNotEstablished(_ImmutableValue):
    """IDSB produced no formula; this is not a nonrecoverability conclusion."""
    graph: RecoveryGraph
    data: Probability
    population_data: Probability | None
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    population_variables: tuple[str, ...]
    reason: Literal["idsb_failed"] = "idsb_failed"
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("external recovery results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ExternalRecoveryNotEstablished cannot be subclassed")


ExternalRecoveryResult: TypeAlias = RecoveredExternalEffect | ExternalRecoveryNotEstablished


def _marginal(expression: ProbabilityExpression, variables: tuple[str, ...],
              retained: frozenset[str]) -> ProbabilityExpression:
    return _summation(tuple(v for v in variables if v not in retained), expression)


def _view(raw: DirectedBidirectedGraph, vertices: frozenset[str]
          ) -> AcyclicDirectedBidirectedGraph:
    return AcyclicDirectedBidirectedGraph.from_graph(induced_subgraph(raw, vertices))


def _separated_extension(raw: DirectedBidirectedGraph, vertices: frozenset[str],
                         context: frozenset[str], base: frozenset[str], budget: _Budget
                         ) -> frozenset[str]:
    view = _view(raw, vertices)
    added: set[str] = set()
    for v in raw.nodes:
        if v in vertices - context - base:
            budget.take()
            if m_separated_sets(view, context, (v,), base):
                added.add(v)
    return base | frozenset(added)


def _selection_order(raw: DirectedBidirectedGraph, vertices: frozenset[str],
                     context: frozenset[str], ancestors: frozenset[str]) -> tuple[str, ...]:
    order = _topological_order(induced_subgraph(raw, vertices))
    return tuple(v for v in order if v in ancestors - context) + tuple(
        v for v in order if v not in ancestors | context)


@dataclass(slots=True)
class _Sources:
    raw: DirectedBidirectedGraph
    vertices: frozenset[str]
    context: frozenset[str]
    population_variables: frozenset[str]
    population: _Kernel
    budget: _Budget


def _theorem_three(sources: _Sources, vertices: frozenset[str],
                   expression: ProbabilityExpression
                   ) -> tuple[tuple[frozenset[str], _Kernel], ...]:
    """Construct Eq. (6)/(86), not an inferred closure of available sources."""
    raw, context, budget = sources.raw, sources.context, sources.budget
    districts = _districts_within(raw, vertices)
    selection_component = next(d for d in districts if context <= d)
    ancestors = _ancestors_within(raw, selection_component, vertices)
    affected: set[str] = set()
    for v in sources.vertices - vertices:
        affected.update(_descendants_within(raw, v, sources.vertices))
    base = sources.population_variables - affected
    available = _separated_extension(raw, vertices, context, base, budget)
    unavailable = vertices - available
    # R-38's Ch/Pa include their seeds (p. 2); omitting R_H is unsound.
    children = unavailable | frozenset(b for a, b in raw.directed
                                     if a in unavailable and b in vertices)
    eligible = tuple(d for d in districts if not d & context
                     and not d & ancestors & children)
    if not eligible:
        return ()
    order = _selection_order(raw, vertices, context, ancestors)
    selected_kernel = _Kernel(order, expression, None)
    selected_factors = dict(zip(order, _kernel_chain_factors(selected_kernel)))
    # Lemma 9, Eq. (69): Q[T_H] under interventions on the removed vertices.
    population_base = _marginal_kernel(sources.population, base).expression
    recovered_population = _product((
        Ratio(_marginal_kernel(selected_kernel, available).expression,
              _marginal_kernel(selected_kernel, base).expression), population_base))
    population_order = tuple(v for v in order if v in available)
    population_kernel = _Kernel(population_order, recovered_population, None)
    population_factors = dict(zip(population_order, _kernel_chain_factors(population_kernel)))
    result: list[tuple[frozenset[str], _Kernel]] = []
    for district in eligible:
        budget.take()
        factors = tuple(population_factors[v] if v in ancestors else selected_factors[v]
                        for v in order if v in district)
        result.append((district, _Kernel(tuple(v for v in order if v in district),
                                        _product(factors), None)))
    return tuple(result)


@dataclass(slots=True)
class _RCEFrame:
    target: frozenset[str]
    vertices: frozenset[str]
    kernel: _Kernel
    candidates: tuple[tuple[frozenset[str], frozenset[str]], ...] | None = None
    next_candidate: int = 0
    waiting: frozenset[str] | None = None


def _rce(target: frozenset[str], initial: _Kernel, sources: _Sources
         ) -> ProbabilityExpression | None:
    """Algorithm 2 with an explicit stack; every nested call shrinks its graph."""
    raw, context, budget = sources.raw, sources.context, sources.budget
    stack = [_RCEFrame(target, sources.vertices, initial)]
    completed = False
    answer: ProbabilityExpression | None = None
    while stack:
        frame = stack[-1]
        if completed:
            removed = frame.waiting
            if removed is None:
                raise AssertionError("RCE continuation has no pending district")
            frame.waiting = None
            if answer is not None:
                frame.vertices -= removed
                frame.kernel = _Kernel(tuple(v for v in frame.kernel.nodes if v not in removed),
                                       Ratio(frame.kernel.expression, answer), None)
                frame.candidates = None
                frame.next_candidate = 0
            completed = False
            continue
        if frame.candidates is None:
            budget.take()
            keep = _ancestors_within(raw, frame.target | context, frame.vertices)
            if keep != frame.vertices:
                frame.kernel = _marginal_kernel(frame.kernel, keep - context)
                frame.vertices = keep
                continue
            factors = _theorem_three(sources, frame.vertices, frame.kernel.expression)
            if factors:
                containing = next((k for d, k in factors if frame.target <= d), None)
                if containing is not None:
                    answer = _tian_identify(frame.target, containing, raw, budget)
                    stack.pop()
                    completed = True
                    continue
                removed = frozenset().union(*(d for d, _ in factors))
                frame.kernel = _Kernel(tuple(v for v in frame.kernel.nodes if v not in removed),
                    Ratio(frame.kernel.expression, _product(tuple(k.expression for _, k in factors))),
                    None)
                frame.vertices -= removed
                continue
            candidates: list[tuple[frozenset[str], frozenset[str]]] = []
            for district in _districts_within(raw, frame.vertices):
                if district & context or frame.target <= district:
                    continue
                keep = _ancestors_within(raw, district | context, frame.vertices)
                if keep != frame.vertices:
                    candidates.append((district, keep))
            frame.candidates = tuple(candidates)
        if frame.next_candidate < len(frame.candidates):
            district, keep = frame.candidates[frame.next_candidate]
            frame.next_candidate += 1
            frame.waiting = district
            child = _RCEFrame(district, keep, _marginal_kernel(frame.kernel, keep - context))
            stack.append(child)
        else:
            stack.pop()
            answer = None
            completed = True
    return answer


def _factor_groups(raw: DirectedBidirectedGraph, d: frozenset[str], b: frozenset[str]
                   ) -> tuple[tuple[frozenset[str], frozenset[str],
                                    tuple[frozenset[str], ...]], ...]:
    """Eq. (12): merge districts joined by arrows out of marginalized B nodes.

    This requires IDSB's ancestral D and B⊆D minus Y: every hidden
    B vertex has a directed route in D to a retained vertex. Hidden collider
    routes therefore connect through retained descendants in the projection.
    Without that premise, arbitrary projection may split original districts
    and this contraction is not valid. This derives the source's F_j/B_j.
    """
    if not b <= d or _ancestors_within(raw, d - b, d) != d:
        raise RecoveryDomainError("factor grouping requires the ancestral IDSB domain")
    districts = _districts_within(raw, d)
    owner = {v: i for i, district in enumerate(districts) for v in district}
    adjacent: dict[int, set[int]] = {i: set() for i in range(len(districts))}
    for a, child in raw.directed:
        if a in b and child in d:
            i, j = owner[a], owner[child]
            adjacent[i].add(j)
            adjacent[j].add(i)
    remaining = set(adjacent)
    groups = []
    while remaining:
        first = min(remaining)
        reached, pending = {first}, [first]
        while pending:
            current = pending.pop()
            for neighbor in sorted(adjacent[current] - reached):
                reached.add(neighbor)
                pending.append(neighbor)
        remaining -= reached
        factors = tuple(districts[i] for i in sorted(reached))
        members = frozenset().union(*factors)
        retained = members - b
        if retained:
            groups.append((retained, members & b, factors))
    return tuple(groups)


def recover_causal_effect_with_external_data(
    graph: RecoveryGraph, data: Probability, *, outcomes: str | Iterable[str],
    interventions: str | Iterable[str] = (), population_data: Probability | None = None,
    max_states: int = 100_000,
) -> ExternalRecoveryResult:
    """Recover P(Y|do(X)) with R-38 IDSB and an explicit joint margin P(T0).

    Requires a known DAG/ADMG and the full selected joint at one fixed ordinary
    sink event. The optional unconditional margin is from the same population.
    Positivity, causal model compatibility and source correctness are caller
    premises, not inferable from symbolic inputs. Success returns a formula;
    failure is unresolved. Exhaustion raises RecoveryBudgetExceeded separately.
    ``max_states`` counts processing/extension states, not time or output size.
    """
    context, x, y = _query(graph, data, interventions, outcomes)
    if not context:
        raise RecoveryDomainError("external selection recovery requires one fixed sink event")
    t0 = _population(data, population_data)
    budget = _Budget(max_states)
    raw, order = graph.topology, graph.topology.nodes
    all_vertices = frozenset(order)
    budget.take()
    w = _ancestors_within(raw, y | context, all_vertices)
    observed_w = w - context
    kernel_order = tuple(v for v in _topological_order(raw) if v in observed_w)
    selected = _marginal(data, data.variables, observed_w)
    initial = _Kernel(kernel_order, selected, None)
    base = t0 & w
    t = _separated_extension(raw, w, context, base, budget)
    margin = (_product(()) if population_data is None else
              _marginal(population_data, population_data.variables, base))
    expanded_population = _product((Ratio(_marginal_kernel(initial, t).expression,
                                         _marginal_kernel(initial, base).expression), margin))
    population_order = tuple(v for v in kernel_order if v in t)
    population = _Kernel(population_order, expanded_population, None)
    sources = _Sources(raw, w, context, t, population, budget)
    d = _ancestors_within(raw, y, observed_w - x)
    b = (d - y) - t
    r = observed_w - t
    projected = latent_projection(_view(raw, observed_w), t)
    projected_districts = projected.districts()
    projected_kernels = tuple((district, _extract(population, district))
                              for district in projected_districts)
    recovered: list[ProbabilityExpression] = []
    for c, bj, fj in _factor_groups(raw, d, b):
        budget.take()
        parents = c | frozenset(a for a, child in raw.directed if child in c and a in observed_w)
        factor = None
        if b & parents == r & parents:
            containing = next((k for district, k in projected_kernels if c <= district), None)
            if containing is not None:
                factor = _tian_identify(c, containing, projected.topology, budget)
        if factor is None:
            parts: list[ProbabilityExpression] = []
            for district in fj:
                part = _rce(district, initial, sources)
                if part is None:
                    return ExternalRecoveryNotEstablished(
                        graph, data, population_data, _ordered_subset(order, y),
                        _ordered_subset(order, x), _ordered_subset(order, t), _token=_TOKEN)
                parts.append(part)
            factor = _summation(_ordered_subset(order, bj), _product(tuple(parts)))
        recovered.append(factor)
    formula = _summation(_ordered_subset(order, (d - y) & t), _product(tuple(recovered)))
    extras = _ordered_subset(order, formula.free_variables - x - y)
    if extras:
        # The effect is invariant in these fixed kernel parameters. Bind them
        # against a normalized available selected marginal, without extra data.
        weight = _marginal(data, data.variables, frozenset(extras))
        formula = _summation(extras, _product((formula, weight)))
    if not y <= formula.free_variables <= x | y:
        raise AssertionError("IDSB formula is not bound to its requested query")
    return RecoveredExternalEffect(graph, data, population_data, _ordered_subset(order, y),
                                   _ordered_subset(order, x), formula,
                                   _ordered_subset(order, t), _token=_TOKEN)
