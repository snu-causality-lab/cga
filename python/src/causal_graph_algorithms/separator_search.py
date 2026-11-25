"""Exact positive-cost cuts and constrained ListSep enumeration.

van der Zander, Liśkiewicz and Textor (2019), sections 3.2 and 3.5.
The implementation uses exact rational Edmonds–Karp flow; it does not inherit
the source's cubic-time bound. Optional minimal-output filtering is exhaustive
ListSep followed by I-minimality, not Takata's polynomial-delay ListMinSep.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import InitVar, dataclass
from fractions import Fraction
from typing import TypeAlias, final

from ._immutable import _ImmutableValue
from ._validation import node_positions, normalize_seeds
from .adjustment import (
    AdjustmentDomainError, AdjustmentGraph, AdjustmentNotFound, AdjustmentSet,
    _original_conditions, _proper_backdoor, _query,
)
from .errors import GraphValidationError
from .separation import SeparationView
from .separators import (
    SeparatorDomainError, SeparatorNotFound, SeparatorSet, _augmented_reduction,
    _ordered, _sets, find_m_separator, m_separator_is_minimal,
)
from .adjustment import _TOKEN as _ADJUSTMENT_TOKEN
from .separators import _TOKEN as _SEPARATOR_TOKEN


Cost: TypeAlias = int | Fraction
_TOKEN = object()


class SearchBudgetExceeded(RuntimeError):
    """Search did not finish; no impossibility or completeness conclusion follows."""


@final
@dataclass(frozen=True, slots=True)
class MinimumCostSeparator(_ImmutableValue):
    """A globally minimum positive-cost separator under the stored constraints."""
    graph: SeparationView
    left: tuple[str, ...]
    right: tuple[str, ...]
    required: tuple[str, ...]
    allowed: tuple[str, ...]
    vertices: tuple[str, ...]
    costs: tuple[tuple[str, Fraction], ...]
    total_cost: Fraction
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("minimum-cost results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MinimumCostSeparator cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class MinimumCostAdjustment(_ImmutableValue):
    """A minimum-cost ordinary adjustment set, without variance optimality."""
    graph: AdjustmentGraph
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    required: tuple[str, ...]
    allowed: tuple[str, ...]
    vertices: tuple[str, ...]
    forbidden_vertices: tuple[str, ...]
    costs: tuple[tuple[str, Fraction], ...]
    total_cost: Fraction
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("minimum-cost results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MinimumCostAdjustment cannot be subclassed")


class _Budget:
    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit <= 0:
            raise SeparatorDomainError("max_states must be a positive integer")
        self.remaining = limit

    def step(self) -> None:
        if not self.remaining:
            raise SearchBudgetExceeded("separator search exhausted max_states")
        self.remaining -= 1


def _costs(nodes: tuple[str, ...], allowed: frozenset[str],
           costs: Mapping[str, Cost] | None) -> dict[str, Fraction]:
    if costs is not None and (not isinstance(costs, Mapping) or set(costs) != set(allowed)):
        raise SeparatorDomainError("costs must specify exactly the allowed vertices")
    result: dict[str, Fraction] = {}
    for v in nodes:
        if v not in allowed:
            continue
        value = 1 if costs is None else costs[v]
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)) or value <= 0:
            raise SeparatorDomainError("costs must be strictly positive integers or Fractions")
        result[v] = Fraction(value)
    return result


def _minimum_cut(
    adjacency: dict[str, set[str]], nodes: tuple[str, ...], x: frozenset[str],
    y: frozenset[str], required: frozenset[str], allowed: frozenset[str],
    costs: dict[str, Fraction], budget: _Budget,
) -> frozenset[str] | None:
    """Split vertices; finite infinity exceeds every admissible cut combined."""
    active = tuple(v for v in nodes if v in adjacency and v not in required)
    index = {v: j for j, v in enumerate(active)}
    source, sink = 2 * len(active), 2 * len(active) + 1
    residual: list[dict[int, Fraction]] = [{} for _ in range(sink + 1)]
    infinity = sum(costs.values(), Fraction()) + 1

    def arc(a: int, b: int, capacity: Fraction) -> None:
        residual[a][b] = residual[a].get(b, Fraction()) + capacity
        residual[b].setdefault(a, Fraction())

    for v in active:
        a, b = 2 * index[v], 2 * index[v] + 1
        arc(a, b, costs[v] if v in allowed else infinity)
        if v in x:
            arc(source, a, infinity)
        if v in y:
            arc(b, sink, infinity)
        for w in nodes:
            if w in index and w in adjacency[v]:
                arc(b, 2 * index[w], infinity)
    flow = Fraction()
    while True:
        parent = {source: -1}
        pending = deque([source])
        while pending and sink not in parent:
            budget.step()
            a = pending.popleft()
            for b, capacity in residual[a].items():
                if capacity > 0 and b not in parent:
                    parent[b] = a
                    pending.append(b)
        if sink not in parent:
            return frozenset(v for v in active
                             if 2 * index[v] in parent and 2 * index[v] + 1 not in parent)
        amount = infinity
        b = sink
        while b != source:
            a = parent[b]
            amount = min(amount, residual[a][b])
            b = a
        flow += amount
        if flow >= infinity:
            return None
        b = sink
        while b != source:
            a = parent[b]
            residual[a][b] -= amount
            residual[b][a] += amount
            b = a


def find_minimum_cost_m_separator(
    graph: SeparationView, left: str | Iterable[str], right: str | Iterable[str], *,
    costs: Mapping[str, Cost] | None = None, required: str | Iterable[str] = (),
    allowed: str | Iterable[str] | None = None, max_states: int = 100_000,
) -> MinimumCostSeparator | SeparatorNotFound:
    """Find a global minimum-cost m-separator I ⊆ Z ⊆ R.

    Costs are exact, strictly positive and specified for every allowed vertex;
    omitted costs are all one. DAG, acyclic bow-capable ADMG and validated
    ancestral/MAG separation views are accepted. Ties are deterministic in node
    order. A resource exception is distinct from a proven absent separator.
    """
    nodes, x, y, i, r = _sets(graph, left, right, required, allowed)
    weights = _costs(nodes, r, costs)
    budget = _Budget(max_states)
    relevant, adjacency = _augmented_reduction(graph, x | y, i)
    # The ADMG reduction may introduce latent roots, which must remain uncuttable.
    augmented_nodes = nodes + tuple(sorted(relevant - set(nodes)))
    cut = _minimum_cut(adjacency, augmented_nodes, x, y, i, r, weights, budget)
    metadata = (graph, _ordered(nodes, x), _ordered(nodes, y), _ordered(nodes, i), _ordered(nodes, r))
    if cut is None:
        return SeparatorNotFound(*metadata, _token=_SEPARATOR_TOKEN)
    vertices = _ordered(nodes, cut | i)
    return MinimumCostSeparator(*metadata, vertices=vertices,
                                costs=tuple(weights.items()),
                                total_cost=sum((weights[v] for v in vertices), Fraction()),
                                _token=_TOKEN)


def iter_m_separators(
    graph: SeparationView, left: str | Iterable[str], right: str | Iterable[str], *,
    required: str | Iterable[str] = (), allowed: str | Iterable[str] | None = None,
    minimal: bool = False, max_states: int = 100_000,
) -> Iterator[SeparatorSet]:
    """List every constrained separator once by feasibility-pruned ListSep.

    Validation is eager, traversal lazy and deterministic (include before exclude
    in graph order). ``minimal=True`` filters I-minimal sets relative to the
    original required set, with no polynomial-delay promise. Exhaustion raises,
    including after earlier yields; an interrupted stream is never complete.
    """
    nodes, x, y, i, r = _sets(graph, left, right, required, allowed)
    budget = _Budget(max_states)
    if type(minimal) is not bool:
        raise SeparatorDomainError("minimal must be bool")

    def generate() -> Iterator[SeparatorSet]:
        pending = [(i, r)]
        while pending:
            budget.step()
            included, available = pending.pop()
            if isinstance(find_m_separator(graph, x, y, required=included, allowed=available), SeparatorNotFound):
                continue
            remaining = available - included
            if remaining:
                vertex = next(v for v in nodes if v in remaining)
                pending.append((included, available - {vertex}))
                pending.append((included | {vertex}, available))
            elif not minimal or m_separator_is_minimal(graph, x, y, included, required=i):
                yield SeparatorSet(graph, _ordered(nodes, x), _ordered(nodes, y),
                                   _ordered(nodes, i), _ordered(nodes, r),
                                   _ordered(nodes, included), minimal, _token=_SEPARATOR_TOKEN)
    return generate()


def _adjustment_constraints(
    graph: AdjustmentGraph, interventions: str | Iterable[str], outcomes: str | Iterable[str],
    required: str | Iterable[str], allowed: str | Iterable[str] | None,
) -> tuple[tuple[str, ...], frozenset[str], frozenset[str], frozenset[str], frozenset[str], bool, frozenset[str]]:
    nodes, x, y = _query(graph, interventions, outcomes)
    try:
        positions = node_positions(nodes)
        i = normalize_seeds(positions, required)
        r = frozenset(nodes) - x - y if allowed is None else normalize_seeds(positions, allowed)
    except GraphValidationError as error:
        raise AdjustmentDomainError(str(error)) from error
    if (i | r) & (x | y) or not i <= r:
        raise AdjustmentDomainError("require I ⊆ R, with both excluding query endpoints")
    amenable, forbidden = _original_conditions(graph, x, y)
    return nodes, x, y, i, r, amenable, forbidden


def find_minimum_cost_adjustment_set(
    graph: AdjustmentGraph, *, interventions: str | Iterable[str], outcomes: str | Iterable[str],
    costs: Mapping[str, Cost] | None = None, required: str | Iterable[str] = (),
    allowed: str | Iterable[str] | None = None, max_states: int = 100_000,
) -> MinimumCostAdjustment | AdjustmentNotFound:
    """Minimize positive covariate cost among all ordinary valid adjustment sets.

    Uses the existing generalized-adjustment graph domain and premises. A failed
    adjustment search does not establish general causal nonidentification.
    """
    nodes, x, y, i, r, amenable, forbidden = _adjustment_constraints(
        graph, interventions, outcomes, required, allowed)
    try:
        weights = _costs(nodes, r, costs)
        _Budget(max_states)
    except SeparatorDomainError as error:
        raise AdjustmentDomainError(str(error)) from error
    metadata = (graph, _ordered(nodes, x), _ordered(nodes, y), _ordered(nodes, i), _ordered(nodes, r))
    forbidden_vertices = _ordered(nodes, forbidden)
    if not amenable or i & forbidden:
        return AdjustmentNotFound(*metadata, reason="not_amenable" if not amenable else "required_forbidden",
                                  forbidden_vertices=forbidden_vertices, _token=_ADJUSTMENT_TOKEN)
    result = find_minimum_cost_m_separator(_proper_backdoor(graph, x, y), x, y, required=i,
                                          allowed=r - forbidden,
                                          costs={v: w for v, w in weights.items() if v not in forbidden},
                                          max_states=max_states)
    if isinstance(result, SeparatorNotFound):
        return AdjustmentNotFound(*metadata, reason="no_separator", forbidden_vertices=forbidden_vertices,
                                  _token=_ADJUSTMENT_TOKEN)
    return MinimumCostAdjustment(*metadata, vertices=result.vertices,
                                 forbidden_vertices=forbidden_vertices, costs=tuple(weights.items()),
                                 total_cost=result.total_cost, _token=_TOKEN)


def iter_adjustment_sets(
    graph: AdjustmentGraph, *, interventions: str | Iterable[str], outcomes: str | Iterable[str],
    required: str | Iterable[str] = (), allowed: str | Iterable[str] | None = None,
    minimal: bool = False, max_states: int = 100_000,
) -> Iterator[AdjustmentSet]:
    """List all valid ordinary adjustment sets within the inclusion constraints.

    ``minimal=True`` selects I-minimal sets. Normal exhaustion establishes
    completeness of this adjustment-set listing only, not of general ID.
    """
    nodes, x, y, i, r, amenable, forbidden = _adjustment_constraints(
        graph, interventions, outcomes, required, allowed)
    try:
        _Budget(max_states)
        if type(minimal) is not bool:
            raise SeparatorDomainError("minimal must be bool")
    except SeparatorDomainError as error:
        raise AdjustmentDomainError(str(error)) from error
    if not amenable or i & forbidden:
        return iter(())
    separators = iter_m_separators(_proper_backdoor(graph, x, y), x, y, required=i,
                                  allowed=r - forbidden, minimal=minimal, max_states=max_states)

    def generate() -> Iterator[AdjustmentSet]:
        for separator in separators:
            yield AdjustmentSet(graph, _ordered(nodes, x), _ordered(nodes, y),
                                _ordered(nodes, i), _ordered(nodes, r), separator.vertices,
                                minimal, _ordered(nodes, forbidden), _token=_ADJUSTMENT_TOKEN)
    return generate()
