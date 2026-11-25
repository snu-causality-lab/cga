"""Efficient adjustment in explicit DAGs with observed and hidden vertices.

Smucler, Sapienza and Rotnitzky, arXiv:2004.10521v3 (2020), Definition 6,
Theorem 2 and Algorithms 1--3. See docs/SMUCLER_ADJUSTMENT.md for scope,
source-version discrepancies and independent verification.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, NoReturn, Self, SupportsIndex, TypeAlias, final

from .directed_bidirected import DAG


SmuclerObjective: TypeAlias = Literal["optimal", "optimal_minimal", "optimal_minimum"]
SmuclerFailureReason: TypeAlias = Literal[
    "no_observed_adjustment_set", "global_optimality_not_established"
]
_TOKEN = object()


class SmuclerAdjustmentDomainError(ValueError):
    """A request violates the paper's explicit-DAG inclusion assumptions."""


class _ImmutableValue:
    __slots__ = ()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("pickle is unsupported for Smucler adjustment results")


@final
@dataclass(frozen=True, init=False)
class SmuclerReduction(_ImmutableValue):
    """Immutable H0/H1 construction, retaining declaration order for inspection."""

    __slots__ = ("h0_vertices", "h0_edges", "vertices", "edges", "forbidden", "ignored")

    h0_vertices: tuple[str, ...]
    h0_edges: tuple[tuple[str, str], ...]
    vertices: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    forbidden: tuple[str, ...]
    ignored: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("SmuclerReduction cannot be subclassed")

    def __init__(
        self, token: object, *, h0_vertices: tuple[str, ...],
        h0_edges: tuple[tuple[str, str], ...], vertices: tuple[str, ...],
        edges: tuple[tuple[str, str], ...], forbidden: tuple[str, ...],
        ignored: tuple[str, ...],
    ) -> None:
        if token is not _TOKEN:
            raise TypeError("SmuclerReduction values are factory-created")
        object.__setattr__(self, "h0_vertices", h0_vertices)
        object.__setattr__(self, "h0_edges", h0_edges)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "forbidden", forbidden)
        object.__setattr__(self, "ignored", ignored)


@final
@dataclass(frozen=True, init=False)
class SmuclerAdjustmentSet(_ImmutableValue):
    """A valid observed adjustment set with the requested optimality guarantee."""

    __slots__ = ("graph", "treatment", "outcome", "observed", "policy_covariates",
                 "objective", "vertices", "reduction")

    graph: DAG
    treatment: str
    outcome: str
    observed: tuple[str, ...]
    policy_covariates: tuple[str, ...]
    objective: SmuclerObjective
    vertices: tuple[str, ...]
    reduction: SmuclerReduction

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("SmuclerAdjustmentSet cannot be subclassed")

    def __init__(
        self, token: object, *, graph: DAG, treatment: str, outcome: str,
        observed: tuple[str, ...], policy_covariates: tuple[str, ...],
        objective: SmuclerObjective, vertices: tuple[str, ...], reduction: SmuclerReduction,
    ) -> None:
        if token is not _TOKEN:
            raise TypeError("SmuclerAdjustmentSet values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "policy_covariates", policy_covariates)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "reduction", reduction)


@final
@dataclass(frozen=True, init=False)
class SmuclerAdjustmentNotApplicable(_ImmutableValue):
    """No observed adjustment, or no global guarantee from the paper's condition.

    ``valid_candidate`` is ``None`` in the first case; in the second it is
    the valid neighbor-of-Y set, without a global optimality assertion.
    """

    __slots__ = ("graph", "treatment", "outcome", "observed", "policy_covariates",
                 "objective", "reason", "valid_candidate", "reduction")

    graph: DAG
    treatment: str
    outcome: str
    observed: tuple[str, ...]
    policy_covariates: tuple[str, ...]
    objective: SmuclerObjective
    reason: SmuclerFailureReason
    valid_candidate: tuple[str, ...] | None
    reduction: SmuclerReduction

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("SmuclerAdjustmentNotApplicable cannot be subclassed")

    def __init__(
        self, token: object, *, graph: DAG, treatment: str, outcome: str,
        observed: tuple[str, ...], policy_covariates: tuple[str, ...],
        objective: SmuclerObjective, reason: SmuclerFailureReason,
        valid_candidate: tuple[str, ...] | None, reduction: SmuclerReduction,
    ) -> None:
        if token is not _TOKEN:
            raise TypeError("SmuclerAdjustmentNotApplicable values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "policy_covariates", policy_covariates)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "valid_candidate", valid_candidate)
        object.__setattr__(self, "reduction", reduction)


SmuclerAdjustmentResult: TypeAlias = SmuclerAdjustmentSet | SmuclerAdjustmentNotApplicable


def _ordered(nodes: tuple[str, ...], members: Iterable[str]) -> tuple[str, ...]:
    selected = set(members)
    return tuple(v for v in nodes if v in selected)


def _selector(nodes: tuple[str, ...], values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (bytes, bytearray)):
        raise SmuclerAdjustmentDomainError(f"{name} requires string vertex labels, not bytes")
    try:
        raw = (values,) if isinstance(values, str) else tuple(values)
    except TypeError as error:
        raise SmuclerAdjustmentDomainError(f"{name} requires an iterable of vertex labels") from error
    if any(not isinstance(v, str) or v not in nodes for v in raw):
        raise SmuclerAdjustmentDomainError(f"{name} contains an unknown or non-string vertex")
    if len(set(raw)) != len(raw):
        raise SmuclerAdjustmentDomainError(f"{name} contains a duplicate vertex")
    return _ordered(nodes, raw)


def _closure(seeds: Iterable[str], adjacency: dict[str, set[str]]) -> set[str]:
    reached = set(seeds)
    pending = list(reached)
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    return reached


def _edges(nodes: tuple[str, ...], adjacency: dict[str, set[str]]) -> tuple[tuple[str, str], ...]:
    return tuple((a, b) for a, b in combinations(nodes, 2) if b in adjacency[a])


def _reduce(
    graph: DAG, treatment: str, outcome: str, observed: tuple[str, ...],
    policy: tuple[str, ...], children: dict[str, set[str]],
) -> tuple[SmuclerReduction, dict[str, set[str]]]:
    nodes = graph.topology.nodes
    parents = {v: set[str]() for v in nodes}
    for a, b in graph.topology.directed:
        parents[b].add(a)
    ancestors_y = _closure((outcome,), parents)
    ancestors = _closure((treatment, outcome, *policy), parents)
    causal = (_closure((treatment,), children) - {treatment}) & ancestors_y
    forbidden = _closure(causal, children) | {treatment}

    # H0: ancestral restriction, proper-back-door edge deletion, moralization.
    h0_vertices = _ordered(nodes, ancestors)
    proper_parents = {v: {p for p in parents[v] & ancestors
                         if not (p == treatment and v in ancestors_y)} for v in h0_vertices}
    h0 = {v: set[str]() for v in h0_vertices}
    for vertex, vertex_parents in proper_parents.items():
        for parent in vertex_parents:
            h0[vertex].add(parent)
            h0[parent].add(vertex)
        for a, b in combinations(vertex_parents, 2):
            h0[a].add(b)
            h0[b].add(a)

    ignored = (ancestors - {treatment, outcome}) & ((set(nodes) - set(observed)) | forbidden)
    retained = ancestors - ignored
    h1 = {v: h0[v] & retained for v in h0_vertices if v in retained}
    # Every connected ignored component induces a clique on its retained boundary.
    unseen = set(ignored)
    while unseen:
        root = next(v for v in h0_vertices if v in unseen)
        pending = [root]
        boundary: set[str] = set()
        unseen.remove(root)
        while pending:
            current = pending.pop()
            boundary.update(h0[current] & retained)
            for neighbor in h0[current] & unseen:
                unseen.remove(neighbor)
                pending.append(neighbor)
        for a, b in combinations(boundary, 2):
            h1[a].add(b)
            h1[b].add(a)
    for vertex in policy:
        for endpoint in (treatment, outcome):
            h1[endpoint].add(vertex)
            h1[vertex].add(endpoint)

    vertices = _ordered(nodes, retained)
    reduction = SmuclerReduction(
        _TOKEN, h0_vertices=h0_vertices, h0_edges=_edges(h0_vertices, h0),
        vertices=vertices, edges=_edges(vertices, h1),
        forbidden=_ordered(nodes, forbidden), ignored=_ordered(nodes, ignored),
    )
    return reduction, h1


def _optimal_minimal(treatment: str, outcome: str, adjacency: dict[str, set[str]]) -> set[str]:
    """Paper Algorithm 3: stop DFS upon first reaching a neighbor of Y."""
    neighbors_y = adjacency[outcome]
    selected: set[str] = set()
    visited: set[str] = set()
    pending = [treatment]
    while pending:
        vertex = pending.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        if vertex in neighbors_y:
            selected.add(vertex)
        else:
            pending.extend(adjacency[vertex] - visited)
    return selected


def _augmenting_path(
    adjacency: tuple[tuple[int, ...], ...], amounts: list[dict[int, int]],
    source: int, sink: int,
) -> list[tuple[int, int]] | None:
    predecessors = {source: source}
    pending = deque((source,))
    while pending and sink not in predecessors:
        vertex = pending.popleft()
        for neighbor in adjacency[vertex]:
            if neighbor not in predecessors and amounts[vertex].get(neighbor, 0) > 0:
                predecessors[neighbor] = vertex
                pending.append(neighbor)
    if sink not in predecessors:
        return None
    path: list[tuple[int, int]] = []
    current = sink
    while current != source:
        previous = predecessors[current]
        path.append((previous, current))
        current = previous
    path.reverse()
    return path


def _maximum_vertex_flow(
    nodes: tuple[str, ...], adjacency: dict[str, set[str]], treatment: str, outcome: str,
    *, extract_paths: bool = False,
) -> tuple[int, tuple[tuple[str, ...], ...]]:
    """Integral node-split max flow with unit internal-vertex capacities.

    BFS augmentations implement Edmonds--Karp. The only calls are on graphs
    without an A--Y edge, so each augmentation has value one and at most n-2
    augmentations are possible. Opposite original undirected arcs are distinct
    after node splitting; they cannot be mistaken for residual reverse arcs.
    """
    positions = {v: i for i, v in enumerate(nodes)}
    capacities: list[dict[int, int]] = [{} for _ in range(2 * len(nodes))]
    infinity = len(nodes) + 1

    def arc(a: int, b: int, capacity: int) -> None:
        capacities[a][b] = capacity
        capacities[b].setdefault(a, 0)

    for vertex, index in positions.items():
        arc(2 * index, 2 * index + 1,
            infinity if vertex in (treatment, outcome) else 1)
    for from_vertex in nodes:
        for to_vertex in adjacency[from_vertex]:
            arc(2 * positions[from_vertex] + 1, 2 * positions[to_vertex], infinity)
    residual = [row.copy() for row in capacities]
    neighbors = tuple(tuple(sorted(row)) for row in capacities)
    source, sink = 2 * positions[treatment] + 1, 2 * positions[outcome]
    value = 0
    while (path := _augmenting_path(neighbors, residual, source, sink)) is not None:
        amount = min(residual[a][b] for a, b in path)
        for a, b in path:
            residual[a][b] -= amount
            residual[b][a] += amount
        value += amount
    if not extract_paths:
        return value, ()

    remaining_flow = [{b: max(0, capacity - residual[a][b]) for b, capacity in row.items()}
                      for a, row in enumerate(capacities)]
    paths: list[tuple[str, ...]] = []
    for _ in range(value):
        path = _augmenting_path(neighbors, remaining_flow, source, sink)
        if path is None:
            raise RuntimeError("cannot decompose integral maximum vertex flow")
        vertices = [treatment]
        for a, b in path:
            remaining_flow[a][b] -= 1
            if b % 2 == 0:
                vertices.append(nodes[b // 2])
        paths.append(tuple(vertices))
    return value, tuple(paths)


def _optimal_minimum(
    nodes: tuple[str, ...], adjacency: dict[str, set[str]], treatment: str, outcome: str,
) -> set[str]:
    """Paper Algorithms 1--2, caching the unchanged base minimum-cut size."""
    minimum_size, paths = _maximum_vertex_flow(nodes, adjacency, treatment, outcome, extract_paths=True)
    selected: set[str] = set()
    for path in paths:
        for vertex in reversed(path[1:-1]):
            augmented = {v: neighbors.copy() for v, neighbors in adjacency.items()}
            for endpoint in (treatment, outcome):
                augmented[vertex].add(endpoint)
                augmented[endpoint].add(vertex)
            forced_size, _ = _maximum_vertex_flow(nodes, augmented, treatment, outcome)
            if forced_size == minimum_size:
                selected.add(vertex)
                break
        else:
            raise RuntimeError("no minimum-cut vertex on a maximum-flow path")
    if len(selected) != minimum_size:
        raise RuntimeError("optimal-minimum cut has inconsistent cardinality")
    return selected


def smucler_adjustment_set(
    graph: DAG, treatment: str, outcome: str, *, observed: Iterable[str] | None = None,
    policy_covariates: Iterable[str] = (), objective: SmuclerObjective = "optimal_minimal",
) -> SmuclerAdjustmentResult:
    """Compute a published efficient observed adjustment set for a single exposure.

    ``graph`` includes hidden vertices explicitly; ``observed=None`` means all
    vertices are observed. Policy covariates must be observed non-descendants
    of treatment and are mandatory in the returned set. Outcome must be a
    strict descendant of treatment, as assumed throughout the source paper.

    ``optimal_minimal`` and ``optimal_minimum`` respectively optimize among
    inclusion-minimal and smallest-cardinality observed adjustment sets.
    ``optimal`` asserts global optimality only when all vertices are observed
    or observed vertices lie among ancestors of treatment, outcome and policy.
    Failure of that sufficient condition does not assert nonexistence.

    Selectors accept distinct labels; a string means one vertex, not characters.
    Every returned vertex tuple follows the graph's declared order. No search
    enumerates adjustment sets or has an exponential fallback.
    """
    if not isinstance(graph, DAG):
        raise SmuclerAdjustmentDomainError("Smucler adjustment requires a validated DAG with explicit hidden vertices")
    nodes = graph.topology.nodes
    for name, value in (("treatment", treatment), ("outcome", outcome)):
        if not isinstance(value, str) or value not in nodes:
            raise SmuclerAdjustmentDomainError(f"{name} must be a declared string vertex")
    if treatment == outcome:
        raise SmuclerAdjustmentDomainError("treatment and outcome must be distinct")
    if not isinstance(objective, str) or objective not in ("optimal", "optimal_minimal", "optimal_minimum"):
        raise SmuclerAdjustmentDomainError("unknown Smucler adjustment objective")
    observed_order = nodes if observed is None else _selector(nodes, observed, "observed")
    policy = _selector(nodes, policy_covariates, "policy_covariates")
    if not {treatment, outcome, *policy} <= set(observed_order):
        raise SmuclerAdjustmentDomainError("treatment, outcome and policy covariates must be observed")
    children = {v: set[str]() for v in nodes}
    for a, b in graph.topology.directed:
        children[a].add(b)
    descendants_a = _closure((treatment,), children)
    if outcome not in descendants_a:
        raise SmuclerAdjustmentDomainError("outcome must be a strict descendant of treatment")
    if set(policy) & descendants_a:
        raise SmuclerAdjustmentDomainError("policy covariates must be non-descendants of treatment")

    reduction, h1 = _reduce(graph, treatment, outcome, observed_order, policy, children)
    if outcome in h1[treatment]:
        return SmuclerAdjustmentNotApplicable(
            _TOKEN, graph=graph, treatment=treatment, outcome=outcome, observed=observed_order,
            policy_covariates=policy, objective=objective,
            reason="no_observed_adjustment_set", valid_candidate=None, reduction=reduction,
        )
    if objective == "optimal":
        selected = h1[outcome]
        if set(observed_order) != set(nodes) and not set(observed_order) <= set(reduction.h0_vertices):
            return SmuclerAdjustmentNotApplicable(
                _TOKEN, graph=graph, treatment=treatment, outcome=outcome, observed=observed_order,
                policy_covariates=policy, objective=objective,
                reason="global_optimality_not_established",
                valid_candidate=_ordered(nodes, selected), reduction=reduction,
            )
    elif objective == "optimal_minimal":
        selected = _optimal_minimal(treatment, outcome, h1)
    else:
        selected = _optimal_minimum(reduction.vertices, h1, treatment, outcome)
    return SmuclerAdjustmentSet(
        _TOKEN, graph=graph, treatment=treatment, outcome=outcome, observed=observed_order,
        policy_covariates=policy, objective=objective, vertices=_ordered(nodes, selected), reduction=reduction,
    )
