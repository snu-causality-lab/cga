"""Runge's ADMG O-set and Theorem 3 graphical optimality conditions.

Reference: Runge (NeurIPS 2021), arXiv:2102.10324v4, Definitions 3--4,
Algorithm C.1 and Theorem 3. See docs/RUNGE_ADJUSTMENT.md for assumptions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass, replace
from typing import NoReturn, Self, SupportsIndex, final

from ._validation import node_positions, normalize_seeds, require_vertex
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .errors import GraphValidationError
from .separation import m_separated


_TOKEN = object()


class RungeAdjustmentDomainError(ValueError):
    """The query is outside Runge's single-treatment/single-outcome domain."""


class RungeAdjustmentResourceLimitError(RuntimeError):
    """A caller-supplied Theorem 3 path or partial-state budget was exhausted."""


@final
@dataclass(frozen=True, slots=True)
class RungeAdjustmentResult:
    """Declaration-ordered O-set components and separately evaluated claims.

    ``adjustment_set`` contains additional covariates (fixed conditions excluded).
    ``None`` means that no valid adjustment set exists. An empty tuple is a valid
    empty set. ``graphical_optimality=None`` means not evaluated or inapplicable;
    construction alone never implies graphical optimality.
    """

    treatment: str
    outcome: str
    conditioned_on: tuple[str, ...]
    adjustment_set: tuple[str, ...] | None
    o_set: tuple[str, ...] | None
    parents: tuple[str, ...]
    colliders: tuple[str, ...]
    collider_parents: tuple[str, ...]
    mediators: tuple[str, ...]
    forbidden: tuple[str, ...]
    valid_ancestors: tuple[str, ...]
    graphical_optimality: bool | None = None
    criterion_i: bool | None = None
    criterion_ii: bool | None = None
    unique_adjustment_set: bool | None = None
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("Runge adjustment results are factory-created")

    @property
    def vertices(self) -> tuple[str, ...] | None:
        """Additional adjustment vertices, excluding ``conditioned_on``."""
        return self.adjustment_set

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("RungeAdjustmentResult cannot be subclassed")

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("pickle is unsupported for Runge adjustment results")


@dataclass(frozen=True)
class _Query:
    graph: AcyclicDirectedBidirectedGraph | DAG
    treatment: str
    outcome: str
    fixed: frozenset[str]
    mediators: frozenset[str]
    forbidden: frozenset[str]
    backdoor: AcyclicDirectedBidirectedGraph

    def ordered(self, vertices: Iterable[str]) -> tuple[str, ...]:
        members = set(vertices)
        return tuple(v for v in self.graph.topology.nodes if v in members)

    def valid(self, additional: Iterable[str], fixed: frozenset[str] | None = None) -> bool:
        total = frozenset(additional) | (self.fixed if fixed is None else fixed)
        return not (total & self.forbidden) and m_separated(
            self.backdoor, self.treatment, self.outcome, total
        )


def _query(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    treatment: str,
    outcome: str,
    conditioned_on: str | Iterable[str],
) -> _Query:
    if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
        raise RungeAdjustmentDomainError("Runge adjustment requires an acyclic directed-bidirected or DAG view")
    topology = graph.topology
    positions = node_positions(topology.nodes)
    try:
        require_vertex(positions, treatment)
        require_vertex(positions, outcome)
        fixed = normalize_seeds(positions, conditioned_on)
    except GraphValidationError as error:
        raise RungeAdjustmentDomainError(str(error)) from error
    if treatment == outcome:
        raise RungeAdjustmentDomainError("treatment and outcome must be distinct")
    descendants = topology.directed_descendants_strict(treatment)
    if outcome not in descendants:
        raise RungeAdjustmentDomainError("Runge adjustment requires a directed causal path from treatment to outcome")
    mediators = descendants & topology.directed_ancestors_strict(outcome) - {treatment, outcome}
    forbidden = topology.directed_descendants_reflexive(mediators | {outcome}) | {treatment}
    if fixed & forbidden:
        raise RungeAdjustmentDomainError("conditioned_on intersects treatment or descendants of outcome/mediators")
    if fixed & descendants:
        raise RungeAdjustmentDomainError(
            "conditioned_on must exclude descendants of treatment: posttreatment "
            "conditioning is outside the causal-estimand guarantee"
        )
    # Delete exactly the first directed edge of every proper causal path X -> Y.
    causal_children = topology.directed_children(treatment) & (mediators | {outcome})
    backdoor = AcyclicDirectedBidirectedGraph(DirectedBidirectedGraph(
        nodes=topology.nodes,
        directed=(edge for edge in topology.directed if not (edge[0] == treatment and edge[1] in causal_children)),
        bidirected=topology.bidirected,
    ))
    return _Query(graph, treatment, outcome, fixed, mediators, forbidden, backdoor)


def _construct(query: _Query, fixed: frozenset[str]) -> RungeAdjustmentResult:
    topology = query.graph.topology
    targets = query.mediators | {query.outcome}
    ancestors = topology.directed_ancestors_reflexive({query.treatment, query.outcome} | fixed) - query.forbidden
    parents = {a for a, b in topology.directed if b in targets} - query.forbidden
    colliders: set[str] = set()
    rejected = set(query.forbidden)
    pending = list(query.ordered(targets))
    impossible = False
    # Def. 3 eligibility depends only on vancs, not on the expansion order.
    while pending:
        current = pending.pop()
        for candidate in query.ordered(topology.bidirected_neighbors(current)):
            if candidate == query.treatment:
                impossible = True  # Algorithm C.1 lines 10--11.
            if candidate in colliders or candidate in rejected:
                continue
            if candidate in ancestors or m_separated(query.graph, candidate, query.treatment, ancestors):
                colliders.add(candidate)
                pending.append(candidate)
            else:
                rejected.add(candidate)
    collider_parents = {a for a, b in topology.directed if b in colliders}
    if query.treatment in collider_parents:
        impossible = True  # Algorithm C.1 lines 20--21.
    oset = parents | colliders | collider_parents
    # The generalized adjustment criterion also verifies the fixed-S query.
    valid = not impossible and query.valid(oset, fixed)
    return RungeAdjustmentResult(
        _token=_TOKEN,
        treatment=query.treatment,
        outcome=query.outcome,
        conditioned_on=query.ordered(fixed),
        adjustment_set=query.ordered(oset - fixed) if valid else None,
        o_set=query.ordered(oset) if valid else None,
        parents=query.ordered(parents),
        colliders=query.ordered(colliders),
        collider_parents=query.ordered(collider_parents),
        mediators=query.ordered(query.mediators),
        forbidden=query.ordered(query.forbidden),
        valid_ancestors=query.ordered(ancestors),
    )


def runge_o_set(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    treatment: str,
    outcome: str,
    *,
    conditioned_on: str | Iterable[str] = (),
) -> RungeAdjustmentResult:
    """Construct the O-set using Algorithm C.1; do not evaluate optimality.

    Graphical assumptions: a directed X-to-Y path, single X and Y, and fixed S
    disjoint from X and all directed descendants of X. This is deliberately
    narrower than the paper's S-disjoint-from-forbidden condition, which does
    not suffice for its conditional causal-effect/OLS correspondence.
    Markovness, faithfulness and a nonzero causal effect are distributional
    assumptions, not tested by code.
    """
    query = _query(graph, treatment, outcome, conditioned_on)
    return _construct(query, query.fixed)


def _unique_adjustment(query: _Query, adjustment: set[str]) -> bool:
    """Test uniqueness with constrained ancestral separators, without a powerset.

    Any different set either adds an eligible node or omits a member of O.
    For each such constraint I subset Z subset R, the ancestral separator
    an(X,Y,I) intersect R is complete (van der Zander et al., 2019).
    """
    allowed = frozenset(query.graph.topology.nodes) - query.forbidden

    def exists(required: frozenset[str], permitted: frozenset[str]) -> bool:
        relevant = query.backdoor.topology.directed_ancestors_reflexive(
            required | {query.treatment, query.outcome}
        )
        candidate = relevant & permitted
        return query.valid(candidate, required)

    for vertex in query.ordered(allowed - query.fixed - adjustment):
        if exists(query.fixed | {vertex}, allowed):
            return False
    for vertex in query.ordered(adjustment):
        if exists(query.fixed, allowed - {vertex}):
            return False
    return True


class _RequiredAdjustmentCheck:
    """Existence of a valid extension, not validity of the required set itself.

    FindSep (van der Zander et al., Lemma 1 / Proposition 2) tests the allowed
    ancestral candidate. For a bow-capable ADMG, expand each bidirected edge
    into an unavailable latent parent and apply the DAG result. The observed
    candidate and m-separation agree. Fixed S remains mandatory; internal
    requirements may be treatment descendants outside the forbidden set.
    """

    def __init__(self, query: _Query) -> None:
        self._query = query
        topology = query.backdoor.topology
        self._base = topology.directed_ancestors_reflexive(
            {query.treatment, query.outcome} | query.fixed
        ) - query.forbidden
        self._ancestors: dict[str, frozenset[str]] = {}
        self._cache: dict[frozenset[str], bool] = {}

    def feasible(self, required: frozenset[str]) -> bool:
        # Check before removing forbidden vertices from the cache key.
        if required & self._query.forbidden:
            return False
        return self.feasible_closure(self.closure(required))

    def ancestors(self, vertex: str) -> frozenset[str]:
        if vertex not in self._ancestors:
            self._ancestors[vertex] = (
                self._query.backdoor.topology.directed_ancestors_reflexive(vertex)
                - self._query.forbidden
            )
        return self._ancestors[vertex]

    def closure(self, required: Iterable[str]) -> frozenset[str]:
        """Canonical allowed ancestral candidate; caller excludes forbidden Q."""
        candidate = set(self._base)
        for vertex in required:
            candidate.update(self.ancestors(vertex))
        return frozenset(candidate)

    def feasible_closure(
        self, candidate: frozenset[str],
        oracle_graph: AcyclicDirectedBidirectedGraph | None = None,
    ) -> bool:
        """Query a canonical closure in the original proper-backdoor H or H[U].

        The cache key deliberately omits oracle_graph. The caller must supply
        H itself or H[U], where U is ancestral in H and contains candidate and
        both query endpoints. H[U] retains every induced directed/bidirected
        edge and all forbidden vertices in U. These preconditions preserve
        m-separation, so cached answers can be shared across these graphs;
        arbitrary edge or vertex pruning is unsupported.
        """
        if candidate not in self._cache:
            query = self._query
            self._cache[candidate] = m_separated(
                query.backdoor if oracle_graph is None else oracle_graph,
                query.treatment, query.outcome, candidate,
            )
        return self._cache[candidate]


@dataclass(frozen=True)
class _ColliderSource:
    terminal: str
    neighbors: dict[str, tuple[str, ...]]
    labels: dict[str, frozenset[str]]
    base: frozenset[str]
    oracle_graph: AcyclicDirectedBidirectedGraph


def _delete_labelled_detours(
    neighbors: dict[str, set[str]], labels: dict[str, frozenset[str]],
) -> None:
    """Reduce this private traversal adjacency, never the separation graph.

    Sequentially delete ab if a detour adds no label outside L_a union L_b.
    Replacement, truncation at the terminal and loop erasure preserve a
    feasible path. Endpoints may occur inside detours; their labels are empty.
    Fixed labels make one lexical pass sufficient, not minimum-edge optimal.
    """
    edges = sorted((a, b) for a in neighbors for b in neighbors[a] if a < b)
    for start, finish in edges:
        allowed = labels[start] | labels[finish]
        reached = {start}
        pending = [start]
        found = False
        while pending and not found:
            current = pending.pop()
            for neighbor in sorted(neighbors[current]):
                if (current == start and neighbor == finish) or (current == finish and neighbor == start):
                    continue
                if neighbor in reached or not labels[neighbor] <= allowed:
                    continue
                if neighbor == finish:
                    found = True
                    break
                reached.add(neighbor)
                pending.append(neighbor)
        if found:
            neighbors[start].remove(finish)
            neighbors[finish].remove(start)


def _prepare_collider_source(
    query: _Query, result: RungeAdjustmentResult, source: str,
    check: _RequiredAdjustmentCheck,
) -> _ColliderSource:
    """Polynomial source/vertex/edge filtering and labelled traversal reduction.

    Original C/O/N/F and H stay fixed. The synthetic terminal merges only
    bidirected adjacencies to Y or mediators and contributes no requirement.
    This implements D={source}, not a mandatory-articulation fixed point.
    """
    topology = query.graph.topology
    terminal = "__condition_i_terminal__"
    while terminal in topology.nodes:
        terminal += "_"
    base = check.closure((source,))
    kept = {source}
    source_feasible = check.feasible_closure(base)
    if source_feasible:
        kept.update(v for v in result.colliders if check.feasible(frozenset({source, v})))
    neighbors = {v: set[str]() for v in kept | {terminal}}
    targets = query.mediators | {query.outcome}
    if source_feasible:
        for a, b in topology.bidirected:
            if a in kept and b in kept and check.feasible(frozenset({source, a, b})):
                neighbors[a].add(b)
                neighbors[b].add(a)
        for vertex in kept:
            if topology.bidirected_neighbors(vertex) & targets:
                neighbors[vertex].add(terminal)
                neighbors[terminal].add(vertex)
    reached = {source}
    pending = [source]
    while pending:
        for neighbor in neighbors[pending.pop()] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    if terminal not in reached:
        kept = {source}
        neighbors = {source: set(), terminal: set()}
    else:
        kept &= reached
        neighbors = {v: adjacent & reached for v, adjacent in neighbors.items() if v in reached}
    labels = {v: check.ancestors(v) - base for v in kept}
    labels[terminal] = frozenset()
    _delete_labelled_detours(neighbors, labels)
    # Forbidden vertices remain graph vertices. Only conditioning candidates
    # exclude F. Keep every original edge induced by this ancestral set.
    universal = query.backdoor.topology.directed_ancestors_reflexive(
        {query.treatment, query.outcome} | query.fixed | kept)
    original = query.backdoor.topology
    oracle_graph = AcyclicDirectedBidirectedGraph(DirectedBidirectedGraph(
        nodes=query.ordered(universal),
        directed=(e for e in original.directed if set(e) <= universal),
        bidirected=(e for e in original.bidirected if set(e) <= universal),
    ))
    return _ColliderSource(
        terminal, {v: tuple(sorted(adjacent)) for v, adjacent in neighbors.items()},
        labels, base, oracle_graph,
    )


def _criterion_i(
    query: _Query,
    result: RungeAdjustmentResult,
    *,
    max_collider_paths: int,
    max_collider_states: int | None,
) -> bool:
    topology = query.graph.topology
    targets = query.mediators | {query.outcome}
    inside = set(result.colliders)
    n_nodes = set().union(*(topology.bidirected_neighbors(w) for w in targets | inside))
    n_nodes -= query.forbidden | set(result.o_set or ()) | query.fixed
    if not n_nodes:
        return True
    check = _RequiredAdjustmentCheck(query)
    examined_states = 0
    for source in query.ordered(n_nodes):
        prepared = _prepare_collider_source(query, result, source, check)
        if not prepared.neighbors[source]:
            continue
        first = (source, prepared.base)
        pending = [first]
        seen = {first}
        minimal: dict[str, list[frozenset[str]]] = {v: [] for v in prepared.neighbors}
        while pending:
            if max_collider_states is not None and examined_states >= max_collider_states:
                raise RungeAdjustmentResourceLimitError(
                    f"Theorem 3 condition I exceeds max_collider_states={max_collider_states}"
                )
            vertex, closure = pending.pop()
            examined_states += 1  # Unique (endpoint, closure), including rejections.
            if any(old <= closure for old in minimal[vertex]):
                continue
            if not check.feasible_closure(closure, prepared.oracle_graph):
                # Every continuation requires a superset of these vertices.
                # No valid extension now implies no valid extension later.
                continue
            minimal[vertex] = [old for old in minimal[vertex] if not closure < old]
            minimal[vertex].append(closure)
            for neighbor in reversed(prepared.neighbors[vertex]):
                if neighbor == prepared.terminal:
                    if max_collider_paths == 0:
                        raise RungeAdjustmentResourceLimitError(
                            f"Theorem 3 condition I exceeds max_collider_paths={max_collider_paths}"
                        )
                    # The terminal Y/mediator is NOT a required covariate.
                    # Cycle erasure turns this feasible walk into a simple
                    # indexed path with a subset of the required vertices.
                    return False
                if neighbor != source:
                    state = (neighbor, closure | prepared.labels[neighbor])
                    if state not in seen and not any(old <= state[1] for old in minimal[neighbor]):
                        # Walk continuations ignore history. At a fixed
                        # endpoint equal closures have identical continuations;
                        # neither union nor intersection of unequal states is safe.
                        # A real smaller feasible closure at the same endpoint
                        # dominates a larger one: it requires fewer covariates.
                        seen.add(state)
                        pending.append(state)
    return True


def _criterion_ii(query: _Query, result: RungeAdjustmentResult) -> bool:
    topology = query.graph.topology
    oset = set(result.o_set or ())
    ancestors = set(result.valid_ancestors)
    inside = set(result.colliders) & ancestors
    targets = query.mediators | {query.outcome}
    for vertex in query.ordered(oset - set(result.parents) - query.fixed):
        if m_separated(query.graph, vertex, query.treatment, (oset | query.fixed) - {vertex}):
            continue
        # A direct E <-> W needs no internal collider; longer paths require
        # every internal collider in C intersect vancs. Only the first edge
        # may be directed E -> C; all subsequent edges are bidirected.
        if topology.bidirected_neighbors(vertex) & targets:
            continue
        reached = {vertex}
        pending = list((topology.directed_children(vertex) | topology.bidirected_neighbors(vertex)) & inside)
        found = False
        while pending and not found:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            neighbors = topology.bidirected_neighbors(current)
            if neighbors & targets:
                found = True
            else:
                pending.extend(neighbors & inside - reached)
        if not found:
            return False
    return True


def runge_optimal_adjustment(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    treatment: str,
    outcome: str,
    *,
    conditioned_on: str | Iterable[str] = (),
    max_collider_paths: int = 100_000,
    max_collider_states: int | None = None,
) -> RungeAdjustmentResult:
    """Construct O and evaluate Runge's exact Theorem 3 graphical criterion.

    Condition I uses safe source/vertex/edge preprocessing, then exact walk
    states (endpoint, allowed ancestor closure). It merges equal states and
    prunes infeasible or dominated closures. General search can be exponential.
    ``max_collider_states`` bounds dequeued distinct states across all N-nodes,
    including cache hits, rejected states and states dominated after enqueueing;
    preprocessing is NOT counted. None leaves state search unbounded.
    ``max_collider_paths`` bounds completed feasible witnesses: search stops at
    the first, so only zero versus positive matters. Zero raises if a witness
    is found, but preprocessing can establish I even with both limits zero.
    Neither limit bounds preprocessing, memory, wall time, construction,
    Condition II or uniqueness. Exhaustion raises, never returns a verdict.
    The result concerns adjustment information; it is not a general estimator
    variance guarantee or a numerical variance estimate. As for ``runge_o_set``,
    caller-supplied fixed S must exclude all directed descendants of treatment.
    """
    if isinstance(max_collider_paths, bool) or not isinstance(max_collider_paths, int) or max_collider_paths < 0:
        raise RungeAdjustmentDomainError("max_collider_paths must be a nonnegative integer")
    if max_collider_states is not None and (
        isinstance(max_collider_states, bool)
        or not isinstance(max_collider_states, int)
        or max_collider_states < 0
    ):
        raise RungeAdjustmentDomainError("max_collider_states must be None or a nonnegative integer")
    query = _query(graph, treatment, outcome, conditioned_on)
    result = _construct(query, query.fixed)
    if result.adjustment_set is None:
        return result
    criterion_i = _criterion_i(query, result, max_collider_paths=max_collider_paths,
                               max_collider_states=max_collider_states)
    criterion_ii = _criterion_ii(query, result)
    unique = _unique_adjustment(query, set(result.adjustment_set))
    return replace(
        result,
        _token=_TOKEN,
        graphical_optimality=unique or (criterion_i and criterion_ii),
        criterion_i=criterion_i,
        criterion_ii=criterion_ii,
        unique_adjustment_set=unique,
    )
