"""Known-DAG adjustment and Henckel–Witte optimal covariate construction."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from typing import Literal, NoReturn, Self, SupportsIndex, final

from ._validation import node_positions, normalize_seeds
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .projection import latent_projection
from .separation import d_separated_sets


class DAGAdjustmentDomainError(ValueError):
    """A query is outside the declared known-DAG adjustment domain."""


_TOKEN = object()


class _ImmutableResult:
    __slots__ = ()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError('pickle is unsupported for DAG adjustment results')


@final
@dataclass(frozen=True, slots=True)
class DAGOptimalAdjustmentSet(_ImmutableResult):
    """A valid O-set for the declared treatment and outcome sets.

    Optimality concerns adjusted estimators in the cited model classes, not all
    estimators or finite-sample mean squared error.
    """

    graph: DAG
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    vertices: tuple[str, ...]
    causal_nodes: tuple[str, ...]
    forbidden_vertices: tuple[str, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError('DAG adjustment results are factory-created')

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError('DAGOptimalAdjustmentSet cannot be subclassed')


@final
@dataclass(frozen=True, slots=True)
class DAGAdjustmentNotApplicable(_ImmutableResult):
    """A theorem-domain or adjustment-existence result, not nonidentification."""

    graph: DAG
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    reason: Literal['outcomes_not_descendants', 'no_valid_adjustment_set']
    candidate_vertices: tuple[str, ...]
    causal_nodes: tuple[str, ...]
    forbidden_vertices: tuple[str, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError('DAG adjustment results are factory-created')

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError('DAGAdjustmentNotApplicable cannot be subclassed')


def _ordered(graph: DAG, vertices: Iterable[str]) -> tuple[str, ...]:
    selected = frozenset(vertices)
    return tuple(v for v in graph.topology.nodes if v in selected)


def _query(
    graph: DAG, interventions: str | Iterable[str], outcomes: str | Iterable[str]
) -> tuple[frozenset[str], frozenset[str]]:
    if not isinstance(graph, DAG):
        raise DAGAdjustmentDomainError('DAG adjustment requires a validated DAG')
    positions = node_positions(graph.topology.nodes)
    x = normalize_seeds(positions, interventions)
    y = normalize_seeds(positions, outcomes)
    if not x or not y:
        raise DAGAdjustmentDomainError('interventions and outcomes must be nonempty')
    if x & y:
        raise DAGAdjustmentDomainError('interventions and outcomes must be disjoint')
    return x, y


def _causal_and_forbidden(
    graph: DAG, x: frozenset[str], y: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """Proper paths exclude further treatments after their first vertex."""
    topology = graph.topology
    reduced = DirectedBidirectedGraph(
        nodes=(v for v in topology.nodes if v not in x),
        directed=((a, b) for a, b in topology.directed if a not in x and b not in x),
    )
    first = frozenset(b for a, b in topology.directed if a in x and b not in x)
    reached = reduced.directed_descendants_reflexive(first)
    causal = reached & reduced.directed_ancestors_reflexive(y)
    forbidden = topology.directed_descendants_reflexive(causal) | x
    return causal, forbidden


def _valid(
    graph: DAG, x: frozenset[str], y: frozenset[str], z: frozenset[str],
    causal: frozenset[str], forbidden: frozenset[str],
) -> bool:
    if z & (forbidden | y):
        return False
    # The proper back-door graph removes only first edges of proper causal paths.
    # Removing every arrow out of X would incorrectly allow causal mediators.
    topology = graph.topology
    backdoor = DAG.from_graph(DirectedBidirectedGraph(
        nodes=topology.nodes,
        directed=((a, b) for a, b in topology.directed if not (a in x and b in causal)),
    ))
    return d_separated_sets(backdoor, x, y, z)


def dag_adjustment_set_is_valid(
    graph: DAG, *, interventions: str | Iterable[str], outcomes: str | Iterable[str],
    adjustment: str | Iterable[str],
) -> bool:
    """Check the complete adjustment criterion, including multi-node queries.

    The supplied DAG may include unobserved vertices; callers can check a candidate
    containing only measured covariates. This function does not infer observability.
    Non-descendant outcomes are supported here even though O-set optimality assumes
    outcomes are descendants of the intervention set.
    """
    x, y = _query(graph, interventions, outcomes)
    z = normalize_seeds(node_positions(graph.topology.nodes), adjustment)
    causal, forbidden = _causal_and_forbidden(graph, x, y)
    return _valid(graph, x, y, z, causal, forbidden)


def dag_optimal_adjustment_set(
    graph: DAG, *, interventions: str | Iterable[str], outcomes: str | Iterable[str],
) -> DAGOptimalAdjustmentSet | DAGAdjustmentNotApplicable:
    """Construct Henckel–Perković–Maathuis's O-set (Def. 3.12, Thm. 3.13).

    Every vertex is available for adjustment in this operation. For incomplete
    observability use the Smucler operation with an explicit observed set.
    A non-descendant outcome returns an explicit theorem-domain result; it is never
    silently replaced by an empty adjustment set.
    """
    x, y = _query(graph, interventions, outcomes)
    causal, forbidden = _causal_and_forbidden(graph, x, y)
    candidate = graph.topology.external_directed_parents(causal) - forbidden
    if not y <= causal:
        reason: Literal['outcomes_not_descendants', 'no_valid_adjustment_set'] = 'outcomes_not_descendants'
    elif not _valid(graph, x, y, candidate, causal, forbidden):
        reason = 'no_valid_adjustment_set'
    else:
        return DAGOptimalAdjustmentSet(
            graph, _ordered(graph, x), _ordered(graph, y), _ordered(graph, candidate),
            _ordered(graph, causal), _ordered(graph, forbidden),
            _token=_TOKEN,
        )
    return DAGAdjustmentNotApplicable(
        graph, _ordered(graph, x), _ordered(graph, y), reason,
        _ordered(graph, candidate), _ordered(graph, causal), _ordered(graph, forbidden),
        _token=_TOKEN,
    )


def prune_dag_adjustment_set(
    graph: DAG, *, interventions: str | Iterable[str], outcomes: str | Iterable[str],
    adjustment: str | Iterable[str],
) -> tuple[str, ...]:
    """Apply Henckel et al. Algorithm 1 to an already valid adjustment set.

    This removes outcome-irrelevant covariates while preserving validity and not
    increasing asymptotic OLS variance. It need not produce the global O-set or an
    inclusion-minimal adjustment set.
    """
    x, y = _query(graph, interventions, outcomes)
    z = normalize_seeds(node_positions(graph.topology.nodes), adjustment)
    causal, forbidden = _causal_and_forbidden(graph, x, y)
    if not _valid(graph, x, y, z, causal, forbidden):
        raise DAGAdjustmentDomainError('pruning requires a valid adjustment set')
    remaining = set(z)
    for vertex in _ordered(graph, z):
        if d_separated_sets(graph, y, (vertex,), x | (remaining - {vertex})):
            remaining.remove(vertex)
    return _ordered(graph, remaining)


def dag_forbidden_projection(
    graph: DAG, *, interventions: str | Iterable[str], outcomes: str | Iterable[str],
) -> AcyclicDirectedBidirectedGraph:
    """Witte et al. Definition 5: latent-project forbidden non-query vertices.

    For descendant outcomes, parents of Y outside X and Y equal the O-set.
    The output can have bidirected edges or bows; it is not necessarily a DAG.
    """
    x, y = _query(graph, interventions, outcomes)
    _, forbidden = _causal_and_forbidden(graph, x, y)
    retained = (frozenset(graph.topology.nodes) - forbidden) | x | y
    return latent_projection(graph, retained)
