"""Corrected sequential adjustment in a known causal DAG or latent-projection ADMG.

This implements the proposed correction to Jung, Park and Lee (2024),
not their printed forbidden-set condition. See docs/SEQUENTIAL_ADJUSTMENT.md.
The reference constructor uses repeated separation tests; it does not implement
the cited linear-time constrained-separator routine.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import InitVar, dataclass
from typing import Literal, NoReturn, Self, SupportsIndex, TypeAlias, final

from ._validation import node_positions, normalize_seeds, require_vertex
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .separation import m_separated_sets

Graph: TypeAlias = AcyclicDirectedBidirectedGraph | DAG
Blocks: TypeAlias = tuple[frozenset[str], ...]
_TOKEN = object()


class SequentialAdjustmentDomainError(ValueError):
    """A query is outside the declared admissible sequential-formula domain."""


class _ImmutableResult:
    __slots__ = ()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError('pickle is unsupported for sequential adjustment results')


@final
@dataclass(frozen=True, slots=True)
class SequentialAdjustmentSequence(_ImmutableResult):
    """A valid, componentwise inclusion-minimal covariate sequence."""

    graph: Graph
    treatment_order: tuple[str, ...]
    outcome_blocks: tuple[tuple[str, ...], ...]
    covariate_blocks: tuple[tuple[str, ...], ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError('sequential adjustment results are factory-created')

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError('SequentialAdjustmentSequence cannot be subclassed')


@final
@dataclass(frozen=True, slots=True)
class SequentialAdjustmentNotFound(_ImmutableResult):
    """No admissible sequence exists for this order; not nonidentification."""

    graph: Graph
    treatment_order: tuple[str, ...]
    outcome_blocks: tuple[tuple[str, ...], ...]
    failed_stage: int
    reason: Literal['no_valid_sequence'] = 'no_valid_sequence'
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError('sequential adjustment results are factory-created')

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError('SequentialAdjustmentNotFound cannot be subclassed')


SequentialAdjustmentResult: TypeAlias = SequentialAdjustmentSequence | SequentialAdjustmentNotFound


def _ordered(topology: DirectedBidirectedGraph, vertices: Iterable[str]) -> tuple[str, ...]:
    chosen = frozenset(vertices)
    return tuple(v for v in topology.nodes if v in chosen)


def _query(graph: Graph, treatment_order: Sequence[str], outcomes: str | Iterable[str]
           ) -> tuple[tuple[str, ...], frozenset[str], Blocks]:
    if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
        raise SequentialAdjustmentDomainError('sequential adjustment requires a validated DAG or ADMG')
    if isinstance(treatment_order, str) or not isinstance(treatment_order, Sequence):
        raise SequentialAdjustmentDomainError('treatment_order must be an ordered sequence')
    topology = graph.topology
    positions = node_positions(topology.nodes)
    xs = tuple(require_vertex(positions, x) for x in treatment_order)
    ys = normalize_seeds(positions, outcomes)
    if not xs or not ys:
        raise SequentialAdjustmentDomainError('treatment_order and outcomes must be nonempty')
    if len(frozenset(xs)) != len(xs):
        raise SequentialAdjustmentDomainError('treatments must be distinct')
    if frozenset(xs) & ys:
        raise SequentialAdjustmentDomainError('treatments and outcomes must be disjoint')
    descendants = tuple(topology.directed_descendants_reflexive(x) for x in xs)
    if any(descendants[i] & frozenset(xs[:i]) for i in range(len(xs))):
        raise SequentialAdjustmentDomainError('treatment_order must be topological')
    yblocks: list[set[str]] = [set() for _ in range(len(xs) + 1)]
    for y in ys:
        last = max([0] + [i + 1 for i, desc in enumerate(descendants) if y in desc])
        yblocks[last].add(y)
    return xs, ys, tuple(frozenset(y) for y in yblocks)


def _covariates(topology: DirectedBidirectedGraph, xs: tuple[str, ...], ys: frozenset[str],
                covariate_blocks: Sequence[str | Iterable[str]]) -> Blocks:
    if isinstance(covariate_blocks, str) or not isinstance(covariate_blocks, Sequence):
        raise SequentialAdjustmentDomainError('covariate_blocks must be an ordered sequence')
    if len(covariate_blocks) != len(xs):
        raise SequentialAdjustmentDomainError('one covariate block is required per treatment')
    seen = frozenset(xs) | ys
    blocks = []
    for i, block in enumerate(covariate_blocks):
        z = normalize_seeds(node_positions(topology.nodes), block)
        if z & seen:
            raise SequentialAdjustmentDomainError('covariate blocks must be pairwise disjoint and outside treatments/outcomes')
        if z & topology.directed_descendants_reflexive(xs[i + 1:]):
            raise SequentialAdjustmentDomainError('covariates cannot descend from later treatments')
        blocks.append(z)
        seen |= z
    return tuple(blocks)


def _stage(topology: DirectedBidirectedGraph, x: str, future: tuple[str, ...], targets: frozenset[str]
           ) -> tuple[AcyclicDirectedBidirectedGraph, frozenset[str]]:
    future_set = frozenset(future)
    intervened = DirectedBidirectedGraph(
        nodes=topology.nodes,
        directed=((a, b) for a, b in topology.directed if b not in future_set),
        bidirected=((a, b) for a, b in topology.bidirected if not ({a, b} & future_set)),
    )
    causal = (intervened.directed_descendants_reflexive(x) - {x}) & intervened.directed_ancestors_reflexive(targets)
    forbidden = intervened.directed_descendants_reflexive(causal)
    backdoor = AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
        nodes=topology.nodes,
        directed=((a, b) for a, b in intervened.directed if not (a == x and b in causal)),
        bidirected=intervened.bidirected,
    ))
    return backdoor, forbidden


def sequential_adjustment_is_valid(
    graph: Graph, *, treatment_order: Sequence[str], outcomes: str | Iterable[str],
    covariate_blocks: Sequence[str | Iterable[str]],
) -> bool:
    """Check universal validity of one admissible supplied sequential expression.

    All graph vertices are observed; latent confounding is encoded by bidirected
    edges. A false result concerns this allocation, not effect identification.
    Nonadmissible allocations raise a domain error rather than receive a verdict.
    """
    xs, ys, yblocks = _query(graph, treatment_order, outcomes)
    blocks = _covariates(graph.topology, xs, ys, covariate_blocks)
    history = yblocks[0]
    for i, (x, z) in enumerate(zip(xs, blocks, strict=True)):
        targets = frozenset().union(*yblocks[i + 1:])
        if targets:
            backdoor, forbidden = _stage(graph.topology, x, xs[i + 1:], targets)
            if z & forbidden or not m_separated_sets(backdoor, x, targets, history | z):
                return False
        history |= {x} | yblocks[i + 1] | z
    return True


def find_minimal_sequential_adjustment(
    graph: Graph, *, treatment_order: Sequence[str], outcomes: str | Iterable[str],
) -> SequentialAdjustmentResult:
    """Find a sequence using finalized histories and all eligible graph vertices.

    Minimality is stagewise and componentwise inclusion minimality, not minimum
    total size or variance. Nodes are pruned in topology order for deterministic
    ties. Repeated m-separation tests do not have a linear-time-per-stage promise.
    """
    xs, ys, yblocks = _query(graph, treatment_order, outcomes)
    topology = graph.topology
    history = yblocks[0]
    result = []
    ordered_outcomes = tuple(_ordered(topology, y) for y in yblocks)
    for i, x in enumerate(xs):
        targets = frozenset().union(*yblocks[i + 1:])
        if not targets:
            chosen: frozenset[str] = frozenset()
        else:
            backdoor, forbidden = _stage(topology, x, xs[i + 1:], targets)
            excluded = frozenset(xs) | ys | history | forbidden | topology.directed_descendants_reflexive(xs[i + 1:])
            chosen = backdoor.topology.directed_ancestors_reflexive({x} | targets | history) - excluded
            if not m_separated_sets(backdoor, x, targets, history | chosen):
                return SequentialAdjustmentNotFound(graph, xs, ordered_outcomes, i, _token=_TOKEN)
            # Every candidate is ancestral to endpoints or mandatory history, so
            # the ancestral moral domain is fixed. One deterministic deletion pass
            # therefore gives inclusion minimality despite general d-separation
            # being nonmonotone under arbitrary conditioning changes.
            for vertex in _ordered(topology, chosen):
                smaller = chosen - {vertex}
                if m_separated_sets(backdoor, x, targets, history | smaller):
                    chosen = smaller
        result.append(_ordered(topology, chosen))
        history |= {x} | yblocks[i + 1] | chosen
    return SequentialAdjustmentSequence(graph, xs, ordered_outcomes, tuple(result), _token=_TOKEN)
