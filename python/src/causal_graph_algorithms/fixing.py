"""Conditional ADMGs and the graph/kernel fixing operation.

Richardson--Evans--Robins--Shpitser (2023), Definitions 17, 19, 26, 33.
Graph properties are validated; kernel Markov/normalization assumptions are
supplied by the caller. No data fitting or distribution testing occurs here.
"""
from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass
from typing import TypeAlias

from ._validation import node_positions, normalize_seeds
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .errors import GraphValidationError
from .identification import _ImmutableIdentificationValue
from .probability import Probability, ProbabilityExpression, Product, Ratio, Summation

_TOKEN = object()


class FixingDomainError(GraphValidationError):
    """Malformed CADMG/kernel input or a requested invalid fixing sequence."""


class FixingBudgetExceeded(RuntimeError):
    """Atomic intrinsic-set enumeration exceeded its reachable-state budget."""


def _seeds(nodes: tuple[str, ...], values: str | Iterable[str]) -> frozenset[str]:
    try:
        return normalize_seeds(node_positions(nodes), values)
    except GraphValidationError as error:
        raise FixingDomainError(str(error)) from error


@typing.final
@dataclass(frozen=True, init=False)
class CADMG(_ImmutableIdentificationValue):
    """Acyclic directed/bidirected topology with a random/fixed partition.

    All vertices remain in ``topology.nodes``. No edge may have an arrowhead
    at a fixed vertex. Fixed nodes may parent random nodes and are never
    included in a district. Creation does not assert kernel or SCM semantics.
    """
    __slots__ = ("topology", "random_vertices", "fixed_vertices")
    topology: DirectedBidirectedGraph
    random_vertices: tuple[str, ...]
    fixed_vertices: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CADMG cannot be subclassed")

    def __init__(self, token: object, *, topology: DirectedBidirectedGraph,
                 random_vertices: tuple[str, ...], fixed_vertices: tuple[str, ...]) -> None:
        if token is not _TOKEN:
            raise TypeError("CADMG values are factory-created")
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "random_vertices", random_vertices)
        object.__setattr__(self, "fixed_vertices", fixed_vertices)

    @classmethod
    def from_graph(cls, graph: AcyclicDirectedBidirectedGraph | DAG, *,
                   random_vertices: str | Iterable[str] | None = None) -> CADMG:
        """Validate a CADMG partition of an already acyclic topology.

        Omitted ``random_vertices`` means every vertex is random. Its
        complement is fixed, including any isolated fixed parameter.
        """
        if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
            raise FixingDomainError("CADMG requires a validated DAG/ADMG")
        topology = graph.topology
        random = frozenset(topology.nodes) if random_vertices is None else _seeds(topology.nodes, random_vertices)
        fixed = frozenset(topology.nodes)-random
        if any(b in fixed for _,b in topology.directed) or any(a in fixed or b in fixed for a,b in topology.bidirected):
            raise FixingDomainError("no arrowhead may be incident at a fixed vertex")
        return cls(_TOKEN, topology=topology,
                   random_vertices=tuple(v for v in topology.nodes if v in random),
                   fixed_vertices=tuple(v for v in topology.nodes if v in fixed))


def _require_graph(graph: CADMG) -> None:
    if not isinstance(graph, CADMG):
        raise FixingDomainError("fixing operations require a CADMG")


def _descendants(graph: CADMG, vertex: str) -> frozenset[str]:
    children: dict[str, list[str]] = {v: [] for v in graph.random_vertices}
    for parent, child in graph.topology.directed:
        if parent in children:
            children[parent].append(child)
    reached = {vertex}
    pending = [vertex]
    while pending:
        for child in children[pending.pop()]:
            if child not in reached:
                reached.add(child)
                pending.append(child)
    return frozenset(reached)


def _districts(graph: CADMG) -> tuple[frozenset[str], ...]:
    neighbors: dict[str, list[str]] = {v: [] for v in graph.random_vertices}
    for a,b in graph.topology.bidirected:
        neighbors[a].append(b)
        neighbors[b].append(a)
    seen: set[str] = set()
    result = []
    for seed in graph.random_vertices:
        if seed in seen:
            continue
        district = {seed}
        pending = [seed]
        while pending:
            for other in neighbors[pending.pop()]:
                if other not in district:
                    district.add(other)
                    pending.append(other)
        seen.update(district)
        result.append(frozenset(district))
    return tuple(result)


def fixable_vertices(graph: CADMG) -> tuple[str, ...]:
    """Definition 17: district(v) intersects descendants(v) only at v."""
    _require_graph(graph)
    district_of = {v:d for d in _districts(graph) for v in d}
    return tuple(v for v in graph.random_vertices if district_of[v] & _descendants(graph,v) == {v})


def _sequence(graph: CADMG, sequence: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(sequence,(Mapping,Set)):
        raise FixingDomainError("a fixing sequence must be ordered; mappings and sets are not accepted")
    try:
        values = (sequence,) if isinstance(sequence,str) else tuple(sequence)
    except TypeError as error:
        raise FixingDomainError("a fixing sequence must be an ordered iterable") from error
    _seeds(graph.topology.nodes, values)
    return values


def _fix_one(graph: CADMG, vertex: str) -> CADMG:
    topology = DirectedBidirectedGraph(nodes=graph.topology.nodes,
        directed=((a,b) for a,b in graph.topology.directed if b != vertex),
        bidirected=((a,b) for a,b in graph.topology.bidirected if vertex not in (a,b)))
    random = tuple(v for v in graph.random_vertices if v != vertex)
    return CADMG(_TOKEN, topology=topology, random_vertices=random,
                 fixed_vertices=tuple(v for v in topology.nodes if v not in random))


def fixing_sequence_is_valid(graph: CADMG, sequence: str | Iterable[str]) -> bool:
    """Check Definition 26 in order; repeated/already-fixed vertices are invalid.

    Unknown labels and malformed inputs raise ``FixingDomainError``.
    """
    _require_graph(graph)
    for vertex in _sequence(graph,sequence):
        if vertex not in fixable_vertices(graph):
            return False
        graph = _fix_one(graph,vertex)
    return True


def fix_vertices(graph: CADMG, sequence: str | Iterable[str]) -> CADMG:
    """Apply a validated ordered sequence; invalid steps raise before returning."""
    _require_graph(graph)
    for index,vertex in enumerate(_sequence(graph,sequence)):
        if vertex not in fixable_vertices(graph):
            raise FixingDomainError(f"invalid fixing step {index}: {vertex!r} is not fixable")
        graph = _fix_one(graph,vertex)
    return graph


@typing.final
@dataclass(frozen=True, init=False)
class ValidFixingSequence(_ImmutableIdentificationValue):
    __slots__ = ("source", "sequence", "result")
    source: CADMG
    sequence: tuple[str, ...]
    result: CADMG

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ValidFixingSequence cannot be subclassed")

    def __init__(self, token: object, *, source: CADMG, sequence: tuple[str, ...], result: CADMG) -> None:
        if token is not _TOKEN:
            raise TypeError("fixing witnesses are factory-created")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "result", result)


@typing.final
@dataclass(frozen=True, init=False)
class UnreachableCADMG(_ImmutableIdentificationValue):
    """The requested random set cannot be reached by valid fixings."""
    __slots__ = ("source", "random_vertices", "blocked_vertices", "partial_sequence")
    source: CADMG
    random_vertices: tuple[str, ...]
    blocked_vertices: tuple[str, ...]
    partial_sequence: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("UnreachableCADMG cannot be subclassed")

    def __init__(self, token: object, *, source: CADMG, random_vertices: tuple[str, ...],
                 blocked_vertices: tuple[str, ...], partial_sequence: tuple[str, ...]) -> None:
        if token is not _TOKEN:
            raise TypeError("fixing witnesses are factory-created")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "random_vertices", random_vertices)
        object.__setattr__(self, "blocked_vertices", blocked_vertices)
        object.__setattr__(self, "partial_sequence", partial_sequence)


FixingSequenceResult: TypeAlias = ValidFixingSequence | UnreachableCADMG


def find_fixing_sequence(graph: CADMG, *, random_vertices: str | Iterable[str]) -> FixingSequenceResult:
    """Find a canonical sequence for an exact remaining random set.

    Lemma 22 (persistence of fixability) makes greedy removal complete. A
    negative is graph reachability only, not a general causal-ID verdict.
    """
    _require_graph(graph)
    target = _seeds(graph.topology.nodes, random_vertices)
    if not target <= frozenset(graph.random_vertices):
        raise FixingDomainError("a fixed vertex cannot become random")
    normalized = tuple(v for v in graph.random_vertices if v in target)
    current = graph
    sequence: list[str] = []
    while frozenset(current.random_vertices) != target:
        candidate = next((v for v in fixable_vertices(current) if v not in target),None)
        if candidate is None:
            return UnreachableCADMG(_TOKEN, source=graph, random_vertices=normalized,
                blocked_vertices=tuple(v for v in current.random_vertices if v not in target),
                partial_sequence=tuple(sequence))
        current = _fix_one(current,candidate)
        sequence.append(candidate)
    return ValidFixingSequence(_TOKEN, source=graph, sequence=tuple(sequence), result=current)


def intrinsic_set_is_valid(graph: CADMG, vertices: str | Iterable[str]) -> bool:
    """Is the nonempty set a district in a reachable graph (Definition 33)?"""
    _require_graph(graph)
    target = _seeds(graph.topology.nodes, vertices)
    if not target or not target <= frozenset(graph.random_vertices):
        return False
    reached = find_fixing_sequence(graph, random_vertices=target)
    return isinstance(reached, ValidFixingSequence) and len(_districts(reached.result)) == 1


def intrinsic_sets(graph: CADMG, *, max_states: int = 100_000) -> tuple[tuple[str, ...], ...]:
    """Atomically enumerate districts of all reachable graphs, canonically.

    This extends Definition 33 to a starting CADMG by valid additional
    fixings. Distinct random sets determine distinct graph states. The
    positive integer cap counts discovered states including the initial one;
    exponential output/work is possible. Exhaustion returns no partial tuple.
    """
    _require_graph(graph)
    if isinstance(max_states,bool) or not isinstance(max_states,int) or max_states <= 0:
        raise FixingDomainError("max_states must be a positive integer")
    pending = [graph]
    seen = {frozenset(graph.random_vertices)}
    intrinsic: set[frozenset[str]] = set()
    while pending:
        current = pending.pop()
        intrinsic.update(_districts(current))
        for vertex in fixable_vertices(current):
            following = _fix_one(current,vertex)
            key = frozenset(following.random_vertices)
            if key in seen:
                continue
            if len(seen) >= max_states:
                raise FixingBudgetExceeded("intrinsic-set reachable-state budget exhausted")
            seen.add(key)
            pending.append(following)
    positions = node_positions(graph.topology.nodes)
    ordered = (tuple(v for v in graph.random_vertices if v in values) for values in intrinsic)
    return tuple(sorted(ordered,key=lambda values:(len(values),tuple(positions[v] for v in values))))


@typing.final
@dataclass(frozen=True, init=False)
class CADMGKernel(_ImmutableIdentificationValue):
    """A symbolic kernel with explicit random/fixed roles and caller premises.

    ``from_expression`` checks expression type/scope only. Normalization,
    positivity and nested-Markov membership are caller assertions, not facts
    inferred by constructing this object. These are required for repeated
    graph-compatible fixing and its order-invariance guarantee.
    """
    __slots__ = ("graph", "formula")
    graph: CADMG
    formula: ProbabilityExpression

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CADMGKernel cannot be subclassed")

    def __init__(self, token: object, *, graph: CADMG, formula: ProbabilityExpression) -> None:
        if token is not _TOKEN:
            raise TypeError("CADMGKernel values are factory-created")
        object.__setattr__(self,"graph",graph)
        object.__setattr__(self,"formula",formula)

    @classmethod
    def from_expression(cls, graph: CADMG, formula: ProbabilityExpression) -> CADMGKernel:
        _require_graph(graph)
        if not isinstance(formula,(Probability,Product,Ratio,Summation)):
            raise FixingDomainError("kernel formula must be a ProbabilityExpression")
        if not formula.free_variables <= frozenset(graph.topology.nodes):
            raise FixingDomainError("kernel expression has free variables outside its CADMG")
        if not frozenset(graph.random_vertices) <= formula.free_variables:
            raise FixingDomainError("every random kernel coordinate must occur free in its expression")
        return cls(_TOKEN,graph=graph,formula=formula)


def _sum_random(variables: tuple[str, ...], formula: ProbabilityExpression) -> ProbabilityExpression:
    return Summation(variables,formula) if variables else formula


def fix_kernel(kernel: CADMGKernel, sequence: str | Iterable[str]) -> CADMGKernel:
    """Apply Definition 19's symbolic kernel transformation along a valid sequence.

    The denominator is ``q(r, mb_random | W) / q(mb_random | W)``.
    The nested-Markov caller premise permits suppressing fixed parameters
    outside the blanket; they remain syntactically present and are never
    summed. Empty sequences preserve identity. Formulas are unsimplified.
    """
    if not isinstance(kernel,CADMGKernel):
        raise FixingDomainError("fix_kernel requires a CADMGKernel")
    steps = _sequence(kernel.graph,sequence)
    # Validate the complete requested sequence before constructing formulas.
    fix_vertices(kernel.graph,steps)
    current = kernel
    for vertex in steps:
        graph = current.graph
        district = next(d for d in _districts(graph) if vertex in d)
        blanket = (district-{vertex}) | {a for a,b in graph.topology.directed if b in district}
        blanket_random = blanket & frozenset(graph.random_vertices)
        joint = _sum_random(tuple(v for v in graph.random_vertices if v not in blanket_random | {vertex}),current.formula)
        denominator = Ratio(joint,Summation((vertex,),joint))
        current = CADMGKernel(_TOKEN,graph=_fix_one(graph,vertex),formula=Ratio(current.formula,denominator))
    return current
