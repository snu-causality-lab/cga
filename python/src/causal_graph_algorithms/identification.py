from __future__ import annotations

import heapq
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NoReturn, SupportsIndex, TypeAlias

from ._validation import node_positions, normalize_seeds
from .directed_bidirected import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
)
from .errors import IdentificationDomainError
from .probability import (
    Probability,
    ProbabilityExpression,
    Product,
    Ratio,
    Summation,
)


_DOMAIN_MESSAGE = (
    "causal-effect identification requires an acyclic directed-bidirected or DAG view"
)
_OVERLAP_MESSAGE = "outcomes and interventions must be disjoint"
_FACTORY_TOKEN = object()


class _ImmutableIdentificationValue:
    __slots__ = ()

    def __copy__(self) -> typing.Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> typing.Self:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("pickle is unsupported for identification results")


@typing.final
@dataclass(frozen=True, init=False)
class HedgeWitness(_ImmutableIdentificationValue):
    """An exact-query hedge represented by its nested C-forest vertex sets."""

    __slots__ = (
        "graph",
        "outcomes",
        "interventions",
        "inner_vertices",
        "outer_vertices",
        "roots",
        "fixing_sequence",
    )

    graph: AcyclicDirectedBidirectedGraph
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    inner_vertices: tuple[str, ...]
    outer_vertices: tuple[str, ...]
    roots: tuple[str, ...]
    fixing_sequence: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("HedgeWitness cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: AcyclicDirectedBidirectedGraph,
        outcomes: tuple[str, ...],
        interventions: tuple[str, ...],
        inner_vertices: tuple[str, ...],
        outer_vertices: tuple[str, ...],
        roots: tuple[str, ...],
        fixing_sequence: tuple[str, ...],
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("HedgeWitness values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "inner_vertices", inner_vertices)
        object.__setattr__(self, "outer_vertices", outer_vertices)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "fixing_sequence", fixing_sequence)


@typing.final
@dataclass(frozen=True, init=False)
class IdentifiedEffect(_ImmutableIdentificationValue):
    """An identified causal query and its observational probability formula."""

    __slots__ = ("graph", "outcomes", "interventions", "formula")

    graph: AcyclicDirectedBidirectedGraph | DAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    formula: ProbabilityExpression

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("IdentifiedEffect cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: AcyclicDirectedBidirectedGraph | DAG,
        outcomes: tuple[str, ...],
        interventions: tuple[str, ...],
        formula: ProbabilityExpression,
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("IdentifiedEffect values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "formula", formula)


@typing.final
@dataclass(frozen=True, init=False)
class NonIdentifiedEffect(_ImmutableIdentificationValue):
    """A nonidentified causal query with an exact-query hedge witness."""

    __slots__ = ("witness",)

    witness: HedgeWitness

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("NonIdentifiedEffect cannot be subclassed")

    def __init__(self, factory_token: object, *, witness: HedgeWitness) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("NonIdentifiedEffect values are factory-created")
        object.__setattr__(self, "witness", witness)


IdentificationResult: TypeAlias = IdentifiedEffect | NonIdentifiedEffect


def _ordered_subset(
    source_order: tuple[str, ...], vertices: frozenset[str]
) -> tuple[str, ...]:
    return tuple(vertex for vertex in source_order if vertex in vertices)


def _topological_order(graph: DirectedBidirectedGraph) -> tuple[str, ...]:
    positions = node_positions(graph.nodes)
    outgoing: dict[str, list[str]] = {node: [] for node in graph.nodes}
    indegree = {node: 0 for node in graph.nodes}
    for parent, child in graph.directed:
        outgoing[parent].append(child)
        indegree[child] += 1
    for children in outgoing.values():
        children.sort(key=positions.__getitem__)

    pending = [positions[node] for node in graph.nodes if indegree[node] == 0]
    heapq.heapify(pending)
    ordered: list[str] = []
    while pending:
        current = graph.nodes[heapq.heappop(pending)]
        ordered.append(current)
        for child in outgoing[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(pending, positions[child])
    if len(ordered) != len(graph.nodes):
        raise AssertionError("validated acyclic graph has no topological order")
    return tuple(ordered)


def _ancestors_within(
    graph: DirectedBidirectedGraph,
    seeds: frozenset[str],
    allowed: frozenset[str],
) -> frozenset[str]:
    parents: dict[str, list[str]] = {node: [] for node in allowed}
    for parent, child in graph.directed:
        if parent in allowed and child in allowed:
            parents[child].append(parent)

    reached = set(seeds)
    pending = list(seeds)
    while pending:
        current = pending.pop()
        for parent in parents[current]:
            if parent not in reached:
                reached.add(parent)
                pending.append(parent)
    return frozenset(reached)


def _descendants_within(
    graph: DirectedBidirectedGraph,
    seed: str,
    allowed: frozenset[str],
) -> frozenset[str]:
    children: dict[str, list[str]] = {node: [] for node in allowed}
    for parent, child in graph.directed:
        if parent in allowed and child in allowed:
            children[parent].append(child)

    reached = {seed}
    pending = [seed]
    while pending:
        current = pending.pop()
        for child in children[current]:
            if child not in reached:
                reached.add(child)
                pending.append(child)
    return frozenset(reached)


def _district_within(
    graph: DirectedBidirectedGraph,
    seed: str,
    allowed: frozenset[str],
) -> frozenset[str]:
    adjacency: dict[str, list[str]] = {node: [] for node in allowed}
    for left, right in graph.bidirected:
        if left in allowed and right in allowed:
            adjacency[left].append(right)
            adjacency[right].append(left)

    reached = {seed}
    pending = [seed]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return frozenset(reached)


def _districts_within(
    graph: DirectedBidirectedGraph,
    allowed: frozenset[str],
) -> tuple[frozenset[str], ...]:
    unseen = set(allowed)
    result: list[frozenset[str]] = []
    for node in graph.nodes:
        if node not in unseen:
            continue
        district = _district_within(graph, node, allowed)
        result.append(district)
        unseen.difference_update(district)
    return tuple(result)


def _is_fixable(
    graph: DirectedBidirectedGraph,
    vertex: str,
    random_vertices: frozenset[str],
) -> bool:
    district = _district_within(graph, vertex, random_vertices)
    descendants = _descendants_within(graph, vertex, random_vertices)
    return district.intersection(descendants) == {vertex}


def _reachable_closure(
    graph: DirectedBidirectedGraph,
    target: frozenset[str],
) -> tuple[frozenset[str], tuple[str, ...]]:
    random_vertices = frozenset(graph.nodes)
    fixing_sequence: list[str] = []
    while True:
        next_vertex = next(
            (
                vertex
                for vertex in graph.nodes
                if vertex in random_vertices
                and vertex not in target
                and _is_fixable(graph, vertex, random_vertices)
            ),
            None,
        )
        if next_vertex is None:
            return random_vertices, tuple(fixing_sequence)
        fixing_sequence.append(next_vertex)
        random_vertices = random_vertices - {next_vertex}


def _exact_query_hedge(
    graph: AcyclicDirectedBidirectedGraph,
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
) -> HedgeWitness | None:
    topology = graph.topology
    outcome_set = frozenset(outcomes)
    intervention_set = frozenset(interventions)
    post_intervention_vertices = frozenset(topology.nodes) - intervention_set
    ancestral_outcomes = _ancestors_within(
        topology,
        outcome_set,
        post_intervention_vertices,
    )

    for district in _districts_within(topology, ancestral_outcomes):
        closure, fixing_sequence = _reachable_closure(topology, district)
        if closure == district:
            continue
        roots = frozenset(
            vertex
            for vertex in district
            if not any(
                parent == vertex and child in district
                for parent, child in topology.directed
            )
        )
        return HedgeWitness(
            _FACTORY_TOKEN,
            graph=graph,
            outcomes=outcomes,
            interventions=interventions,
            inner_vertices=_ordered_subset(topology.nodes, district),
            outer_vertices=_ordered_subset(topology.nodes, closure),
            roots=_ordered_subset(topology.nodes, roots),
            fixing_sequence=fixing_sequence,
        )
    return None


def _product(
    factors: tuple[ProbabilityExpression, ...],
) -> ProbabilityExpression:
    if len(factors) == 1:
        return factors[0]
    return Product(factors)


def _summation(
    variables: tuple[str, ...],
    body: ProbabilityExpression,
) -> ProbabilityExpression:
    if not variables:
        return body
    return Summation(variables, body)


@dataclass(frozen=True, slots=True)
class _Kernel:
    nodes: tuple[str, ...]
    expression: ProbabilityExpression
    chain_factors: tuple[ProbabilityExpression, ...] | None


def _kernel_expression_for(
    kernel: _Kernel,
    retained: frozenset[str],
) -> ProbabilityExpression:
    retained_order = tuple(node for node in kernel.nodes if node in retained)
    if retained_order == kernel.nodes:
        return kernel.expression

    if (
        kernel.chain_factors is not None
        and retained_order == kernel.nodes[: len(retained_order)]
    ):
        return _product(kernel.chain_factors[: len(retained_order)])

    removed = tuple(node for node in kernel.nodes if node not in retained)
    return _summation(removed, kernel.expression)


def _marginal_kernel(kernel: _Kernel, retained: frozenset[str]) -> _Kernel:
    retained_order = tuple(node for node in kernel.nodes if node in retained)
    expression = _kernel_expression_for(kernel, retained)
    factors: tuple[ProbabilityExpression, ...] | None = None
    if (
        kernel.chain_factors is not None
        and retained_order == kernel.nodes[: len(retained_order)]
    ):
        factors = kernel.chain_factors[: len(retained_order)]
    return _Kernel(retained_order, expression, factors)


def _kernel_chain_factors(kernel: _Kernel) -> tuple[ProbabilityExpression, ...]:
    if kernel.chain_factors is not None:
        return kernel.chain_factors

    factors: list[ProbabilityExpression] = []
    previous: ProbabilityExpression | None = None
    for prefix_length in range(1, len(kernel.nodes) + 1):
        retained = frozenset(kernel.nodes[:prefix_length])
        marginal = _kernel_expression_for(kernel, retained)
        factor = marginal if previous is None else Ratio(marginal, previous)
        factors.append(factor)
        previous = marginal
    return tuple(factors)


@dataclass(slots=True)
class _IDState:
    vertices: frozenset[str]
    outcomes: frozenset[str]
    interventions: frozenset[str]
    kernel: _Kernel


@dataclass(slots=True)
class _DistrictProductFrame:
    parent_vertices: frozenset[str]
    parent_outcomes: frozenset[str]
    parent_interventions: frozenset[str]
    parent_kernel: _Kernel
    districts: tuple[frozenset[str], ...]
    next_index: int
    results: list[ProbabilityExpression]


@dataclass(frozen=True, slots=True)
class _ClassicalIDOutput:
    formula: ProbabilityExpression
    line_three_interventions: frozenset[str]


def _initial_kernel(
    topology: DirectedBidirectedGraph,
    topological_order: tuple[str, ...],
) -> _Kernel:
    factors: list[ProbabilityExpression] = []
    predecessors: list[str] = []
    for vertex in topological_order:
        factors.append(Probability(vertex, conditioned_on=tuple(predecessors)))
        predecessors.append(vertex)
    normalized_factors = tuple(factors)
    return _Kernel(
        topological_order,
        _product(normalized_factors),
        normalized_factors,
    )


def _classical_id_formula(
    topology: DirectedBidirectedGraph,
    *,
    outcomes: frozenset[str],
    interventions: frozenset[str],
) -> _ClassicalIDOutput:
    """Run Figure 3 of Shpitser--Pearl (2006) with an explicit work stack."""

    source_topological_order = _topological_order(topology)
    state = _IDState(
        frozenset(topology.nodes),
        outcomes,
        interventions,
        _initial_kernel(topology, source_topological_order),
    )
    continuations: list[_DistrictProductFrame] = []
    line_three_interventions: set[str] = set()
    completed: ProbabilityExpression | None = None

    while True:
        if completed is not None:
            if not continuations:
                return _ClassicalIDOutput(
                    completed,
                    frozenset(line_three_interventions),
                )
            frame = continuations[-1]
            frame.results.append(completed)
            if frame.next_index < len(frame.districts):
                district = frame.districts[frame.next_index]
                frame.next_index += 1
                state = _IDState(
                    frame.parent_vertices,
                    district,
                    frame.parent_vertices - district,
                    frame.parent_kernel,
                )
                completed = None
                continue

            continuations.pop()
            product = _product(tuple(frame.results))
            summed = _ordered_subset(
                source_topological_order,
                frame.parent_vertices
                - frame.parent_outcomes
                - frame.parent_interventions,
            )
            completed = _summation(summed, product)
            continue

        if not state.interventions:
            completed = _kernel_expression_for(state.kernel, state.outcomes)
            continue

        ancestors = _ancestors_within(topology, state.outcomes, state.vertices)
        if ancestors != state.vertices:
            state = _IDState(
                ancestors,
                state.outcomes,
                state.interventions.intersection(ancestors),
                _marginal_kernel(state.kernel, ancestors),
            )
            continue

        not_intervened = state.vertices - state.interventions
        post_intervention_ancestors = _ancestors_within(
            topology,
            state.outcomes,
            not_intervened,
        )
        added_interventions = not_intervened - post_intervention_ancestors
        if added_interventions:
            line_three_interventions.update(added_interventions)
            state.interventions = state.interventions.union(added_interventions)
            continue

        remaining_districts = _districts_within(topology, not_intervened)
        if len(remaining_districts) > 1:
            first = remaining_districts[0]
            continuations.append(
                _DistrictProductFrame(
                    state.vertices,
                    state.outcomes,
                    state.interventions,
                    state.kernel,
                    remaining_districts,
                    1,
                    [],
                )
            )
            state = _IDState(
                state.vertices,
                first,
                state.vertices - first,
                state.kernel,
            )
            continue

        if len(remaining_districts) != 1:
            raise AssertionError("ID reached an empty remaining district")
        remaining_district = remaining_districts[0]
        current_districts = _districts_within(topology, state.vertices)
        if len(current_districts) == 1:
            raise AssertionError(
                "classical ID reached line 5 after the exact-query intrinsic check"
            )

        chain_factors = _kernel_chain_factors(state.kernel)
        if remaining_district in current_districts:
            selected = tuple(
                factor
                for node, factor in zip(state.kernel.nodes, chain_factors, strict=True)
                if node in remaining_district
            )
            summed = _ordered_subset(
                source_topological_order,
                remaining_district - state.outcomes,
            )
            completed = _summation(summed, _product(selected))
            continue

        containing = tuple(
            district for district in current_districts if remaining_district < district
        )
        if len(containing) != 1:
            raise AssertionError("ID line 7 has no unique containing district")
        containing_district = containing[0]
        selected = tuple(
            factor
            for node, factor in zip(state.kernel.nodes, chain_factors, strict=True)
            if node in containing_district
        )
        containing_order = tuple(
            node for node in state.kernel.nodes if node in containing_district
        )
        state = _IDState(
            containing_district,
            state.outcomes,
            state.interventions.intersection(containing_district),
            _Kernel(containing_order, _product(selected), selected),
        )


def _average_added_interventions(
    formula: ProbabilityExpression,
    *,
    line_three_interventions: frozenset[str],
    source_order: tuple[str, ...],
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
) -> ProbabilityExpression:
    """Bind line-3 auxiliary interventions with an observational kernel.

    ID may add irrelevant observed vertices to its recursive intervention set.  Its
    returned kernel is invariant in those vertices on the causal model, but leaving
    them free would not be an exact symbolic representation of the requested query.
    Averaging against their observational conditional distribution binds them without
    choosing a reference value.  This requires only the support of the requested
    intervention event, rather than positivity at every auxiliary assignment, under
    the usual regular-conditional/support interpretation.
    """

    requested = frozenset((*outcomes, *interventions))
    unexpected = formula.free_variables - requested - line_three_interventions
    if unexpected:
        raise AssertionError(
            "classical ID produced free variables that were not introduced at line 3"
        )
    added = tuple(
        vertex
        for vertex in source_order
        if vertex in formula.free_variables
        and vertex in line_three_interventions
        and vertex not in requested
    )
    if not added:
        return formula
    intervention_set = frozenset(interventions)
    ordered_interventions = tuple(
        vertex for vertex in source_order if vertex in intervention_set
    )
    weight = Probability(added, conditioned_on=ordered_interventions)
    return Summation(added, _product((formula, weight)))


def identify_causal_effect(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    *,
    outcomes: str | Iterable[str],
    interventions: str | Iterable[str] = (),
) -> IdentificationResult:
    """Identify ``P(outcomes | do(interventions))`` in a semi-Markovian ADMG."""

    if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
        raise IdentificationDomainError(_DOMAIN_MESSAGE)

    topology = graph.topology
    positions = node_positions(topology.nodes)
    outcome_set = normalize_seeds(positions, outcomes)
    intervention_set = normalize_seeds(positions, interventions)
    if outcome_set.intersection(intervention_set):
        raise IdentificationDomainError(_OVERLAP_MESSAGE)

    normalized_outcomes = _ordered_subset(topology.nodes, outcome_set)
    normalized_interventions = _ordered_subset(topology.nodes, intervention_set)
    if not normalized_outcomes:
        return IdentifiedEffect(
            _FACTORY_TOKEN,
            graph=graph,
            outcomes=normalized_outcomes,
            interventions=normalized_interventions,
            formula=Product(()),
        )

    if isinstance(graph, AcyclicDirectedBidirectedGraph) and topology.bidirected:
        witness = _exact_query_hedge(
            graph,
            normalized_outcomes,
            normalized_interventions,
        )
        if witness is not None:
            return NonIdentifiedEffect(_FACTORY_TOKEN, witness=witness)

    classical_output = _classical_id_formula(
        topology,
        outcomes=outcome_set,
        interventions=intervention_set,
    )
    formula = _average_added_interventions(
        classical_output.formula,
        line_three_interventions=classical_output.line_three_interventions,
        source_order=_topological_order(topology),
        outcomes=normalized_outcomes,
        interventions=normalized_interventions,
    )
    requested_variables = outcome_set.union(intervention_set)
    if not outcome_set <= formula.free_variables <= requested_variables:
        raise AssertionError("ID formula is not bound to the exact requested query")
    return IdentifiedEffect(
        _FACTORY_TOKEN,
        graph=graph,
        outcomes=normalized_outcomes,
        interventions=normalized_interventions,
        formula=formula,
    )
