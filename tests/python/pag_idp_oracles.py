"""Literal test-only oracle for no-selection PAG marginal IDP.

This module imports no production graph, visibility, or identification code.  It
implements Jaber--Ribeiro--Zhang--Bareinboim (2022), Definition 5,
Definition 6, Proposition 2, and Algorithm 1 directly over endpoint-mark atoms.
The deliberately factorial path searches are suitable only for small tests.

Two source details are intentionally visible here.  PC-relatedness is a
pairwise, nontransitive relation, and a directed edge keeps the visibility it
has in the original PAG when an induced graph is considered.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache
from typing import TypeAlias

from .oracles import (
    EndpointMarkAtom,
    directed_edge_visibility_witness_by_simple_paths,
)


VertexSet = frozenset[str]


@dataclass(frozen=True, slots=True)
class TinyPAG:
    """A validated endpoint-marked value used only by the IDP oracle."""

    nodes: tuple[str, ...]
    edges: tuple[EndpointMarkAtom, ...]

    def __post_init__(self) -> None:
        positions = {node: index for index, node in enumerate(self.nodes)}
        if len(positions) != len(self.nodes):
            raise ValueError("oracle PAG nodes must be unique")
        if any(not node for node in self.nodes):
            raise ValueError("oracle PAG nodes must be nonempty")
        seen: set[frozenset[str]] = set()
        for left, left_mark, right, right_mark in self.edges:
            if (
                left not in positions
                or right not in positions
                or left == right
                or left_mark not in {"tail", "arrow", "circle"}
                or right_mark not in {"tail", "arrow", "circle"}
            ):
                raise ValueError("invalid oracle PAG edge")
            pair = frozenset((left, right))
            if pair in seen:
                raise ValueError("oracle PAG repeats an adjacency")
            seen.add(pair)
            if positions[left] >= positions[right]:
                raise ValueError("oracle PAG edges must follow declared node order")


class _OracleExpression:
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class OracleProbability(_OracleExpression):
    """An observational probability atom."""

    variables: tuple[str, ...]
    conditioned_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OracleUnit(_OracleExpression):
    pass


@dataclass(frozen=True, slots=True)
class OracleProduct(_OracleExpression):
    factors: tuple[OracleExpression, ...]


@dataclass(frozen=True, slots=True)
class OracleSummation(_OracleExpression):
    variables: tuple[str, ...]
    body: OracleExpression


@dataclass(frozen=True, slots=True)
class OracleRatio(_OracleExpression):
    numerator: OracleExpression
    denominator: OracleExpression


OracleExpression: TypeAlias = (
    OracleProbability | OracleUnit | OracleProduct | OracleSummation | OracleRatio
)


@dataclass(frozen=True, slots=True)
class OracleIDPStep:
    """One definition-level recursion decision."""

    action: str
    target_vertices: tuple[str, ...]
    kernel_vertices: tuple[str, ...]
    bucket: tuple[str, ...] = ()
    pc_vertices: tuple[str, ...] = ()
    possible_descendants: tuple[str, ...] = ()
    left_region: tuple[str, ...] = ()
    right_region: tuple[str, ...] = ()
    intersection: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OracleIDPResult:
    graph: TinyPAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    possible_ancestors: tuple[str, ...]
    unbound_formula: OracleExpression | None
    formula: OracleExpression | None
    terminal_target_vertices: tuple[str, ...] | None
    terminal_kernel_vertices: tuple[str, ...] | None
    trace: tuple[OracleIDPStep, ...]

    @property
    def identified(self) -> bool:
        return self.formula is not None


@dataclass(frozen=True, slots=True)
class _Kernel:
    vertices: tuple[str, ...]
    expression: OracleExpression


@dataclass(frozen=True, slots=True)
class _Failure(Exception):
    target_vertices: tuple[str, ...]
    kernel_vertices: tuple[str, ...]


def ordered(graph: TinyPAG, vertices: set[str] | VertexSet) -> tuple[str, ...]:
    return tuple(node for node in graph.nodes if node in vertices)


@cache
def _edge_maps(
    graph: TinyPAG,
) -> tuple[
    dict[tuple[str, str], tuple[str, str]],
    dict[str, tuple[str, ...]],
]:
    positions = {node: index for index, node in enumerate(graph.nodes)}
    marks: dict[tuple[str, str], tuple[str, str]] = {}
    neighbors: dict[str, list[str]] = {node: [] for node in graph.nodes}
    for left, left_mark, right, right_mark in graph.edges:
        marks[(left, right)] = (left_mark, right_mark)
        marks[(right, left)] = (right_mark, left_mark)
        neighbors[left].append(right)
        neighbors[right].append(left)
    return marks, {
        node: tuple(sorted(values, key=positions.__getitem__))
        for node, values in neighbors.items()
    }


@cache
def visible_directed_edges_by_definition(
    graph: TinyPAG,
) -> frozenset[tuple[str, str]]:
    """Apply Zhang's directed-edge visibility definition in the full PAG."""

    visible: set[tuple[str, str]] = set()
    for left, left_mark, right, right_mark in graph.edges:
        if (left_mark, right_mark) == ("tail", "arrow"):
            source, target = left, right
        elif (left_mark, right_mark) == ("arrow", "tail"):
            source, target = right, left
        else:
            continue
        if (
            directed_edge_visibility_witness_by_simple_paths(
                graph.nodes, graph.edges, source, target
            )
            is not None
        ):
            visible.add((source, target))
    return frozenset(visible)


@cache
def circle_buckets_by_paths(
    graph: TinyPAG, ambient: VertexSet
) -> tuple[tuple[str, ...], ...]:
    """Return maximal circle--circle path components in source order."""

    marks, neighbors = _edge_maps(graph)
    unassigned = set(ambient)
    result: list[tuple[str, ...]] = []
    for seed in graph.nodes:
        if seed not in unassigned:
            continue
        reached = {seed}
        pending = [seed]
        while pending:
            current = pending.pop()
            for neighbor in neighbors[current]:
                if neighbor not in ambient or neighbor in reached:
                    continue
                if marks[(current, neighbor)] != ("circle", "circle"):
                    continue
                reached.add(neighbor)
                pending.append(neighbor)
        unassigned.difference_update(reached)
        result.append(ordered(graph, reached))
    return tuple(result)


@cache
def possible_descendants_reflexive_by_simple_paths(
    graph: TinyPAG, seeds: VertexSet, ambient: VertexSet
) -> VertexSet:
    """Literal reflexive possible descendants in the induced PAG."""

    marks, neighbors = _edge_maps(graph)
    reached: set[str] = set(seeds)

    def visit(current: str, path: VertexSet) -> None:
        for neighbor in neighbors[current]:
            if (
                neighbor not in ambient
                or neighbor in path
                or marks[(current, neighbor)][0] == "arrow"
            ):
                continue
            reached.add(neighbor)
            visit(neighbor, path | {neighbor})

    for seed in seeds:
        visit(seed, frozenset({seed}))
    return frozenset(reached)


@cache
def possible_ancestors_reflexive_by_simple_paths(
    graph: TinyPAG, targets: VertexSet, ambient: VertexSet
) -> VertexSet:
    """Literal candidate-by-candidate reflexive possible ancestry."""

    return frozenset(
        candidate
        for candidate in ambient
        if possible_descendants_reflexive_by_simple_paths(
            graph, frozenset({candidate}), ambient
        ).intersection(targets)
    )


@cache
def pc_related_by_simple_paths(
    graph: TinyPAG,
    left: str,
    right: str,
    ambient: VertexSet,
) -> bool:
    """Test literal Definition 5 without taking a transitive closure."""

    if left not in ambient or right not in ambient:
        raise ValueError("pc-related endpoints must lie in the induced PAG")
    if left == right:
        return True
    marks, neighbors = _edge_maps(graph)
    visible = visible_directed_edges_by_definition(graph)

    def edge_visible(first: str, second: str) -> bool:
        pair_marks = marks[(first, second)]
        if pair_marks == ("tail", "arrow"):
            return (first, second) in visible
        if pair_marks == ("arrow", "tail"):
            return (second, first) in visible
        return False

    available_internal = tuple(
        node for node in graph.nodes if node in ambient and node not in (left, right)
    )
    for internal_count in range(len(available_internal) + 1):
        for internal in itertools.permutations(available_internal, internal_count):
            path = (left, *internal, right)
            if any(
                (first, second) not in marks or edge_visible(first, second)
                for first, second in itertools.pairwise(path)
            ):
                continue
            if all(
                marks[(current, previous)][0] == "arrow"
                and marks[(current, following)][0] == "arrow"
                for previous, current, following in zip(path, path[1:], path[2:])
            ):
                return True
    return False


@cache
def pc_vertices_one_shot(
    graph: TinyPAG, seeds: VertexSet, ambient: VertexSet
) -> VertexSet:
    """Union the pairwise pc-neighborhoods of seeds exactly once."""

    return frozenset(
        candidate
        for candidate in ambient
        if any(
            pc_related_by_simple_paths(graph, seed, candidate, ambient)
            for seed in seeds
        )
    )


@cache
def region_by_definition(
    graph: TinyPAG, seeds: VertexSet, ambient: VertexSet
) -> VertexSet:
    """Bucket-saturate the one-shot pc-neighborhood inside ``P_ambient``."""

    if not seeds <= ambient:
        raise ValueError("region seeds must lie in the ambient PAG")
    pc_vertices = pc_vertices_one_shot(graph, seeds, ambient)
    return frozenset(
        node
        for bucket in circle_buckets_by_paths(graph, ambient)
        if pc_vertices.intersection(bucket)
        for node in bucket
    )


def expression_free_variables(expression: OracleExpression) -> VertexSet:
    if isinstance(expression, OracleProbability):
        return frozenset((*expression.variables, *expression.conditioned_on))
    if isinstance(expression, OracleUnit):
        return frozenset()
    if isinstance(expression, OracleProduct):
        return frozenset().union(
            *(expression_free_variables(factor) for factor in expression.factors)
        )
    if isinstance(expression, OracleSummation):
        return expression_free_variables(expression.body).difference(
            expression.variables
        )
    return expression_free_variables(expression.numerator).union(
        expression_free_variables(expression.denominator)
    )


def _sum_out(
    graph: TinyPAG, expression: OracleExpression, variables: VertexSet
) -> OracleExpression:
    free = expression_free_variables(expression)
    if not variables <= free:
        raise AssertionError("oracle summation requested a non-free binder")
    normalized = ordered(graph, set(variables))
    return expression if not normalized else OracleSummation(normalized, expression)


def _product(*expressions: OracleExpression) -> OracleExpression:
    factors = tuple(
        expression
        for expression in expressions
        if not isinstance(expression, OracleUnit)
    )
    if not factors:
        return OracleUnit()
    if len(factors) == 1:
        return factors[0]
    return OracleProduct(factors)


def _ratio(
    numerator: OracleExpression, denominator: OracleExpression
) -> OracleExpression:
    return OracleRatio(numerator, denominator)


def identify_pag_by_definition(
    graph: TinyPAG,
    *,
    outcomes: VertexSet,
    interventions: VertexSet = frozenset(),
) -> OracleIDPResult:
    """Run the 2022 marginal IDP recursion in deterministic source order."""

    vertices = frozenset(graph.nodes)
    if not outcomes <= vertices or not interventions <= vertices:
        raise ValueError("oracle IDP query contains an unknown vertex")
    if outcomes.intersection(interventions):
        raise ValueError("oracle IDP outcomes and interventions overlap")
    normalized_outcomes = ordered(graph, set(outcomes))
    normalized_interventions = ordered(graph, set(interventions))
    if not outcomes:
        return OracleIDPResult(
            graph,
            normalized_outcomes,
            normalized_interventions,
            (),
            OracleUnit(),
            OracleUnit(),
            None,
            None,
            (),
        )
    if not interventions:
        ancestors = possible_ancestors_reflexive_by_simple_paths(
            graph, outcomes, vertices
        )
        observational = _sum_out(
            graph,
            OracleProbability(graph.nodes),
            vertices.difference(outcomes),
        )
        return OracleIDPResult(
            graph,
            normalized_outcomes,
            normalized_interventions,
            ordered(graph, set(ancestors)),
            observational,
            observational,
            None,
            None,
            (),
        )

    retained = vertices.difference(interventions)
    ancestors = possible_ancestors_reflexive_by_simple_paths(graph, outcomes, retained)
    normalized_ancestors = ordered(graph, set(ancestors))
    trace: list[OracleIDPStep] = []
    active: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    def identify(target: VertexSet, kernel: _Kernel) -> _Kernel:
        target_tuple = ordered(graph, set(target))
        ambient = frozenset(kernel.vertices)
        key = (target_tuple, kernel.vertices)
        if key in active:
            raise AssertionError("definition-level IDP repeated an active frame")
        active.add(key)
        try:
            if not target:
                trace.append(OracleIDPStep("empty", target_tuple, kernel.vertices))
                return _Kernel((), OracleUnit())
            if target == ambient:
                trace.append(OracleIDPStep("equal", target_tuple, kernel.vertices))
                return kernel

            # Algorithm 1 line 6: B may equal all of T \\ C.  Both the
            # pc-neighborhood and reflexive PossDe are computed in P_T.
            for bucket_tuple in circle_buckets_by_paths(graph, ambient):
                bucket = frozenset(bucket_tuple)
                if not bucket <= ambient.difference(target):
                    continue
                pc_vertices = pc_vertices_one_shot(graph, bucket, ambient)
                descendants = possible_descendants_reflexive_by_simple_paths(
                    graph, bucket, ambient
                )
                if not pc_vertices.intersection(descendants) <= bucket:
                    continue
                trace.append(
                    OracleIDPStep(
                        "eliminate",
                        target_tuple,
                        kernel.vertices,
                        bucket_tuple,
                        ordered(graph, set(pc_vertices)),
                        ordered(graph, set(descendants)),
                    )
                )
                descendant_marginal = OracleSummation(
                    ordered(graph, set(descendants)), kernel.expression
                )
                denominator = (
                    kernel.expression
                    if descendants == bucket
                    else OracleSummation(
                        ordered(graph, set(descendants.difference(bucket))),
                        kernel.expression,
                    )
                )
                reduced_vertices = ambient.difference(bucket)
                reduced = _Kernel(
                    ordered(graph, set(reduced_vertices)),
                    OracleRatio(
                        OracleProduct((kernel.expression, descendant_marginal)),
                        denominator,
                    ),
                )
                return identify(target, reduced)

            # Algorithm 1 line 9: choose buckets from P_C and compute both
            # regions in P_C.  Region construction is not iterated.
            for bucket_tuple in circle_buckets_by_paths(graph, target):
                bucket = frozenset(bucket_tuple)
                left = region_by_definition(graph, bucket, target)
                if left == target:
                    continue
                right_seed = target.difference(left)
                right = region_by_definition(graph, right_seed, target)
                intersection = left.intersection(right)
                trace.append(
                    OracleIDPStep(
                        "region",
                        target_tuple,
                        kernel.vertices,
                        bucket_tuple,
                        left_region=ordered(graph, set(left)),
                        right_region=ordered(graph, set(right)),
                        intersection=ordered(graph, set(intersection)),
                    )
                )
                left_kernel = identify(left, kernel)
                right_kernel = identify(right, kernel)
                intersection_kernel = identify(intersection, kernel)
                return _Kernel(
                    target_tuple,
                    _ratio(
                        _product(left_kernel.expression, right_kernel.expression),
                        intersection_kernel.expression,
                    ),
                )

            trace.append(OracleIDPStep("fail", target_tuple, kernel.vertices))
            raise _Failure(target_tuple, kernel.vertices)
        finally:
            active.remove(key)

    try:
        kernel = identify(
            ancestors,
            _Kernel(graph.nodes, OracleProbability(graph.nodes)),
        )
    except _Failure as failure:
        return OracleIDPResult(
            graph,
            normalized_outcomes,
            normalized_interventions,
            normalized_ancestors,
            None,
            None,
            failure.target_vertices,
            failure.kernel_vertices,
            tuple(trace),
        )

    unbound_formula = _sum_out(graph, kernel.expression, ancestors.difference(outcomes))
    formula = unbound_formula
    free = expression_free_variables(formula)
    extras = free.intersection(vertices.difference(ancestors.union(interventions)))
    if extras:
        formula = OracleSummation(
            ordered(graph, set(extras)),
            _product(
                formula,
                OracleProbability(
                    ordered(graph, set(extras)), normalized_interventions
                ),
            ),
        )
    free = expression_free_variables(formula)
    if not outcomes <= free <= outcomes.union(interventions):
        raise AssertionError("oracle IDP formula is not bound to the exact query")
    return OracleIDPResult(
        graph,
        normalized_outcomes,
        normalized_interventions,
        normalized_ancestors,
        unbound_formula,
        formula,
        None,
        None,
        tuple(trace),
    )


def all_nonempty_outcome_queries(
    nodes: tuple[str, ...],
) -> tuple[tuple[VertexSet, VertexSet], ...]:
    """All ordered roles ``(Y, X)`` with nonempty Y and disjoint X."""

    queries: list[tuple[VertexSet, VertexSet]] = []
    for roles in itertools.product(
        ("other", "outcome", "intervention"), repeat=len(nodes)
    ):
        outcomes = frozenset(
            node for node, role in zip(nodes, roles, strict=True) if role == "outcome"
        )
        if not outcomes:
            continue
        interventions = frozenset(
            node
            for node, role in zip(nodes, roles, strict=True)
            if role == "intervention"
        )
        queries.append((outcomes, interventions))
    return tuple(queries)
