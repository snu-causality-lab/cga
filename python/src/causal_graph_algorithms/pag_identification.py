from __future__ import annotations

import typing
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NoReturn, SupportsIndex, TypeAlias

from ._validation import node_positions, normalize_seeds
from .endpoint_marked import Mark
from .errors import IdentificationDomainError
from .pag import CausalPAG
from .probability import (
    Probability,
    ProbabilityExpression,
    Product,
    Ratio,
    Summation,
)
from .visibility import _directed_edge_visibility_path


_DOMAIN_MESSAGE = "PAG causal-effect identification requires a causal PAG proof view"
_OVERLAP_MESSAGE = "outcomes and interventions must be disjoint"
_FACTORY_TOKEN = object()


class _ImmutablePAGIdentificationValue:
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
class IdentifiedPAGEffect(_ImmutablePAGIdentificationValue):
    """An identified PAG causal query and its observational formula."""

    __slots__ = ("graph", "outcomes", "interventions", "formula")

    graph: CausalPAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    formula: ProbabilityExpression

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("IdentifiedPAGEffect cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: CausalPAG,
        outcomes: tuple[str, ...],
        interventions: tuple[str, ...],
        formula: ProbabilityExpression,
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("IdentifiedPAGEffect values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "formula", formula)


@typing.final
@dataclass(frozen=True, init=False)
class NonIdentifiedPAGEffect(_ImmutablePAGIdentificationValue):
    """A nonidentified PAG query and its terminal IDP frame."""

    __slots__ = (
        "graph",
        "outcomes",
        "interventions",
        "terminal_target_vertices",
        "terminal_kernel_vertices",
    )

    graph: CausalPAG
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]
    terminal_target_vertices: tuple[str, ...]
    terminal_kernel_vertices: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("NonIdentifiedPAGEffect cannot be subclassed")

    def __init__(
        self,
        factory_token: object,
        *,
        graph: CausalPAG,
        outcomes: tuple[str, ...],
        interventions: tuple[str, ...],
        terminal_target_vertices: tuple[str, ...],
        terminal_kernel_vertices: tuple[str, ...],
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("NonIdentifiedPAGEffect values are factory-created")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(
            self,
            "terminal_target_vertices",
            terminal_target_vertices,
        )
        object.__setattr__(
            self,
            "terminal_kernel_vertices",
            terminal_kernel_vertices,
        )


PAGIdentificationResult: TypeAlias = IdentifiedPAGEffect | NonIdentifiedPAGEffect


class _PAGContext:
    """Original-PAG relations inherited by every induced IDP frame."""

    __slots__ = (
        "nodes",
        "positions",
        "marks",
        "neighbors",
        "visible_directed_edges",
        "circle_bucket_cache",
        "possible_closure_cache",
        "pc_related_cache",
        "pc_vertices_cache",
        "region_cache",
    )

    nodes: tuple[str, ...]
    positions: dict[str, int]
    marks: dict[tuple[str, str], Mark]
    neighbors: dict[str, tuple[str, ...]]
    visible_directed_edges: frozenset[tuple[str, str]]
    circle_bucket_cache: dict[frozenset[str], tuple[tuple[str, ...], ...]]
    possible_closure_cache: dict[
        tuple[bool, frozenset[str], frozenset[str]], frozenset[str]
    ]
    pc_related_cache: dict[tuple[str, str, frozenset[str]], bool]
    pc_vertices_cache: dict[tuple[frozenset[str], frozenset[str]], frozenset[str]]
    region_cache: dict[tuple[frozenset[str], frozenset[str]], frozenset[str]]

    def __init__(self, graph: CausalPAG) -> None:
        topology = graph.topology
        self.nodes = topology.nodes
        self.positions = node_positions(self.nodes)
        neighbor_lists: dict[str, list[str]] = {node: [] for node in self.nodes}
        self.marks = {}
        directed_edges: list[tuple[str, str]] = []
        for left, left_mark, right, right_mark in topology.edges:
            neighbor_lists[left].append(right)
            neighbor_lists[right].append(left)
            self.marks[(left, right)] = left_mark
            self.marks[(right, left)] = right_mark
            if left_mark is Mark.TAIL and right_mark is Mark.ARROW:
                directed_edges.append((left, right))
            elif left_mark is Mark.ARROW and right_mark is Mark.TAIL:
                directed_edges.append((right, left))
        self.neighbors = {
            node: tuple(sorted(values, key=self.positions.__getitem__))
            for node, values in neighbor_lists.items()
        }
        self.visible_directed_edges = frozenset(
            (source, target)
            for source, target in directed_edges
            if _directed_edge_visibility_path(topology, source, target) is not None
        )
        self.circle_bucket_cache = {}
        self.possible_closure_cache = {}
        self.pc_related_cache = {}
        self.pc_vertices_cache = {}
        self.region_cache = {}

    def ordered(self, vertices: frozenset[str]) -> tuple[str, ...]:
        return tuple(node for node in self.nodes if node in vertices)

    def edge_is_visible(self, left: str, right: str) -> bool:
        left_mark = self.marks[(left, right)]
        right_mark = self.marks[(right, left)]
        return (
            left_mark is Mark.TAIL
            and right_mark is Mark.ARROW
            and (left, right) in self.visible_directed_edges
        ) or (
            left_mark is Mark.ARROW
            and right_mark is Mark.TAIL
            and (right, left) in self.visible_directed_edges
        )

    def circle_buckets(self, ambient: frozenset[str]) -> tuple[tuple[str, ...], ...]:
        cached = self.circle_bucket_cache.get(ambient)
        if cached is not None:
            return cached

        unassigned = set(ambient)
        buckets: list[tuple[str, ...]] = []
        for seed in self.nodes:
            if seed not in unassigned:
                continue
            reached = {seed}
            pending = [seed]
            while pending:
                current = pending.pop()
                for neighbor in self.neighbors[current]:
                    if neighbor not in ambient or neighbor in reached:
                        continue
                    if not (
                        self.marks[(current, neighbor)] is Mark.CIRCLE
                        and self.marks[(neighbor, current)] is Mark.CIRCLE
                    ):
                        continue
                    reached.add(neighbor)
                    pending.append(neighbor)
            unassigned.difference_update(reached)
            buckets.append(self.ordered(frozenset(reached)))
        result = tuple(buckets)
        self.circle_bucket_cache[ambient] = result
        return result

    def _possible_closure(
        self,
        seeds: frozenset[str],
        ambient: frozenset[str],
        *,
        reverse: bool,
    ) -> frozenset[str]:
        if not seeds <= ambient:
            raise AssertionError("possible-path seeds left the induced PAG")
        key = (reverse, seeds, ambient)
        cached = self.possible_closure_cache.get(key)
        if cached is not None:
            return cached

        reached = set(seeds)
        pending = list(reversed(self.ordered(seeds)))
        while pending:
            current = pending.pop()
            for neighbor in self.neighbors[current]:
                if neighbor not in ambient or neighbor in reached:
                    continue
                departure_mark = (
                    self.marks[(neighbor, current)]
                    if reverse
                    else self.marks[(current, neighbor)]
                )
                if departure_mark is Mark.ARROW:
                    continue
                reached.add(neighbor)
                pending.append(neighbor)
        result = frozenset(reached)
        self.possible_closure_cache[key] = result
        return result

    def possible_descendants(
        self, seeds: frozenset[str], ambient: frozenset[str]
    ) -> frozenset[str]:
        return self._possible_closure(seeds, ambient, reverse=False)

    def possible_ancestors(
        self, targets: frozenset[str], ambient: frozenset[str]
    ) -> frozenset[str]:
        return self._possible_closure(targets, ambient, reverse=True)

    def pc_related(
        self,
        left: str,
        right: str,
        ambient: frozenset[str],
    ) -> bool:
        if left not in ambient or right not in ambient:
            raise AssertionError("PC endpoints left the induced PAG")
        if self.positions[left] > self.positions[right]:
            left, right = right, left
        key = (left, right, ambient)
        if key in self.pc_related_cache:
            return self.pc_related_cache[key]
        if left == right:
            self.pc_related_cache[key] = True
            return True

        if right in self.neighbors[left] and not self.edge_is_visible(left, right):
            self.pc_related_cache[key] = True
            return True

        internal = ambient.difference((left, right))
        reached: set[str] = set()
        pending: list[str] = []
        for neighbor in reversed(self.neighbors[left]):
            if (
                neighbor in internal
                and self.marks[(neighbor, left)] is Mark.ARROW
                and not self.edge_is_visible(left, neighbor)
            ):
                reached.add(neighbor)
                pending.append(neighbor)

        while pending:
            current = pending.pop()
            if (
                right in self.neighbors[current]
                and self.marks[(current, right)] is Mark.ARROW
                and not self.edge_is_visible(current, right)
            ):
                self.pc_related_cache[key] = True
                return True
            for neighbor in reversed(self.neighbors[current]):
                if neighbor not in internal or neighbor in reached:
                    continue
                if not (
                    self.marks[(current, neighbor)] is Mark.ARROW
                    and self.marks[(neighbor, current)] is Mark.ARROW
                    and not self.edge_is_visible(current, neighbor)
                ):
                    continue
                reached.add(neighbor)
                pending.append(neighbor)

        self.pc_related_cache[key] = False
        return False

    def pc_vertices(
        self, seeds: frozenset[str], ambient: frozenset[str]
    ) -> frozenset[str]:
        if not seeds <= ambient:
            raise AssertionError("PC seeds left the induced PAG")
        key = (seeds, ambient)
        cached = self.pc_vertices_cache.get(key)
        if cached is not None:
            return cached
        result = frozenset(
            candidate
            for candidate in self.nodes
            if candidate in ambient
            and any(
                self.pc_related(seed, candidate, ambient)
                for seed in self.ordered(seeds)
            )
        )
        self.pc_vertices_cache[key] = result
        return result

    def region(self, seeds: frozenset[str], ambient: frozenset[str]) -> frozenset[str]:
        if not seeds <= ambient:
            raise AssertionError("region seeds left the induced PAG")
        key = (seeds, ambient)
        cached = self.region_cache.get(key)
        if cached is not None:
            return cached
        pc_vertices = self.pc_vertices(seeds, ambient)
        result = frozenset(
            node
            for bucket in self.circle_buckets(ambient)
            if pc_vertices.intersection(bucket)
            for node in bucket
        )
        self.region_cache[key] = result
        return result


@dataclass(frozen=True, slots=True)
class _Kernel:
    vertices: tuple[str, ...]
    expression: ProbabilityExpression


@dataclass(frozen=True, slots=True, eq=False)
class _MemoKey:
    target_vertices: tuple[str, ...]
    kernel_vertices: tuple[str, ...]
    expression: ProbabilityExpression

    def __hash__(self) -> int:
        return hash(
            (
                self.target_vertices,
                self.kernel_vertices,
                id(self.expression),
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _MemoKey)
            and self.target_vertices == other.target_vertices
            and self.kernel_vertices == other.kernel_vertices
            and self.expression is other.expression
        )


@dataclass(frozen=True, slots=True)
class _EvaluateTask:
    target: frozenset[str]
    kernel: _Kernel


@dataclass(frozen=True, slots=True)
class _ForwardTask:
    key: _MemoKey


@dataclass(frozen=True, slots=True)
class _RegionTask:
    key: _MemoKey
    target_vertices: tuple[str, ...]


_Task: TypeAlias = _EvaluateTask | _ForwardTask | _RegionTask


@dataclass(frozen=True, slots=True)
class _IdentificationFailure(Exception):
    target_vertices: tuple[str, ...]
    kernel_vertices: tuple[str, ...]


def _store_kernel(
    key: _MemoKey,
    result: _Kernel,
    *,
    active: set[_MemoKey],
    memo: dict[_MemoKey, _Kernel],
    completed: list[_Kernel],
) -> None:
    if key not in active:
        raise AssertionError("PAG IDP completed an inactive frame")
    active.remove(key)
    memo[key] = result
    completed.append(result)


def _require_summation(
    context: _PAGContext,
    variables: frozenset[str],
    body: ProbabilityExpression,
) -> Summation:
    if not variables:
        raise AssertionError("PAG IDP requested an empty structural summation")
    if not variables <= body.free_variables:
        raise AssertionError("PAG IDP requested a non-free summation binder")
    return Summation(context.ordered(variables), body)


def _sum_out(
    context: _PAGContext,
    variables: frozenset[str],
    body: ProbabilityExpression,
) -> ProbabilityExpression:
    if not variables:
        return body
    return _require_summation(context, variables, body)


def _identify_kernel(
    context: _PAGContext,
    target: frozenset[str],
    initial_kernel: _Kernel,
) -> _Kernel:
    tasks: list[_Task] = [_EvaluateTask(target, initial_kernel)]
    completed: list[_Kernel] = []
    memo: dict[_MemoKey, _Kernel] = {}
    active: set[_MemoKey] = set()

    while tasks:
        task = tasks.pop()
        if isinstance(task, _ForwardTask):
            if not completed:
                raise AssertionError("PAG IDP forwarding task lost its child result")
            result = completed.pop()
            _store_kernel(
                task.key,
                result,
                active=active,
                memo=memo,
                completed=completed,
            )
            continue

        if isinstance(task, _RegionTask):
            if len(completed) < 3:
                raise AssertionError("PAG IDP region task lost a child result")
            intersection_kernel = completed.pop()
            right_kernel = completed.pop()
            left_kernel = completed.pop()
            result = _Kernel(
                task.target_vertices,
                Ratio(
                    Product(
                        (
                            left_kernel.expression,
                            right_kernel.expression,
                        )
                    ),
                    intersection_kernel.expression,
                ),
            )
            _store_kernel(
                task.key,
                result,
                active=active,
                memo=memo,
                completed=completed,
            )
            continue

        target_vertices = context.ordered(task.target)
        ambient = frozenset(task.kernel.vertices)
        if not task.target <= ambient:
            raise AssertionError("PAG IDP target left its kernel vertices")
        key = _MemoKey(
            target_vertices,
            task.kernel.vertices,
            task.kernel.expression,
        )
        cached = memo.get(key)
        if cached is not None:
            completed.append(cached)
            continue
        if key in active:
            raise AssertionError("PAG IDP repeated an active frame")
        active.add(key)

        if not task.target:
            _store_kernel(
                key,
                _Kernel((), Product(())),
                active=active,
                memo=memo,
                completed=completed,
            )
            continue
        if task.target == ambient:
            _store_kernel(
                key,
                task.kernel,
                active=active,
                memo=memo,
                completed=completed,
            )
            continue

        scheduled = False
        removable = ambient.difference(task.target)
        for bucket_vertices in context.circle_buckets(ambient):
            bucket = frozenset(bucket_vertices)
            if not bucket <= removable:
                continue
            pc_vertices = context.pc_vertices(bucket, ambient)
            descendants = context.possible_descendants(bucket, ambient)
            if not pc_vertices.intersection(descendants) <= bucket:
                continue

            expression = task.kernel.expression
            descendant_marginal = _require_summation(
                context,
                descendants,
                expression,
            )
            denominator: ProbabilityExpression
            if descendants == bucket:
                denominator = expression
            else:
                denominator = _require_summation(
                    context,
                    descendants.difference(bucket),
                    expression,
                )
            reduced_vertices = ambient.difference(bucket)
            reduced = _Kernel(
                context.ordered(reduced_vertices),
                Ratio(
                    Product((expression, descendant_marginal)),
                    denominator,
                ),
            )
            tasks.append(_ForwardTask(key))
            tasks.append(_EvaluateTask(task.target, reduced))
            scheduled = True
            break
        if scheduled:
            continue

        for bucket_vertices in context.circle_buckets(task.target):
            bucket = frozenset(bucket_vertices)
            left = context.region(bucket, task.target)
            if left == task.target:
                continue
            right = context.region(task.target.difference(left), task.target)
            intersection = left.intersection(right)
            tasks.append(_RegionTask(key, target_vertices))
            tasks.append(_EvaluateTask(intersection, task.kernel))
            tasks.append(_EvaluateTask(right, task.kernel))
            tasks.append(_EvaluateTask(left, task.kernel))
            scheduled = True
            break
        if scheduled:
            continue

        raise _IdentificationFailure(target_vertices, task.kernel.vertices)

    if active:
        raise AssertionError("PAG IDP left active frames after completion")
    if len(completed) != 1:
        raise AssertionError("PAG IDP produced an invalid result stack")
    return completed[0]


def _assert_observational_formula(
    expression: ProbabilityExpression,
    source_vertices: frozenset[str],
) -> None:
    pending = [expression]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, Probability):
            labels = frozenset((*current.variables, *current.conditioned_on))
            if current.interventions or not labels <= source_vertices:
                raise AssertionError(
                    "PAG IDP produced a non-observational probability atom"
                )
        elif isinstance(current, Product):
            pending.extend(current.factors)
        elif isinstance(current, Summation):
            if not frozenset(current.variables) <= source_vertices:
                raise AssertionError("PAG IDP produced an unknown summation binder")
            pending.append(current.body)
        else:
            pending.append(current.numerator)
            pending.append(current.denominator)


def identify_pag_causal_effect(
    graph: CausalPAG,
    *,
    outcomes: str | Iterable[str],
    interventions: str | Iterable[str] = (),
) -> PAGIdentificationResult:
    """Identify a no-selection PAG marginal effect.

    Returned functionals use the usual regular-conditional and query-support
    interpretation; this symbolic operation does not define off-support values.
    """

    if not isinstance(graph, CausalPAG):
        raise IdentificationDomainError(_DOMAIN_MESSAGE)

    context = _PAGContext(graph)
    vertices = frozenset(context.nodes)
    outcome_set = normalize_seeds(context.positions, outcomes)
    intervention_set = normalize_seeds(context.positions, interventions)
    if outcome_set.intersection(intervention_set):
        raise IdentificationDomainError(_OVERLAP_MESSAGE)

    normalized_outcomes = context.ordered(outcome_set)
    normalized_interventions = context.ordered(intervention_set)
    if not outcome_set:
        return IdentifiedPAGEffect(
            _FACTORY_TOKEN,
            graph=graph,
            outcomes=normalized_outcomes,
            interventions=normalized_interventions,
            formula=Product(()),
        )

    observational = Probability(context.nodes)
    if not intervention_set:
        formula = _sum_out(
            context,
            vertices.difference(outcome_set),
            observational,
        )
        return IdentifiedPAGEffect(
            _FACTORY_TOKEN,
            graph=graph,
            outcomes=normalized_outcomes,
            interventions=normalized_interventions,
            formula=formula,
        )

    retained = vertices.difference(intervention_set)
    ancestors = context.possible_ancestors(outcome_set, retained)
    try:
        kernel = _identify_kernel(
            context,
            ancestors,
            _Kernel(context.nodes, observational),
        )
    except _IdentificationFailure as failure:
        return NonIdentifiedPAGEffect(
            _FACTORY_TOKEN,
            graph=graph,
            outcomes=normalized_outcomes,
            interventions=normalized_interventions,
            terminal_target_vertices=failure.target_vertices,
            terminal_kernel_vertices=failure.kernel_vertices,
        )

    formula = _sum_out(
        context,
        ancestors.difference(outcome_set),
        kernel.expression,
    )
    free_variables = formula.free_variables
    extras = free_variables.intersection(
        vertices.difference(ancestors.union(intervention_set))
    )
    if extras:
        ordered_extras = context.ordered(extras)
        formula = Summation(
            ordered_extras,
            Product(
                (
                    formula,
                    Probability(
                        ordered_extras,
                        conditioned_on=normalized_interventions,
                    ),
                )
            ),
        )

    _assert_observational_formula(formula, vertices)
    free_variables = formula.free_variables
    requested = outcome_set.union(intervention_set)
    if not outcome_set <= free_variables <= requested:
        raise AssertionError(
            "PAG IDP formula is not bound to the exact requested query"
        )
    return IdentifiedPAGEffect(
        _FACTORY_TOKEN,
        graph=graph,
        outcomes=normalized_outcomes,
        interventions=normalized_interventions,
        formula=formula,
    )
