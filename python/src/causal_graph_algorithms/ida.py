"""Graphical parent-set enumeration for population IDA and joint-IDA."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from itertools import combinations, product
from math import prod
from typing import TypeAlias

from ._validation import node_positions, require_vertex
from .endpoint_marked import EndpointEdge, EndpointMarkedGraph, Mark
from .errors import GraphValidationError
from .pdag import CPDAG, PDAGDomainError, _consistent_extension, _skeleton

ParentSet: TypeAlias = frozenset[str]
ParentTuple: TypeAlias = tuple[ParentSet, ...]


def _cpdag_topology(graph: CPDAG) -> EndpointMarkedGraph:
    if not isinstance(graph, CPDAG):
        raise PDAGDomainError("IDA requires a validated CPDAG")
    return graph.topology


def ida_parent_sets(graph: CPDAG, treatment: str) -> frozenset[ParentSet]:
    """Return the distinct parent sets of one treatment in its DAG class.

    Maathuis, Kalisch and Bühlmann (2009), Algorithm 3 and Lemma 3.1.
    The compelled parents are joined by any clique of undirected neighbors.
    """
    raw = _cpdag_topology(graph)
    try:
        require_vertex(node_positions(raw.nodes), treatment)
    except GraphValidationError as error:
        raise PDAGDomainError(str(error)) from error
    fixed = raw.directed_parents(treatment)
    neighbors = tuple(sorted(raw.undirected_neighbors(treatment)))
    skeleton = _skeleton(raw)
    return frozenset(
        fixed.union(chosen)
        for size in range(len(neighbors) + 1)
        for chosen in combinations(neighbors, size)
        if all(frozenset(pair) in skeleton for pair in combinations(chosen, 2))
    )


def _component_targets(
    raw: EndpointMarkedGraph, targets: tuple[str, ...],
) -> Iterable[tuple[frozenset[str], tuple[str, ...]]]:
    assigned: set[str] = set()
    for target in targets:
        if target in assigned:
            continue
        component = {target}
        queue = [target]
        while queue:
            new_neighbors = raw.undirected_neighbors(queue.pop()) - component
            component.update(new_neighbors)
            queue.extend(new_neighbors)
        assigned.update(component)
        yield frozenset(component), tuple(v for v in targets if v in component)


def _component_parent_counts(
    raw: EndpointMarkedGraph, component: frozenset[str], targets: tuple[str, ...],
) -> Counter[ParentTuple]:
    nodes = tuple(v for v in raw.nodes if v in component)
    lines = tuple(edge for edge in raw.edges
                  if edge[0] in component and edge[2] in component and edge[1] == edge[3] == Mark.TAIL)
    pending: list[tuple[EndpointEdge, ...]] = [()]
    counts: Counter[ParentTuple] = Counter()
    while pending:
        chosen = pending.pop()
        next_edge = len(chosen)
        partial = EndpointMarkedGraph(nodes=nodes, edges=chosen + lines[next_edge:])
        extension = _consistent_extension(partial, original_colliders=frozenset())
        if extension is None:
            continue
        if next_edge == len(lines):
            counts[tuple(extension.topology.directed_parents(v) for v in targets)] += 1
            continue
        left, _, right, _ = lines[next_edge]
        pending.append(chosen + ((left, Mark.ARROW, right, Mark.TAIL),))
        pending.append(chosen + ((left, Mark.TAIL, right, Mark.ARROW),))
    return counts


def joint_ida_parent_counts(
    graph: CPDAG, treatments: str | Iterable[str],
) -> dict[ParentTuple, int]:
    """Return semi-local parent tuple counts in the supplied treatment order.

    Nandy, Maathuis and Richardson (2017), Algorithm 5.1. The ratios
    agree with full DAG enumeration. Untouched chain components contribute
    a common multiplier to the full counts.
    """
    raw = _cpdag_topology(graph)
    positions = node_positions(raw.nodes)
    try:
        values = (treatments,) if isinstance(treatments, str) else tuple(treatments)
        targets = tuple(require_vertex(positions, value) for value in values)
    except (TypeError, GraphValidationError) as error:
        raise PDAGDomainError("treatments must be known vertices in the CPDAG") from error
    if not targets or len(targets) != len(set(targets)):
        raise PDAGDomainError("treatments must be distinct and nonempty")

    fixed = {v: raw.directed_parents(v) for v in targets}
    local = [(group, _component_parent_counts(raw, component, group))
             for component, group in _component_targets(raw, targets)]
    result: dict[ParentTuple, int] = {}
    for selections in product(*(counts.items() for _, counts in local)):
        by_treatment = {
            v: fixed[v] | incoming
            for (group, _), (parent_sets, _) in zip(local, selections, strict=True)
            for v, incoming in zip(group, parent_sets, strict=True)
        }
        result[tuple(by_treatment[v] for v in targets)] = prod(count for _, count in selections)
    return result


def joint_ida_parent_sets(
    graph: CPDAG, treatments: str | Iterable[str],
) -> frozenset[ParentTuple]:
    """Return the distinct tuples of parents realized by a shared DAG."""
    return frozenset(joint_ida_parent_counts(graph, treatments))
