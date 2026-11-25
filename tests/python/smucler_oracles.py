"""Exhaustive definition-level checks, independent of production reductions/flow."""

from __future__ import annotations

import itertools
from collections.abc import Iterable


def subsets(vertices: Iterable[str]) -> list[frozenset[str]]:
    order = tuple(vertices)
    return [frozenset(s) for k in range(len(order) + 1)
            for s in itertools.combinations(order, k)]


def simple_paths(
    nodes: tuple[str, ...], edges: frozenset[tuple[str, str]], start: str, end: str,
    *, directed: bool = False,
) -> list[tuple[str, ...]]:
    answer: list[tuple[str, ...]] = []
    frontier: list[tuple[str, ...]] = [(start,)]
    while frontier:
        path = frontier.pop()
        if path[-1] == end:
            answer.append(path)
            continue
        for vertex in nodes:
            if vertex in path:
                continue
            forward = (path[-1], vertex) in edges
            backward = not directed and (vertex, path[-1]) in edges
            if forward or backward:
                frontier.append((*path, vertex))
    return answer


def descendants(
    nodes: tuple[str, ...], edges: frozenset[tuple[str, str]], vertex: str,
) -> frozenset[str]:
    return frozenset(other for other in nodes
                     if simple_paths(nodes, edges, vertex, other, directed=True))


def _open_path(
    nodes: tuple[str, ...], edges: frozenset[tuple[str, str]],
    path: tuple[str, ...], conditioned: frozenset[str],
) -> bool:
    for left, middle, right in zip(path, path[1:], path[2:]):
        collider = (left, middle) in edges and (right, middle) in edges
        if collider:
            if not descendants(nodes, edges, middle) & conditioned:
                return False
        elif middle in conditioned:
            return False
    return True


def d_separated(
    nodes: tuple[str, ...], edges: frozenset[tuple[str, str]],
    left: Iterable[str], right: Iterable[str], conditioned: frozenset[str],
) -> bool:
    return not any(_open_path(nodes, edges, path, conditioned)
                   for a in left for b in right
                   for path in simple_paths(nodes, edges, a, b))


def valid_sets(
    nodes: tuple[str, ...], edges: frozenset[tuple[str, str]],
    treatment: str, outcome: str, observed: frozenset[str],
    policy: frozenset[str] = frozenset(),
) -> list[frozenset[str]]:
    paths = simple_paths(nodes, edges, treatment, outcome)
    causal = [path for path in paths
              if all(pair in edges for pair in zip(path, path[1:]))]
    forbidden = {treatment}
    for path in causal:
        for vertex in path[1:]:
            forbidden.update(descendants(nodes, edges, vertex))
    noncausal = [path for path in paths if path not in causal]
    return [candidate for candidate in subsets(v for v in nodes
                                               if v in observed - {treatment, outcome})
            if policy <= candidate and not candidate & forbidden
            and not any(_open_path(nodes, edges, path, candidate)
                        for path in noncausal)]


def binary_joint(
    nodes: tuple[str, ...], edges: frozenset[tuple[str, str]], seed: int,
) -> dict[tuple[int, ...], float]:
    """Positive independent binary CPTs; no graph-library operations."""
    import random

    rng = random.Random(seed)
    parents = {v: tuple(i for i, u in enumerate(nodes) if (u, v) in edges)
               for v in nodes}
    cpts = {v: {bits: rng.uniform(0.1, 0.9)
                for bits in itertools.product((0, 1), repeat=len(parents[v]))}
            for v in nodes}
    joint: dict[tuple[int, ...], float] = {}
    for assignment in itertools.product((0, 1), repeat=len(nodes)):
        probability = 1.0
        for i, vertex in enumerate(nodes):
            p = cpts[vertex][tuple(assignment[j] for j in parents[vertex])]
            probability *= p if assignment[i] else 1 - p
        joint[assignment] = probability
    return joint


def efficiency_bound(
    nodes: tuple[str, ...], joint: dict[tuple[int, ...], float],
    treatment: str, outcome: str, adjustment: frozenset[str],
    policy: frozenset[str] = frozenset(),
) -> float:
    """Exact binary summation of the NP-adjustment influence-function variance."""
    indices = tuple(i for i, v in enumerate(nodes) if v in adjustment)
    a_index, y_index = nodes.index(treatment), nodes.index(outcome)
    policy_indices = tuple(indices.index(nodes.index(v)) for v in nodes if v in policy)
    mean = 0.0
    second_moment = 0.0
    residual = 0.0
    for assignment in itertools.product((0, 1), repeat=len(indices)):
        action = sum(assignment[i] for i in policy_indices) % 2 if policy else 1
        mass = action_mass = outcome_mass = 0.0
        for row, p in joint.items():
            if tuple(row[i] for i in indices) != assignment:
                continue
            mass += p
            if row[a_index] == action:
                action_mass += p
                outcome_mass += p * row[y_index]
        response = outcome_mass / action_mass
        propensity = action_mass / mass
        mean += mass * response
        second_moment += mass * response * response
        residual += mass * response * (1 - response) / propensity
    return residual + second_moment - mean * mean
