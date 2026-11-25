"""Literal path and exact linear-model checks; no production imports."""
from __future__ import annotations

import itertools
from fractions import Fraction

EdgeSet = frozenset[tuple[str, str]]


def paths(nodes: tuple[str, ...], edges: EdgeSet, start: str, end: str) -> list[tuple[str, ...]]:
    pending = [(start,)]
    found = []
    while pending:
        path = pending.pop()
        if path[-1] == end:
            found.append(path)
            continue
        for v in nodes:
            if v not in path and ((path[-1], v) in edges or (v, path[-1]) in edges):
                pending.append((*path, v))
    return found


def directed(path: tuple[str, ...], edges: EdgeSet) -> bool:
    return all(pair in edges for pair in zip(path, path[1:]))


def descendant(nodes: tuple[str, ...], edges: EdgeSet, a: str, b: str) -> bool:
    return a == b or any(directed(p, edges) for p in paths(nodes, edges, a, b))


def causal_nodes(nodes: tuple[str, ...], edges: EdgeSet, x: set[str], y: set[str]) -> set[str]:
    return {
        v for a in x for b in y for p in paths(nodes, edges, a, b)
        if directed(p, edges) and not (set(p[1:]) & x) for v in p[1:]
    }


def valid(nodes: tuple[str, ...], edges: EdgeSet, x: set[str], y: set[str], z: set[str]) -> bool:
    if z & (x | y):
        return False
    cn = causal_nodes(nodes, edges, x, y)
    if any(descendant(nodes, edges, c, v) for c in cn for v in z):
        return False
    for a in x:
        for b in y:
            for p in paths(nodes, edges, a, b):
                if set(p[1:]) & x or directed(p, edges):
                    continue
                blocked = False
                for left, v, right in zip(p, p[1:], p[2:]):
                    collider = (left, v) in edges and (right, v) in edges
                    if collider:
                        if not any(descendant(nodes, edges, v, w) for w in z):
                            blocked = True
                    elif v in z:
                        blocked = True
                if not blocked:
                    return False
    return True


def subsets(vertices: tuple[str, ...]) -> list[set[str]]:
    return [set(s) for k in range(len(vertices) + 1) for s in itertools.combinations(vertices, k)]


def all_dags(nodes: tuple[str, ...]) -> list[EdgeSet]:
    result: set[EdgeSet] = set()
    for ordering in itertools.permutations(nodes):
        pairs = tuple(itertools.combinations(ordering, 2))
        for mask in range(1 << len(pairs)):
            result.add(frozenset(e for i, e in enumerate(pairs) if mask & (1 << i)))
    return sorted(result, key=lambda e: sorted(e))


def covariance(nodes: tuple[str, ...], edges: EdgeSet, seed: int) -> list[list[Fraction]]:
    """Expand each variable into independent exogenous noises, with rational weights."""
    coefficient: dict[str, list[Fraction]] = {}
    while len(coefficient) < len(nodes):
        for v in nodes:
            if v in coefficient:
                continue
            parents = [u for u in nodes if (u, v) in edges]
            if not all(u in coefficient for u in parents):
                continue
            row = [Fraction(int(w == v)) for w in nodes]
            for u in parents:
                i, j = nodes.index(u), nodes.index(v)
                weight = Fraction((i * 7 + j * 3 + seed) % 7 + 1, 3)
                if (i + seed) % 2:
                    weight = -weight
                row = [a + weight * b for a, b in zip(row, coefficient[u])]
            coefficient[v] = row
    variance = [Fraction((i + seed) % 5 + 1, 2) for i in range(len(nodes))]
    return [[sum((a * b * s for a, b, s in zip(coefficient[u], coefficient[v], variance)), Fraction())
             for v in nodes] for u in nodes]


def residual_variance(cov: list[list[Fraction]], target: int, given: list[int]) -> Fraction:
    if not given:
        return cov[target][target]
    matrix = [[cov[i][j] for j in given] + [cov[i][target]] for i in given]
    for k in range(len(given)):
        pivot = matrix[k][k]
        matrix[k] = [v / pivot for v in matrix[k]]
        for i in range(len(given)):
            if i != k:
                factor = matrix[i][k]
                matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[k])]
    return cov[target][target] - sum((cov[target][v] * matrix[i][-1] for i, v in enumerate(given)), Fraction())


def avar(cov: list[list[Fraction]], x: int, y: int, z: list[int]) -> Fraction:
    return residual_variance(cov, y, [x, *z]) / residual_variance(cov, x, z)
