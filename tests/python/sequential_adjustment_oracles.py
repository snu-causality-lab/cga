"""Independent raw-edge, latent-DAG moralization and finite-SCM SCA checks.

No production graph traversal, stage transformation or constructor is imported.
The structural oracle uses all directed paths and latent-DAG moralization;
the numerical oracle evaluates the supplied observational expression directly.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product

Edges = frozenset[tuple[str, str]]


def directed_paths(edges: Edges, start: str, end: str) -> list[tuple[str, ...]]:
    pending = [(start,)]
    found = []
    while pending:
        path = pending.pop()
        if path[-1] == end:
            found.append(path)
        else:
            pending.extend((*path, b) for a, b in edges if a == path[-1] and b not in path)
    return found


def desc(nodes: tuple[str, ...], edges: Edges, seeds: set[str]) -> set[str]:
    return {v for u in seeds for v in nodes if directed_paths(edges, u, v)}


def outcome_blocks(nodes: tuple[str, ...], edges: Edges, treatments: tuple[str, ...],
                   outcomes: set[str]) -> tuple[set[str], ...]:
    blocks = [set() for _ in range(len(treatments) + 1)]
    for y in outcomes:
        stage = max([0] + [i + 1 for i, x in enumerate(treatments) if directed_paths(edges, x, y)])
        blocks[stage].add(y)
    return tuple(blocks)


def separated(nodes: tuple[str, ...], directed: Edges, bidirected: Edges,
              x: str, ys: set[str], given: set[str]) -> bool:
    # Integer IDs avoid collisions between arbitrary observed labels and latent IDs.
    ids = {v: i for i, v in enumerate(nodes)}
    edges = {(ids[a], ids[b]) for a, b in directed}
    for j, (a, b) in enumerate(sorted(bidirected), len(nodes)):
        edges.update(((j, ids[a]), (j, ids[b])))
    ancestral = {ids[v] for v in {x} | ys | given}
    while True:
        larger = ancestral | {a for a, b in edges if b in ancestral}
        if larger == ancestral:
            break
        ancestral = larger
    moral = {v: set() for v in ancestral}
    for child in ancestral:
        parents = {a for a, b in edges if b == child and a in ancestral}
        for a in parents:
            moral[a].add(child)
            moral[child].add(a)
        for a, b in combinations(parents, 2):
            moral[a].add(b)
            moral[b].add(a)
    reached, pending = {ids[x]}, [ids[x]]
    blocked = {ids[v] for v in given}
    while pending:
        for v in moral[pending.pop()] - reached - blocked:
            reached.add(v)
            pending.append(v)
    return reached.isdisjoint(ids[y] for y in ys)


def stage_valid(nodes: tuple[str, ...], directed: Edges, bidirected: Edges,
                x: str, future: set[str], targets: set[str], history: set[str],
                z: set[str], *, printed: bool = False) -> bool:
    cut = frozenset((a, b) for a, b in directed if b not in future)
    hidden = frozenset((a, b) for a, b in bidirected if not ({a, b} & future))
    paths = [p for y in targets for p in directed_paths(cut, x, y)]
    forbidden_edges = directed if printed else cut
    forbidden_paths = [p for y in targets for p in directed_paths(forbidden_edges, x, y)]
    causal = {v for p in forbidden_paths for v in p[1:]}
    if z & desc(nodes, forbidden_edges, causal):
        return False
    first_edges = {(p[0], p[1]) for p in paths}
    return not targets or separated(nodes, cut - first_edges, hidden, x, targets, history | z)


def valid(nodes: tuple[str, ...], directed: Edges, bidirected: Edges,
          treatments: tuple[str, ...], outcomes: set[str],
          blocks: tuple[set[str], ...], *, printed: bool = False) -> bool:
    ys = outcome_blocks(nodes, directed, treatments, outcomes)
    history = set(ys[0])
    seen = set(treatments) | outcomes
    if len(blocks) != len(treatments):
        return False
    for i, (x, z) in enumerate(zip(treatments, blocks, strict=True)):
        future = set(treatments[i + 1:])
        if z & (seen | desc(nodes, directed, future)):
            return False
        seen |= z
        targets = set().union(*ys[i + 1:])
        if not stage_valid(nodes, directed, bidirected, x, future, targets, history, z, printed=printed):
            return False
        history |= {x} | ys[i + 1] | z
    return True


def allocations(nodes: tuple[str, ...], treatments: tuple[str, ...], outcomes: set[str]):
    eligible = tuple(v for v in nodes if v not in set(treatments) | outcomes)
    for assignment in product(range(len(treatments) + 1), repeat=len(eligible)):
        yield tuple({v for v, s in zip(eligible, assignment) if s == i + 1}
                    for i in range(len(treatments)))


def functional(table: tuple[tuple[tuple[int, ...], Fraction], ...],
               nodes: tuple[str, ...], treatments: tuple[str, ...],
               ys: tuple[set[str], ...], zs: tuple[set[str], ...],
               values: dict[str, int]) -> Fraction:
    positions = {v: i for i, v in enumerate(nodes)}
    covariates = tuple(v for v in nodes if any(v in z for z in zs))

    def event(vertices: set[str], environment: dict[str, int]) -> Fraction:
        return sum((prob for bits, prob in table
                    if all(bits[positions[v]] == environment[v] for v in vertices)), Fraction())

    result = Fraction()
    for bits in product((0, 1), repeat=len(covariates)):
        environment = values | dict(zip(covariates, bits))
        history = set(ys[0])
        term = event(history, environment)
        for x, y, z in zip(treatments, ys[1:], zs, strict=True):
            term *= event(history | z, environment) / event(history, environment)
            history |= {x} | z
            term *= event(history | y, environment) / event(history, environment)
            history |= y
        result += term
    return result
