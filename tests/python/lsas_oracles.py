"""LSAS test evidence from a latent-DAG moralization definition.

No production graph, separation, blanket or selection code is imported here.
Each bidirected edge gets its own independent latent parent. These routines
describe acyclic directed/bidirected graphs; they do not model selection bias.
"""
from __future__ import annotations

from itertools import combinations, product


def separated(nodes, directed, bidirected, left, right, given):
    parents = {node: set() for node in nodes}
    for source, target in directed:
        parents[target].add(source)
    for index, (first, second) in enumerate(bidirected):
        latent = ("latent", index)
        parents[latent] = set()
        parents[first].add(latent)
        parents[second].add(latent)
    ancestral = {left, right, *given}
    pending = list(ancestral)
    while pending:
        for parent in parents[pending.pop()]:
            if parent not in ancestral:
                ancestral.add(parent)
                pending.append(parent)
    neighbors = {node: set() for node in ancestral}
    for child in ancestral:
        current = parents[child] & ancestral
        for parent in current:
            neighbors[parent].add(child)
            neighbors[child].add(parent)
        for first, second in combinations(current, 2):
            neighbors[first].add(second)
            neighbors[second].add(first)
    reached = {left}
    pending = [left]
    while pending:
        for neighbor in neighbors[pending.pop()] - set(given):
            if neighbor == right:
                return False
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return True


def ci_oracle(nodes, directed=(), bidirected=()):
    def independent(left, right, given):
        assert left != right
        assert left not in given and right not in given
        assert {left, right, *given} <= set(nodes)
        assert isinstance(given, frozenset)
        return separated(nodes, directed, bidirected, left, right, given)
    return independent


def has_directed_path(directed, source, target):
    pending = [source]
    reached = {source}
    while pending:
        current = pending.pop()
        for parent, child in directed:
            if parent == current:
                if child == target:
                    return True
                if child not in reached:
                    reached.add(child)
                    pending.append(child)
    return False


def adjustment_valid(nodes, directed, bidirected, treatment, outcome, adjustment):
    # All other observed vertices precede X and Y in the enumerated universe.
    assert all(not has_directed_path(directed, treatment, z) for z in adjustment)
    backdoor = tuple((a, b) for a, b in directed if a != treatment)
    return separated(nodes, backdoor, bidirected, treatment, outcome, adjustment)


def pretreatment_graphs(nodes):
    """Every bow-capable ADMG in one fixed topological order, with X,Y last."""
    pairs = tuple(combinations(nodes, 2))
    for states in product(range(4), repeat=len(pairs)):
        directed = tuple(pair for pair, state in zip(pairs, states) if state & 1)
        bidirected = tuple(pair for pair, state in zip(pairs, states) if state & 2)
        yield directed, bidirected


def printed_rules_apply(nodes, directed, bidirected, treatment="X", outcome="Y"):
    """Literal paper R1/R2 witness quantifier, for the tiny source fixtures."""
    ci = ci_oracle(nodes, directed, bidirected)
    mb_x = [s for s in nodes if s != treatment
            and not ci(treatment, s, frozenset(nodes) - {treatment, s})]
    mb_y = [z for z in nodes if z != outcome
            and not ci(outcome, z, frozenset(nodes) - {outcome, z})]
    for s in mb_x:
        if s == outcome:
            continue
        candidates = [z for z in mb_y if z != treatment]
        for size in range(len(candidates) + 1):
            for subset in combinations(candidates, size):
                z = frozenset(subset)
                # Printed R2(i) never queries S; S in Z is harmless here.
                if ci(treatment, outcome, z):
                    return True
                if s in z:
                    continue
                if not ci(s, outcome, z) and ci(s, outcome, z | {treatment}):
                    return True
                if not ci(s, treatment, z) and ci(s, outcome, z):
                    return True
    return False
