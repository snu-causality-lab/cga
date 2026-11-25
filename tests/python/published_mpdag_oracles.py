"""Independent finite DAG controls; no production path, closure, or planner helper."""
from __future__ import annotations

from fractions import Fraction
from itertools import product

from causal_graph_algorithms.probability import Probability, Product, Ratio, Summation


def joint_distribution(nodes, directed, variant):
    table = {}
    for bits in product((0, 1), repeat=len(nodes)):
        assignment = dict(zip(nodes, bits))
        mass = Fraction(1)
        for i, vertex in enumerate(nodes):
            parents = tuple(v for v in nodes if (v, vertex) in directed)
            code = sum((j + 2) * (assignment[v] + 1) for j, v in enumerate(parents))
            numerator = 1 + ((i + 2) * 7 + code * (variant + 2) + variant * (i + 3)) % 19
            probability = Fraction(numerator, 21)
            mass *= probability if assignment[vertex] else 1 - probability
        table[bits] = mass
    assert sum(table.values()) == 1
    return table


def marginal(nodes, table, assignment):
    return sum((mass for bits, mass in table.items()
                if all(bits[nodes.index(v)] == value for v, value in assignment.items())), Fraction())


def conditional(nodes, table, variables, given, assignment):
    event = {v: assignment[v] for v in (*variables, *given)}
    evidence = {v: assignment[v] for v in given}
    return marginal(nodes, table, event) / marginal(nodes, table, evidence)


def intervention_probability(nodes, table, directed, x_values, y_values):
    total = Fraction()
    for bits in product((0, 1), repeat=len(nodes)):
        assignment = dict(zip(nodes, bits))
        if any(assignment[v] != b for v, b in {**x_values, **y_values}.items()):
            continue
        mass = Fraction(1)
        for vertex in nodes:
            if vertex in x_values:
                continue
            parents = tuple(v for v in nodes if (v, vertex) in directed)
            mass *= conditional(nodes, table, (vertex,), parents, assignment)
        total += mass
    return total


def evaluate(nodes, table, expression, assignment):
    if isinstance(expression, Probability):
        assert not expression.interventions and not expression.conditioned_values
        return conditional(nodes, table, expression.variables, expression.conditioned_on, assignment)
    if isinstance(expression, Product):
        value = Fraction(1)
        for factor in expression.factors:
            value *= evaluate(nodes, table, factor, assignment)
        return value
    if isinstance(expression, Summation):
        return sum((evaluate(nodes, table, expression.body,
                             {**assignment, **dict(zip(expression.variables, bits))})
                    for bits in product((0, 1), repeat=len(expression.variables))), Fraction())
    assert isinstance(expression, Ratio)
    return evaluate(nodes, table, expression.numerator, assignment) / evaluate(nodes, table, expression.denominator, assignment)


def directed_descendants(nodes, directed, sources):
    reached = set(sources)
    for _ in nodes:
        reached.update(b for a, b in directed if a in reached)
    return reached


def sparse_members(raw):
    """Orient only existing undirected edges; compare literal original colliders."""
    nodes = raw.nodes
    fixed, undirected = set(), []
    skeleton = {frozenset((a, b)) for a, _, b, _ in raw.edges}
    for a, ma, b, mb in raw.edges:
        if ma.value == mb.value:
            undirected.append((a, b))
        else:
            fixed.add((a, b) if ma.value == "tail" else (b, a))
    def colliders(edges):
        return {(frozenset((a, c)), b) for a in nodes for b in nodes for c in nodes
                if a != c and (a, b) in edges and (c, b) in edges
                and frozenset((a, c)) not in skeleton}
    expected = colliders(fixed)
    members = []
    for flips in product((0, 1), repeat=len(undirected)):
        edges = fixed | {(a, b) if bit else (b, a) for (a, b), bit in zip(undirected, flips)}
        if colliders(edges) != expected:
            continue
        remaining = set(nodes)
        while remaining:
            sources = {v for v in remaining if not any(b == v and a in remaining for a, b in edges)}
            if not sources:
                break
            remaining.difference_update(sources)
        if not remaining:
            members.append(frozenset(edges))
    return tuple(members)
