"""Independent small-graph definitions and finite SCMs; no production traversals.

Graph tests enumerate literal simple paths. Numeric truth deletes intervention
factors before summing explicit independent latent roots. Formula evaluation only
receives declared input tables, never the interventional truth table.
"""
from itertools import product

import numpy as np

from causal_graph_algorithms import Probability, Product, Ratio, Summation


def closure(seeds, directed, vertices):
    reached = set(seeds)
    while True:
        enlarged = reached | {b for a, b in directed if a in reached and b in vertices}
        if enlarged == reached:
            return frozenset(reached)
        reached = enlarged


def paths(raw, starts, ends, directed=None):
    directed = raw.directed if directed is None else directed
    adjacent = {v: [] for v in raw.nodes}
    for a, b in directed:
        adjacent[a].append((b, False, True))
        adjacent[b].append((a, True, False))
    for a, b in raw.bidirected:
        adjacent[a].append((b, True, True))
        adjacent[b].append((a, True, True))
    for start in starts:
        stack = [((start,), ())]
        while stack:
            vertices, marks = stack.pop()
            for v, at_old, at_new in adjacent[vertices[-1]]:
                if v in vertices:
                    continue
                next_vertices, next_marks = vertices + (v,), marks + ((at_old, at_new),)
                if v in ends:
                    yield next_vertices, next_marks
                else:
                    stack.append((next_vertices, next_marks))


def active_path(vertices, marks, given, ancestor_given):
    for i, v in enumerate(vertices[1:-1], 1):
        collider = marks[i - 1][1] and marks[i][0]
        if (collider and v not in ancestor_given) or (not collider and v in given):
            return False
    return True


def separated(raw, left, right, given, directed=None):
    directed = raw.directed if directed is None else directed
    ancestor_given = closure(given, tuple((b, a) for a, b in directed), set(raw.nodes))
    return not any(active_path(v, m, given, ancestor_given)
                   for v, m in paths(raw, left, right, directed))


def pair_definition(raw, context, x, y, z, zt):
    """2018 Definition 9, with Dpcp computed from literal proper causal paths."""
    if x & y or z & (x | y | context) or not zt <= z:
        return False
    first_edges, causal_vertices = set(), set()
    for vertices, marks in paths(raw, x, y):
        if set(vertices[1:]) & x or not all(not a and b for a, b in marks):
            continue
        first_edges.add(vertices[:2])
        causal_vertices.update(vertices[1:])
    incoming_cut = tuple((a, b) for a, b in raw.directed if b not in x)
    forbidden = closure(causal_vertices, incoming_cut, set(raw.nodes))
    pbd = tuple(e for e in raw.directed if e not in first_edges)
    return (not z & forbidden and separated(raw, x, y, z | context, pbd)
            and (not context or separated(raw, y, context, zt, pbd)))


def subsets(items):
    ordered = tuple(items)
    for flags in product((False, True), repeat=len(ordered)):
        yield frozenset(v for v, flag in zip(ordered, flags) if flag)


def all_pairs(raw, context, x, y, population=(), required=(), allowed=None):
    candidates = set(raw.nodes) - set(context) - x - y
    if allowed is not None:
        candidates &= set(allowed)
    result = set()
    for z in subsets(v for v in raw.nodes if v in candidates):
        if not set(required) <= z:
            continue
        for zt in subsets(v for v in raw.nodes if v in z and v in population):
            if pair_definition(raw, frozenset(context), x, y, z, zt):
                result.add((z, zt))
    return result


class BinarySCM:
    def __init__(self, raw, rng):
        self.nodes = raw.nodes
        n = len(self.nodes)
        edges = tuple(sorted(raw.bidirected, key=lambda e: tuple(self.nodes.index(v) for v in e)))
        self.bits = np.array(list(product((0, 1), repeat=n + len(edges))), dtype=np.int8)
        self.index = self.bits[:, :n] @ (2 ** np.arange(n - 1, -1, -1))
        self.roots = np.ones(len(self.bits))
        self.factors = []
        for j in range(len(edges)):
            p = rng.integers(1, 10) / 10
            self.roots *= np.where(self.bits[:, n + j], p, 1 - p)
        for i, vertex in enumerate(self.nodes):
            parents = sorted(self.nodes.index(a) for a, b in raw.directed if b == vertex)
            parents += [n + j for j, edge in enumerate(edges) if vertex in edge]
            table = rng.integers(1, 10, size=2 ** len(parents)) / 10
            indices = self.bits[:, parents] @ (2 ** np.arange(len(parents) - 1, -1, -1))
            p1 = table[indices] if parents else np.full(len(self.bits), table[0])
            self.factors.append(np.where(self.bits[:, i], p1, 1 - p1))
        self.joint = self.intervene({})

    def intervene(self, assignments):
        weights = self.roots.copy()
        for i, vertex in enumerate(self.nodes):
            weights *= (self.bits[:, i] == assignments[vertex]
                        if vertex in assignments else self.factors[i])
        result = np.bincount(self.index, weights=weights,
                             minlength=2 ** len(self.nodes)).reshape((2,) * len(self.nodes))
        assert abs(result.sum() - 1) < 2e-12
        return result

    def source_table(self, data):
        fixed = dict(data.conditioned_values)
        slicing = tuple(fixed.get(v, slice(None)) for v in self.nodes)
        table = self.joint[slicing]
        remaining = tuple(v for v in self.nodes if v not in fixed)
        table = table.sum(axis=tuple(i for i, v in enumerate(remaining)
                                    if v not in data.variables), keepdims=False)
        order = tuple(v for v in remaining if v in data.variables)
        if len(order) > 1:
            table = table.transpose(tuple(order.index(v) for v in data.variables))
        return table / table.sum()

    def target(self, data, x, y):
        order = data.variables
        result = np.empty((2,) * len(order))
        target_by_x = {}
        for values in product((0, 1), repeat=len(x)):
            joint = self.intervene(dict(zip(x, values)))
            marginal = joint.sum(axis=tuple(i for i, v in enumerate(self.nodes) if v not in y))
            y_order = tuple(v for v in self.nodes if v in y)
            target_by_x[values] = (marginal, y_order)
        for values in product((0, 1), repeat=len(order)):
            assignment = dict(zip(order, values))
            marginal, y_order = target_by_x[tuple(assignment[v] for v in x)]
            result[values] = marginal[tuple(assignment[v] for v in y_order)]
        return result


def evaluate(formula, data, table, population_data=None, population_table=None):
    """Only declared conditional data and the explicitly declared population table."""
    order = data.variables
    shape = (1,) * len(order)

    def lifted_marginal(source, source_order, retained):
        axes = tuple(i for i, v in enumerate(source_order) if v not in retained)
        reduced = source.sum(axis=axes)
        retained_order = tuple(v for v in source_order if v in retained)
        desired = tuple(v for v in order if v in retained)
        if len(desired) > 1:
            reduced = reduced.transpose(tuple(retained_order.index(v) for v in desired))
        return reduced.reshape(tuple(2 if v in retained else 1 for v in order))

    cache = {}

    def visit(expression):
        if id(expression) in cache:
            return cache[id(expression)]
        if isinstance(expression, Probability):
            if expression.interventions:
                raise AssertionError("an unavailable interventional source leaf")
            fixed = dict(expression.conditioned_values)
            unfixed = set(expression.conditioned_on) - set(fixed)
            labels = set(expression.variables) | unfixed
            if fixed == dict(data.conditioned_values) and set(fixed) == set(data.conditioned_on):
                source, source_order = table, data.variables
            elif (not fixed and population_data is not None and population_table is not None
                  and labels <= set(population_data.variables)):
                source, source_order = population_table, population_data.variables
            else:
                raise AssertionError("the formula uses a distribution absent from its declared inputs")
            if not labels <= set(source_order):
                raise AssertionError("source leaf uses unavailable variables")
            value = (lifted_marginal(source, source_order, labels)
                     / lifted_marginal(source, source_order, unfixed))
        elif isinstance(expression, Product):
            value = np.ones(shape)
            for factor in expression.factors:
                value = value * visit(factor)
        elif isinstance(expression, Summation):
            value = visit(expression.body).sum(
                axis=tuple(order.index(v) for v in expression.variables), keepdims=True)
        elif isinstance(expression, Ratio):
            value = visit(expression.numerator) / visit(expression.denominator)
        else:
            raise AssertionError("unknown formula syntax")
        cache[id(expression)] = value
        return value

    return visit(formula)
