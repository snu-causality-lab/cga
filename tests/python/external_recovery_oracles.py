"""Exact finite SCM truth and a source-only evaluator, independent of IDSB.

No production graph traversal or recovery routine is used. Each bidirected edge
gets its own independent binary root. All local probabilities are positive
rationals; intervention truth omits the corresponding structural factors.
"""
from fractions import Fraction
from itertools import product

from causal_graph_algorithms import Probability, Product, Ratio, Summation


class RationalSCM:
    def __init__(self, raw, seed=0):
        self.nodes = raw.nodes
        self.directed = tuple(raw.directed)
        self.edges = tuple(sorted(tuple(sorted(e)) for e in raw.bidirected))
        self.seed = seed
        self._joint_cache = {}

    def joint(self, interventions=None):
        interventions = interventions or {}
        key = tuple(sorted(interventions.items()))
        if key in self._joint_cache:
            return self._joint_cache[key]
        result = {}
        n = len(self.nodes)
        for bits in product((0, 1), repeat=n + len(self.edges)):
            a = dict(zip(self.nodes, bits[:n]))
            if any(a[v] != value for v, value in interventions.items()):
                continue
            weight = Fraction(1)
            for j, _ in enumerate(self.edges):
                p = Fraction(1 + (3 * j + self.seed) % 5, 7)
                weight *= p if bits[n + j] else 1 - p
            for i, v in enumerate(self.nodes):
                if v in interventions:
                    continue
                parents = [a[u] for u, child in sorted(self.directed) if child == v]
                parents += [bits[n + j] for j, edge in enumerate(self.edges) if v in edge]
                code = sum((1 << j) * value for j, value in enumerate(parents))
                p = Fraction(1 + (code * (2 * i + 1) + i + self.seed) % 7, 9)
                weight *= p if a[v] else 1 - p
            row = bits[:n]
            result[row] = result.get(row, Fraction()) + weight
        assert sum(result.values()) == 1
        self._joint_cache[key] = result
        return result

    def source(self, source):
        fixed = dict(source.conditioned_values)
        table = {}
        mass = Fraction()
        for bits, weight in self.joint().items():
            row = dict(zip(self.nodes, bits))
            if any(row[v] != value for v, value in fixed.items()):
                continue
            key = tuple(row[v] for v in source.variables)
            table[key] = table.get(key, Fraction()) + weight
            mass += weight
        return {key: value / mass for key, value in table.items()}

    def target(self, outcomes, interventions, assignment):
        return sum(weight for bits, weight in self.joint(
            {v: assignment[v] for v in interventions}).items()
            if all(bits[self.nodes.index(v)] == assignment[v] for v in outcomes))


def evaluate_sources(expression, assignment, data, selected, population=None, margin=None):
    """Every leaf must use one declared source; no access to SCM or causal truth."""
    cache = {}

    def visit(expr, a):
        key = (id(expr), tuple(sorted(a.items())))
        if key in cache:
            return cache[key]
        if isinstance(expr, Probability):
            if expr.interventions:
                raise AssertionError("undeclared interventional input")
            fixed = dict(expr.conditioned_values)
            given = set(expr.conditioned_on) - fixed.keys()
            labels = set(expr.variables) | given
            if fixed == dict(data.conditioned_values):
                source, table = data, selected
            elif not fixed and population is not None:
                source, table = population, margin
            else:
                raise AssertionError("undeclared source event")
            if not labels <= set(source.variables):
                raise AssertionError("unavailable source variables")
            numerator = denominator = Fraction()
            for row, weight in table.items():
                values = dict(zip(source.variables, row))
                if all(values[v] == a[v] for v in given):
                    denominator += weight
                    if all(values[v] == a[v] for v in expr.variables):
                        numerator += weight
            value = numerator / denominator
        elif isinstance(expr, Product):
            value = Fraction(1)
            for factor in expr.factors:
                value *= visit(factor, a)
        elif isinstance(expr, Ratio):
            value = visit(expr.numerator, a) / visit(expr.denominator, a)
        elif isinstance(expr, Summation):
            value = sum(visit(expr.body, a | dict(zip(expr.variables, bits)))
                        for bits in product((0, 1), repeat=len(expr.variables)))
        else:
            raise AssertionError("unknown probability expression")
        cache[key] = value
        return value

    return visit(expression, assignment)


def assert_exact_formula(g, data, result, population, seed=0):
    scm = RationalSCM(g.topology, seed)
    selected = scm.source(data)
    margin = None if population is None else scm.source(population)
    assert set(result.outcomes) <= result.formula.free_variables <= set(
        result.outcomes + result.interventions)
    for bits in product((0, 1), repeat=len(result.outcomes + result.interventions)):
        assignment = dict(zip(result.outcomes + result.interventions, bits))
        actual = evaluate_sources(result.formula, assignment, data, selected, population, margin)
        assert actual == scm.target(result.outcomes, result.interventions, assignment)
