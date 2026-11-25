"""Axiom deductions checked against independent finite probability tables."""
from fractions import Fraction
from itertools import combinations, product

import pytest

from causal_graph_algorithms import DAG, EndpointMarkedGraph
from causal_graph_algorithms.conditional_independence import (
    CIStatement, CIDomainError, CIScope, CIResourceLimitError, GraphoidAxioms,
    SeparationCIOracle, CachedCIOracle, derive_ci_closure,
)


def ci(left, right, given=()):
    return CIStatement(left, right, given)


def test_statement_symmetry_sets_validation_and_oracle_cache():
    assert ci("X", ("Y", "Z")) == ci(("Z", "Y"), "X")
    assert hash(ci("X", "Y")) == hash(ci("Y", "X"))
    with pytest.raises(CIDomainError):
        ci("X", "Y", "X")
    with pytest.raises(CIDomainError):
        ci("", "Y")
    oracle = CachedCIOracle(SeparationCIOracle(DAG.from_text("X --> M <-- Y")))
    assert oracle(ci("X", "Y")) is True
    assert oracle(ci("Y", "X")) is True
    assert oracle.calls == 1 and oracle.cache_hits == 1
    assert oracle(ci("X", "Y", "M")) is False
    with pytest.raises(CIDomainError):
        oracle(ci("X", "unknown"))
    with pytest.raises(CIDomainError):
        SeparationCIOracle(EndpointMarkedGraph.from_text("X --- Y"))
    with pytest.raises(CIDomainError):
        CachedCIOracle(lambda _: 0)(ci("X", "Y"))


def test_semigraphoid_rules_and_explicit_optional_assumptions():
    scope = CIScope(("X", "Y", "W", "Z"))
    result = derive_ci_closure(scope, (ci("X", ("Y", "W"), "Z"),), max_statements=100)
    assert ci("X", "Y", "Z") in result.statements
    assert ci("X", "Y", ("Z", "W")) in result.statements
    contracted = derive_ci_closure(scope, (ci("X", "Y", "Z"),
        ci("X", "W", ("Z", "Y"))), max_statements=100)
    assert ci("X", ("Y", "W"), "Z") in contracted.statements
    antecedents = (ci("X", "Y", ("Z", "W")), ci("X", "W", ("Z", "Y")))
    assert ci("X", ("Y", "W"), "Z") not in derive_ci_closure(
        scope, antecedents, max_statements=100).statements
    intersected = derive_ci_closure(scope, antecedents,
        axioms=GraphoidAxioms(intersection=True), max_statements=100)
    assert ci("X", ("Y", "W"), "Z") in intersected.statements
    antecedents = (ci("X", "Y", "Z"), ci("X", "W", "Z"))
    assert ci("X", ("Y", "W"), "Z") not in derive_ci_closure(
        scope, antecedents, max_statements=100).statements
    composed = derive_ci_closure(scope, antecedents,
        axioms=GraphoidAxioms(composition=True), max_statements=100)
    assert ci("X", ("Y", "W"), "Z") in composed.statements
    assert composed.axioms.composition


def test_resource_limit_unknown_is_not_dependence_and_trivial_ci():
    scope = CIScope(("X", "Y", "Z"))
    with pytest.raises(CIResourceLimitError):
        derive_ci_closure(scope, (ci("X", ("Y", "Z")),), max_statements=1)
    result = derive_ci_closure(scope, (), max_statements=1)
    assert result.contains(ci((), "X"))
    assert not result.contains(ci("X", "Y"))  # underivable, not dependent
    with pytest.raises(CIDomainError):
        derive_ci_closure(scope, (ci("X", "missing"),), max_statements=100)
    with pytest.raises(CIResourceLimitError):
        derive_ci_closure(scope, (ci("X", "Y"),), max_statements=100, max_rule_checks=1)


def all_statements(nodes):
    for choices in product(range(4), repeat=len(nodes)):
        sets = [tuple(n for n, c in zip(nodes, choices) if c == k) for k in (1, 2, 3)]
        if sets[0] and sets[1]:
            yield ci(*sets)


def probability_ci(nodes, table, statement):
    # Literal marginal sums and P(a,b,c)P(c)=P(a,c)P(b,c).
    positions = {n: i for i, n in enumerate(nodes)}
    def marginal(fixed):
        return sum((p for values, p in table.items()
                    if all(values[positions[n]] == v for n, v in fixed.items())), Fraction())
    a, b, c = statement.left, statement.right, statement.given
    for values in product((0, 1), repeat=len(a | b | c)):
        fixed = dict(zip(sorted(a | b | c), values))
        ac = {n: v for n, v in fixed.items() if n in a | c}
        bc = {n: v for n, v in fixed.items() if n in b | c}
        cc = {n: v for n, v in fixed.items() if n in c}
        if marginal(fixed) * marginal(cc) != marginal(ac) * marginal(bc):
            return False
    return True


def test_semigraphoid_soundness_on_all_three_binary_variable_supports():
    nodes = ("A", "B", "C")
    assignments = tuple(product((0, 1), repeat=3))
    statements = frozenset(all_statements(nodes))
    for size in range(1, len(assignments) + 1):
        for support in combinations(assignments, size):
            table = {row: Fraction(1, size) for row in support}
            valid = frozenset(s for s in statements if probability_ci(nodes, table, s))
            result = derive_ci_closure(CIScope(nodes), valid, max_statements=100)
            assert result.statements <= valid
