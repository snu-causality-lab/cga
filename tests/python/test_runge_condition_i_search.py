"""Counterexamples and operation-count contracts for Condition I optimization."""
from itertools import combinations, permutations

import pytest

from causal_graph_algorithms import AcyclicDirectedBidirectedGraph, DirectedBidirectedGraph
from causal_graph_algorithms import runge_adjustment as runge
from tests.python import runge_oracles as oracle


def graph(nodes, directed, bidirected=()):
    return AcyclicDirectedBidirectedGraph(DirectedBidirectedGraph(
        nodes=nodes, directed=directed, bidirected=bidirected))


def cycle_graph():
    return graph(("X", "Y", "N", "A", "B", "T"),
                 (("X", "Y"), ("X", "N"), ("A", "B")),
                 (("N", "A"), ("N", "B"), ("A", "B"),
                  ("A", "T"), ("B", "T"), ("T", "Y")))


def test_required_check_caches_closures_not_just_required_sets(monkeypatch):
    view = cycle_graph()
    query = runge._query(view, "X", "Y", ())
    check = runge._RequiredAdjustmentCheck(query)
    calls = []
    original = runge.m_separated

    def counted(g, x, y, z):
        calls.append(frozenset(z))
        return original(g, x, y, z)

    monkeypatch.setattr(runge, "m_separated", counted)
    for required in ({"N", "B"}, {"N", "A", "B"}, {"N", "B"}):
        assert check.feasible(frozenset(required)) is True
    assert len(calls) == 1
    for required in ({"N", "B", "T"}, {"N", "A", "B", "T"}):
        assert check.feasible(frozenset(required)) is False
    assert len(calls) == 2
    # Another graph must not inherit this graph's cache.
    other = graph(view.topology.nodes, (("X", "Y"),), (("X", "Y"),))
    assert runge._RequiredAdjustmentCheck(runge._query(other, "X", "Y", ())).feasible(
        frozenset({"N", "B"})) is False
    assert len(calls) == 3


@pytest.mark.parametrize("text,fixed,expected", [
    ("X -> Y; X -> N; N <-> Y", (), True),
    ("X -> Y; X -> A -> N; N <-> Y", (), False),
    ("X -> M -> Y; X -> A -> N; N <-> M", (), False),
    ("X -> Y; X <-> S <-> N <-> Y", ("S",), True),
    ("X -> Y; X -> N; N <-> C <-> Y", (), True),
    ("X -> Y; X -> A -> N; A <-> N; N <-> Y", (), True),
])
def test_no_auxiliary_oset_reconstruction_and_correct_required_vertices(text, fixed, expected, monkeypatch):
    view = AcyclicDirectedBidirectedGraph.from_text(text)
    assert oracle.criteria(view.topology, "X", "Y", set(fixed))[0] is expected
    original = runge._construct
    calls = []

    def counted(query, conditions):
        calls.append(conditions)
        return original(query, conditions)

    monkeypatch.setattr(runge, "_construct", counted)
    result = runge.runge_optimal_adjustment(view, "X", "Y", conditioned_on=fixed)
    assert result.criterion_i is expected
    assert calls == [frozenset(fixed)]  # Only the original requested O-set.


def test_preprocessing_discards_cyclic_histories_before_state_search():
    view = cycle_graph()
    expected = oracle.criteria(view.topology, "X", "Y", set())
    assert expected == (True, True)
    # The earlier literal DFS visited nine prefixes. Required-set preprocessing
    # now separates the source and terminal, before any budgeted walk state.
    result = runge.runge_optimal_adjustment(view, "X", "Y", max_collider_states=0)
    assert (result.criterion_i, result.criterion_ii) == expected


def test_pruning_cuts_factorially_many_dead_completions(monkeypatch):
    interior = tuple(f"C{i}" for i in range(6))
    view = graph(("X", "Y", "N", *interior), (("X", "Y"),),
                 (("X", "N"), *(("N", c) for c in interior),
                  *(("Y", c) for c in interior), *combinations(interior, 2)))
    # Each path forces N and a C: X <-> N <-> C <-> Y is then open.
    paths = [("N", *middle, "Y") for k in range(1, 7) for middle in permutations(interior, k)]
    assert len(paths) == 1956
    for c in interior:
        # Exhaustive supersets, using the independent latent-DAG oracle.
        remaining = tuple(v for v in interior if v != c)
        assert not any(oracle.latent_dag_adjustment_valid(
            view.topology, "X", "Y", {"N", c, *extra})
            for k in range(6) for extra in combinations(remaining, k))
    original = runge._RequiredAdjustmentCheck.feasible
    calls = []

    def counted(self, required):
        calls.append(required)
        return original(self, required)

    monkeypatch.setattr(runge._RequiredAdjustmentCheck, "feasible", counted)
    result = runge.runge_optimal_adjustment(
        view, "X", "Y", max_collider_paths=0, max_collider_states=0)
    assert result.criterion_i is True
    assert len(calls) == 6  # Six infeasible vertices are excluded in preprocessing.


@pytest.mark.parametrize("limit", [True, False, -1, 1.5, "1"])
def test_bad_partial_search_limits_fail_before_graph_use(limit):
    with pytest.raises(runge.RungeAdjustmentDomainError, match="max_collider_states"):
        runge.runge_optimal_adjustment(object(), "X", "Y", max_collider_states=limit)


def test_zero_limits_vacuity_exhaustion_and_exact_boundaries():
    empty_n = AcyclicDirectedBidirectedGraph.from_text("X -> Y")
    assert runge.runge_optimal_adjustment(
        empty_n, "X", "Y", max_collider_paths=0, max_collider_states=0).criterion_i is True
    witness = AcyclicDirectedBidirectedGraph.from_text("X -> Y; X -> A -> N; N <-> Y")
    for options in ({"max_collider_states": 0}, {"max_collider_paths": 0}):
        with pytest.raises(runge.RungeAdjustmentResourceLimitError):
            runge.runge_optimal_adjustment(witness, "X", "Y", **options)
    assert runge.runge_optimal_adjustment(
        witness, "X", "Y", max_collider_paths=1, max_collider_states=1).criterion_i is False
    impossible_n = AcyclicDirectedBidirectedGraph.from_text("X -> Y; X -> N; N <-> Y")
    assert runge.runge_optimal_adjustment(
        impossible_n, "X", "Y", max_collider_paths=0, max_collider_states=1).criterion_i is True


def test_preprocessing_can_reject_all_sources_without_spending_state_budget():
    view = AcyclicDirectedBidirectedGraph.from_text("X -> Y; X -> N1; X -> N2; N1 <-> Y; N2 <-> Y")
    assert runge.runge_optimal_adjustment(view, "X", "Y", max_collider_states=0).criterion_i is True
