"""Required-set extension checks against independent, exhaustive validity oracles.

Expected answers enumerate every observed candidate set in the original graph.
They do not use the production proper-backdoor graph, ancestor sets, separation,
or O-set construction. These are literal graphical extension tests; internal
requirements may contain treatment descendants outside the forbidden set.
"""

from __future__ import annotations

from itertools import combinations, product
import random

import pytest

from causal_graph_algorithms import runge_adjustment
from causal_graph_algorithms.directed_bidirected import (
    AcyclicDirectedBidirectedGraph,
    DirectedBidirectedGraph,
)
from tests.python import runge_oracles as oracle


def _graph(nodes, directed=(), bidirected=()):
    return AcyclicDirectedBidirectedGraph(DirectedBidirectedGraph(
        nodes=nodes, directed=directed, bidirected=bidirected,
    ))


def _subsets(nodes):
    ordered = tuple(nodes)
    return tuple(
        frozenset(subset)
        for size in range(len(ordered) + 1)
        for subset in combinations(ordered, size)
    )


def _valid_totals(view, *, compare_paths=False):
    """Compute truth once per graph, including candidates containing forbidden nodes."""
    topology = view.topology
    candidates = _subsets(node for node in topology.nodes if node not in {"X", "Y"})
    forbidden = oracle.forbidden_nodes(topology, "X", "Y")
    valid = []
    for total in candidates:
        expected = oracle.latent_dag_adjustment_valid(topology, "X", "Y", set(total))
        if compare_paths:
            assert expected == (
                not (total & forbidden)
                and not oracle.open_path_exists(topology, "X", "Y", set(total), noncausal=True)
            ), (topology, total)
        if expected:
            valid.append(total)
    return tuple(valid)


def _check_all_requirements(view, *, compare_paths=False):
    topology = view.topology
    valid = _valid_totals(view, compare_paths=compare_paths)
    forbidden = oracle.forbidden_nodes(topology, "X", "Y")
    permitted = tuple(node for node in topology.nodes if node not in forbidden)
    fixed_candidates = tuple(
        node for node in permitted if node not in oracle.closure(topology, {"X"})
    )
    calls = 0
    for fixed in _subsets(fixed_candidates):
        query = runge_adjustment._query(view, "X", "Y", fixed)
        check = runge_adjustment._RequiredAdjustmentCheck(query)
        # Reversing the second pass exposes order-dependent cached verdicts.
        required_sets = _subsets(permitted)
        for required in (*required_sets, *reversed(required_sets)):
            expected = any((fixed | required) <= total for total in valid)
            assert check.feasible(required) is expected, (topology, fixed, required, valid)
            calls += 1
    return calls


def test_required_collider_can_be_invalid_alone_but_repairable():
    view = _graph(
        ("X", "Y", "U", "V", "C"),
        (("X", "Y"), ("U", "X"), ("U", "C"), ("V", "C"), ("V", "Y")),
    )
    valid = _valid_totals(view, compare_paths=True)
    assert frozenset({"C"}) not in valid
    assert frozenset({"C", "U"}) in valid
    assert frozenset({"C", "V"}) in valid
    query = runge_adjustment._query(view, "X", "Y", ())
    assert runge_adjustment._RequiredAdjustmentCheck(query).feasible(frozenset({"C"})) is True


def test_required_latent_collider_is_not_repairable():
    view = _graph(("X", "Y", "C"), (("X", "Y"),), (("X", "C"), ("C", "Y")))
    assert _valid_totals(view, compare_paths=True) == (frozenset(),)
    query = runge_adjustment._query(view, "X", "Y", ())
    check = runge_adjustment._RequiredAdjustmentCheck(query)
    assert check.feasible(frozenset()) is True
    assert check.feasible(frozenset({"C"})) is False
    assert check.feasible(frozenset()) is True


@pytest.mark.parametrize("restricted_first", [False, True])
def test_closure_cache_agrees_with_fresh_ancestral_induced_queries(restricted_first):
    view = AcyclicDirectedBidirectedGraph.from_text(
        "h -> X -> M -> Y; h -> Y; h <-> Y; "
        "X <-> C <-> Y; C -> N; N <-> Y; Y -> junk")
    valid = _valid_totals(view, compare_paths=True)
    query = runge_adjustment._query(view, "X", "Y", {"h"})
    prepared = runge_adjustment._prepare_collider_source(
        query, runge_adjustment._construct(query, query.fixed), "N",
        runge_adjustment._RequiredAdjustmentCheck(query))
    full = query.backdoor.topology
    small = prepared.oracle_graph.topology
    kept = set(small.nodes)
    # Check the producer's restriction, independently of its ancestry helper.
    assert kept < set(full.nodes) and "junk" not in kept
    assert oracle.closure(full, kept, reverse=True) == kept
    assert {"X", "M", "Y"} <= kept
    assert small.directed == frozenset(e for e in full.directed if set(e) <= kept)
    assert small.bidirected == frozenset(e for e in full.bidirected if set(e) <= kept)

    required_sets = (frozenset(), frozenset({"N"}))
    expected = {
        required: any(query.fixed | required <= total for total in valid)
        for required in required_sets
    }
    assert tuple(expected.values()) == (True, False)
    shared = runge_adjustment._RequiredAdjustmentCheck(query)
    candidates = {required: shared.closure(required) for required in required_sets}
    query_graphs = (None, prepared.oracle_graph)
    if restricted_first:
        query_graphs = tuple(reversed(query_graphs))
    for required, candidate in candidates.items():
        assert candidate | {query.treatment, query.outcome} <= kept
        # Fresh checks ensure a cached full-graph verdict cannot hide a bad
        # induced-graph query (or vice versa).
        for oracle_graph in query_graphs:
            fresh = runge_adjustment._RequiredAdjustmentCheck(query)
            assert fresh.feasible_closure(candidate, oracle_graph) is expected[required]
        for oracle_graph in query_graphs:
            assert shared.feasible_closure(candidate, oracle_graph) is expected[required]
    assert set(shared._cache) == set(candidates.values())


@pytest.mark.parametrize("required", [frozenset(), frozenset({"Z"})])
def test_direct_causal_bow_has_no_repair(required):
    view = _graph(("X", "Y", "Z"), (("X", "Y"),), (("X", "Y"),))
    assert _valid_totals(view, compare_paths=True) == ()
    query = runge_adjustment._query(view, "X", "Y", ())
    assert runge_adjustment._RequiredAdjustmentCheck(query).feasible(required) is False


def test_fixed_conditions_are_automatically_required():
    view = _graph(("X", "Y", "S", "Z"), (("X", "Y"),), (("X", "S"), ("S", "Y")))
    valid = _valid_totals(view, compare_paths=True)
    assert valid and all("S" not in total for total in valid)
    query = runge_adjustment._query(view, "X", "Y", {"S"})
    check = runge_adjustment._RequiredAdjustmentCheck(query)
    assert check.feasible(frozenset()) is False
    assert check.feasible(frozenset({"Z"})) is False
    assert check.feasible(frozenset({"S"})) is False


def test_internal_posttreatment_requirement_can_be_repaired():
    view = _graph(
        ("X", "Y", "D", "U"),
        (("X", "Y"), ("X", "D"), ("U", "D"), ("U", "Y")),
    )
    valid = _valid_totals(view, compare_paths=True)
    assert frozenset({"D"}) not in valid
    assert frozenset({"D", "U"}) in valid
    query = runge_adjustment._query(view, "X", "Y", ())
    check = runge_adjustment._RequiredAdjustmentCheck(query)
    assert check.feasible(frozenset({"D"})) is True
    # The internal extension domain must not widen the caller's fixed-S domain.
    with pytest.raises(runge_adjustment.RungeAdjustmentDomainError, match="descendants of treatment"):
        runge_adjustment._query(view, "X", "Y", {"D"})


def test_unrelated_outgoing_treatment_edge_must_remain_for_extension_check():
    view = _graph(("X", "Y", "D"), (("X", "Y"), ("X", "D")), (("D", "Y"),))
    assert _valid_totals(view, compare_paths=True) == (frozenset(),)
    query = runge_adjustment._query(view, "X", "Y", ())
    check = runge_adjustment._RequiredAdjustmentCheck(query)
    assert check.feasible(frozenset()) is True
    assert check.feasible(frozenset({"D"})) is False


@pytest.mark.parametrize("forbidden", ("X", "M", "Y", "D"))
def test_forbidden_requirement_is_rejected_even_after_cached_success(forbidden):
    view = _graph(
        ("X", "M", "Y", "D", "A"),
        (("X", "M"), ("M", "Y"), ("M", "D")),
    )
    assert _valid_totals(view, compare_paths=True) == (frozenset(), frozenset({"A"}))
    query = runge_adjustment._query(view, "X", "Y", ())
    check = runge_adjustment._RequiredAdjustmentCheck(query)
    assert check.feasible(frozenset({"A"})) is True
    assert check.feasible(frozenset({"A", forbidden})) is False
    assert check.feasible(frozenset({"A"})) is True


def test_causal_path_is_required_in_original_graph_not_proper_backdoor_graph():
    view = _graph(("U", "X", "M", "Y"), (("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")))
    assert _valid_totals(view, compare_paths=True) == (frozenset({"U"}),)
    query = runge_adjustment._query(view, "X", "Y", ())
    assert runge_adjustment._RequiredAdjustmentCheck(query).feasible(frozenset()) is True
    without_causal_path = _graph(("U", "X", "M", "Y"), (("U", "X"), ("U", "Y"), ("M", "Y")))
    with pytest.raises(runge_adjustment.RungeAdjustmentDomainError, match="directed causal path"):
        runge_adjustment._query(without_causal_path, "X", "Y", ())


@pytest.mark.parametrize("nodes", [
    ("X", "Y"),
    ("X", "Y", "A"), ("X", "A", "Y"), ("A", "X", "Y"),
    ("X", "Y", "A", "B"), ("X", "A", "Y", "B"), ("X", "A", "B", "Y"),
    ("A", "X", "Y", "B"), ("A", "X", "B", "Y"), ("A", "B", "X", "Y"),
])
def test_all_ordered_admgs_through_four_vertices(nodes):
    """All topologies up to relabeling of non-query vertices, including bows.

    Every topological placement of X before Y is present; optional vertices are
    interchangeable labels A and B. A graph can appear in multiple orders.
    """
    pairs = tuple(combinations(nodes, 2))
    graphs = calls = 0
    for states in product(range(4), repeat=len(pairs)):
        view = _graph(
            nodes,
            (edge for edge, state in zip(pairs, states) if state & 1),
            (edge for edge, state in zip(pairs, states) if state & 2),
        )
        if "Y" not in oracle.closure(view.topology, {"X"}):
            continue
        calls += _check_all_requirements(view, compare_paths=len(nodes) <= 3)
        graphs += 1
    assert graphs > 0 and calls >= 2 * graphs


def test_seeded_five_vertex_admgs_all_permitted_required_and_fixed_sets():
    rng = random.Random(21021032413)
    nodes = ("A", "X", "B", "C", "Y")
    pairs = tuple(combinations(nodes, 2))
    calls = 0
    for _ in range(100):
        directed = {edge for edge in pairs if rng.random() < .35}
        directed.add(("X", "Y"))
        bidirected = {edge for edge in pairs if rng.random() < .3}
        view = _graph(nodes, directed, bidirected)
        calls += _check_all_requirements(view, compare_paths=True)
    assert calls >= 200


def _check_auxiliary_oset_independently(view, fixed):
    """Do not define auxiliary validity by extension existence (the claim tested)."""
    g = view.topology
    totals = _valid_totals(view)
    if not any(fixed <= total for total in totals):
        return 0
    p, c, pc, _, targets = oracle.components(g, "X", "Y", set(fixed))
    sources = {v for v in g.nodes if any(
        (True, True) in oracle.edge_choices(g, v, w) for w in targets | c)}
    sources -= oracle.forbidden_nodes(g, "X", "Y") | p | c | pc | fixed
    count = 0
    for source in sources:
        for path in oracle.bidirected_paths(g, source, targets, c):
            required = fixed | frozenset(path[:-1])
            ap, ac, apc, _, _ = oracle.components(g, "X", "Y", set(required))
            # Original Definition-3 path enumeration, then a separate latent-DAG
            # validity oracle; neither calls production _construct or criteria.
            actual = oracle.latent_dag_adjustment_valid(
                g, "X", "Y", ap | ac | apc | set(required))
            expected = any(required <= total for total in totals)
            assert actual == expected, (g, fixed, path, required)
            count += 1
    return count


def test_auxiliary_oset_equivalence_on_ordered_four_node_bow_capable_graphs():
    nodes = ("A", "X", "B", "Y")
    pairs = tuple(combinations(nodes, 2))
    checked = with_fixed = with_bow = 0
    for states in product(range(4), repeat=len(pairs)):
        view = _graph(nodes, [e for e, s in zip(pairs, states) if s & 1],
                      [e for e, s in zip(pairs, states) if s & 2])
        if "Y" not in oracle.closure(view.topology, {"X"}):
            continue
        for fixed in (frozenset(), frozenset({"A"})):
            count = _check_auxiliary_oset_independently(view, fixed)
            checked += count
            with_fixed += count if fixed else 0
            with_bow += count if any(s == 3 for s in states) else 0
    assert checked > 0 and with_fixed > 0 and with_bow > 0


def test_auxiliary_posttreatment_bow_fixture():
    view = _graph(("X", "Y", "A", "N"),
                  (("X", "Y"), ("X", "A"), ("A", "N")),
                  (("A", "N"), ("N", "Y")))
    assert _check_auxiliary_oset_independently(view, frozenset()) > 0


def test_fixed_s_changes_ii_and_must_not_reuse_unconditioned_predicate():
    view = _graph(("b", "X", "d", "Y"),
                  (("b", "X"), ("X", "Y"), ("b", "d")), (("d", "Y"),))
    plain = runge_adjustment.runge_optimal_adjustment(view, "X", "Y")
    fixed = runge_adjustment.runge_optimal_adjustment(view, "X", "Y", conditioned_on={"b"})
    assert (plain.criterion_i, plain.criterion_ii) == oracle.criteria(view.topology, "X", "Y", set()) == (True, False)
    assert (fixed.criterion_i, fixed.criterion_ii) == oracle.criteria(view.topology, "X", "Y", {"b"}) == (True, True)
    assert set(plain.o_set) == set(fixed.o_set) == {"b", "d"}
    assert fixed.adjustment_set == ("d",)


def test_fixed_s_can_change_the_original_oset():
    view = _graph(("X", "s", "n", "Y"), (("X", "Y"),),
                  (("X", "s"), ("s", "n"), ("n", "Y")))
    plain = runge_adjustment.runge_o_set(view, "X", "Y")
    fixed = runge_adjustment.runge_o_set(view, "X", "Y", conditioned_on={"s"})
    assert plain.o_set == ("n",) and fixed.o_set == ()
    assert oracle.latent_dag_adjustment_valid(view.topology, "X", "Y", {"s"})
    assert not oracle.latent_dag_adjustment_valid(view.topology, "X", "Y", {"s", "n"})
