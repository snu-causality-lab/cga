"""Independent simple-path/set oracles for the private search preparation.

Only the implementation under test supplies the reduced traversal adjacency.
Expected sources, full paths and valid additional sets come from runge_oracles.
Fixed S is added back before comparing required vertices with those sets.
"""
from dataclasses import replace
from itertools import combinations, product
import random

import pytest

from causal_graph_algorithms import AcyclicDirectedBidirectedGraph, DirectedBidirectedGraph
from causal_graph_algorithms import runge_adjustment as runge
from tests.python import runge_oracles as oracle


def graph(nodes, directed, bidirected=()):
    return AcyclicDirectedBidirectedGraph(DirectedBidirectedGraph(
        nodes=nodes, directed=directed, bidirected=bidirected))


def prepare(view, source, fixed=()):
    query = runge._query(view, "X", "Y", fixed)
    result = runge._construct(query, query.fixed)
    return runge._prepare_collider_source(
        query, result, source, runge._RequiredAdjustmentCheck(query))


def reduced_requirements(prepared, source):
    """Independent literal DFS, with no production search helper."""
    pending = [(source,)]
    while pending:
        path = pending.pop()
        for neighbor in prepared.neighbors[path[-1]]:
            if neighbor == prepared.terminal:
                yield frozenset(path)
            elif neighbor not in path:
                pending.append((*path, neighbor))


def assert_preserved(view, fixed=()):
    g = view.topology
    valid = oracle.valid_sets(g, "X", "Y", set(fixed))
    assert valid
    p, c, pc, _, targets = oracle.components(g, "X", "Y", set(fixed))
    sources = {v for v in g.nodes if any(
        (True, True) in oracle.edge_choices(g, v, w) for w in targets | c)}
    sources -= oracle.forbidden_nodes(g, "X", "Y") | p | c | pc | set(fixed)
    outcomes = {}
    for source in sorted(sources):
        expected = any(set(path[:-1]) <= total | set(fixed)
                       for path in oracle.bidirected_paths(g, source, targets, c)
                       for total in valid)
        prepared = prepare(view, source, fixed)
        actual = any(required <= total | set(fixed)
                     for required in reduced_requirements(prepared, source)
                     for total in valid)
        assert actual == expected, source
        outcomes[source] = actual
    assert runge.runge_optimal_adjustment(view, "X", "Y", conditioned_on=fixed).criterion_i == (
        not any(outcomes.values()))
    return outcomes


def safe_graph():
    return graph(("a", "X", "Y", "K", "N", "c"),
                 (("a", "X"), ("X", "Y"), ("K", "N")),
                 (("X", "K"), ("N", "a"), ("N", "c"), ("a", "c"), ("c", "Y")))


def test_safe_detour_reduces_actual_graph_and_retains_fixed_s_path():
    view = safe_graph()
    prepared = prepare(view, "N", ("a",))
    assert prepared.labels["a"] == frozenset()
    assert prepared.labels["c"] == frozenset({"c"})
    assert "c" not in prepared.neighbors["N"]
    assert list(reduced_requirements(prepared, "N")) == [frozenset({"N", "a", "c"})]
    assert assert_preserved(view, ("a",))["N"] is True


def test_district_spanning_tree_can_remove_the_only_feasible_index():
    nodes = ("X", "Y", "K", "N", "U", "a")
    directed = (("X", "Y"), ("K", "N"), ("U", "a"))
    bidirected = (("X", "K"), ("K", "U"), ("U", "Y"),
                  ("N", "a"), ("a", "Y"), ("N", "Y"))
    full, tree = graph(nodes, directed, bidirected), graph(nodes, directed, bidirected[:-1])
    assert oracle.criteria(full.topology, "X", "Y", set())[0] is False
    assert oracle.criteria(tree.topology, "X", "Y", set())[0] is True
    prepared = prepare(full, "N")
    assert prepared.terminal in prepared.neighbors["N"]
    assert assert_preserved(full)["N"] is True
    assert not any(assert_preserved(tree).values())


def test_transitive_reduction_preserves_ancestry_but_changes_separation():
    nodes = ("h", "X", "b", "n", "Y")
    directed = (("h", "X"), ("X", "Y"), ("X", "b"), ("b", "n"))
    full = graph(nodes, (*directed, ("X", "n")), (("n", "Y"),))
    reduced = graph(nodes, directed, (("n", "Y"),))
    assert all(oracle.closure(full.topology, {v}) == oracle.closure(reduced.topology, {v}) for v in nodes)
    prepared = prepare(full, "n")
    assert ("X", "n") in prepared.oracle_graph.topology.directed
    assert assert_preserved(full) == {"n": False}
    assert assert_preserved(reduced) == {"n": True}


def test_pair_infeasibility_cannot_be_inferred_from_individual_vertices():
    view = graph(("X", "Y", "a", "b", "c"), (("X", "Y"),),
                 (("X", "a"), ("a", "b"), ("b", "c"), ("c", "Y")))
    totals = oracle.valid_sets(view.topology, "X", "Y", set())
    assert any({"a", "b"} <= total for total in totals)
    assert any({"a", "c"} <= total for total in totals)
    assert not any({"a", "b", "c"} <= total for total in totals)
    assert not list(reduced_requirements(prepare(view, "a"), "a"))
    assert assert_preserved(view) == {"a": False}


def test_per_source_oracle_detects_loss_hidden_by_another_source(monkeypatch):
    view = AcyclicDirectedBidirectedGraph.from_text(
        "h -> X -> Y; X -> b -> n1; b -> n2; n1 <-> Y; n2 <-> Y")
    assert assert_preserved(view) == {"n1": True, "n2": True}
    original = prepare

    def loses_one(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, neighbors={v: () for v in result.neighbors}) if args[1] == "n1" else result

    monkeypatch.setitem(assert_preserved.__globals__, "prepare", loses_one)
    with pytest.raises(AssertionError, match="n1"):
        assert_preserved(view)


def test_induced_oracle_keeps_forbidden_vertices_and_every_induced_edge():
    view = AcyclicDirectedBidirectedGraph.from_text(
        "h -> X -> Y; X -> M -> Y; X -> b -> n; n <-> Y; Y -> junk")
    before = (view.topology.nodes, view.topology.directed, view.topology.bidirected)
    prepared = prepare(view, "n")
    full = runge._query(view, "X", "Y", ()).backdoor.topology
    small = prepared.oracle_graph.topology
    kept = set(small.nodes)
    assert {"X", "M", "Y"} <= kept and "junk" not in kept
    assert small.directed == frozenset(e for e in full.directed if set(e) <= kept)
    assert small.bidirected == frozenset(e for e in full.bidirected if set(e) <= kept)
    for x, y in combinations(small.nodes, 2):
        rest = sorted(kept - {x, y})
        for size in range(len(rest) + 1):
            for fixed in combinations(rest, size):
                assert oracle.open_path_exists(full, x, y, set(fixed)) == oracle.open_path_exists(
                    small, x, y, set(fixed))
    assert before == (view.topology.nodes, view.topology.directed, view.topology.bidirected)


def test_terminal_name_collision_and_multiple_targets_never_conditioned():
    view = AcyclicDirectedBidirectedGraph.from_text(
        "h -> X -> M -> Y; X -> b -> n; n <-> M; n <-> Y; __condition_i_terminal__")
    prepared = prepare(view, "n")
    assert prepared.terminal not in view.topology.nodes
    assert prepared.labels[prepared.terminal] == frozenset()
    assert list(reduced_requirements(prepared, "n")) == [frozenset({"n"})]
    assert assert_preserved(view) == {"n": True}


def test_all_adjustable_ordered_four_vertex_admgs():
    nodes = ("X", "A", "B", "Y")
    pairs = tuple(combinations(nodes, 2))
    adjustable = 0
    for states in product(range(4), repeat=len(pairs)):
        view = graph(nodes, [e for e, s in zip(pairs, states) if s & 1],
                     [e for e, s in zip(pairs, states) if s & 2])
        if "Y" in oracle.closure(view.topology, {"X"}) and oracle.valid_sets(view.topology, "X", "Y", set()):
            assert_preserved(view)
            adjustable += 1
    assert adjustable == 800


def test_seeded_fixed_s_samples():
    rng = random.Random(20260914)
    nodes = ("S", "X", "A", "B", "C", "N", "Y")
    pairs = tuple(combinations(nodes, 2))
    adjustable = active = 0
    for _ in range(100):
        view = graph(nodes, [e for e in pairs if e == ("X", "Y") or rng.random() < .2],
                     [e for e in pairs if rng.random() < .2])
        if oracle.valid_sets(view.topology, "X", "Y", {"S"}):
            active += bool(assert_preserved(view, ("S",)))
            adjustable += 1
    assert (adjustable, active) == (67, 20)


@pytest.mark.parametrize("two_conflicts", [False, True])
def test_endpoint_only_union_and_intersection_merging_are_not_valid(two_conflicts):
    nodes = ("X", "Y", "K", "N", "a", "b", "t", "z", "Ua", "Va")
    directed = [("X", "Y"), ("K", "N"), ("Ua", "a"), ("Va", "z")]
    bidirected = [("X", "K"), ("K", "Ua"), ("Ua", "Va"), ("Va", "Y"),
                  ("N", "a"), ("N", "b"), ("a", "t"), ("b", "t"), ("t", "z"), ("z", "Y")]
    if two_conflicts:
        nodes += ("Ub", "Vb")
        directed += [("Ub", "b"), ("Vb", "z")]
        bidirected += [("K", "Ub"), ("Ub", "Vb"), ("Vb", "Y")]
    view = graph(nodes, directed, bidirected)
    eligible = tuple(v for v in nodes if v not in {"X", "Y"})
    valid = [set(given) for size in range(len(eligible) + 1)
             for given in combinations(eligible, size)
             if oracle.latent_dag_adjustment_valid(view.topology, "X", "Y", set(given))]
    left, right, suffix = {"N", "a", "t"}, {"N", "b", "t"}, {"z"}
    exists = lambda required: any(required <= total for total in valid)
    assert not exists(left | suffix)
    assert exists(right | suffix) is (not two_conflicts)
    assert not exists(left | right | suffix)
    assert exists((left & right) | suffix)
    expected_c = {"a", "b", "t", "z", "Ua", "Va"} | ({"Ub", "Vb"} if two_conflicts else set())
    result = runge.runge_optimal_adjustment(view, "X", "Y")
    assert set(result.colliders) == set(result.o_set) == expected_c
    assert result.criterion_i is two_conflicts
    prepared = prepare(view, "N")
    assert any(exists(required) for required in reduced_requirements(prepared, "N")) is (not two_conflicts)


def test_forest_backtracking_is_dominated_and_budget_is_global():
    view = AcyclicDirectedBidirectedGraph.from_text(
        "X -> Y; X <-> N1 <-> b <-> c <-> d <-> Y; X <-> N2 <-> b")
    assert assert_preserved(view) == {"N1": False, "N2": False}
    # Each source has one indexed path and four nonterminal endpoints. Returning
    # along its tree edge only adds requirements to an already reached endpoint.
    assert runge.runge_optimal_adjustment(view, "X", "Y", max_collider_states=8).criterion_i is True
    for limit in (0, 1, 4, 7):
        with pytest.raises(runge.RungeAdjustmentResourceLimitError, match="max_collider_states"):
            runge.runge_optimal_adjustment(view, "X", "Y", max_collider_states=limit)


@pytest.mark.parametrize("size,expected_cases", [(2, 2), (3, 32), (4, 1024)])
def test_detour_rule_preserves_every_two_bit_downward_predicate(size, expected_cases):
    nodes = tuple(str(i) for i in range(size))
    pairs = tuple(combinations(nodes, 2))
    labels_possible = (frozenset(), frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"}))
    predicates = [frozenset(s for i, s in enumerate(labels_possible) if mask >> i & 1)
                  for mask in range(16)
                  if all(not (mask >> i & 1) or all(mask >> j & 1 for j, t in enumerate(labels_possible) if t <= s)
                         for i, s in enumerate(labels_possible))]
    assert len(predicates) == 6

    def unions(adjacency, labels):
        result = set()
        pending = [(nodes[0], frozenset({nodes[0]}), frozenset())]
        while pending:
            vertex, visited, label = pending.pop()
            if vertex == nodes[-1]:
                result.add(label)
                continue
            for neighbor in adjacency[vertex] - visited:
                pending.append((neighbor, visited | {neighbor}, label | labels[neighbor]))
        return result

    cases = 0
    for mask in range(1 << len(pairs)):
        original = {v: set() for v in nodes}
        for i, (a, b) in enumerate(pairs):
            if mask >> i & 1:
                original[a].add(b)
                original[b].add(a)
        for middle in product(labels_possible, repeat=size - 2):
            labels = dict(zip(nodes[1:-1], middle))
            labels[nodes[0]] = labels[nodes[-1]] = frozenset()
            reduced = {v: set(adjacent) for v, adjacent in original.items()}
            runge._delete_labelled_detours(reduced, labels)
            before, after = unions(original, labels), unions(reduced, labels)
            assert all(bool(before & predicate) == bool(after & predicate) for predicate in predicates)
            cases += 1
    assert cases == expected_cases
