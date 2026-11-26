from fractions import Fraction
from itertools import combinations

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
    GraphValidationError,
    MarkovEquivalenceDomainError,
    dag_intervention_markov_equivalent,
    d_separated,
    find_minimum_cost_intervention,
    hedge_hull,
)


def _subsets(vertices):
    for size in range(len(vertices) + 1):
        yield from combinations(vertices, size)


def _district_hedges(graph, target):
    """Enumerate the defining vertex sets, using independent transitive closures."""
    target = set(target)
    result = []
    for extra in _subsets(tuple(node for node in graph.nodes if node not in target)):
        if not extra:
            continue
        retained = target | set(extra)
        directed = {(a, b) for a, b in graph.directed if {a, b} <= retained}
        bidirected = {(a, b) for a, b in graph.bidirected if {a, b} <= retained}
        bidirected |= {(b, a) for a, b in bidirected}
        for relation in (directed, bidirected):
            relation.update((node, node) for node in retained)
            for midpoint in retained:
                relation.update(
                    (a, b)
                    for a in retained
                    for b in retained
                    if (a, midpoint) in relation and (midpoint, b) in relation
                )
        if all(any((node, s) in directed for s in target) for node in retained):
            if all((a, b) in bidirected for a in retained for b in retained):
                result.append(frozenset(extra))
    return result


def _akbari_figure_2a():
    return AcyclicDirectedBidirectedGraph(
        DirectedBidirectedGraph(
            nodes=("v1", "v2", "v3", "s1", "s2", "x"),
            directed=(
                ("v1", "v2"), ("v2", "s1"), ("s1", "s2"), ("s2", "x"), ("v3", "x"),
            ),
            bidirected=(
                ("v2", "s2"), ("s1", "s2"), ("v1", "s2"),
                ("v1", "x"), ("s1", "x"), ("v3", "x"),
            ),
        )
    )


def test_akbari_figures_2_and_3():
    graph = _akbari_figure_2a()
    target = ("s1", "s2")
    assert _district_hedges(graph.topology, target) == [
        frozenset({"v2"}), frozenset({"v1", "v2"}),
    ]
    assert hedge_hull(graph, target) == ("v1", "v2", "s1", "s2")
    design = find_minimum_cost_intervention(
        graph, target, costs={"v1": 0, "v2": Fraction(1, 3)}
    )
    assert design is not None
    assert design.vertices == ("v2",)
    assert design.total_cost == Fraction(1, 3)
    assert find_minimum_cost_intervention(graph, target, allowed=("v1",)) is None


def test_hedge_free_hull_contains_the_target():
    graph = AcyclicDirectedBidirectedGraph(DirectedBidirectedGraph(nodes=("s",)))
    assert hedge_hull(graph, "s") == ("s",)
    assert find_minimum_cost_intervention(graph, "s").vertices == ()


@pytest.mark.parametrize("costs", [{"v2": -1}, {"v2": True}, {"unknown": 1}, {"v2": 0.5}])
def test_rejects_invalid_intervention_costs(costs):
    with pytest.raises(GraphValidationError):
        find_minimum_cost_intervention(_akbari_figure_2a(), ("s1", "s2"), costs=costs)


def test_rejects_empty_or_disconnected_targets():
    graph = _akbari_figure_2a()
    for target in ((), ("s1", "v2")):
        with pytest.raises(GraphValidationError):
            hedge_hull(graph, target)


def test_single_district_design_against_definition_on_three_vertices():
    nodes = ("a", "b", "c")
    pairs = tuple(combinations(nodes, 2))
    for arrows in _subsets(pairs):
        for confounding in _subsets(pairs):
            graph = AcyclicDirectedBidirectedGraph(
                DirectedBidirectedGraph(nodes=nodes, directed=arrows, bidirected=confounding)
            )
            for target in _subsets(nodes):
                if not target:
                    continue
                induced_pairs = set(combinations(target, 2))
                connected = len(target) == 1 or sum(
                    edge in induced_pairs for edge in confounding
                ) >= len(target) - 1
                if not connected:
                    continue
                hedges = _district_hedges(graph.topology, target)
                extras = tuple(node for node in nodes if node not in target)
                assert set(hedge_hull(graph, target)) == set(target).union(*hedges)
                costs = {"a": 0, "b": 2, "c": 1}
                for allowed in _subsets(extras):
                    feasible = [
                        assignment for assignment in _subsets(allowed)
                        if all(set(assignment) & hedge for hedge in hedges)
                    ]
                    design = find_minimum_cost_intervention(
                        graph, target, allowed=allowed, costs=costs
                    )
                    if not feasible:
                        assert design is None
                        continue
                    assert design is not None
                    optimum = min(
                        feasible,
                        key=lambda k: (sum(costs[v] for v in k), len(k), k),
                    )
                    assert design.vertices == optimum
                    assert design.total_cost == sum(costs[v] for v in optimum)


def _dag(nodes, edges):
    return DAG(DirectedBidirectedGraph(nodes=nodes, directed=edges))


def test_hauser_buhlmann_figure_2():
    nodes = tuple(map(str, range(1, 8)))
    edges = {
        ("2", "1"), ("1", "5"), ("2", "5"), ("2", "3"), ("2", "6"),
        ("3", "6"), ("3", "7"), ("3", "4"), ("4", "7"), ("5", "6"),
    }
    d = _dag(nodes, edges)
    d1 = _dag(nodes, edges - {("2", "1"), ("1", "5"), ("2", "5")} | {
        ("1", "2"), ("5", "1"), ("5", "2"),
    })
    d2 = _dag(nodes, edges - {("2", "3"), ("3", "7"), ("3", "4"), ("4", "7")} | {
        ("3", "2"), ("7", "3"), ("4", "3"), ("7", "4"),
    })
    assert dag_intervention_markov_equivalent(d, d1, ((), "4"))
    assert dag_intervention_markov_equivalent(d, d2, ((),))
    assert not dag_intervention_markov_equivalent(d, d2, ((), "4"))
    with pytest.raises(MarkovEquivalenceDomainError):
        dag_intervention_markov_equivalent(d, d2, ("4",))


def test_cut_signatures_against_separation_with_no_observational_regime():
    nodes = ("a", "b", "c")
    arrows = tuple((a, b) for a in nodes for b in nodes if a != b)
    graphs = []
    for edges in _subsets(arrows):
        try:
            graphs.append(_dag(nodes, edges))
        except GraphValidationError:
            continue
    targets = (("a", "b"), ("b", "c"), ("c", "a"))
    # Use a conservative family without an observational regime.
    def independences(graph):
        result = []
        for target in targets:
            mutilated = _dag(nodes, [(a, b) for a, b in graph.topology.directed if b not in target])
            result.extend(
                d_separated(mutilated, a, b, given=given)
                for a, b in combinations(nodes, 2)
                for given in ((), tuple(set(nodes) - {a, b}))
            )
        return result

    signatures = [independences(graph) for graph in graphs]
    for left_idx, left in enumerate(graphs):
        for right_idx, right in enumerate(graphs):
            assert dag_intervention_markov_equivalent(left, right, targets) == (
                signatures[left_idx] == signatures[right_idx]
            )
