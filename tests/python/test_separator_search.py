"""Exact-cost searches and listing, checked against literal-path definitions."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path
import copy
import pickle

import pytest

from causal_graph_algorithms import AcyclicDirectedBidirectedGraph, AncestralGraph, DAG, DirectedBidirectedGraph
from causal_graph_algorithms.separator_search import (
    MinimumCostAdjustment, MinimumCostSeparator, SearchBudgetExceeded,
    find_minimum_cost_adjustment_set, find_minimum_cost_m_separator,
    iter_adjustment_sets, iter_m_separators,
)
from causal_graph_algorithms.separators import SeparatorDomainError, SeparatorNotFound
from tests.python.oracles import m_separated_by_simple_edge_paths


def test_weighted_chain_and_exact_fraction() -> None:
    graph = DAG.from_text('X --> A --> B --> Y')
    result = find_minimum_cost_m_separator(graph, 'X', 'Y', costs={'A': 7, 'B': Fraction(1, 3)})
    assert isinstance(result, MinimumCostSeparator)
    assert result.vertices == ('B',) and result.total_cost == Fraction(1, 3)


def test_joint_paths_choose_combined_cost_not_cardinality() -> None:
    graph = DAG.from_text('X --> A --> C --> Y; X --> B --> C')
    result = find_minimum_cost_m_separator(graph, 'X', 'Y', costs={'A': 1, 'B': 1, 'C': 5})
    assert isinstance(result, MinimumCostSeparator)
    assert result.vertices == ('A', 'B') and result.total_cost == 2
    forced = find_minimum_cost_m_separator(graph, 'X', 'Y', required='C', costs={'A': 1, 'B': 1, 'C': 5})
    assert isinstance(forced, MinimumCostSeparator)
    assert forced.vertices == ('C',) and forced.total_cost == 5


def test_exhausted_iterator_is_distinct_from_finished_empty() -> None:
    graph = DAG.from_text('X --> A --> Y')
    with pytest.raises(SearchBudgetExceeded):
        list(iter_m_separators(graph, 'X', 'Y', max_states=1))
    assert list(iter_m_separators(DAG.from_text('X --> Y'), 'X', 'Y')) == []


def test_listing_includes_nonancestral_separators_and_original_minimality() -> None:
    graph = DAG.from_text('X --> A --> Y; A --> D')
    assert {r.vertices for r in iter_m_separators(graph, 'X', 'Y')} == {('A',), ('A', 'D')}
    assert [r.vertices for r in iter_m_separators(graph, 'X', 'Y', minimal=True)] == [('A',)]
    assert [r.vertices for r in iter_m_separators(graph, 'X', 'Y', required='D', minimal=True)] == [('A', 'D')]


def test_collider_inclusion_recomputes_ancestral_reduction() -> None:
    graph = DAG.from_text('X --> C <-- Y; C --> D')
    assert {r.vertices for r in iter_m_separators(graph, 'X', 'Y')} == {()}
    assert isinstance(find_minimum_cost_m_separator(graph, 'X', 'Y', required='D'), SeparatorNotFound)


@pytest.mark.parametrize('costs', [{'A': 0}, {'A': -1}, {'A': True}, {'A': 1.0}, {}, {'A': 1, 'Z': 2}])
def test_costs_are_complete_positive_exact_allowed_weights(costs: dict[str, object]) -> None:
    with pytest.raises(SeparatorDomainError):
        find_minimum_cost_m_separator(DAG.from_text('X --> A --> Y'), 'X', 'Y', costs=costs)  # type: ignore[arg-type]


def test_adjustment_cost_and_all_sets() -> None:
    graph = DAG.from_text('Z --> X --> Y; Z --> Y; X --> M --> Y; W --> Z')
    result = find_minimum_cost_adjustment_set(graph, interventions='X', outcomes='Y', costs={'Z': 2, 'M': 1, 'W': 1})
    assert isinstance(result, MinimumCostAdjustment)
    assert result.vertices == ('Z',) and result.total_cost == 2
    assert {frozenset(r.vertices) for r in iter_adjustment_sets(graph, interventions='X', outcomes='Y')} == {frozenset({'Z'}), frozenset({'Z', 'W'})}


def test_paper_figure_five_and_reported_minimal_sets() -> None:
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/papers/zander_2019_separator_figure5.json').read_text())
    graph = DAG.from_text(fixture['graph'])
    optional = tuple(v for v in graph.topology.nodes if v not in {'X', 'Y'})
    subsets = {frozenset(v for v, bit in zip(optional, bits) if bit)
               for bits in product((False, True), repeat=len(optional))}
    truth = {s for s in subsets if _literal_separated(graph.topology.nodes, graph.topology.directed, 'X', 'Y', s)}
    minimal_truth = {s for s in truth if not any(t < s for t in truth)}
    listed = {frozenset(r.vertices) for r in iter_m_separators(graph, 'X', 'Y', minimal=True)}
    assert listed == minimal_truth == {frozenset(s) for s in fixture['adjudicated_minimal_sets']}
    forced = {frozenset(r.vertices) for r in iter_m_separators(graph, 'X', 'Y', required='V1', minimal=True)}
    forced_truth = {s for s in truth if 'V1' in s}
    assert forced == {s for s in forced_truth if not any(t < s for t in forced_truth)}
    assert forced == {frozenset({'Z1', 'V1'}), frozenset({'Z2', 'V1'})}
    weighted = find_minimum_cost_m_separator(graph, 'X', 'Y', costs={'V1': 1, 'V2': 1, 'Z1': 3, 'Z2': 2})
    assert frozenset(weighted.vertices) == frozenset({'Z2'}) and weighted.total_cost == 2
    weights = {'V1': 1, 'V2': 1, 'Z1': 3, 'Z2': 2}
    assert weighted.total_cost == min(sum(weights[v] for v in s) for s in truth)


def test_joint_endpoint_latent_weighted_and_constrained_literal_control() -> None:
    graph = AcyclicDirectedBidirectedGraph.from_text('X1 <-> A --> Y1; X2 --> B --> Y2; A <-> B')
    atoms = tuple((a, False, b, True) for a, b in graph.topology.directed) + tuple((a, True, b, True) for a, b in graph.topology.bidirected)
    truth = {frozenset(s) for s in (('B',), ('A', 'B')) if all(
        m_separated_by_simple_edge_paths(graph.topology.nodes, atoms, (x,), (y,), frozenset(s))
        for x in ('X1', 'X2') for y in ('Y1', 'Y2'))}
    listed = {frozenset(s.vertices) for s in iter_m_separators(graph, ('X1', 'X2'), ('Y1', 'Y2'), required='B', allowed=('A', 'B'))}
    assert listed == truth
    result = find_minimum_cost_m_separator(graph, ('X1', 'X2'), ('Y1', 'Y2'), required='B', allowed=('A', 'B'), costs={'A': Fraction(1, 3), 'B': Fraction(2, 7)})
    assert frozenset(result.vertices) in truth and result.total_cost == Fraction(13, 21)


def test_validation_is_eager_and_results_are_frozen() -> None:
    graph = DAG.from_text('X --> A --> Y')
    with pytest.raises(SeparatorDomainError):
        iter_m_separators(graph, 'X', 'Y', allowed='unknown')
    with pytest.raises(SeparatorDomainError):
        iter_m_separators(graph, 'X', 'Y', max_states=True)
    result = find_minimum_cost_m_separator(graph, 'X', 'Y')
    assert isinstance(result, MinimumCostSeparator)
    assert copy.copy(result) is result and copy.deepcopy(result) is result
    with pytest.raises(TypeError):
        pickle.dumps(result)
    with pytest.raises((AttributeError, TypeError)):
        result.total_cost = Fraction(100)
    with pytest.raises(TypeError):
        MinimumCostSeparator(graph, ('X',), ('Y',), (), ('A',), ('A',), (('A', Fraction(1)),), Fraction(1))


def test_latent_roots_are_never_cut_and_undirected_ancestral_graphs() -> None:
    graph = AcyclicDirectedBidirectedGraph.from_text('X <-> Y')
    assert isinstance(find_minimum_cost_m_separator(graph, 'X', 'Y'), SeparatorNotFound)
    graph = AncestralGraph.from_text('X --- A --- B --- Y')
    found = find_minimum_cost_m_separator(graph, 'X', 'Y', costs={'A': 3, 'B': 2})
    assert found.vertices == ('B',)
    assert {r.vertices for r in iter_m_separators(graph, 'X', 'Y')} == {('A',), ('B',), ('A', 'B')}


def test_three_vertex_bow_capable_graphs_against_literal_edge_paths() -> None:
    nodes = ('X', 'M', 'Y')
    pairs = tuple(combinations(nodes, 2))
    for directed_flags in product((False, True), repeat=3):
        directed = tuple(e for e, flag in zip(pairs, directed_flags) if flag)
        for bidirected_flags in product((False, True), repeat=3):
            bidirected = tuple(e for e, flag in zip(pairs, bidirected_flags) if flag)
            graph = AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=bidirected))
            atoms = tuple((a, False, b, True) for a, b in directed) + tuple((a, True, b, True) for a, b in bidirected)
            truth = {z for z in (frozenset(), frozenset({'M'}))
                     if m_separated_by_simple_edge_paths(nodes, atoms, ('X',), ('Y',), z)}
            assert {frozenset(r.vertices) for r in iter_m_separators(graph, 'X', 'Y')} == truth
            found = find_minimum_cost_m_separator(graph, 'X', 'Y')
            if truth:
                assert frozenset(found.vertices) in truth and found.total_cost == min(map(len, truth))
            else:
                assert isinstance(found, SeparatorNotFound)


def _literal_separated(nodes: tuple[str, ...], directed: tuple[tuple[str, str], ...],
                       x: str, y: str, z: frozenset[str]) -> bool:
    """Enumerate simple paths, then apply the original d-separation definition."""
    parents = {v: {a for a, b in directed if b == v} for v in nodes}
    ancestors = set(z)
    frontier = list(z)
    while frontier:
        for p in parents[frontier.pop()]:
            if p not in ancestors:
                ancestors.add(p)
                frontier.append(p)
    arrows = set(directed)
    neighbors = {v: {b if a == v else a for a, b in directed if v in (a, b)} for v in nodes}
    pending = [(x,)]
    while pending:
        path = pending.pop()
        if path[-1] == y:
            active = True
            for a, b, c in zip(path, path[1:], path[2:]):
                collider = (a, b) in arrows and (c, b) in arrows
                if (collider and b not in ancestors) or (not collider and b in z):
                    active = False
                    break
            if active:
                return False
        else:
            pending.extend(path + (v,) for v in neighbors[path[-1]] if v not in path)
    return True


def test_four_vertex_dag_universe_against_literal_paths() -> None:
    """64 order-compatible DAGs, all singleton endpoints and allowed/required sets."""
    nodes = ('A', 'B', 'C', 'D')
    potential = tuple(combinations(nodes, 2))
    for flags in product((False, True), repeat=len(potential)):
        edges = tuple(edge for edge, flag in zip(potential, flags) if flag)
        graph = DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=edges))
        for x, y in combinations(nodes, 2):
            optional = tuple(v for v in nodes if v not in (x, y))
            subsets = [frozenset(v for v, bit in zip(optional, bits) if bit)
                       for bits in product((False, True), repeat=len(optional))]
            truth = {s for s in subsets if _literal_separated(nodes, edges, x, y, s)}
            assert {frozenset(r.vertices) for r in iter_m_separators(graph, x, y)} == truth
            for allowed in subsets:
                for required in subsets:
                    if not required <= allowed:
                        continue
                    candidates = {s for s in truth if required <= s <= allowed}
                    found = find_minimum_cost_m_separator(graph, x, y, required=required, allowed=allowed,
                                                          costs={v: i + 1 for i, v in enumerate(optional) if v in allowed})
                    assert {frozenset(r.vertices) for r in iter_m_separators(graph, x, y, required=required, allowed=allowed)} == candidates
                    minimal = {s for s in candidates if not any(t < s for t in candidates)}
                    assert {frozenset(r.vertices) for r in iter_m_separators(graph, x, y, required=required, allowed=allowed, minimal=True)} == minimal
                    if not candidates:
                        assert isinstance(found, SeparatorNotFound)
                    else:
                        assert isinstance(found, MinimumCostSeparator)
                        assert frozenset(found.vertices) in candidates
                        assert found.total_cost == min(sum(i + 1 for i, v in enumerate(optional) if v in s) for s in candidates)
