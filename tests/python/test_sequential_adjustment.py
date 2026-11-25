from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from itertools import combinations, product
from pathlib import Path
import random

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph, EndpointMarkedGraph,
    GraphValidationError, SequentialAdjustmentDomainError,
    SequentialAdjustmentNotFound, SequentialAdjustmentSequence,
    find_minimal_sequential_adjustment, sequential_adjustment_is_valid,
)
from . import sequential_adjustment_oracles as oracle
from .id_oracles import (
    TinyADMG, exact_binary_interventional_probability,
    exact_binary_observational_distribution,
)


def _graph(nodes, directed=(), bidirected=()):
    return AcyclicDirectedBidirectedGraph.from_graph(
        DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=bidirected))


def _valid(graph, xs, ys, zs):
    return sequential_adjustment_is_valid(
        graph, treatment_order=xs, outcomes=ys, covariate_blocks=zs)


def _find(graph, xs, ys):
    return find_minimal_sequential_adjustment(graph, treatment_order=xs, outcomes=ys)


def test_future_intervention_changes_printed_forbidden_verdict():
    graph = DAG.from_text('X1 -> Z -> X2 -> Y')
    nodes, edges = graph.topology.nodes, frozenset(graph.topology.directed)
    zs = ({'Z'}, set())
    assert not oracle.valid(nodes, edges, frozenset(), ('X1', 'X2'), {'Y'}, zs, printed=True)
    assert oracle.valid(nodes, edges, frozenset(), ('X1', 'X2'), {'Y'}, zs)
    assert _valid(graph, ('X1', 'X2'), 'Y', zs)


def test_paper_fixture_preserves_printed_and_corrected_answers():
    path = Path(__file__).parents[1] / 'fixtures/papers/jung_park_lee_2024_sequential.json'
    fixture = json.loads(path.read_text())
    for case in fixture['cases']:
        graph = _graph(case['nodes'], case['directed'], case['bidirected'])
        xs, ys = tuple(case['treatment_order']), set(case['outcomes'])
        zs = tuple(set(z) for z in case['covariate_blocks'])
        assert oracle.valid(graph.topology.nodes, frozenset(graph.topology.directed),
                            frozenset(graph.topology.bidirected), xs, ys, zs,
                            printed=True) == case['printed_criterion_accepts']
        assert _valid(graph, xs, ys, zs) == case['corrected_criterion_accepts']


def test_constructor_finalizes_history_before_searching_next_stage():
    graph = DAG.from_text('X1 -> X2 -> Y; C -> X2; C -> Y')
    assert _valid(graph, ('X1', 'X2'), 'Y', ({'C'}, set()))
    assert not _valid(graph, ('X1', 'X2'), 'Y', ((), ()))
    result = _find(graph, ('X1', 'X2'), 'Y')
    assert isinstance(result, SequentialAdjustmentSequence)
    assert result.covariate_blocks == ((), ('C',))


def test_mandatory_history_activates_collider_outside_endpoint_ancestors():
    graph = DAG.from_text('X1; A -> C <- B; A -> X2 -> Y; B -> Y')
    result = _find(graph, ('X1', 'X2'), ('C', 'Y'))
    assert isinstance(result, SequentialAdjustmentSequence)
    assert result.outcome_blocks == (('C',), (), ('Y',))
    assert result.covariate_blocks[0] == ()
    assert result.covariate_blocks[1] in (('A',), ('B',))
    assert not _valid(graph, ('X1', 'X2'), ('C', 'Y'), ((), ()))


def test_no_solution_is_formula_class_failure_not_nonidentification():
    graph = _graph(('X', 'M', 'Y'), (('X', 'M'), ('M', 'Y')), (('X', 'Y'),))
    result = _find(graph, ('X',), 'Y')
    assert isinstance(result, SequentialAdjustmentNotFound)
    assert result.reason == 'no_valid_sequence'
    assert result.failed_stage == 0
    assert result.graph is graph
    assert not _valid(graph, ('X',), 'Y', ((),))
    assert not _valid(graph, ('X',), 'Y', (('M',),))


def test_topology_order_only_breaks_constructor_ties_and_inputs_are_immutable():
    graph = DAG.from_text('Y; Z -> X -> Y; Z -> Y')
    result = _find(graph, ('X',), 'Y')
    assert isinstance(result, SequentialAdjustmentSequence)
    assert result.covariate_blocks == (('Z',),)
    assert copy.copy(result) is result
    assert copy.deepcopy(result) is result
    with pytest.raises(FrozenInstanceError):
        result.covariate_blocks = ()
    with pytest.raises(TypeError, match='factory'):
        SequentialAdjustmentSequence(graph, ('X',), ((), ('Y',)), (('Z',),))
    # Outcomes unaffected by every treatment are retained in Y0, with vacuous stages.
    independent = DAG.from_text('Y; X1 -> X2; Z')
    result = _find(independent, ('X1', 'X2'), 'Y')
    assert isinstance(result, SequentialAdjustmentSequence)
    assert result.covariate_blocks == ((), ())
    assert result.outcome_blocks == (('Y',), (), ())


@pytest.mark.parametrize('xs, ys, zs, message', [
    ((), 'Y', (), 'nonempty'),
    (('X1', 'X1'), 'Y', ((), ()), 'distinct'),
    (('X2', 'X1'), 'Y', ((), ()), 'topological'),
    (('X1', 'X2'), (), ((), ()), 'nonempty'),
    (('X1', 'X2'), 'X1', ((), ()), 'disjoint'),
    (('X1', 'X2'), 'Y', ((),), 'one covariate block'),
    (('X1', 'X2'), 'Y', (('Z',), ('Z',)), 'disjoint'),
    (('X1', 'X2'), 'Y', (('Y',), ()), 'disjoint'),
    (('X1', 'X2'), 'Y', (('D',), ()), 'later treatments'),
    ({'X1', 'X2'}, 'Y', ((), ()), 'ordered sequence'),
    ('X1', 'Y', ((),), 'ordered sequence'),
])
def test_domain_validation(xs, ys, zs, message):
    graph = DAG.from_text('X1 -> X2 -> D -> Y; Z')
    with pytest.raises(SequentialAdjustmentDomainError, match=message):
        _valid(graph, xs, ys, zs)


def test_unknown_vertices_and_raw_views_fail_before_graphical_classification():
    graph = DAG.from_text('X -> Y')
    with pytest.raises(GraphValidationError, match='unknown vertex'):
        _valid(graph, ('missing',), 'Y', ((),))
    for raw in (graph.topology, EndpointMarkedGraph(nodes=('X', 'Y'))):
        with pytest.raises(SequentialAdjustmentDomainError, match='validated'):
            _find(raw, ('X',), 'Y')


def test_invalid_later_block_is_not_hidden_by_earlier_graphical_failure():
    graph = _graph(('X1', 'X2', 'Y'), (('X1', 'Y'),), (('X1', 'Y'),))
    with pytest.raises(GraphValidationError, match='unknown vertex'):
        _valid(graph, ('X1', 'X2'), 'Y', ((), ('missing',)))


def _check_universe(graph, xs, ys):
    topology = graph.topology
    nodes, directed, bidirected = topology.nodes, frozenset(topology.directed), frozenset(topology.bidirected)
    feasible = []
    count = 0
    for zs in oracle.allocations(nodes, xs, ys):
        admissible = all(not (z & oracle.desc(nodes, directed, set(xs[i + 1:]))) for i, z in enumerate(zs))
        if not admissible:
            continue
        expected = oracle.valid(nodes, directed, bidirected, xs, ys, zs)
        assert _valid(graph, xs, ys, zs) == expected, (topology, xs, ys, zs)
        count += 1
        if expected:
            feasible.append(zs)
    result = _find(graph, xs, ys)
    assert isinstance(result, SequentialAdjustmentSequence) == bool(feasible)
    if feasible:
        blocks = tuple(set(z) for z in result.covariate_blocks)
        assert blocks in feasible
        # All componentwise subsets, not merely individual single removals.
        assert not any(other != blocks and all(a <= b for a, b in zip(other, blocks))
                       for other in feasible)
        yblocks = oracle.outcome_blocks(nodes, directed, xs, ys)
        history = set(yblocks[0])
        for i, (x, block) in enumerate(zip(xs, blocks)):
            targets = set().union(*yblocks[i + 1:])
            vertices = tuple(sorted(block))
            for bits in range((1 << len(vertices)) - 1):
                smaller = {v for j, v in enumerate(vertices) if bits >> j & 1}
                assert not oracle.stage_valid(nodes, directed, bidirected, x,
                                              set(xs[i + 1:]), targets, history, smaller)
            history |= {x} | yblocks[i + 1] | block
        assert _find(graph, xs, ys) == result
    return count


def test_all_ordered_four_node_dags_all_two_treatment_roles_and_allocations():
    nodes = ('A', 'B', 'C', 'D')
    pairs = tuple(combinations(nodes, 2))
    count = 0
    for mask in range(1 << len(pairs)):
        graph = _graph(nodes, tuple(e for i, e in enumerate(pairs) if mask >> i & 1))
        for xs in combinations(nodes, 2):
            remaining = tuple(v for v in nodes if v not in xs)
            for bits in range(1, 1 << len(remaining)):
                ys = {v for i, v in enumerate(remaining) if bits >> i & 1}
                count += _check_universe(graph, xs, ys)
    assert count == 2552


def test_bow_admg_draws_include_three_treatments_and_early_outcomes():
    rng = random.Random(20260919)
    nodes = tuple('ABCDEF')
    pairs = tuple(combinations(nodes, 2))
    for _ in range(80):
        directed = tuple(e for e in pairs if rng.randrange(4) == 0)
        bidirected = tuple(e for e in pairs if rng.randrange(7) == 0)
        graph = _graph(nodes, directed, bidirected)
        xs = tuple(sorted(rng.sample(nodes, 3)))
        remaining = tuple(v for v in nodes if v not in xs)
        ys = set(rng.sample(remaining, rng.randint(1, 2)))
        _check_universe(graph, xs, ys)


def test_all_ordered_four_node_bow_admgs_fixed_roles():
    nodes = ('A', 'B', 'C', 'D')
    pairs = tuple(combinations(nodes, 2))
    count = 0
    for directed_mask, hidden_mask in product(range(64), repeat=2):
        graph = _graph(nodes,
                       tuple(e for i, e in enumerate(pairs) if directed_mask >> i & 1),
                       tuple(e for i, e in enumerate(pairs) if hidden_mask >> i & 1))
        count += _check_universe(graph, ('A', 'C'), {'D'})
    assert count == 12288


@pytest.mark.parametrize('nodes, edges, hidden, xs, ys, zs, expect_equal', [
    (('X', 'Z', 'Y'), ((0, 1), (1, 2)), (), ('X',), {'Y'}, ({'Z'},), False),
    (('X1', 'Z', 'X2', 'Y'), ((0, 1), (1, 2), (2, 3)), (), ('X1', 'X2'), {'Y'}, ({'Z'}, set()), True),
    (('X1', 'C', 'X2', 'Y'), ((0, 2), (1, 2), (1, 3), (2, 3)), (), ('X1', 'X2'), {'Y'}, (set(), set()), False),
    (('X1', 'A', 'B', 'C', 'X2', 'Y'), ((1, 3), (2, 3), (1, 4), (2, 5), (4, 5)), (), ('X1', 'X2'), {'Y'}, ({'C'}, set()), False),
    (('X1', 'Z', 'X2', 'Y'), ((1, 2), (2, 3)), ((0, 1),), ('X1', 'X2'), {'Y'}, (set(), {'Z'}), True),
])
def test_exact_observational_functional_against_truncated_scm(nodes, edges, hidden, xs, ys, zs, expect_equal):
    tiny = TinyADMG(tuple(range(len(nodes))), frozenset(edges), frozenset(hidden))
    directed = frozenset((nodes[a], nodes[b]) for a, b in edges)
    yblocks = oracle.outcome_blocks(nodes, directed, xs, ys)
    comparisons = []
    labels = xs + tuple(y for y in nodes if y in ys)
    for variant in (0, 1):
        table = exact_binary_observational_distribution(tiny, variant)
        for bits in product((0, 1), repeat=len(labels)):
            values = dict(zip(labels, bits))
            supplied = oracle.functional(table, nodes, xs, yblocks, zs, values)
            target = exact_binary_interventional_probability(
                tiny, outcomes=tuple(nodes.index(y) for y in ys),
                interventions=tuple(nodes.index(x) for x in xs),
                values={nodes.index(v): bit for v, bit in values.items()}, variant=variant)
            comparisons.append(supplied == target)
    assert all(comparisons) == expect_equal
    assert _valid(_graph(nodes, directed, tuple((nodes[a], nodes[b]) for a, b in hidden)), xs, ys, zs) == expect_equal
