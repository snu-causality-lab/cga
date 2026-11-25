from __future__ import annotations

import itertools
import random
import copy
import pickle

import pytest

from causal_graph_algorithms import DAG, DirectedBidirectedGraph
from causal_graph_algorithms.dag_adjustment import (
    DAGAdjustmentDomainError,
    DAGAdjustmentNotApplicable,
    DAGOptimalAdjustmentSet,
    dag_adjustment_set_is_valid,
    dag_forbidden_projection,
    dag_optimal_adjustment_set,
    prune_dag_adjustment_set,
)
from .dag_adjustment_oracles import all_dags, avar, causal_nodes, covariance, subsets, valid


def test_oset_avoids_instruments_and_includes_precision_parent() -> None:
    graph = DAG.from_text('I --> X --> M --> Y; C --> X; C --> M; P --> Y')
    result = dag_optimal_adjustment_set(graph, interventions='X', outcomes='Y')
    assert isinstance(result, DAGOptimalAdjustmentSet)
    assert result.vertices == ('C', 'P')
    assert set(result.causal_nodes) == {'M', 'Y'}
    assert set(result.forbidden_vertices) == {'X', 'M', 'Y'}
    assert prune_dag_adjustment_set(graph, interventions='X', outcomes='Y', adjustment=('I', 'C', 'P')) == ('C', 'P')


def test_no_causal_path_is_not_empty_adjustment() -> None:
    graph = DAG.from_text('C --> X; C --> Y')
    result = dag_optimal_adjustment_set(graph, interventions='X', outcomes='Y')
    assert isinstance(result, DAGAdjustmentNotApplicable)
    assert result.reason == 'outcomes_not_descendants'
    assert not dag_adjustment_set_is_valid(graph, interventions='X', outcomes='Y', adjustment=())
    assert dag_adjustment_set_is_valid(graph, interventions='X', outcomes='Y', adjustment=('C',))


def test_joint_interventions_use_proper_paths_and_can_lack_adjustment() -> None:
    graph = DAG.from_text('X1 --> M --> X2 --> Y; M --> Y')
    result = dag_optimal_adjustment_set(graph, interventions=('X1', 'X2'), outcomes='Y')
    assert isinstance(result, DAGAdjustmentNotApplicable)
    assert result.reason == 'no_valid_adjustment_set'
    assert not dag_adjustment_set_is_valid(graph, interventions=('X1', 'X2'), outcomes='Y', adjustment=('M',))
    graph2 = DAG.from_text('X1 --> M --> X2 --> Y; C --> M')
    result2 = dag_optimal_adjustment_set(graph2, interventions=('X1', 'X2'), outcomes='Y')
    assert isinstance(result2, DAGOptimalAdjustmentSet)
    assert result2.vertices == ()
    assert result2.causal_nodes == ('Y',)


def test_multiple_outcomes_do_not_include_outcomes_in_oset() -> None:
    graph = DAG.from_text('X --> Y1 --> Y2; C --> Y1')
    result = dag_optimal_adjustment_set(graph, interventions='X', outcomes=('Y1', 'Y2'))
    assert isinstance(result, DAGOptimalAdjustmentSet)
    assert result.vertices == ('C',)
    projected = dag_forbidden_projection(graph, interventions='X', outcomes=('Y1', 'Y2'))
    assert projected.topology.nodes == graph.topology.nodes


def test_rejects_invalid_domain_and_invalid_pruning_input() -> None:
    graph = DAG.from_text('X --> M --> Y')
    with pytest.raises(DAGAdjustmentDomainError):
        dag_optimal_adjustment_set(graph.topology, interventions='X', outcomes='Y')  # type: ignore[arg-type]
    for x, y in [((), 'Y'), ('X', ()), ('X', 'X')]:
        with pytest.raises(DAGAdjustmentDomainError):
            dag_optimal_adjustment_set(graph, interventions=x, outcomes=y)
    with pytest.raises(DAGAdjustmentDomainError):
        prune_dag_adjustment_set(graph, interventions='X', outcomes='Y', adjustment=('M',))


def test_result_values_are_sealed_factory_created_and_immutable() -> None:
    graph = DAG.from_text('X --> Y')
    result = dag_optimal_adjustment_set(graph, interventions='X', outcomes='Y')
    for cls in (DAGOptimalAdjustmentSet, DAGAdjustmentNotApplicable):
        with pytest.raises(TypeError):
            cls()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            type('ForgedAdjustment', (cls,), {})
    assert copy.copy(result) is result
    assert copy.deepcopy(result) is result
    with pytest.raises(TypeError):
        pickle.dumps(result)
    with pytest.raises((AttributeError, TypeError)):
        result.vertices = ()  # type: ignore[misc,union-attr]


def test_all_four_vertex_dags_against_literal_path_criterion() -> None:
    nodes = ('A', 'B', 'C', 'D')
    graphs = all_dags(nodes)
    assert len(graphs) == 543
    for edges in graphs:
        graph = DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=edges))
        for x, y in itertools.permutations(nodes, 2):
            for z in subsets(tuple(v for v in nodes if v not in (x, y))):
                assert dag_adjustment_set_is_valid(graph, interventions=x, outcomes=y, adjustment=z) == valid(nodes, edges, {x}, {y}, z)
            cn = causal_nodes(nodes, edges, {x}, {y})
            result = dag_optimal_adjustment_set(graph, interventions=x, outcomes=y)
            if not cn:
                assert isinstance(result, DAGAdjustmentNotApplicable)
                continue
            assert isinstance(result, DAGOptimalAdjustmentSet)
            assert valid(nodes, edges, {x}, {y}, set(result.vertices))
            projection = dag_forbidden_projection(graph, interventions=x, outcomes=y)
            projected_parents = {a for a, b in projection.topology.directed if b == y} - {x, y}
            assert set(result.vertices) == projected_parents


def test_small_joint_queries_against_literal_paths() -> None:
    nodes = ('A', 'B', 'C', 'D')
    rng = random.Random(301)
    for edges in rng.sample(all_dags(nodes), 70):
        graph = DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=edges))
        for x in itertools.combinations(nodes, 2):
            rest = tuple(v for v in nodes if v not in x)
            for y in subsets(rest):
                if not y:
                    continue
                candidates = subsets(tuple(v for v in rest if v not in y))
                for z in candidates:
                    assert dag_adjustment_set_is_valid(graph, interventions=x, outcomes=y, adjustment=z) == valid(nodes, edges, set(x), y, z)
                result = dag_optimal_adjustment_set(graph, interventions=x, outcomes=y)
                cn = causal_nodes(nodes, edges, set(x), y)
                if y <= cn:
                    exists = any(valid(nodes, edges, set(x), y, z) for z in candidates)
                    assert isinstance(result, DAGOptimalAdjustmentSet) == exists


def test_exact_linear_variance_and_pruning_order() -> None:
    nodes = ('A', 'B', 'C', 'D')
    rng = random.Random(91)
    for edges in rng.sample(all_dags(nodes), 65):
        graph = DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=edges))
        for x, y in itertools.permutations(nodes, 2):
            if not causal_nodes(nodes, edges, {x}, {y}):
                continue
            result = dag_optimal_adjustment_set(graph, interventions=x, outcomes=y)
            assert isinstance(result, DAGOptimalAdjustmentSet)
            for z in subsets(tuple(v for v in nodes if v not in (x, y))):
                if not valid(nodes, edges, {x}, {y}, z):
                    continue
                pruned = prune_dag_adjustment_set(graph, interventions=x, outcomes=y, adjustment=z)
                assert set(pruned) <= z
                assert valid(nodes, edges, {x}, {y}, set(pruned))
                for seed in (1, 4):
                    cov = covariance(nodes, edges, seed)
                    def variance(zz):
                        return avar(cov, nodes.index(x), nodes.index(y), [nodes.index(v) for v in zz])
                    assert variance(result.vertices) <= variance(z)
                    assert variance(pruned) <= variance(z)
                reordered = DAG.from_graph(DirectedBidirectedGraph(nodes=tuple(reversed(nodes)), directed=edges))
                assert set(prune_dag_adjustment_set(reordered, interventions=x, outcomes=y, adjustment=z)) == set(pruned)
