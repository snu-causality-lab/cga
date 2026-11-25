"""Pinned source examples, with reported expectations and independent adjudication."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from causal_graph_algorithms import CausalPAG, MaximalAncestralGraph

from causal_graph_algorithms.conditional_adjustment import (
    ConditionalAdjustmentNotFound, ConditionalAdjustmentSet, conditional_adjustment_set_is_valid,
    find_conditional_adjustment_set,
)
from causal_graph_algorithms.endpoint_marked import EndpointMarkedGraph, Mark
from causal_graph_algorithms.mpdag_identification import IdentifiedMPDAGEffect, enumerate_mpdag_effects, identify_mpdag_causal_effect
from causal_graph_algorithms.pdag import MPDAG

from .pdag_oracles import consistent_members
from .published_mpdag_oracles import evaluate, intervention_probability, joint_distribution, sparse_members


def _fixtures():
    return json.loads((Path(__file__).parents[1] / "fixtures/papers/published_mpdag_20261116.json").read_text())["fixtures"]


def _graph(fixture):
    return MPDAG.from_graph(EndpointMarkedGraph(nodes=fixture["nodes"], edges=(
        [(a, Mark.TAIL, b, Mark.ARROW) for a, b in fixture["directed"]]
        + [(a, Mark.TAIL, b, Mark.TAIL) for a, b in fixture["undirected"]])))


def test_laplante_reported_examples_1_to_4() -> None:
    fixtures = _fixtures()
    for name in ("laplante_fig2a", "laplante_fig2b", "laplante_fig2c"):
        fixture = fixtures[name]
        graph = _graph(fixture)
        for row in fixture["reported"]:
            kwargs = dict(interventions=row["x"], outcomes=row["y"], conditioned_on=row["z"])
            if row.get("no_adjustment"):
                assert isinstance(find_conditional_adjustment_set(graph, **kwargs), ConditionalAdjustmentNotFound)
            else:
                assert conditional_adjustment_set_is_valid(graph, adjustment=row["s"], **kwargs) is row["valid"]


def test_perkovic_example_3_8_semidirected_cycle_and_joint_outcomes() -> None:
    fixture = _fixtures()["perkovic_fig4a"]
    graph = _graph(fixture)
    row = fixture["reported"]
    result = identify_mpdag_causal_effect(graph, interventions=row["x"], outcomes=row["y"])
    assert isinstance(result, IdentifiedMPDAGEffect)
    assert result.buckets == tuple(tuple(b) for b in row["buckets"])
    assert result.formula.free_variables == frozenset(("X", "Y1", "Y2"))


def test_laplante_constructed_formula_against_original_dag_laws() -> None:
    fixtures = _fixtures()
    for name in ("laplante_fig2a", "laplante_fig2b"):
        fixture = fixtures[name]
        graph = _graph(fixture)
        members = sparse_members(graph.topology)
        assert members
        for row in fixture["reported"]:
            if not row["valid"]:
                continue
            result = find_conditional_adjustment_set(graph,
                interventions=row["x"], outcomes=row["y"], conditioned_on=row["z"])
            assert isinstance(result, ConditionalAdjustmentSet)
            for variant in range(2):
                table = joint_distribution(graph.topology.nodes, members[0], variant)
                assignment = {v: (variant + i) % 2 for i, v in enumerate((*row["x"], *row["y"], *row["z"]))}
                value = evaluate(graph.topology.nodes, table, result.formula, assignment)
                x_values = {v: assignment[v] for v in row["x"]}
                z_values = {v: assignment[v] for v in row["z"]}
                yz_values = {v: assignment[v] for v in (*row["y"], *row["z"])}
                for dag in members:
                    numerator = intervention_probability(graph.topology.nodes, table, dag, x_values, yz_values)
                    denominator = intervention_probability(graph.topology.nodes, table, dag, x_values, z_values)
                    assert value == numerator / denominator


def test_laplante_figure3_example6_validated_pag_and_printed_adjustment() -> None:
    fixture = _fixtures()["laplante_fig3"]
    mark = {"tail": Mark.TAIL, "arrow": Mark.ARROW, "circle": Mark.CIRCLE}
    printed = EndpointMarkedGraph(nodes=fixture["nodes"], edges=(
        (a, mark[ma], b, mark[mb]) for a, ma, b, mb in fixture["endpoint_edges"]))
    representative = MaximalAncestralGraph.from_graph(EndpointMarkedGraph(
        nodes=fixture["nodes"], edges=(
            [(a, Mark.TAIL, b, Mark.ARROW) for a, b in fixture["representative_directed"]]
            + [(a, Mark.ARROW, b, Mark.ARROW) for a, b in fixture["representative_bidirected"]])))
    graph = CausalPAG.from_mag(representative)
    assert graph.topology == printed
    row = fixture["reported"]
    result = find_conditional_adjustment_set(graph,
        interventions=row["x"], outcomes=row["y"], conditioned_on=row["z"])
    assert isinstance(result, ConditionalAdjustmentSet)
    assert result.vertices == tuple(row["s"])
    assert conditional_adjustment_set_is_valid(graph, interventions=row["x"],
        outcomes=row["y"], conditioned_on=row["z"], adjustment=row["s"])


def test_guo_figure_2_seven_dags_three_distinct_effects() -> None:
    fixture = _fixtures()["guo_fig2a"]
    graph = _graph(fixture)
    row = fixture["reported"]
    members = consistent_members(graph.topology)
    assert len(members) == row["dag_count"]
    partition = enumerate_mpdag_effects(graph, interventions=row["x"], outcomes=row["y"], max_results=10)
    assert len(partition) == row["distinct_effect_count"]
    represented = [d for result in partition for d in consistent_members(result.graph.topology)]
    assert len(represented) == len(set(represented))
    assert set(represented) == set(members)
    fingerprints = []
    for result in partition:
        fingerprint = []
        for variant in range(3):
            table = joint_distribution(graph.topology.nodes, members[0], variant)
            for a, y in product((0, 1), repeat=2):
                assignment = {"A": a, "Y": y}
                value = evaluate(graph.topology.nodes, table, result.formula, assignment)
                fingerprint.append(value)
                for dag in consistent_members(result.graph.topology):
                    assert value == intervention_probability(graph.topology.nodes, table, dag, {"A": a}, {"Y": y})
        fingerprints.append(tuple(fingerprint))
    assert len(set(fingerprints)) == 3
