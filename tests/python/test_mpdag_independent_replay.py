"""Server-only substantive replay; three-node universes and exact rational laws."""
from __future__ import annotations

from itertools import combinations, product

from causal_graph_algorithms.conditional_adjustment import (
    ConditionalAdjustmentDomainError, ConditionalAdjustmentSet,
    conditional_adjustment_set_is_valid, find_conditional_adjustment_set,
)
from causal_graph_algorithms.endpoint_marked import EndpointMarkedGraph, Mark
from causal_graph_algorithms.mpdag_identification import (
    IdentifiedMPDAGEffect, enumerate_mpdag_effects, identify_mpdag_causal_effect,
)
from causal_graph_algorithms.pdag import MPDAG, PDAGDomainError

from .gac_oracles import literal_gac
from .pdag_oracles import consistent_members
from .published_mpdag_oracles import (
    directed_descendants, evaluate, intervention_probability, joint_distribution,
)


def _mpdags():
    nodes = ("A", "B", "C")
    for states in product(range(4), repeat=3):
        edges = []
        for (a, b), state in zip(combinations(nodes, 2), states):
            if state:
                marks = ((Mark.TAIL, Mark.TAIL), (Mark.TAIL, Mark.ARROW), (Mark.ARROW, Mark.TAIL))[state - 1]
                edges.append((a, marks[0], b, marks[1]))
        raw = EndpointMarkedGraph(nodes=nodes, edges=edges)
        try:
            graph = MPDAG.from_graph(raw)
        except PDAGDomainError:
            continue
        members = consistent_members(raw)
        assert members
        yield graph, members


def test_all_three_node_mpdag_queries_partition_and_exact_values() -> None:
    for graph, members in _mpdags():
        nodes = graph.topology.nodes
        tables = tuple(joint_distribution(nodes, members[0], variant) for variant in range(4))
        for roles in product(range(3), repeat=3):
            x = tuple(v for v, role in zip(nodes, roles) if role == 1)
            y = tuple(v for v, role in zip(nodes, roles) if role == 2)
            if not x or not y:
                continue
            result = identify_mpdag_causal_effect(graph, interventions=x, outcomes=y)
            partition = enumerate_mpdag_effects(graph, interventions=x, outcomes=y, max_results=6)
            fingerprints = []
            covered = []
            for subclass in partition:
                subclass_members = consistent_members(subclass.graph.topology)
                covered.extend(subclass_members)
                fingerprint = []
                for table in tables:
                    for bits in product((0, 1), repeat=len(x) + len(y)):
                        assignment = dict(zip((*x, *y), bits))
                        value = evaluate(nodes, table, subclass.formula, assignment)
                        fingerprint.append(value)
                        for member in subclass_members:
                            expected = intervention_probability(nodes, table, member,
                                {v: assignment[v] for v in x}, {v: assignment[v] for v in y})
                            assert value == expected
                fingerprints.append(tuple(fingerprint))
            assert len(covered) == len(set(covered))
            assert set(covered) == set(members)
            assert len(set(fingerprints)) == len(partition)
            assert isinstance(result, IdentifiedMPDAGEffect) == (len(partition) == 1)
            if isinstance(result, IdentifiedMPDAGEffect):
                for table in tables:
                    assignment = {v: 1 for v in (*x, *y)}
                    assert evaluate(nodes, table, result.formula, assignment) == evaluate(
                        nodes, table, partition[0].formula, assignment)


def test_all_three_node_conditional_sets_against_every_original_dag() -> None:
    for graph, members in _mpdags():
        nodes = graph.topology.nodes
        for roles in product(range(5), repeat=3):
            x, y, z, s = (frozenset(v for v, role in zip(nodes, roles) if role == k) for k in range(1, 5))
            if not x or not y:
                continue
            supported = all(not z & directed_descendants(nodes, d, x) for d in members)
            if not supported:
                try:
                    conditional_adjustment_set_is_valid(
                        graph, interventions=x, outcomes=y, conditioned_on=z, adjustment=s)
                except ConditionalAdjustmentDomainError:
                    continue
                raise AssertionError("possible-descendant conditioning was accepted")
            expected = all(literal_gac(nodes, tuple((a, "tail", b, "arrow") for a, b in d),
                x, y, z | s, "dag") for d in members)
            assert conditional_adjustment_set_is_valid(
                graph, interventions=x, outcomes=y, conditioned_on=z, adjustment=s) == expected
            constructed = find_conditional_adjustment_set(
                graph, interventions=x, outcomes=y, conditioned_on=z)
            possible_sets = tuple(frozenset(v for v, take in zip(tuple(set(nodes) - x - y - z), flags) if take)
                for flags in product((0, 1), repeat=len(set(nodes) - x - y - z)))
            existence = any(all(literal_gac(nodes, tuple((a, "tail", b, "arrow") for a, b in d),
                x, y, z | candidate, "dag") for d in members) for candidate in possible_sets)
            assert isinstance(constructed, ConditionalAdjustmentSet) == existence
