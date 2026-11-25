"""MPDAG identification and IDGraphs contracts."""
from __future__ import annotations

import pytest

from causal_graph_algorithms import DAG
from causal_graph_algorithms.endpoint_marked import EndpointMarkedGraph, Mark
from causal_graph_algorithms.mpdag_identification import (
    IdentifiedMPDAGEffect,
    MPDAGIdentificationResourceLimitError,
    MPDAGIdentificationDomainError,
    NonIdentifiedMPDAGEffect,
    enumerate_mpdag_effects,
    identify_mpdag_causal_effect,
)
from causal_graph_algorithms.pdag import MPDAG


def test_identified_dag_smoke() -> None:
    graph = DAG.from_text("X -> M -> Y; X -> Y")
    result = identify_mpdag_causal_effect(graph, interventions="X", outcomes="Y")
    assert isinstance(result, IdentifiedMPDAGEffect)
    assert result.formula.free_variables == frozenset(("X", "Y"))
    assert result.buckets == (("M",), ("Y",))


def test_undirected_edge_idgraphs() -> None:
    graph = MPDAG.from_graph(EndpointMarkedGraph(nodes=("X", "Y"), edges=(
        ("X", Mark.TAIL, "Y", Mark.TAIL),)))
    negative = identify_mpdag_causal_effect(graph, interventions="X", outcomes="Y")
    assert isinstance(negative, NonIdentifiedMPDAGEffect)
    assert negative.path == ("X", "Y")
    results = enumerate_mpdag_effects(graph, interventions="X", outcomes="Y", max_results=2)
    assert len(results) == 2
    assert all(isinstance(row, IdentifiedMPDAGEffect) for row in results)
    with pytest.raises(MPDAGIdentificationResourceLimitError):
        enumerate_mpdag_effects(graph, interventions="X", outcomes="Y", max_results=1)


def test_proper_paths_exclude_other_joint_treatments() -> None:
    graph = MPDAG.from_graph(EndpointMarkedGraph(nodes=("X1", "X2", "Y"), edges=(
        ("X1", Mark.TAIL, "X2", Mark.TAIL),
        ("X1", Mark.TAIL, "Y", Mark.ARROW),
        ("X2", Mark.TAIL, "Y", Mark.ARROW))))
    result = identify_mpdag_causal_effect(graph, interventions=("X1", "X2"), outcomes="Y")
    assert isinstance(result, IdentifiedMPDAGEffect)


def test_backward_chord_is_not_a_possible_path() -> None:
    graph = MPDAG.from_graph(EndpointMarkedGraph(nodes=("A", "B", "C"), edges=(
        ("A", Mark.TAIL, "B", Mark.TAIL),
        ("B", Mark.TAIL, "C", Mark.TAIL),
        ("C", Mark.TAIL, "A", Mark.ARROW))))
    result = identify_mpdag_causal_effect(graph, interventions="A", outcomes="C")
    assert isinstance(result, IdentifiedMPDAGEffect)


def test_query_validation_and_factory_boundaries() -> None:
    graph = DAG.from_text("X -> Y")
    with pytest.raises(MPDAGIdentificationDomainError):
        identify_mpdag_causal_effect(graph.topology, interventions="X", outcomes="Y")  # type: ignore[arg-type]
    with pytest.raises(MPDAGIdentificationDomainError):
        identify_mpdag_causal_effect(graph, interventions="X", outcomes="X")
    with pytest.raises(MPDAGIdentificationDomainError):
        enumerate_mpdag_effects(graph, interventions="X", outcomes="Y", max_results=True)
    with pytest.raises(TypeError):
        NonIdentifiedMPDAGEffect(graph, ("X",), ("Y",), ("X", "Y"))
