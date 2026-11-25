"""Conditional adjustment public contracts and independent finite replay."""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import FrozenInstanceError

import pytest

from causal_graph_algorithms import DAG, CausalPAG, MaximalAncestralGraph
from causal_graph_algorithms.conditional_adjustment import (
    ConditionalAdjustmentDomainError,
    ConditionalAdjustmentNotFound,
    ConditionalAdjustmentResourceLimitError,
    ConditionalAdjustmentSet,
    conditional_adjustment_set_is_valid,
    find_conditional_adjustment_set,
)
from causal_graph_algorithms.endpoint_marked import EndpointMarkedGraph, Mark
from causal_graph_algorithms.pdag import MPDAG


def test_conditional_adjustment_smoke() -> None:
    graph = DAG.from_text("Z -> X; Z -> Y; X -> Y")
    result = find_conditional_adjustment_set(
        graph, interventions="X", outcomes="Y", conditioned_on="Z")
    assert isinstance(result, ConditionalAdjustmentSet)
    assert result.vertices == ()
    assert result.formula.free_variables == frozenset(("X", "Y", "Z"))
    assert copy(result) is result and deepcopy(result) is result
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.vertices = ("Z",)  # type: ignore[misc]


def test_conditioning_can_activate_a_collider() -> None:
    graph = DAG.from_text("X -> Y; A -> X; A -> Z; B -> Z; B -> Y")
    assert not conditional_adjustment_set_is_valid(
        graph, interventions="X", outcomes="Y", conditioned_on="Z", adjustment=())
    result = find_conditional_adjustment_set(
        graph, interventions="X", outcomes="Y", conditioned_on="Z")
    assert isinstance(result, ConditionalAdjustmentSet)
    assert conditional_adjustment_set_is_valid(
        graph, interventions="X", outcomes="Y", conditioned_on="Z", adjustment=result.vertices)


def test_conditioning_on_possible_descendant_is_domain_error() -> None:
    graph = DAG.from_text("X -> Z -> Y")
    with pytest.raises(ConditionalAdjustmentDomainError, match="possible descendant"):
        find_conditional_adjustment_set(
            graph, interventions="X", outcomes="Y", conditioned_on="Z")


def test_nonamenable_mpdag_is_an_adjustment_negative() -> None:
    graph = MPDAG.from_graph(EndpointMarkedGraph(nodes=("X", "Y", "Z"), edges=(
        ("X", Mark.TAIL, "Y", Mark.TAIL),)))
    result = find_conditional_adjustment_set(
        graph, interventions="X", outcomes="Y", conditioned_on="Z")
    assert isinstance(result, ConditionalAdjustmentNotFound)
    assert result.reason == "not_amenable"


def test_pag_visibility_and_empty_conditioning() -> None:
    raw = EndpointMarkedGraph(nodes=("X", "Y"), edges=(("X", Mark.TAIL, "Y", Mark.ARROW),))
    graph = CausalPAG.from_mag(MaximalAncestralGraph.from_graph(raw))
    result = find_conditional_adjustment_set(
        graph, interventions="X", outcomes="Y", conditioned_on=())
    assert isinstance(result, ConditionalAdjustmentNotFound)


def test_pag_positive_visible_edge_with_nondescendant_conditioning() -> None:
    raw = EndpointMarkedGraph(nodes=("A", "B", "X", "Y"), edges=(
        ("A", Mark.TAIL, "X", Mark.ARROW),
        ("B", Mark.TAIL, "X", Mark.ARROW),
        ("X", Mark.TAIL, "Y", Mark.ARROW)))
    graph = CausalPAG.from_mag(MaximalAncestralGraph.from_graph(raw))
    assert conditional_adjustment_set_is_valid(
        graph, interventions="X", outcomes="Y", conditioned_on="A", adjustment=())
    result = find_conditional_adjustment_set(
        graph, interventions="X", outcomes="Y", conditioned_on="A")
    assert isinstance(result, ConditionalAdjustmentSet)


def test_budget_and_unvalidated_input_are_not_negatives() -> None:
    graph = DAG.from_text("X -> Y; Z -> Y")
    with pytest.raises(ConditionalAdjustmentResourceLimitError):
        find_conditional_adjustment_set(
            graph, interventions="X", outcomes="Y", conditioned_on="Z", max_path_states=1)
    with pytest.raises(ConditionalAdjustmentDomainError):
        find_conditional_adjustment_set(
            graph.topology, interventions="X", outcomes="Y", conditioned_on="Z")  # type: ignore[arg-type]
