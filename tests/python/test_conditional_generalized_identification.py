from __future__ import annotations

import copy
import itertools
import json
import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph,
    ConditionalGeneralizedBudgetExceeded, IdentifiedConditionalGeneralizedEffect,
    NonIdentifiedConditionalGeneralizedEffect, IdentificationDomainError,
    Ratio, identify_conditional_generalized_causal_effect,
    identify_generalized_causal_effect, NonIdentifiedGeneralizedEffect,
)
from .gid_oracles import evaluate_generalized_probability_expression
from .id_oracles import TinyADMG, exact_binary_conditional_interventional_probability
from .published_fixing_oracles import conditional_generalized_oracle


def _effect(graph, *, y="Y", x=(), z=(), sources=((),), **kwargs):
    return identify_conditional_generalized_causal_effect(
        graph, outcomes=y, interventions=x, conditioned_on=z,
        experiment_interventions=sources, **kwargs,
    )


def test_maxbi_uses_candidate_rule_two_cut_and_not_printed_complement_cut():
    direct = DAG.from_text("Z --> Y")
    result = _effect(direct, z="Z")
    assert isinstance(result, IdentifiedConditionalGeneralizedEffect)
    assert result.rule_two_interventions == ("Z",)
    # With a confounded incoming edge, an action/observation exchange fails.
    confounded = AcyclicDirectedBidirectedGraph.from_text("Z --> Y; Z <-> Y")
    result = _effect(confounded, z="Z")
    assert result.rule_two_interventions == ()
    chain = AcyclicDirectedBidirectedGraph.from_text("X --> Z --> Y; X <-> Z")
    result = _effect(chain, x="X", z="Z")
    assert isinstance(result, IdentifiedConditionalGeneralizedEffect)
    assert result.rule_two_interventions == ("Z",)
    assert isinstance(identify_generalized_causal_effect(
        chain, outcomes=("Y", "Z"), interventions="X", experiment_interventions=((),)
    ), NonIdentifiedGeneralizedEffect)
    collider = DAG.from_text("Z --> C <-- Y")
    result = _effect(collider, z=("Z", "C"))
    assert result.rule_two_interventions == ("Z",)


def test_sources_are_exact_copied_and_not_closed_under_intervention_subsets():
    graph = AcyclicDirectedBidirectedGraph.from_text("X --> Y; X <-> Y; Z")
    sources = [["X"], ["X"], []]
    result = _effect(graph, x="X", z="Z", sources=sources)
    sources[0].append("Y")
    assert isinstance(result, IdentifiedConditionalGeneralizedEffect)
    assert result.experiment_interventions == ((), ("X",))
    assert isinstance(_effect(graph, x="X", z="Z", sources=()), NonIdentifiedConditionalGeneralizedEffect)
    assert isinstance(_effect(graph, x="X", sources=(("X", "Y"),)), NonIdentifiedConditionalGeneralizedEffect)


def test_nonidentified_result_keeps_original_query_and_reduced_source_hedges():
    graph = AcyclicDirectedBidirectedGraph.from_text("X --> Y; X <-> Y; Z")
    result = _effect(graph, x="X", z="Z")
    assert isinstance(result, NonIdentifiedConditionalGeneralizedEffect)
    assert result.outcomes == ("Y",)
    assert result.conditioned_on == ("Z",)
    assert result.witness.interventions == ("X", "Z")
    assert result.witness.outcomes == ("Y",)
    assert result.witness.source_hedges


def test_negative_bow_has_independently_constructed_positive_counterpair():
    # Model A: X=U, Y=U xor E. Model B: X=U, Y=X xor E, with independent
    # fair U,Z and P(E=1)=1/4. Permitted graph inputs may be inactive.
    graph = AcyclicDirectedBidirectedGraph.from_text("X --> Y; X <-> Y; Z")
    tables = []
    do_values = []
    for use_parent in (False, True):
        table = {}
        do_y = Fraction(0)
        for u,e,z in itertools.product((0,1), repeat=3):
            weight = Fraction(1,4) * (Fraction(1,4) if e else Fraction(3,4))
            x = u
            y = (x if use_parent else u) ^ e
            table[(x,y,z)] = table.get((x,y,z),Fraction(0))+weight
            # Conditional on Z=0 after do(X=0); multiply by two to remove P(Z=0).
            if z == 0 and ((0 if use_parent else u) ^ e) == 1:
                do_y += 2*weight
        tables.append(table)
        do_values.append(do_y)
    assert tables[0] == tables[1] and min(tables[0].values()) > 0
    assert do_values == [Fraction(1,2), Fraction(1,4)]
    assert isinstance(_effect(graph,x="X",z="Z"), NonIdentifiedConditionalGeneralizedEffect)


def test_results_are_immutable_factory_bound_and_deterministic():
    graph = DAG.from_text("X --> Y; Z --> Y")
    left = _effect(graph, x="X", z="Z")
    right = _effect(graph, x=("X",), z=("Z",), sources=[[], []])
    assert left == right
    assert isinstance(left.formula, Ratio)
    assert copy.copy(left) is left and copy.deepcopy(left) is left
    with pytest.raises(FrozenInstanceError):
        left.outcomes = ()
    with pytest.raises(TypeError):
        pickle.dumps(left)
    with pytest.raises(TypeError):
        IdentifiedConditionalGeneralizedEffect(object())


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5])
def test_rule_two_check_budget_validation_and_exhaustion(bad):
    graph = DAG.from_text("X --> Y; Z --> Y; W --> Y")
    with pytest.raises(IdentificationDomainError):
        _effect(graph, x="X", z=("Z", "W"), max_rule_two_checks=bad)
    with pytest.raises(ConditionalGeneralizedBudgetExceeded):
        _effect(graph, x="X", z=("Z", "W"), max_rule_two_checks=1)


def test_domain_errors_are_not_negative_identification():
    graph = DAG.from_text("X --> Y; Z")
    for kwargs in ({"y":"X", "x":"X"}, {"y":"Y", "z":"Y"}, {"x":"X", "z":"X"}):
        with pytest.raises(IdentificationDomainError):
            _effect(graph, **kwargs)
    with pytest.raises(IdentificationDomainError):
        _effect(DirectedBidirectedGraph(nodes=("Y",)))
    with pytest.raises(IdentificationDomainError):
        _effect(graph, sources="X")
    empty = _effect(graph, y=(), sources=())
    assert isinstance(empty, IdentifiedConditionalGeneralizedEffect)
    assert not empty.formula.free_variables


def test_kivva_figure2_and_theorem3_independent_decision_and_exact_values():
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/papers/kivva_2023_cgid.json").read_text())
    case = fixture["figure_2"]
    labels = tuple(case["nodes"])
    positions = {label: i for i, label in enumerate(labels)}
    graph = AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
        nodes=labels, directed=map(tuple, case["directed"]), bidirected=map(tuple, case["bidirected"]),
    ))
    tiny = TinyADMG(tuple(range(len(labels))),
        frozenset((positions[a], positions[b]) for a,b in case["directed"]),
        frozenset(tuple(sorted((positions[a], positions[b]))) for a,b in case["bidirected"]))
    result = _effect(graph, y="Y1", x="X1", z=("Z1", "Z2"))
    expected, moved = conditional_generalized_oracle(tiny, outcomes=frozenset({1}),
        interventions=frozenset({0}), conditioned_on=frozenset({2,3}), sources=(frozenset(),))
    assert expected and isinstance(result, IdentifiedConditionalGeneralizedEffect)
    assert frozenset(positions[v] for v in result.rule_two_interventions) == moved
    joint = identify_generalized_causal_effect(graph, outcomes=("Y1","Z1","Z2"),
        interventions="X1", experiment_interventions=((),))
    assert isinstance(joint, NonIdentifiedGeneralizedEffect)
    for bits in itertools.product((0,1), repeat=len(labels)):
        values = dict(enumerate(bits))
        actual = evaluate_generalized_probability_expression(result.formula, graph=tiny,
            node_labels=labels, experiment_interventions=(frozenset(),),
            environment=dict(zip(labels,bits)), variant=0)
        expected_value = exact_binary_conditional_interventional_probability(tiny,
            outcomes=(1,), interventions=(0,), conditioned_on=(2,3), values=values, variant=0)
        assert actual == expected_value


def test_small_graphs_against_exhaustive_exchange_subset_oracle():
    from .id_oracles import all_bow_admgs_n3
    labels = ("X", "Z", "Y")
    source_collections = ((), (frozenset(),), (frozenset({0}),), (frozenset({0,1}),))
    for tiny in all_bow_admgs_n3():
        graph = AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
            nodes=labels, directed=((labels[a],labels[b]) for a,b in tiny.directed),
            bidirected=((labels[a],labels[b]) for a,b in tiny.bidirected)))
        for sources in source_collections:
            expected, moved = conditional_generalized_oracle(tiny, outcomes=frozenset({2}),
                interventions=frozenset({0}), conditioned_on=frozenset({1}), sources=sources)
            result = _effect(graph, x="X", z="Z", sources=tuple(tuple(labels[i] for i in s) for s in sources))
            assert isinstance(result, IdentifiedConditionalGeneralizedEffect) == expected
            assert frozenset(labels.index(v) for v in result.rule_two_interventions) == moved
