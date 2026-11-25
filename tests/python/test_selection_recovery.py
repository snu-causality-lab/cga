"""Regression, source-availability and independent SCM tests for recovery."""
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal_graph_algorithms import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph, Probability
from tests.python.selection_recovery_oracles import BinarySCM, all_pairs, evaluate, separated


@pytest.fixture
def recovery():
    return importlib.import_module("causal_graph_algorithms.selection_recovery")


def graph(nodes, directed=(), bidirected=()):
    raw = DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=bidirected)
    return AcyclicDirectedBidirectedGraph.from_graph(raw) if bidirected else DAG.from_graph(raw)


def data_for(g, context="S", value=1):
    return Probability(tuple(v for v in g.topology.nodes if v != context),
                       conditioned_on=context, conditioned_values={context: value})


def assert_causal_formula(g, data, result, population_data=None, seed=42):
    assert result.formula.free_variables <= frozenset(result.outcomes + result.interventions)
    scm = BinarySCM(g.topology, np.random.default_rng(seed))
    actual = evaluate(result.formula, data, scm.source_table(data), population_data,
                      None if population_data is None else scm.source_table(population_data))
    expected = scm.target(data, result.interventions, result.outcomes)
    assert np.max(np.abs(actual - expected)) < 2e-11


def test_conditional_recovery_uses_the_input_conditioning_event(recovery):
    g = graph(("X", "Y", "context"), (("X", "Y"), ("X", "context")))
    for value in (0, 1):
        data = data_for(g, "context", value)
        result = recovery.recover_conditional_distribution(g, data, outcomes="Y", conditioned_on="X")
        assert isinstance(result, recovery.RecoveredDistribution)
        assert result.method == "conditional" and result.interventions == ()
        assert result.formula == Probability("Y", conditioned_on=("X", "context"),
                                             conditioned_values={"context": value})
        scm = BinarySCM(g.topology, np.random.default_rng(value + 10))
        actual = evaluate(result.formula, data, scm.source_table(data))
        population = Probability(("X", "Y"))
        expected = evaluate(Probability("Y", conditioned_on="X"), population,
                            scm.source_table(population))
        assert np.max(np.abs(actual - expected)) < 1e-12
    assert not separated(g.topology, {"Y"}, {"context"}, set())


def test_population_conditional_recovery_is_distinct_from_a_causal_effect(recovery):
    g = graph(("X", "Y", "S"), (("X", "S"),), (("X", "Y"),))
    data = data_for(g)
    assert isinstance(recovery.recover_conditional_distribution(
        g, data, outcomes="Y", conditioned_on="X"), recovery.RecoveredDistribution)
    assert isinstance(recovery.recover_causal_effect(g, data, outcomes="Y", interventions="X"),
                      recovery.RecoveryNotEstablished)


def test_joint_treatment_counterexample_to_printed_nonproper_path_condition(recovery):
    for edges in ((("A", "B"), ("A", "Y")),
                  (("A", "B"), ("A", "Y"), ("B", "S"))):
        g = graph(("A", "B", "Y", "S"), edges)
        data = data_for(g)
        assert recovery.selection_adjustment_pair_is_valid(
            g, data, interventions=("A", "B"), outcomes="Y", adjustment=())
        pairs = list(recovery.iter_selection_adjustment_pairs(
            g, data, interventions=("A", "B"), outcomes="Y"))
        assert [(p.adjustment, p.population_adjustment) for p in pairs] == [((), ())]
        assert_causal_formula(g, data, pairs[0])


def test_external_population_covariate_is_required_for_selected_confounder(recovery):
    g = graph(("Z", "X", "Y", "S"),
              (("Z", "X"), ("Z", "Y"), ("Z", "S"), ("X", "Y")))
    data = data_for(g)
    assert isinstance(recovery.find_selection_adjustment_pair(
        g, data, interventions="X", outcomes="Y"), recovery.NoSelectionAdjustmentPair)
    population = Probability("Z")
    pair = recovery.find_selection_adjustment_pair(
        g, data, interventions="X", outcomes="Y", population_data=population)
    assert isinstance(pair, recovery.SelectionAdjustmentPair)
    assert pair.adjustment == pair.population_adjustment == ("Z",)
    assert_causal_formula(g, data, pair, population)
    assert copy.copy(pair) is pair and copy.deepcopy(pair) is pair
    with pytest.raises(AttributeError):
        pair.adjustment = ()
    with pytest.raises(TypeError):
        recovery.SelectionAdjustmentPair(g, data, ("X",), ("Y",), (), (), population, pair.formula)


def test_mixed_selected_and_population_weights(recovery):
    g = graph(("W", "T", "X", "Y", "S"),
              (("W", "X"), ("W", "Y"), ("T", "Y"), ("T", "S"), ("X", "Y")))
    data, population = data_for(g), Probability("T")
    pair = recovery.find_selection_adjustment_pair(
        g, data, interventions="X", outcomes="Y", population_data=population)
    assert pair.adjustment == ("W", "T") and pair.population_adjustment == ("T",)
    assert_causal_formula(g, data, pair, population)
    assert not recovery.selection_adjustment_pair_is_valid(
        g, data, interventions="X", outcomes="Y", adjustment=("W", "T"),
        population_adjustment=("W", "T"), population_data=population)


def test_paper_figure_two_is_independently_adjudicated(recovery):
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/papers/correa_tian_bareinboim_2018_selection.json").read_text())
    g = graph(fixture["nodes"], fixture["directed"], fixture["bidirected"])
    data, population = data_for(g), Probability("Z2")
    expected = all_pairs(g.topology, {"S"}, {"X1", "X2"}, {"Y"}, {"Z2"})
    pairs = list(recovery.iter_selection_adjustment_pairs(
        g, data, interventions=("X1", "X2"), outcomes="Y", population_data=population))
    assert {(frozenset(p.adjustment), frozenset(p.population_adjustment)) for p in pairs} == expected
    assert (frozenset(("Z1", "Z2", "Z3")), frozenset(("Z2",))) in expected
    for pair in pairs:
        assert_causal_formula(g, data, pair, population, 81)


def test_rc_recovers_frontdoor_beyond_adjustment_and_binds_nuisance_parameters(recovery):
    g = graph(("Z", "X", "M", "Y", "S"),
              (("Z", "X"), ("X", "M"), ("M", "Y"), ("Z", "S")), (("X", "Y"),))
    data = data_for(g)
    assert isinstance(recovery.find_selection_adjustment_pair(
        g, data, interventions="X", outcomes="Y"), recovery.NoSelectionAdjustmentPair)
    result = recovery.recover_causal_effect(g, data, interventions="X", outcomes="Y")
    assert isinstance(result, recovery.RecoveredDistribution) and result.method == "rc"
    assert "Z" not in result.formula.free_variables
    for seed in (1, 20, 70):
        assert_causal_formula(g, data, result, seed=seed)


def test_selected_only_failure_does_not_claim_general_nonrecoverability(recovery):
    g = graph(("X", "Y", "S"), (("X", "Y"), ("Y", "S")))
    result = recovery.recover_causal_effect(g, data_for(g), interventions="X", outcomes="Y")
    assert isinstance(result, recovery.RecoveryNotEstablished)
    assert result.reason == "rc_failed"
    # Independent exact counterpair: same selected joint and sampling rate,
    # population causal risks 1/2 and 2/3. No unavailable population fallback.
    from fractions import Fraction as F
    first = (F(1, 2), F(2, 5), F(4, 5))
    second = (F(2, 3), F(3, 5), F(3, 5))
    def full_selected_and_rate(model):
        p, s0, s1 = model
        joint = {(x, y): F(1, 2) * (p if y else 1 - p) * (s1 if y else s0)
                 for x in (0, 1) for y in (0, 1)}
        rate = sum(joint.values())
        return {xy: value / rate for xy, value in joint.items()}, rate
    assert full_selected_and_rate(first) == full_selected_and_rate(second)
    assert full_selected_and_rate(first)[1] == F(3, 5)
    assert first[0] != second[0]


def test_rc_empty_interventions_recover_population_marginals(recovery):
    for bidirected in ((), (("Z", "S"), ("W", "Y"))):
        g = graph(("Z", "W", "Y", "S"), (("Z", "S"), ("W", "Y")), bidirected)
        data = data_for(g)
        result = recovery.recover_causal_effect(g, data, outcomes="Y")
        assert isinstance(result, recovery.RecoveredDistribution) and result.interventions == ()
        assert_causal_formula(g, data, result)
    g = graph(("Y", "S"), (("Y", "S"),))
    assert isinstance(recovery.recover_causal_effect(g, data_for(g), outcomes="Y"),
                      recovery.RecoveryNotEstablished)


def test_graph_and_input_orders_do_not_change_formula_truth(recovery):
    edges = (("W", "X"), ("W", "Y"), ("T", "Y"), ("T", "S"), ("X", "Y"))
    for order in (("W", "T", "X", "Y", "S"), ("Y", "S", "X", "T", "W")):
        g = graph(order, edges)
        data = Probability(("Y", "T", "W", "X"), conditioned_on="S", conditioned_values={"S": 0})
        population = Probability(("T", "W"))
        pairs = list(recovery.iter_selection_adjustment_pairs(
            g, data, interventions="X", outcomes="Y", population_data=population))
        assert {(frozenset(p.adjustment), frozenset(p.population_adjustment)) for p in pairs} == all_pairs(
            g.topology, {"S"}, {"X"}, {"Y"}, {"T", "W"})
        for pair in pairs:
            assert_causal_formula(g, data, pair, population)
    frontdoor = graph(("Y", "S", "M", "X", "Z"),
                      (("Z", "X"), ("X", "M"), ("M", "Y"), ("Z", "S")), (("X", "Y"),))
    data = Probability(("M", "Y", "Z", "X"), conditioned_on="S", conditioned_values={"S": 0})
    result = recovery.recover_causal_effect(frontdoor, data, outcomes="Y", interventions="X")
    assert isinstance(result, recovery.RecoveredDistribution)
    assert_causal_formula(frontdoor, data, result)


def test_formula_evaluator_refuses_unavailable_population_leaves():
    data = Probability(("X", "Y"), conditioned_on="S", conditioned_values={"S": 1})
    with pytest.raises(AssertionError, match="absent"):
        evaluate(Probability("Y", conditioned_on="X"), data, np.full((2, 2), .25))


def test_unconditional_inputs_dispatch_to_ordinary_id(recovery):
    g = graph(("X", "Y"), (("X", "Y"),))
    data = Probability(("X", "Y"))
    result = recovery.recover_causal_effect(g, data, interventions="X", outcomes="Y")
    assert isinstance(result, recovery.RecoveredDistribution) and result.method == "id"
    assert_causal_formula(g, data, result)


def test_constraints_and_budget_are_not_algorithmic_negative_results(recovery):
    g = graph(("Z", "X", "Y", "S"), (("X", "Y"),))
    data = data_for(g)
    pairs = list(recovery.iter_selection_adjustment_pairs(
        g, data, interventions="X", outcomes="Y", required="Z", allowed="Z"))
    assert [(p.adjustment, p.population_adjustment) for p in pairs] == [(("Z",), ())]
    with pytest.raises(recovery.RecoveryDomainError):
        recovery.find_selection_adjustment_pair(g, data, interventions="X", outcomes="Y", required="Z", allowed=())
    with pytest.raises(recovery.RecoveryBudgetExceeded):
        list(recovery.iter_selection_adjustment_pairs(g, data, interventions="X", outcomes="Y", max_states=1))
    with pytest.raises(recovery.RecoveryBudgetExceeded):
        recovery.recover_causal_effect(g, data, interventions="X", outcomes="Y", max_states=1)


@pytest.mark.parametrize("bad_data", [
    Probability("X", conditioned_on="S", conditioned_values={"S": 1}),
    Probability(("X", "Y"), conditioned_on="S"),
    Probability("Y", conditioned_on="S", interventions="X", conditioned_values={"S": 1}),
    Probability("Y", conditioned_on=("X", "S"), conditioned_values={"X": 0, "S": 1}),
])
def test_outside_admitted_source_domain_fails_closed(recovery, bad_data):
    g = graph(("X", "Y", "S"), (("X", "Y"),))
    with pytest.raises(recovery.RecoveryDomainError):
        recovery.recover_causal_effect(g, bad_data, interventions="X", outcomes="Y")


def test_conditioned_graph_vertex_is_ordinary_but_must_satisfy_the_paper_domain(recovery):
    g = graph(("X", "Y", "S"), (("S", "X"), ("X", "Y")))
    with pytest.raises(recovery.RecoveryDomainError, match="sink"):
        recovery.recover_causal_effect(g, data_for(g), interventions="X", outcomes="Y")
    raw = DirectedBidirectedGraph(nodes=("X", "Y", "S"), directed=(("X", "Y"),))
    with pytest.raises(recovery.RecoveryDomainError):
        recovery.recover_causal_effect(raw, Probability(("X", "Y"), conditioned_on="S", conditioned_values={"S": 1}),
                                       interventions="X", outcomes="Y")


def test_public_exports(recovery):
    public = importlib.import_module("causal_graph_algorithms")
    names = ("recover_conditional_distribution", "recover_causal_effect", "selection_adjustment_pair_is_valid",
             "find_selection_adjustment_pair", "iter_selection_adjustment_pairs", "SelectionAdjustmentPair",
             "NoSelectionAdjustmentPair", "RecoveredDistribution", "RecoveryNotEstablished",
             "RecoveryDomainError", "RecoveryBudgetExceeded")
    for name in names:
        assert getattr(public, name) is getattr(recovery, name)
        assert name in public.__all__
