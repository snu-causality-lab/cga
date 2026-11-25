"""Paper figures, source contracts and exact checks for external-data IDSB."""
import copy
import importlib
import json
import os
from pathlib import Path
from itertools import combinations, product

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph, Probability,
    RecoveryBudgetExceeded, RecoveryDomainError, NoSelectionAdjustmentPair,
    find_selection_adjustment_pair,
)
from tests.python.external_recovery_oracles import (
    RationalSCM, assert_exact_formula, evaluate_sources,
)
from tests.python.selection_recovery_oracles import all_pairs


@pytest.fixture
def external():
    return importlib.import_module("causal_graph_algorithms.external_selection_recovery")


def paper_graph(name):
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/papers/"
        "correa_tian_bareinboim_2019_external.json").read_text())["figures"][name]
    raw = DirectedBidirectedGraph(nodes=fixture["nodes"], directed=fixture["directed"],
                                  bidirected=fixture["bidirected"])
    return AcyclicDirectedBidirectedGraph.from_graph(raw), Probability(fixture["population"])


def source(g, value=1):
    return Probability(tuple(v for v in g.topology.nodes if v != "S"),
                       conditioned_on="S", conditioned_values={"S": value})


def test_public_external_recovery_operation_is_available(external):
    assert callable(external.recover_causal_effect_with_external_data)


def test_factor_grouping_rejects_hidden_sinks_outside_ancestral_idsb_domain(external):
    raw = DirectedBidirectedGraph(nodes=("A", "H1", "H2", "C"),
        directed=(("H1", "H2"),), bidirected=(("A", "H1"), ("H2", "C")))
    # H2 is not an ancestor of retained A or C, hence these sets cannot be
    # D=An(Y)_(G\X), B⊆D\Y in Algorithm 3. Arbitrary latent projection can
    # split the original districts; Equation (12) needs the ancestral premise.
    with pytest.raises(RecoveryDomainError, match="ancestral"):
        external._factor_groups(raw, frozenset(raw.nodes), frozenset(("H1", "H2")))


def test_ancestral_hidden_collider_groups_match_actual_latent_projection(external):
    from causal_graph_algorithms import latent_projection
    raw = DirectedBidirectedGraph(nodes=("A", "H1", "H2", "C"),
        directed=(("H1", "H2"), ("H2", "A")),
        bidirected=(("A", "H1"), ("H2", "C")))
    g = AcyclicDirectedBidirectedGraph.from_graph(raw)
    hidden = frozenset(("H1", "H2"))
    projected = latent_projection(g, ("A", "C"))
    groups = external._factor_groups(raw, frozenset(raw.nodes), hidden)
    assert {c for c, _, _ in groups} == set(projected.districts())
    assert len(groups) == 1 and groups[0][0] == frozenset(("A", "C"))


@pytest.mark.parametrize("figure", ["2a", "2b", "3a"])
def test_actual_paper_figures_recover_and_evaluate_exactly(external, figure):
    g, population = paper_graph(figure)
    data = source(g)
    result = external.recover_causal_effect_with_external_data(
        g, data, outcomes="Y", interventions="X", population_data=population)
    assert isinstance(result, external.RecoveredExternalEffect)
    assert result.population_data == population and result.data == data
    assert result.method == "idsb"
    assert_exact_formula(g, data, result, population, seed=17)


def test_nonadjustment_paper_figure_and_both_external_margins(external):
    g, _ = paper_graph("2b")
    assert g.topology.has_directed("W1", "W2")
    assert g.topology.has_directed("W2", "X")
    assert not g.topology.has_directed("W2", "W1")
    for variable in ("W1", "W2"):
        population = Probability(variable)
        assert all_pairs(g.topology, {"S"}, {"X"}, {"Y"}, {variable}) == set()
        assert isinstance(find_selection_adjustment_pair(
            g, source(g), interventions="X", outcomes="Y", population_data=population),
            NoSelectionAdjustmentPair)
        result = external.recover_causal_effect_with_external_data(
            g, source(g), interventions="X", outcomes="Y", population_data=population)
        assert isinstance(result, external.RecoveredExternalEffect)
        assert_exact_formula(g, source(g), result, population, seed=3)


def test_no_population_source_is_not_implicitly_supplied(external):
    g = DAG.from_text("Z --> X --> Y; Z --> Y; Z --> S")
    data = source(g)
    unresolved = external.recover_causal_effect_with_external_data(
        g, data, outcomes="Y", interventions="X")
    assert isinstance(unresolved, external.ExternalRecoveryNotEstablished)
    assert unresolved.reason == "idsb_failed"
    assert unresolved.population_data is None
    result = external.recover_causal_effect_with_external_data(
        g, data, outcomes="Y", interventions="X", population_data=Probability("Z"))
    assert isinstance(result, external.RecoveredExternalEffect)
    assert_exact_formula(g, data, result, Probability("Z"))


def test_selected_only_success_and_empty_intervention(external):
    g = DAG.from_text("X --> Y; S")
    for intervention in ((), "X"):
        result = external.recover_causal_effect_with_external_data(
            g, source(g, 0), outcomes="Y", interventions=intervention)
        assert isinstance(result, external.RecoveredExternalEffect)
        assert_exact_formula(g, source(g, 0), result, None)


def test_full_population_source_uses_only_its_declared_joint(external):
    g, _ = paper_graph("2b")
    population = Probability(("Y", "X", "W1", "W2"))
    result = external.recover_causal_effect_with_external_data(
        g, source(g), outcomes="Y", interventions="X", population_data=population)
    assert isinstance(result, external.RecoveredExternalEffect)
    assert_exact_formula(g, source(g), result, population, seed=9)
    scm = RationalSCM(g.topology)
    with pytest.raises(AssertionError, match="unavailable source variables"):
        evaluate_sources(Probability(("X", "Y")), {"X": 0, "Y": 0},
                         source(g), scm.source(source(g)), Probability("W2"),
                         scm.source(Probability("W2")))


def test_external_recovery_is_deterministic_and_factory_sealed(external):
    g, population = paper_graph("2b")
    kwargs = dict(outcomes="Y", interventions="X", population_data=population)
    first = external.recover_causal_effect_with_external_data(g, source(g), **kwargs)
    assert first == external.recover_causal_effect_with_external_data(g, source(g), **kwargs)
    assert copy.copy(first) is first and copy.deepcopy(first) is first
    with pytest.raises(AttributeError):
        first.outcomes = ()
    with pytest.raises(TypeError):
        external.RecoveredExternalEffect(g, source(g), population, ("Y",), ("X",),
                                         first.formula, first.population_variables)
    with pytest.raises(RecoveryBudgetExceeded):
        external.recover_causal_effect_with_external_data(g, source(g), **kwargs, max_states=1)


@pytest.mark.parametrize("population", [Probability("unknown"), Probability("S"),
    Probability("X", conditioned_on="W2"), Probability("X", interventions="W2"),
    [Probability("W1"), Probability("W2")]])
def test_unsupported_or_unavailable_population_source_fails_closed(external, population):
    g, _ = paper_graph("2b")
    with pytest.raises(RecoveryDomainError):
        external.recover_causal_effect_with_external_data(
            g, source(g), outcomes="Y", interventions="X", population_data=population)


def test_unsupported_selected_inputs_and_queries(external):
    g, population = paper_graph("2b")
    bad = [Probability(g.topology.nodes),
           Probability(("W1", "W2", "X", "Y"), conditioned_on="S"),
           Probability(("X", "Y"), conditioned_on="S", conditioned_values={"S": 1})]
    for data in bad:
        with pytest.raises(RecoveryDomainError):
            external.recover_causal_effect_with_external_data(
                g, data, outcomes="Y", interventions="X", population_data=population)
    for kwargs in (dict(outcomes=()), dict(outcomes="Y", interventions="Y"),
                   dict(outcomes="S"), dict(outcomes="unknown")):
        with pytest.raises(RecoveryDomainError):
            external.recover_causal_effect_with_external_data(g, source(g), **kwargs)
    for limit in (0, -1, True, 1.5):
        with pytest.raises(RecoveryDomainError):
            external.recover_causal_effect_with_external_data(
                g, source(g), outcomes="Y", max_states=limit)


def test_old_selected_only_api_does_not_accept_external_data():
    from causal_graph_algorithms import recover_causal_effect
    g, population = paper_graph("2b")
    with pytest.raises(TypeError):
        recover_causal_effect(g, source(g), outcomes="Y", population_data=population)


def test_seeded_small_graph_positive_formulas_use_only_declared_sources(external):
    """Independent finite SCM replay; unresolved outputs are never called false."""
    import random
    rng = random.Random(201938)
    nodes = ("Z", "X", "Y", "S")
    possible_directed = tuple(combinations(nodes, 2))
    possible_bidirected = tuple(combinations(nodes, 2))
    successes = 0
    for index in range(16):
        directed = tuple(edge for edge in possible_directed if rng.randrange(2))
        bidirected = tuple(edge for edge in possible_bidirected if rng.randrange(4) == 0)
        raw = DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=bidirected)
        g = AcyclicDirectedBidirectedGraph.from_graph(raw)
        for population in (None, Probability("Z"), Probability(("Z", "Y"))):
            result = external.recover_causal_effect_with_external_data(
                g, source(g), outcomes="Y", interventions="X", population_data=population)
            if isinstance(result, external.RecoveredExternalEffect):
                successes += 1
                assert_exact_formula(g, source(g), result, population, seed=index)
            else:
                assert isinstance(result, external.ExternalRecoveryNotEstablished)
    assert successes > 0


@pytest.mark.skipif(os.environ.get("CGA_RUN_EXTERNAL_RECOVERY_EXHAUSTIVE") != "1",
                    reason="opt-in allocated-server scientific replay")
def test_all_two_observed_vertex_selection_graphs_and_population_margins(external):
    """All 96 labeled sink-S graphs, four sources and two queries: 768 rows."""
    nodes = ("X", "Y", "S")
    edges = tuple(combinations(nodes, 2))
    rows = 0
    for xy_edges in ((), (("X", "Y"),), (("Y", "X"),)):
        for sink_flags in product((False, True), repeat=2):
            directed = xy_edges + tuple((v, "S") for v, take in zip(("X", "Y"), sink_flags)
                                        if take)
            for bidirected_flags in product((False, True), repeat=len(edges)):
                bidirected = tuple(e for e, take in zip(edges, bidirected_flags) if take)
                raw = DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=bidirected)
                g = AcyclicDirectedBidirectedGraph.from_graph(raw)
                for margin in (None, Probability("X"), Probability("Y"), Probability(("X", "Y"))):
                    for x in ((), ("X",)):
                        rows += 1
                        result = external.recover_causal_effect_with_external_data(
                            g, source(g), outcomes="Y", interventions=x, population_data=margin)
                        if isinstance(result, external.RecoveredExternalEffect):
                            assert_exact_formula(g, source(g), result, margin, seed=19)
    assert rows == 768
