from fractions import Fraction
from itertools import product

import pytest

from causal_graph_algorithms import AcyclicDirectedBidirectedGraph, DAG
from causal_graph_algorithms.counterfactual import (
    CounterfactualAtom, CounterfactualDomainError, CounterfactualEvent,
    CounterfactualTerm, InterventionAssignment, ValueLiteral,
)
from causal_graph_algorithms.counterfactual_identification import (
    IdentifiedCounterfactual, UndefinedCounterfactualCondition,
    UnresolvedCounterfactual, identify_counterfactual,
    identify_conditional_counterfactual,
)
from causal_graph_algorithms.counterfactual_probability import ValueConstant, ValueProbability
from tests.python.counterfactual_oracles import (
    binary_noise, event_probability, formula_value, response,
)


SPACES = {"X": ("0", "1"), "Y": ("0", "1")}


def term(variable, **assignments):
    return CounterfactualTerm(variable, tuple(
        InterventionAssignment(v, ValueLiteral(v, value))
        for v, value in assignments.items()))


def event(variable="Y", value="1", **assignments):
    return CounterfactualEvent((CounterfactualAtom(term(variable, **assignments),
                                                 ValueLiteral(variable, value)),))


def test_exact_sources_and_single_world_translation():
    graph = AcyclicDirectedBidirectedGraph.from_text("X --> Y; X <-> Y")
    query = event(X="0")
    assert isinstance(identify_counterfactual(graph, query=query, state_spaces=SPACES,
                      experiment_interventions=((),)), UnresolvedCounterfactual)
    result = identify_counterfactual(graph, query=query, state_spaces=SPACES,
                                     experiment_interventions=(("X",),))
    assert isinstance(result, IdentifiedCounterfactual)
    assert result.query == query and result.experiment_interventions == (("X",),)
    assert isinstance(result.formula, ValueProbability)
    assert result.formula.kernel.interventions == ("X",)
    assert isinstance(identify_counterfactual(DAG.from_text("X --> Y"), query=event(),
                      state_spaces=SPACES, experiment_interventions=()), UnresolvedCounterfactual)


def test_logical_zero_self_intervention_and_impossible_condition():
    graph = DAG.from_text("X --> Y")
    query = CounterfactualEvent((CounterfactualAtom(term("X", X="0"), ValueLiteral("X", "1")),))
    result = identify_counterfactual(graph, query=query, state_spaces=SPACES,
                                     experiment_interventions=())
    assert isinstance(result, IdentifiedCounterfactual) and result.formula == ValueConstant(0)
    result = identify_counterfactual(graph, query=event("X", "0", X="0"),
                                     state_spaces=SPACES, experiment_interventions=())
    assert result.formula == ValueConstant(1)
    assert isinstance(identify_conditional_counterfactual(graph, query=event(),
                      conditioned_on=query, state_spaces=SPACES,
                      experiment_interventions=((),)), UndefinedCounterfactualCondition)


def test_cross_world_condition_does_not_inherit_kernel_positivity():
    graph = DAG.from_text("X --> Y")
    result = identify_conditional_counterfactual(graph, query=event(X="0"),
        conditioned_on=event(X="1"), state_spaces=SPACES,
        experiment_interventions=((), ("X",)))
    assert isinstance(result, UnresolvedCounterfactual)
    assert "cross" in result.reason


def test_validation_conflicting_clamps_and_nested_route():
    with pytest.raises(CounterfactualDomainError):
        CounterfactualTerm("Y", (InterventionAssignment("X", ValueLiteral("X", "0")),
                                  InterventionAssignment("X", ValueLiteral("X", "1"))))
    with pytest.raises(CounterfactualDomainError):
        InterventionAssignment("X", ValueLiteral("Y", "0"))
    nested = CounterfactualTerm("Y", (InterventionAssignment("X", term("X", Y="0")),))
    query = CounterfactualEvent((CounterfactualAtom(nested, ValueLiteral("Y", "1")),))
    result = identify_counterfactual(DAG.from_text("X --> Y"), query=query,
                                     state_spaces=SPACES, experiment_interventions=((),))
    assert isinstance(result, UnresolvedCounterfactual)
    with pytest.raises(CounterfactualDomainError):
        identify_counterfactual(DAG.from_text("X --> Y"), query=event(value="2"),
                                 state_spaces=SPACES, experiment_interventions=((),))


def test_frontdoor_value_binding_against_direct_structural_evaluation():
    graph = AcyclicDirectedBidirectedGraph.from_text("X --> M --> Y; X <-> Y")
    spaces = {n: ("0", "1") for n in graph.topology.nodes}
    # Shared U0 confounds X/Y; independent U1/U2 make observational kernels positive.
    model = (("X", "M", "Y"), {
        "X": lambda v, u: str(u[0]),
        "M": lambda v, u: str(int(v["X"]) ^ u[1]),
        "Y": lambda v, u: str(int(v["M"]) ^ u[0] ^ u[2]),
    }, binary_noise((Fraction(1, 3), Fraction(1, 5), Fraction(1, 7))))
    for x, y in product(("0", "1"), repeat=2):
        query = event(value=y, X=x)
        result = identify_counterfactual(graph, query=query, state_spaces=spaces,
                                         experiment_interventions=((),))
        assert isinstance(result, IdentifiedCounterfactual)
        assert not result.formula.free_indices
        assert formula_value(result.formula, model,
                             experiment_interventions=((),)) == event_probability(query, model)


def test_conditional_single_world_ratio_against_original_event():
    graph = DAG.from_text("X --> Y")
    model = (("X", "Y"), {"X": lambda v, u: str(u[0]),
                           "Y": lambda v, u: str(int(v["X"]) ^ u[1])},
             binary_noise((Fraction(1, 3), Fraction(1, 7))))
    query, evidence = event(), event("X", "0")
    result = identify_conditional_counterfactual(graph, query=query,
        conditioned_on=evidence, state_spaces=SPACES, experiment_interventions=((),))
    joint = CounterfactualEvent(query.atoms + evidence.atoms)
    assert formula_value(result.formula, model, experiment_interventions=((),)) == (
        event_probability(joint, model) / event_probability(evidence, model))
    assert result.definedness.positive_condition == evidence


def test_positive_world_kernels_can_have_zero_cross_world_support():
    graph = DAG.from_text("X --> Y")
    model = (("X", "Y"), {"X": lambda v, u: str(u[0]),
                           "Y": lambda v, u: str(int(v["X"]) ^ u[1])},
             binary_noise((Fraction(1, 2), Fraction(1, 2))))
    evidence = CounterfactualEvent(event(X="0").atoms + event(X="1").atoms)
    assert event_probability(evidence, model) == 0
    assert event_probability(event(X="0"), model) == Fraction(1, 2)
    assert event_probability(event(X="1"), model) == Fraction(1, 2)
    result = identify_conditional_counterfactual(graph, query=event("X", "0"),
        conditioned_on=evidence, state_spaces=SPACES, experiment_interventions=((), ("X",)))
    assert isinstance(result, UnresolvedCounterfactual)


def test_value_formula_renderer_preserves_literal_and_bound_roles():
    import causal_graph_algorithms.counterfactual_probability as syntax
    graph = AcyclicDirectedBidirectedGraph.from_text("X --> M --> Y; X <-> Y")
    result = identify_counterfactual(graph, query=event(X="0"),
        state_spaces={v: ("0", "1") for v in graph.topology.nodes},
        experiment_interventions=((),))
    tex = syntax.value_expression_to_tex(result.formula)
    assert "\\sum" in tex and "\\textnormal{0}" in tex
    assert "i_{" in tex  # a bound index has its own name, distinct from X=0


def test_counterfactual_event_renderer_and_typed_index_validation():
    import causal_graph_algorithms.counterfactual as syntax
    from causal_graph_algorithms import Probability
    from causal_graph_algorithms.counterfactual_probability import ValueIndex, ValueSummation
    tex = syntax.counterfactual_event_to_tex(event(X="0"))
    assert "\\textnormal{Y}" in tex and "\\textnormal{X}" in tex
    with pytest.raises(CounterfactualDomainError):
        ValueProbability(Probability("Y"), (("Y", ValueLiteral("X", "0")),))
    index = ValueIndex("Y", ("0", "1"), 0)
    with pytest.raises(CounterfactualDomainError):
        ValueSummation((index,), ValueConstant(1))


def test_empty_index_alphabet_is_rejected_after_iterable_normalization():
    from causal_graph_algorithms.counterfactual_probability import ValueIndex

    with pytest.raises(CounterfactualDomainError, match="nonempty"):
        ValueIndex("Y", iter(()), 0)
    assert ValueIndex("Y", iter(("0", "1")), 0).states == ("0", "1")


def test_formula_oracle_rejects_unsupplied_exact_source_kernel():
    from causal_graph_algorithms import Probability

    model = (("X", "Y"), {"X": lambda v, u: str(u[0]),
                           "Y": lambda v, u: str(int(v["X"]) ^ u[1])},
             binary_noise((Fraction(1, 2), Fraction(1, 2))))
    formula = ValueProbability(Probability("Y", interventions="X"), (
        ("Y", ValueLiteral("Y", "1")), ("X", ValueLiteral("X", "0"))))
    # These sources are the test input, never metadata returned by the planner.
    with pytest.raises(ValueError, match="unsupplied"):
        formula_value(formula, model, experiment_interventions=((),))
    # Having do(X,Y) does not silently admit a different do(X) source either.
    with pytest.raises(ValueError, match="unsupplied"):
        formula_value(formula, model, experiment_interventions=(("X", "Y"),))
    assert formula_value(formula, model,
                         experiment_interventions=(("X",),)) == Fraction(1, 2)


def test_r79_conditional_split_counterpair_uses_the_original_events():
    # Review fixture: X->Y, private Y response rows have the same marginals.
    # Printed split gives 1/2; the original conditional events give 4/5 and 1/5.
    from causal_graph_algorithms import CounterfactualEvent
    graph = DAG.from_text("X --> Y")
    models = []
    for diagonal in (Fraction(2, 5), Fraction(1, 10)):
        weights = {(0, 0): diagonal, (1, 1): diagonal,
                   (0, 1): Fraction(1, 2) - diagonal,
                   (1, 0): Fraction(1, 2) - diagonal}
        noise = tuple(((x, a, b), weight / 2) for x in (0, 1)
                      for (a, b), weight in weights.items())
        models.append((("X", "Y"), {"X": lambda v, u: str(u[0]),
                       "Y": lambda v, u: str(u[1 + int(v["X"])])}, noise))
    for sources in ((), ("X",), ("Y",), ("X", "Y")):
        for clamps in product(("0", "1"), repeat=len(sources)):
            interventions = dict(zip(sources, clamps))
            outcomes = tuple(v for v in ("X", "Y") if v not in sources)
            for values in product(("0", "1"), repeat=len(outcomes)):
                probabilities = [sum((p for u, p in model[2] if all(
                    response(v, interventions, model, u) == value
                    for v, value in zip(outcomes, values))), Fraction()) for model in models]
                assert probabilities[0] == probabilities[1] and probabilities[0] > 0
    query, evidence = event(value="0", X="0"), event(value="0", X="1")
    joint = CounterfactualEvent(query.atoms + evidence.atoms)
    values = [event_probability(joint, m) / event_probability(evidence, m) for m in models]
    assert values == [Fraction(4, 5), Fraction(1, 5)]
    result = identify_conditional_counterfactual(graph, query=query, conditioned_on=evidence,
        state_spaces=SPACES, experiment_interventions=((), ("X",), ("Y",), ("X", "Y")))
    assert isinstance(result, UnresolvedCounterfactual)


def test_nested_cut_fixture_preserves_whole_event_cancellation():
    # On Z->A->B->Y, Y_(A_(Z=0), B)=Y pointwise by effectiveness/composition.
    graph = DAG.from_text("Z --> A --> B --> Y")
    nested = CounterfactualTerm("Y", (
        InterventionAssignment("A", term("A", Z="0")),
        InterventionAssignment("B", term("B"))))
    query = CounterfactualEvent((CounterfactualAtom(nested, ValueLiteral("Y", "1")),))
    natural = event()
    for z in (0, 1):
        for a, b, y in product(range(4), repeat=3):
            model = (("Z", "A", "B", "Y"), {
                "Z": lambda v, u, z=z: str(z),
                "A": lambda v, u, a=a: str((a >> int(v["Z"])) & 1),
                "B": lambda v, u, b=b: str((b >> int(v["A"])) & 1),
                "Y": lambda v, u, y=y: str((y >> int(v["B"])) & 1),
            }, (((), Fraction(1)),))
            assert event_probability(query, model) == event_probability(natural, model)
    result = identify_counterfactual(graph, query=query,
        state_spaces={v: ("0", "1") for v in graph.topology.nodes}, experiment_interventions=((),))
    # This implementation has not admitted the general nested normalization route.
    assert isinstance(result, UnresolvedCounterfactual)
