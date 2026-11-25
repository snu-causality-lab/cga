"""Fixed conditions belong to ordinary conditional probability inputs."""
import copy
import pickle
from typing import get_args

import pytest

from causal_graph_algorithms import (
    Probability, ProbabilityExpression, ProbabilityExpressionValidationError,
    Product, Ratio, Summation, probability_expression_to_tex,
)


def test_conditional_data_are_ordinary_probabilities_without_a_new_leaf_kind():
    source_values = {"context": 1}
    source = Probability(("X", "Y"), conditioned_on="context",
                         conditioned_values=source_values)
    source_values["context"] = 0
    assert type(source) is Probability
    assert get_args(ProbabilityExpression) == (Probability, Product, Summation, Ratio)
    assert source.conditioned_values == (("context", 1),)
    assert source.free_variables == frozenset(("X", "Y"))
    assert copy.copy(source) is source and copy.deepcopy(source) is source
    with pytest.raises(AttributeError):
        source.conditioned_values = ()
    with pytest.raises(TypeError):
        pickle.dumps(source)


def test_partial_fixed_conditions_keep_unfixed_variables_free():
    p = Probability("Y", conditioned_on=("X", "C"),
                    conditioned_values={"C": "available"})
    assert p.free_variables == frozenset(("X", "Y"))
    assert Summation("X", p).free_variables == frozenset(("Y",))
    with pytest.raises(ProbabilityExpressionValidationError):
        Summation("C", p)
    assert Probability("Y", conditioned_on="C").conditioned_values == ()
    assert Probability("Y", conditioned_on="C").free_variables == frozenset(("Y", "C"))


def test_fixed_events_are_part_of_equality_hash_and_rendering():
    first = Probability("Y", conditioned_on=("X", "S"), conditioned_values={"S": 1})
    second = Probability("Y", conditioned_on=("X", "S"), conditioned_values={"S": 0})
    assert first != second
    assert first != Probability("Y", conditioned_on=("X", "S"))
    assert len({first, second}) == 2
    same = Probability("Y", conditioned_on=("X", "S"), conditioned_values={"S": 1})
    assert first == same and hash(first) == hash(same)
    assert r"\textnormal{S}=1" in probability_expression_to_tex(first)
    assert r"\textnormal{S}=0" in probability_expression_to_tex(second)
    escaped = Probability("Y", conditioned_on="C_1", conditioned_values={"C_1": "a&b"})
    assert r"\textnormal{C\_1}=\textnormal{a\&b}" in probability_expression_to_tex(escaped)


def test_nested_bound_variables_do_not_rebind_a_fixed_condition():
    p = Probability("Y", conditioned_on=("X", "S"), conditioned_values={"S": 1})
    q = Probability("X", conditioned_on="S", conditioned_values={"S": 1})
    expression = Ratio(Summation("X", Product((p, q))), Probability("Y"))
    rendered = probability_expression_to_tex(expression)
    assert rendered.count(r"\textnormal{S}=1") == 2
    assert expression.free_variables == frozenset(("Y",))


@pytest.mark.parametrize("values", [
    {"unknown": 1}, {"Y": 1}, {"S": True}, {"S": 1.0},
    {"S": ""}, {"S": "\ud800"}, [("S", 1)],
])
def test_malformed_fixed_conditions_fail_closed(values):
    with pytest.raises(ProbabilityExpressionValidationError):
        Probability("Y", conditioned_on="S", conditioned_values=values)


def test_normalized_fixed_conditions_follow_the_declared_condition_order():
    p = Probability("Y", conditioned_on=("B", "A"),
                    conditioned_values={"A": "yes", "B": -2})
    assert p.conditioned_values == (("B", -2), ("A", "yes"))


def test_value_binding_cannot_silently_override_a_fixed_input_event():
    from causal_graph_algorithms import CounterfactualDomainError, ValueLiteral
    from causal_graph_algorithms.counterfactual_probability import ValueProbability
    p = Probability("Y", conditioned_on="S", conditioned_values={"S": 1})
    with pytest.raises(CounterfactualDomainError, match="fixed"):
        ValueProbability(p, (("Y", ValueLiteral("Y", "0")), ("S", ValueLiteral("S", "0"))))
