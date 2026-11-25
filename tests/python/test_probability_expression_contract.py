from __future__ import annotations

import copy
import pickle
import sys
from collections.abc import Callable
from typing import Any, get_args

import pytest

import causal_graph_algorithms as cga
import causal_graph_algorithms.probability as probability_module
from causal_graph_algorithms import (
    Probability,
    ProbabilityExpression,
    ProbabilityExpressionValidationError,
    Product,
    Ratio,
    Summation,
    probability_expression_to_tex,
)


def test_public_probability_expression_alias_is_the_closed_structural_union() -> None:
    assert get_args(ProbabilityExpression) == (
        Probability,
        Product,
        Summation,
        Ratio,
    )


def test_probability_normalizes_ordered_label_sequences_and_string_singletons() -> None:
    variables = ["Y", "M"]
    conditioned_on = ["X", "Z"]
    interventions = ["A", "B"]
    expression = Probability(
        variables,
        conditioned_on=conditioned_on,
        interventions=interventions,
    )

    variables.reverse()
    conditioned_on.clear()
    interventions.append("C")

    assert expression.variables == ("Y", "M")
    assert expression.conditioned_on == ("X", "Z")
    assert expression.interventions == ("A", "B")
    assert Probability("outcome").variables == ("outcome",)
    assert Probability("Y", conditioned_on="X").conditioned_on == ("X",)
    assert Probability("Y", interventions="X").interventions == ("X",)


def test_composite_expressions_copy_ordered_inputs_without_simplifying() -> None:
    first = Probability("X")
    second = Probability("Y", conditioned_on="X")
    factors: list[ProbabilityExpression] = [first, second]
    product = Product(factors)
    summed_variables = ["Y"]
    summation = Summation(summed_variables, product)
    ratio = Ratio(product, first)

    factors.reverse()
    summed_variables[0] = "changed"

    assert product.factors == (first, second)
    assert summation.variables == ("Y",)
    assert summation.body is product
    assert ratio.numerator is product
    assert ratio.denominator is first

    empty = Product(())
    singleton = Product((first,))
    self_ratio = Ratio(first, first)
    assert empty.factors == ()
    assert singleton.factors == (first,)
    assert singleton != first
    assert isinstance(self_ratio, Ratio)


def test_free_variables_follow_lexical_binding() -> None:
    conditional = Probability(
        ("Y", "M"),
        conditioned_on="X",
        interventions="Z",
    )
    product = Product((conditional, Probability("W")))
    summation = Summation(("M", "W"), product)
    ratio = Ratio(summation, Probability("Q", conditioned_on="R"))

    assert conditional.free_variables == frozenset({"Y", "M", "X", "Z"})
    assert product.free_variables == frozenset({"Y", "M", "X", "Z", "W"})
    assert summation.free_variables == frozenset({"Y", "X", "Z"})
    assert ratio.free_variables == frozenset({"Y", "X", "Z", "Q", "R"})
    assert Product(()).free_variables == frozenset()
    assert isinstance(ratio.free_variables, frozenset)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: Probability(()),
        lambda: Probability([]),
        lambda: Probability({"X"}),
        lambda: Probability(label for label in ("X",)),
        lambda: Probability("Y", conditioned_on={"X"}),
        lambda: Probability("Y", interventions=(label for label in ("X",))),
        lambda: Probability(("X", 1)),
        lambda: Probability(("X", "")),
        lambda: Probability(("X", "\ud800")),
        lambda: Probability(("X", "X")),
        lambda: Probability("Y", conditioned_on=("",)),
        lambda: Probability("Y", conditioned_on=("X", 1)),
        lambda: Probability("Y", conditioned_on=("X", "X")),
        lambda: Probability("Y", interventions=("\ud800",)),
        lambda: Probability("Y", interventions=("X", 1)),
        lambda: Probability("Y", interventions=("X", "X")),
        lambda: Probability("X", conditioned_on="X"),
        lambda: Probability("X", interventions="X"),
        lambda: Probability("Y", conditioned_on="X", interventions="X"),
    ],
)
def test_probability_rejects_invalid_or_overlapping_roles(
    constructor: Callable[[], object],
) -> None:
    with pytest.raises(ProbabilityExpressionValidationError):
        constructor()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: Product(""),
        lambda: Product("XY"),
        lambda: Product(Probability("X")),
        lambda: Product({Probability("X")}),
        lambda: Product(iter((Probability("X"),))),
        lambda: Product((Probability("X"), object())),
        lambda: Summation((), Probability("X")),
        lambda: Summation({"X"}, Probability("X")),
        lambda: Summation((label for label in ("X",)), Probability("X")),
        lambda: Summation(("X", "X"), Probability("X")),
        lambda: Summation("", Probability("X")),
        lambda: Summation("\ud800", Probability("X")),
        lambda: Summation(("X", 1), Probability("X")),
        lambda: Summation("missing", Probability("X")),
        lambda: Summation("X", object()),
        lambda: Ratio(object(), Probability("X")),
        lambda: Ratio(Probability("X"), object()),
    ],
)
def test_composites_reject_unordered_nonsequence_or_invalid_inputs(
    constructor: Callable[[], object],
) -> None:
    with pytest.raises(ProbabilityExpressionValidationError):
        constructor()


def test_probability_expression_validation_error_is_a_value_error() -> None:
    assert issubclass(ProbabilityExpressionValidationError, ValueError)


def test_expression_equality_and_hashing_are_structural_and_order_sensitive() -> None:
    first = Probability(
        ("Y", "M"),
        conditioned_on=("X", "Z"),
        interventions=("A", "B"),
    )
    same = Probability(
        ["Y", "M"],
        conditioned_on=["X", "Z"],
        interventions=["A", "B"],
    )
    reordered = Probability(
        ("M", "Y"),
        conditioned_on=("X", "Z"),
        interventions=("A", "B"),
    )

    assert first == same
    assert hash(first) == hash(same)
    assert first != reordered
    assert Product((first, reordered)) != Product((reordered, first))
    assert Summation(("Y", "M"), first) != Summation(("M", "Y"), first)
    assert Ratio(first, reordered) != Ratio(reordered, first)


@pytest.mark.parametrize(
    "base",
    [Probability, Product, Summation, Ratio],
)
def test_expression_value_classes_are_sealed_and_slotted(base: type[object]) -> None:
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("MutableExpressionSubclass", (base,), {})

    if base is Probability:
        value: ProbabilityExpression = Probability("X")
    elif base is Product:
        value = Product((Probability("X"),))
    elif base is Summation:
        value = Summation("X", Probability("X"))
    else:
        value = Ratio(Probability("X"), Probability("Y"))
    assert not hasattr(value, "__dict__")
    expected_slots = {
        Probability: ("variables", "conditioned_on", "interventions", "conditioned_values"),
        Product: ("factors",),
        Summation: ("variables", "body"),
        Ratio: ("numerator", "denominator"),
    }
    assert base.__slots__ == expected_slots[base]


def test_expression_values_are_frozen_copy_identity_and_not_pickle_stateful() -> None:
    probability = Probability("Y", conditioned_on="X")
    product = Product((probability,))
    summation = Summation("Y", product)
    ratio = Ratio(summation, Probability("X"))

    for value in (probability, product, summation, ratio):
        original_hash = hash(value)
        assert copy.copy(value) is value
        assert copy.deepcopy(value) is value
        assert hash(value) == original_hash
        assert not hasattr(value, "__setstate__")
        with pytest.raises(TypeError, match="pickle is unsupported"):
            pickle.dumps(value)
        with pytest.raises((AttributeError, TypeError)):
            value.free_variables = frozenset()  # type: ignore[misc]


def test_public_surface_has_no_expression_evaluator_or_simplifier() -> None:
    assert not hasattr(cga, "evaluate_probability_expression")
    assert not hasattr(cga, "simplify_probability_expression")
    for value in (
        Probability("X"),
        Product(()),
        Summation("X", Probability("X")),
        Ratio(Probability("X"), Probability("Y")),
    ):
        assert not hasattr(value, "evaluate")
        assert not hasattr(value, "simplify")
        assert not hasattr(value, "distribution")
        assert not hasattr(value, "children")
        assert not hasattr(value, "updated")


def test_free_variables_scale_with_unique_shared_expression_nodes() -> None:
    expression: ProbabilityExpression = Probability("X")
    for _ in range(24):
        expression = Product((expression, expression))

    free_variable_line_events = 0

    def trace_free_variable_work(
        frame: Any, event: str, argument: Any
    ) -> Callable[[Any, str, Any], object] | None:
        del argument
        nonlocal free_variable_line_events
        if (
            frame.f_code is probability_module._free_variables.__code__
            and event == "line"
        ):
            free_variable_line_events += 1
            if free_variable_line_events > 5_000:
                raise AssertionError(
                    "free-variable analysis repeated shared expression occurrences"
                )
        return trace_free_variable_work

    previous_trace = sys.gettrace()
    sys.settrace(trace_free_variable_work)
    try:
        assert expression.free_variables == frozenset({"X"})
    finally:
        sys.settrace(previous_trace)

    assert free_variable_line_events < 5_000


def test_probability_tex_contract_for_all_three_roles_and_order() -> None:
    assert probability_expression_to_tex(Probability("Y")) == (
        r"P\!\left(\textnormal{Y}\right)"
    )
    assert probability_expression_to_tex(
        Probability(("Y", "M"), conditioned_on=("X", "Z"))
    ) == (
        r"P\!\left(\textnormal{Y},\,\textnormal{M} \mid "
        r"\textnormal{X},\,\textnormal{Z}\right)"
    )
    assert probability_expression_to_tex(
        Probability(
            ("Y", "M"),
            conditioned_on=("X", "Z"),
            interventions=("A", "B"),
        )
    ) == (
        r"P_{\textnormal{A},\,\textnormal{B}}\!\left("
        r"\textnormal{Y},\,\textnormal{M} \mid "
        r"\textnormal{X},\,\textnormal{Z}\right)"
    )


def test_composite_tex_contract_has_no_implicit_simplification() -> None:
    probability_x = Probability("X")
    probability_y_given_x = Probability("Y", conditioned_on="X")

    assert probability_expression_to_tex(Product(())) == "1"
    assert probability_expression_to_tex(
        Product((probability_x, probability_y_given_x))
    ) == (
        r"P\!\left(\textnormal{X}\right)\,"
        r"P\!\left(\textnormal{Y} \mid \textnormal{X}\right)"
    )
    assert probability_expression_to_tex(
        Summation("Y", probability_y_given_x)
    ) == (
        r"\sum_{\textnormal{Y}}\left["
        r"P\!\left(\textnormal{Y} \mid \textnormal{X}\right)\right]"
    )
    assert probability_expression_to_tex(
        Ratio(probability_y_given_x, probability_x)
    ) == (
        r"\frac{P\!\left(\textnormal{Y} \mid \textnormal{X}\right)}"
        r"{P\!\left(\textnormal{X}\right)}"
    )

    two_variable_sum = Summation(
        ("W", "M"),
        Product((Probability("W"), Probability("M"))),
    )
    assert probability_expression_to_tex(two_variable_sum) == (
        r"\sum_{\textnormal{W},\,\textnormal{M}}\left["
        r"P\!\left(\textnormal{W}\right)\,"
        r"P\!\left(\textnormal{M}\right)\right]"
    )


def test_intervention_tex_without_conditioning_and_under_a_lexical_binder() -> None:
    assert probability_expression_to_tex(
        Probability("Y", interventions="A")
    ) == r"P_{\textnormal{A}}\!\left(\textnormal{Y}\right)"

    expression = Product(
        (
            Probability("Z", interventions="A"),
            Summation("A", Probability("Y", interventions="A")),
        )
    )
    assert probability_expression_to_tex(expression) == (
        r"P_{\textnormal{A}}\!\left(\textnormal{Z}\right)\,"
        r"\sum_{\textnormal{A}^{\prime}}\left["
        r"P_{\textnormal{A}^{\prime}}\!\left(\textnormal{Y}\right)"
        r"\right]"
    )


def test_tex_renderer_uses_lexical_primes_for_shadowed_labels() -> None:
    frontdoor = Product(
        (
            Probability("Z", conditioned_on="X"),
            Summation(
                "X",
                Product(
                    (
                        Probability("Y", conditioned_on=("X", "Z")),
                        Probability("X"),
                    )
                ),
            ),
        )
    )

    assert probability_expression_to_tex(frontdoor) == (
        r"P\!\left(\textnormal{Z} \mid \textnormal{X}\right)\,"
        r"\sum_{\textnormal{X}^{\prime}}\left["
        r"P\!\left(\textnormal{Y} \mid "
        r"\textnormal{X}^{\prime},\,\textnormal{Z}\right)\,"
        r"P\!\left(\textnormal{X}^{\prime}\right)\right]"
    )


def test_tex_renderer_handles_nested_shadowing_and_copies_sibling_scopes() -> None:
    nested = Summation(
        "X",
        Product(
            (
                Probability("X"),
                Summation(
                    "X",
                    Product(
                        (
                            Probability("X"),
                            Summation("X", Probability("X")),
                        )
                    ),
                ),
            )
        ),
    )
    siblings = Product(
        (
            Summation("X", Probability("X")),
            Summation("X", Probability("X")),
        )
    )

    assert probability_expression_to_tex(nested) == (
        r"\sum_{\textnormal{X}}\left["
        r"P\!\left(\textnormal{X}\right)\,"
        r"\sum_{\textnormal{X}^{\prime}}\left["
        r"P\!\left(\textnormal{X}^{\prime}\right)\,"
        r"\sum_{\textnormal{X}^{\prime\prime}}\left["
        r"P\!\left(\textnormal{X}^{\prime\prime}\right)\right]"
        r"\right]\right]"
    )
    assert probability_expression_to_tex(siblings) == (
        r"\sum_{\textnormal{X}}\left["
        r"P\!\left(\textnormal{X}\right)\right]\,"
        r"\sum_{\textnormal{X}}\left["
        r"P\!\left(\textnormal{X}\right)\right]"
    )


def test_tex_renderer_escapes_labels_without_interpreting_tex() -> None:
    label = "A \\{}#$%&_^~\t\n\x00\u200d\u00a0α"
    expression = Probability(label)

    assert probability_expression_to_tex(expression) == (
        r"P\!\left(\textnormal{A\ \textbackslash{}\{\}\#\$\%\&\_"
        r"\textasciicircum{}\textasciitilde{}"
        r"\texttt{U+0009}\texttt{U+000A}\texttt{U+0000}"
        r"\texttt{U+200D}\texttt{U+00A0}α}\right)"
    )


def test_tex_renderer_escapes_conditioning_intervention_and_binder_labels() -> None:
    expression = Summation(
        "Y_1",
        Probability(
            "Y_1",
            conditioned_on="X#1",
            interventions="A$1",
        ),
    )

    assert probability_expression_to_tex(expression) == (
        r"\sum_{\textnormal{Y\_1}}\left["
        r"P_{\textnormal{A\$1}}\!\left(\textnormal{Y\_1} \mid "
        r"\textnormal{X\#1}\right)\right]"
    )


def test_expression_operations_do_not_recurse_with_tree_depth() -> None:
    first: ProbabilityExpression = Probability("X")
    second: ProbabilityExpression = Probability("X")
    denominator = Probability("Z")
    for _ in range(500):
        first = Ratio(first, denominator)
        second = Ratio(second, denominator)

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        assert first == second
        assert hash(first) == hash(second)
        assert first.free_variables == frozenset({"X", "Z"})
        assert probability_expression_to_tex(first) == (
            probability_expression_to_tex(second)
        )
    finally:
        sys.setrecursionlimit(previous_limit)


@pytest.mark.parametrize("invalid", [None, 1, object()])
def test_tex_renderer_rejects_nonexpressions(invalid: Any) -> None:
    with pytest.raises(TypeError, match="TeX rendering requires"):
        probability_expression_to_tex(invalid)
