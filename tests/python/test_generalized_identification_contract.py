from __future__ import annotations

import copy
import inspect
import itertools
import json
import pickle
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, get_args

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GeneralizedIdentificationResult,
    GeneralizedNonIdentificationWitness,
    GraphValidationError,
    HedgeWitness,
    IdentifiedEffect,
    IdentifiedGeneralizedEffect,
    IdentificationDomainError,
    NonIdentifiedEffect,
    NonIdentifiedGeneralizedEffect,
    Probability,
    ProbabilityExpression,
    Product,
    Ratio,
    Summation,
    identify_causal_effect,
    identify_generalized_causal_effect,
    probability_expression_to_tex,
)

from .gid_oracles import (
    all_intervention_targets,
    evaluate_generalized_probability_expression,
    gid_local_problem,
    minimal_finite_formula_cases_n3,
    n3_gid_query_analyses,
)
from .id_oracles import (
    TinyADMG,
    all_bow_admgs_n3,
    all_nonempty_outcome_queries_n3,
    exact_binary_interventional_probability,
)


DOMAIN_MESSAGE = (
    "generalized causal-effect identification requires an acyclic "
    "directed-bidirected or DAG view"
)
EXPERIMENT_COLLECTION_MESSAGE = (
    "experiment interventions must be an iterable of intervention sets"
)
OVERLAP_MESSAGE = "outcomes and interventions must be disjoint"


def _adbg(
    *,
    nodes: tuple[str, ...] | list[str],
    directed: tuple[tuple[str, str], ...] | list[list[str]] = (),
    bidirected: tuple[tuple[str, str], ...] | list[list[str]] = (),
) -> AcyclicDirectedBidirectedGraph:
    return AcyclicDirectedBidirectedGraph.from_graph(
        DirectedBidirectedGraph(
            nodes=nodes,
            directed=directed,
            bidirected=bidirected,
        )
    )


def _dag(
    *,
    nodes: tuple[str, ...] | list[str],
    directed: tuple[tuple[str, str], ...] | list[list[str]] = (),
) -> DAG:
    return DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=directed))


def _load_fixture(name: str) -> dict[str, Any]:
    path = Path(__file__).parents[1] / "fixtures" / "papers" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_expression(
    expression: ProbabilityExpression,
) -> tuple[ProbabilityExpression, ...]:
    walked: list[ProbabilityExpression] = []
    visited: set[int] = set()
    pending = [expression]
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        walked.append(current)
        if isinstance(current, Product):
            pending.extend(reversed(current.factors))
        elif isinstance(current, Summation):
            pending.append(current.body)
        elif isinstance(current, Ratio):
            pending.append(current.denominator)
            pending.append(current.numerator)
    return tuple(walked)


def _evaluate_explicit_source_expression(
    expression: ProbabilityExpression,
    *,
    node_order: tuple[str, ...],
    domains: dict[str, tuple[int, ...]],
    source_tables: dict[
        tuple[str, ...],
        dict[tuple[int, ...], dict[tuple[int, ...], Fraction]],
    ],
    environment: dict[str, int],
) -> Fraction:
    """Evaluate the public IR on hand-authored finite source distributions."""

    if isinstance(expression, Probability):
        source = expression.interventions
        if source not in source_tables:
            raise AssertionError("formula atom uses an unavailable explicit source")
        action_values = tuple(environment[label] for label in source)
        table = source_tables[source][action_values]
        available = tuple(label for label in node_order if label not in source)
        positions = {label: index for index, label in enumerate(available)}

        def event_probability(labels: tuple[str, ...]) -> Fraction:
            return sum(
                (
                    probability
                    for assignment, probability in table.items()
                    if all(
                        assignment[positions[label]] == environment[label]
                        for label in labels
                    )
                ),
                Fraction(0),
            )

        numerator = event_probability(expression.variables + expression.conditioned_on)
        if not expression.conditioned_on:
            return numerator
        denominator = event_probability(expression.conditioned_on)
        assert denominator > 0
        return numerator / denominator

    if isinstance(expression, Product):
        value = Fraction(1)
        for factor in expression.factors:
            value *= _evaluate_explicit_source_expression(
                factor,
                node_order=node_order,
                domains=domains,
                source_tables=source_tables,
                environment=environment,
            )
        return value

    if isinstance(expression, Summation):
        value = Fraction(0)
        summation_domains = tuple(domains[label] for label in expression.variables)
        for assignment in itertools.product(*summation_domains):
            bound = dict(environment)
            bound.update(zip(expression.variables, assignment, strict=True))
            value += _evaluate_explicit_source_expression(
                expression.body,
                node_order=node_order,
                domains=domains,
                source_tables=source_tables,
                environment=bound,
            )
        return value

    assert isinstance(expression, Ratio)
    denominator = _evaluate_explicit_source_expression(
        expression.denominator,
        node_order=node_order,
        domains=domains,
        source_tables=source_tables,
        environment=environment,
    )
    assert denominator > 0
    return (
        _evaluate_explicit_source_expression(
            expression.numerator,
            node_order=node_order,
            domains=domains,
            source_tables=source_tables,
            environment=environment,
        )
        / denominator
    )


def _assert_formula_uses_only_exact_supplied_sources(
    result: IdentifiedGeneralizedEffect,
) -> None:
    topology = result.graph.topology
    source_set = frozenset(result.experiment_interventions)
    requested = set(result.outcomes) | set(result.interventions)
    assert set(result.outcomes) <= result.formula.free_variables
    assert result.formula.free_variables <= requested

    for expression in _walk_expression(result.formula):
        if isinstance(expression, Probability):
            assert expression.interventions in source_set
            assert (
                set(expression.variables)
                | set(expression.conditioned_on)
                | set(expression.interventions)
            ) <= set(topology.nodes)
        elif isinstance(expression, Summation):
            assert set(expression.variables) <= set(topology.nodes)


def _tiny_to_public(
    graph: TinyADMG,
    labels: tuple[str, ...],
) -> AcyclicDirectedBidirectedGraph:
    node_labels = dict(zip(graph.nodes, labels, strict=True))
    return _adbg(
        nodes=labels,
        directed=tuple(
            (node_labels[parent], node_labels[child])
            for parent, child in sorted(graph.directed)
        ),
        bidirected=tuple(
            (node_labels[left], node_labels[right])
            for left, right in sorted(graph.bidirected)
        ),
    )


def _ordered_labels(
    labels: tuple[str, ...], vertices: frozenset[int]
) -> tuple[str, ...]:
    return tuple(label for index, label in enumerate(labels) if index in vertices)


def _assert_source_hedge_matches_oracle(
    actual: HedgeWitness,
    *,
    full_labels: tuple[str, ...],
    expected_graph: TinyADMG,
    expected_outcomes: frozenset[int],
    expected_interventions: frozenset[int],
    expected_hedge: Any,
) -> None:
    expected_nodes = _ordered_labels(full_labels, frozenset(expected_graph.nodes))
    assert actual.graph.topology.nodes == expected_nodes
    assert actual.graph.topology.directed == frozenset(
        (full_labels[parent], full_labels[child])
        for parent, child in expected_graph.directed
    )
    assert actual.graph.topology.bidirected == frozenset(
        (full_labels[left], full_labels[right])
        for left, right in expected_graph.bidirected
    )
    assert actual.outcomes == _ordered_labels(full_labels, expected_outcomes)
    assert actual.interventions == _ordered_labels(full_labels, expected_interventions)
    assert actual.inner_vertices == _ordered_labels(
        full_labels, expected_hedge.inner.nodes
    )
    assert actual.outer_vertices == _ordered_labels(
        full_labels, expected_hedge.outer.nodes
    )
    assert actual.roots == _ordered_labels(full_labels, expected_hedge.inner.roots)
    assert actual.fixing_sequence == tuple(
        full_labels[node] for node in expected_hedge.fixing_order
    )


def test_expression_walk_visits_shared_subexpressions_once_by_identity() -> None:
    shared = Probability("Y")
    equal_but_distinct = Probability("Y")
    expression = Product((shared, shared, equal_but_distinct))

    walked = _walk_expression(expression)

    assert len(walked) == 3
    assert walked[0] is expression
    assert walked[1] is shared
    assert walked[2] is equal_but_distinct


def test_generalized_public_result_alias_and_domain_error_type() -> None:
    assert get_args(GeneralizedIdentificationResult) == (
        IdentifiedGeneralizedEffect,
        NonIdentifiedGeneralizedEffect,
    )
    assert issubclass(IdentificationDomainError, ValueError)


@pytest.mark.parametrize(
    "invalid_graph",
    [
        DirectedBidirectedGraph(nodes=("X", "Y"), directed=(("X", "Y"),)),
        EndpointMarkedGraph(nodes=("X", "Y")),
        object(),
        None,
    ],
)
def test_generalized_identification_rejects_raw_and_wrong_graph_views(
    invalid_graph: object,
) -> None:
    with pytest.raises(
        IdentificationDomainError,
        match=f"^{re.escape(DOMAIN_MESSAGE)}$",
    ):
        identify_generalized_causal_effect(  # type: ignore[arg-type]
            invalid_graph,
            outcomes="Y",
            interventions="X",
            experiment_interventions=(),
        )


def test_generalized_graph_domain_precedes_every_input_consumption() -> None:
    class ExplodingIterable:
        def __iter__(self) -> Any:
            raise AssertionError("input must not be consumed")

    raw = DirectedBidirectedGraph(nodes=("X", "Y"))
    with pytest.raises(IdentificationDomainError, match="^generalized causal-effect"):
        identify_generalized_causal_effect(
            raw,  # type: ignore[arg-type]
            outcomes=ExplodingIterable(),  # type: ignore[arg-type]
            interventions=ExplodingIterable(),  # type: ignore[arg-type]
            experiment_interventions=ExplodingIterable(),  # type: ignore[arg-type]
        )


def test_generalized_signature_is_keyword_only_and_requires_experiments() -> None:
    signature = inspect.signature(identify_generalized_causal_effect)
    assert tuple(signature.parameters) == (
        "graph",
        "outcomes",
        "experiment_interventions",
        "interventions",
    )
    assert signature.parameters["graph"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["graph"].default is inspect.Parameter.empty
    for required in ("outcomes", "experiment_interventions"):
        assert signature.parameters[required].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[required].default is inspect.Parameter.empty
    assert signature.parameters["interventions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["interventions"].default == ()

    graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    with pytest.raises(TypeError):
        identify_generalized_causal_effect(  # type: ignore[misc]
            graph, "Y", (), "X"
        )
    with pytest.raises(TypeError):
        identify_generalized_causal_effect(  # type: ignore[call-arg]
            graph,
            outcomes="Y",
        )
    for unsupported in (
        {"assume_positive": True},
        {"values": {"X": 0}},
        {"max_experiments": 1},
        {"max_vertices": 3},
    ):
        with pytest.raises(TypeError):
            identify_generalized_causal_effect(  # type: ignore[call-arg]
                graph,
                outcomes="Y",
                experiment_interventions=(),
                **unsupported,
            )


@pytest.mark.parametrize("invalid", ["X", None, 1, object()])
def test_outer_experiment_collection_rejects_bare_string_and_noniterable(
    invalid: object,
) -> None:
    graph = _dag(nodes=("X", "Y"))
    with pytest.raises(
        IdentificationDomainError,
        match=f"^{EXPERIMENT_COLLECTION_MESSAGE}$",
    ):
        identify_generalized_causal_effect(
            graph,
            outcomes="Y",
            experiment_interventions=invalid,  # type: ignore[arg-type]
        )


def test_validation_order_is_y_then_x_then_complete_experiments_then_overlap() -> None:
    graph = _dag(nodes=("X", "Y"))

    class ExplodingExperiments:
        def __iter__(self) -> Any:
            raise AssertionError("experiments must not be consumed yet")

    with pytest.raises(GraphValidationError, match="unknown vertex: 'bad-y'"):
        identify_generalized_causal_effect(
            graph,
            outcomes="bad-y",
            interventions="bad-x",
            experiment_interventions=ExplodingExperiments(),  # type: ignore[arg-type]
        )
    with pytest.raises(GraphValidationError, match="unknown vertex: 'bad-x'"):
        identify_generalized_causal_effect(
            graph,
            outcomes="Y",
            interventions="bad-x",
            experiment_interventions=ExplodingExperiments(),  # type: ignore[arg-type]
        )

    consumed: list[str] = []

    def experiments() -> Any:
        consumed.append("first")
        yield "X"
        consumed.append("second")
        yield ("Y", "Y")
        consumed.append("finished")

    with pytest.raises(
        IdentificationDomainError,
        match=f"^{OVERLAP_MESSAGE}$",
    ):
        identify_generalized_causal_effect(
            graph,
            outcomes=("X", "X"),
            interventions="X",
            experiment_interventions=experiments(),
        )
    assert consumed == ["first", "second", "finished"]

    with pytest.raises(GraphValidationError, match="unknown vertex: 'bad-source'"):
        identify_generalized_causal_effect(
            graph,
            outcomes="X",
            interventions="X",
            experiment_interventions=(("bad-source",),),
        )


def test_inner_experiment_targets_use_existing_graph_validation_errors() -> None:
    graph = _dag(nodes=("X", "Y"))
    with pytest.raises(GraphValidationError, match="seed vertices must be iterable"):
        identify_generalized_causal_effect(
            graph,
            outcomes="Y",
            experiment_interventions=(None,),  # type: ignore[arg-type]
        )
    with pytest.raises(GraphValidationError, match="unknown vertex: 1"):
        identify_generalized_causal_effect(
            graph,
            outcomes="Y",
            experiment_interventions=((1,),),  # type: ignore[list-item]
        )


def test_experiment_collection_is_canonical_deduplicated_and_source_owned() -> None:
    graph = _dag(nodes=("B", "A", "C"))
    first_target = ["C", "B", "C"]
    outer = [first_target, "A", (), ("B",), ("C", "B"), ("A", "A")]
    result = identify_generalized_causal_effect(
        graph,
        outcomes=(),
        experiment_interventions=outer,
    )
    first_target.clear()
    outer.clear()

    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert result.graph is graph
    assert result.outcomes == ()
    assert result.interventions == ()
    assert result.experiment_interventions == (
        (),
        ("B",),
        ("A",),
        ("B", "C"),
    )
    assert result.formula == Product(())

    permuted = identify_generalized_causal_effect(
        graph,
        outcomes=(),
        experiment_interventions={
            frozenset(("A",)),
            frozenset(("C", "B")),
            frozenset(),
            frozenset(("B",)),
        },
    )
    assert result == permuted
    assert hash(result) == hash(permuted)


def test_query_selectors_and_nonempty_source_permutations_are_canonical() -> None:
    selector_graph = _dag(nodes=("B", "A", "C"))
    selector_result = identify_generalized_causal_effect(
        selector_graph,
        outcomes=("C", "A", "C"),
        interventions=(node for node in ("B", "B")),
        experiment_interventions=((),),
    )
    assert isinstance(selector_result, IdentifiedGeneralizedEffect)
    assert selector_result.outcomes == ("A", "C")
    assert selector_result.interventions == ("B",)

    source_graph = _dag(nodes=("X", "Z", "Y"))
    reversed_sources = identify_generalized_causal_effect(
        source_graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(("Z",), ("X",), ("Z", "Z")),
    )
    set_sources = identify_generalized_causal_effect(
        source_graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions={frozenset(("X",)), frozenset(("Z",))},
    )
    assert isinstance(reversed_sources, IdentifiedGeneralizedEffect)
    assert reversed_sources.experiment_interventions == (("X",), ("Z",))
    assert reversed_sources == set_sources
    assert hash(reversed_sources) == hash(set_sources)


def test_empty_outcomes_are_unit_only_after_complete_experiment_validation() -> None:
    graph = _adbg(
        nodes=("X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    with pytest.raises(GraphValidationError, match="unknown vertex: 'missing'"):
        identify_generalized_causal_effect(
            graph,
            outcomes=(),
            interventions="X",
            experiment_interventions=(("X",), ("missing",)),
        )

    result = identify_generalized_causal_effect(
        graph,
        outcomes=(),
        interventions="X",
        experiment_interventions=(("X", "Y"),),
    )
    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert result.graph is graph
    assert result.outcomes == ()
    assert result.interventions == ("X",)
    assert result.experiment_interventions == (("X", "Y"),)
    assert result.formula == Product(())


def test_empty_collection_and_observational_source_are_not_ambiguous() -> None:
    graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    no_sources = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(),
    )
    observational = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=((),),
    )

    assert isinstance(no_sources, NonIdentifiedGeneralizedEffect)
    assert no_sources.witness.experiment_interventions == ()
    assert no_sources.witness.source_hedges == ()
    assert isinstance(observational, IdentifiedGeneralizedEffect)
    assert observational.experiment_interventions == ((),)
    assert observational.formula == Probability("Y", conditioned_on="X")
    _assert_formula_uses_only_exact_supplied_sources(observational)


def test_outer_strings_are_singleton_sources_but_one_inner_tuple_is_joint() -> None:
    graph = _dag(nodes=("X", "Y"))
    singleton_sources = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=["X", "Y"],
    )
    joint_source = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=[("X", "Y")],
    )

    assert isinstance(singleton_sources, IdentifiedGeneralizedEffect)
    assert singleton_sources.experiment_interventions == (("X",), ("Y",))
    assert isinstance(joint_source, NonIdentifiedGeneralizedEffect)
    assert joint_source.witness.experiment_interventions == (("X", "Y"),)
    assert joint_source.witness.source_hedges == ()


def test_experiment_targets_have_no_implicit_upward_or_downward_closure() -> None:
    downward_graph = _dag(nodes=("X", "Y"))
    joint_only = identify_generalized_causal_effect(
        downward_graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(("X", "Y"),),
    )
    exact_singleton = identify_generalized_causal_effect(
        downward_graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(("X",),),
    )
    assert isinstance(joint_only, NonIdentifiedGeneralizedEffect)
    assert isinstance(exact_singleton, IdentifiedGeneralizedEffect)

    upward_graph = _adbg(
        nodes=("X", "Z", "Y"),
        directed=(("Z", "Y"),),
        bidirected=(("Z", "Y"),),
    )
    singleton_only = identify_generalized_causal_effect(
        upward_graph,
        outcomes="Y",
        interventions=("X", "Z"),
        experiment_interventions=(("X",),),
    )
    exact_joint = identify_generalized_causal_effect(
        upward_graph,
        outcomes="Y",
        interventions=("X", "Z"),
        experiment_interventions=(("X", "Z"),),
    )
    assert isinstance(singleton_only, NonIdentifiedGeneralizedEffect)
    assert isinstance(exact_joint, IdentifiedGeneralizedEffect)


def test_all_variable_source_is_valid_but_cannot_cover_nonempty_district() -> None:
    graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    result = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(("X", "Y"),),
    )

    assert isinstance(result, NonIdentifiedGeneralizedEffect)
    witness = result.witness
    assert witness.graph is graph
    assert witness.target_vertices == ("Y",)
    assert witness.unresolved_district == ("Y",)
    assert witness.source_hedges == ()


def test_covering_observational_bow_failure_returns_one_exact_local_hedge() -> None:
    graph = _adbg(
        nodes=("X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    result = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=((),),
    )

    assert isinstance(result, NonIdentifiedGeneralizedEffect)
    witness = result.witness
    assert witness.graph is graph
    assert witness.outcomes == ("Y",)
    assert witness.interventions == ("X",)
    assert witness.experiment_interventions == ((),)
    assert witness.target_vertices == ("Y",)
    assert witness.unresolved_district == ("Y",)
    assert len(witness.source_hedges) == 1
    source_hedge = witness.source_hedges[0]
    assert source_hedge.graph == graph
    assert source_hedge.outcomes == ("Y",)
    assert source_hedge.interventions == ("X",)
    assert source_hedge.inner_vertices == ("Y",)
    assert source_hedge.outer_vertices == ("X", "Y")
    assert source_hedge.roots == ("Y",)


def test_tier_one_binds_only_actual_free_nuisance_with_conditional_leaf() -> None:
    graph = _adbg(
        nodes=("W", "X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    result = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(("Y",), ("W", "X"), ()),
    )
    expected = Summation(
        "W",
        Product(
            (
                Probability("Y", interventions=("W", "X")),
                Probability("W", conditioned_on="X"),
            )
        ),
    )

    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert result.experiment_interventions == ((), ("Y",), ("W", "X"))
    assert result.formula == expected
    assert result.formula.free_variables == frozenset(("X", "Y"))
    _assert_formula_uses_only_exact_supplied_sources(result)


def test_tier_one_normalizer_may_intervene_on_outcome_but_conditions_only_on_x() -> (
    None
):
    graph = _adbg(
        nodes=("W", "X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    result = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(("Y",), ("W", "X")),
    )
    expected = Summation(
        "W",
        Product(
            (
                Probability("Y", interventions=("W", "X")),
                Probability(
                    "W",
                    conditioned_on="X",
                    interventions="Y",
                ),
            )
        ),
    )

    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert result.experiment_interventions == (("Y",), ("W", "X"))
    assert result.formula == expected
    assert result.formula.free_variables == frozenset(("X", "Y"))
    assert isinstance(result.formula, Summation)
    assert isinstance(result.formula.body, Product)
    pointwise, weight = result.formula.body.factors
    assert pointwise == Probability("Y", interventions=("W", "X"))
    assert weight == Probability(
        "W",
        conditioned_on="X",
        interventions="Y",
    )
    _assert_formula_uses_only_exact_supplied_sources(result)


def test_binding_orders_formula_labels_topologically_not_by_declaration() -> None:
    graph = _adbg(
        nodes=("W2", "W1", "X", "Y"),
        directed=(("W1", "W2"), ("X", "Y")),
        bidirected=(("X", "Y"),),
    )
    result = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=((), ("W2", "W1", "X")),
    )
    expected = Summation(
        ("W1", "W2"),
        Product(
            (
                Probability("Y", interventions=("W2", "W1", "X")),
                Probability(("W1", "W2"), conditioned_on="X"),
            )
        ),
    )

    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert result.experiment_interventions == ((), ("W2", "W1", "X"))
    assert result.formula == expected
    assert isinstance(result.formula, Summation)
    assert result.formula.variables == ("W1", "W2")
    _assert_formula_uses_only_exact_supplied_sources(result)


def test_tier_two_uses_shared_normalizer_and_preserves_lexical_scope() -> None:
    graph = _dag(
        nodes=("B", "A", "C"),
        directed=(("B", "A"), ("A", "C")),
    )
    result = identify_generalized_causal_effect(
        graph,
        outcomes="C",
        interventions="A",
        experiment_interventions=(("B",),),
    )
    pointwise = Probability("C", conditioned_on="A", interventions="B")
    source_joint = Probability(("A", "C"), interventions="B")
    score = Summation(("A", "C"), source_joint)
    normalizer = Summation("B", score)
    expected = Summation(
        "B",
        Product((pointwise, Ratio(score, normalizer))),
    )

    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert result.formula == expected
    assert result.formula.free_variables == frozenset(("A", "C"))
    assert isinstance(result.formula, Summation)
    assert isinstance(result.formula.body, Product)
    actual_ratio = result.formula.body.factors[1]
    assert isinstance(actual_ratio, Ratio)
    assert isinstance(actual_ratio.denominator, Summation)
    # The wrapper deliberately preserves one shared score node in its syntax DAG.
    assert actual_ratio.numerator is actual_ratio.denominator.body
    assert isinstance(actual_ratio.numerator, Summation)
    assert actual_ratio.numerator.body is actual_ratio.denominator.body.body
    tex = probability_expression_to_tex(result.formula)
    assert r"\sum_{\textnormal{B}^{\prime}}" in tex
    assert r"P_{\textnormal{B}^{\prime}}" in tex

    labels = ("B", "A", "C")
    experiments = (frozenset({0}),)
    for variant in (0, 1):
        for b, a, c in itertools.product((0, 1), repeat=3):
            environment = {"B": b, "A": a, "C": c}
            expected_value = exact_binary_interventional_probability(
                TinyADMG((0, 1, 2), frozenset(((0, 1), (1, 2))), frozenset()),
                outcomes=(2,),
                interventions=(1,),
                values={1: a, 2: c},
                variant=variant,
            )
            pointwise_value = evaluate_generalized_probability_expression(
                pointwise,
                graph=TinyADMG(
                    (0, 1, 2),
                    frozenset(((0, 1), (1, 2))),
                    frozenset(),
                ),
                node_labels=labels,
                experiment_interventions=experiments,
                environment=environment,
                variant=variant,
            )
            assert pointwise_value == expected_value


def test_tier_two_sums_nonquery_source_actions_before_normalizing() -> None:
    graph = _adbg(
        nodes=("Z", "W", "X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    result = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(("Z",), ("W", "X")),
    )

    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert isinstance(result.formula, Summation)
    assert result.formula.variables == ("W",)
    assert isinstance(result.formula.body, Product)
    pointwise, ratio = result.formula.body.factors
    assert isinstance(ratio, Ratio)
    source_joint = Probability(("W", "X", "Y"), interventions="Z")
    score = Summation(("X", "Y"), source_joint)
    bound_score = Summation("Z", score)
    assert ratio.numerator == bound_score
    assert isinstance(ratio.denominator, Summation)
    assert ratio.denominator.variables == ("W",)
    assert ratio.numerator is ratio.denominator.body
    assert all(
        atom.interventions == ("W", "X")
        for atom in _walk_expression(pointwise)
        if isinstance(atom, Probability)
    )
    tex = probability_expression_to_tex(result.formula)
    assert r"\sum_{\textnormal{Z}}" in tex
    assert r"P_{\textnormal{Z}}" in tex
    assert r"\sum_{\textnormal{W}^{\prime}}" in tex
    assert r"\textnormal{X}^{\prime}" in tex
    assert r"\textnormal{Y}^{\prime}" in tex
    assert result.formula.free_variables == frozenset(("X", "Y"))
    _assert_formula_uses_only_exact_supplied_sources(result)


def test_multi_district_formula_receives_one_global_tier_two_weight() -> None:
    graph = _adbg(
        nodes=("B", "X1", "W", "X2", "Y"),
        directed=(("X1", "W"), ("W", "Y"), ("X2", "Y")),
        bidirected=(("X1", "W"), ("X1", "X2"), ("X2", "Y")),
    )
    result = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions=("X1", "X2"),
        experiment_interventions=(("B", "X2"), ("B", "X1")),
    )
    pointwise = Summation(
        "W",
        Product(
            (
                Probability(
                    "Y",
                    conditioned_on=("X1", "W"),
                    interventions=("B", "X2"),
                ),
                Probability("W", interventions=("B", "X1")),
            )
        ),
    )
    source_joint = Probability(("W", "X2", "Y"), interventions=("B", "X1"))
    score = Summation(("W", "X2", "Y"), source_joint)
    expected = Summation(
        "B",
        Product((pointwise, Ratio(score, Summation("B", score)))),
    )

    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert result.experiment_interventions == (
        ("B", "X1"),
        ("B", "X2"),
    )
    assert result.formula == expected
    assert (
        sum(isinstance(node, Ratio) for node in _walk_expression(result.formula)) == 1
    )
    assert isinstance(result.formula, Summation)
    assert isinstance(result.formula.body, Product)
    ratio = result.formula.body.factors[1]
    assert isinstance(ratio, Ratio)
    assert isinstance(ratio.denominator, Summation)
    # One shared score prevents duplicating a potentially large source expression.
    assert ratio.numerator is ratio.denominator.body
    assert result.formula.free_variables == frozenset(("X1", "X2", "Y"))
    _assert_formula_uses_only_exact_supplied_sources(result)


def test_tier_two_normalizer_scans_all_sources_and_is_value_independent() -> None:
    graph = _adbg(
        nodes=("B", "A", "C"),
        directed=(("A", "C"),),
        bidirected=(("A", "C"),),
    )
    combined = identify_generalized_causal_effect(
        graph,
        outcomes="C",
        interventions="A",
        experiment_interventions=(("B", "A"), ("B",)),
    )
    permuted = identify_generalized_causal_effect(
        graph,
        outcomes="C",
        interventions="A",
        experiment_interventions=(("B",), ("B", "A")),
    )
    successful_only = identify_generalized_causal_effect(
        graph,
        outcomes="C",
        interventions="A",
        experiment_interventions=(("B", "A"),),
    )
    pointwise = Probability("C", interventions=("B", "A"))
    first_joint = Probability(("A", "C"), interventions="B")
    first_score = Summation(("A", "C"), first_joint)
    combined_expected = Summation(
        "B",
        Product((pointwise, Ratio(first_score, Summation("B", first_score)))),
    )
    own_joint = Probability("C", interventions=("B", "A"))
    own_score = Summation("C", own_joint)
    own_expected = Summation(
        "B",
        Product((pointwise, Ratio(own_score, Summation("B", own_score)))),
    )

    assert isinstance(combined, IdentifiedGeneralizedEffect)
    assert isinstance(permuted, IdentifiedGeneralizedEffect)
    assert isinstance(successful_only, IdentifiedGeneralizedEffect)
    assert combined == permuted
    assert combined.formula == combined_expected
    assert successful_only.formula == own_expected
    assert combined.formula != successful_only.formula
    combined_atoms = tuple(
        atom.interventions
        for atom in _walk_expression(combined.formula)
        if isinstance(atom, Probability)
    )
    assert ("B",) in combined_atoms
    assert ("B", "A") in combined_atoms


def test_tier_two_nonbinary_actions_and_denominator_mutations() -> None:
    graph = _dag(
        nodes=("B", "A", "C"),
        directed=(("B", "A"), ("A", "C")),
    )
    result = identify_generalized_causal_effect(
        graph,
        outcomes="C",
        interventions="A",
        experiment_interventions=(("B",),),
    )
    assert isinstance(result, IdentifiedGeneralizedEffect)

    domains = {
        "B": (0, 1, 2),
        "A": (0, 1),
        "C": (0, 1, 2),
    }
    probability_a_one = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    probability_c = {
        0: (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
        1: (Fraction(1, 5), Fraction(3, 10), Fraction(1, 2)),
    }
    source_by_action: dict[tuple[int, ...], dict[tuple[int, ...], Fraction]] = {}
    for b in domains["B"]:
        table: dict[tuple[int, ...], Fraction] = {}
        for a, c in itertools.product(domains["A"], domains["C"]):
            probability_a = probability_a_one[b] if a else 1 - probability_a_one[b]
            table[(a, c)] = probability_a * probability_c[a][c]
        assert sum(table.values(), Fraction(0)) == 1
        assert all(probability > 0 for probability in table.values())
        source_by_action[(b,)] = table
    source_tables = {("B",): source_by_action}

    assert isinstance(result.formula, Summation)
    assert isinstance(result.formula.body, Product)
    pointwise, ratio = result.formula.body.factors
    assert isinstance(ratio, Ratio)
    score = ratio.numerator
    assert isinstance(score, Summation)
    missing_sum_mutation = Summation(
        "B",
        Product((pointwise, Ratio(score, score))),
    )
    assert isinstance(score.body, Probability)
    wrong_body_mutation = Summation(
        "B",
        Product((pointwise, Ratio(score, Summation("B", score.body)))),
    )
    mutation_mismatches = [0, 0]

    for a, c in itertools.product(domains["A"], domains["C"]):
        environment = {"A": a, "C": c}
        expected = probability_c[a][c]
        actual = _evaluate_explicit_source_expression(
            result.formula,
            node_order=("B", "A", "C"),
            domains=domains,
            source_tables=source_tables,
            environment=environment,
        )
        assert actual == expected
        for mutation_index, mutation in enumerate(
            (missing_sum_mutation, wrong_body_mutation)
        ):
            mutated = _evaluate_explicit_source_expression(
                mutation,
                node_order=("B", "A", "C"),
                domains=domains,
                source_tables=source_tables,
                environment=environment,
            )
            mutation_mismatches[mutation_index] += mutated != expected

    assert mutation_mismatches == [6, 6]


def test_generalized_results_and_nested_witnesses_are_immutable_values() -> None:
    identified_graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    identified = identify_generalized_causal_effect(
        identified_graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(("X",),),
    )
    bow = _adbg(
        nodes=("X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    nonidentified = identify_generalized_causal_effect(
        bow,
        outcomes="Y",
        interventions="X",
        experiment_interventions=((),),
    )
    same = identify_generalized_causal_effect(
        bow,
        outcomes="Y",
        interventions="X",
        experiment_interventions=((),),
    )

    assert isinstance(identified, IdentifiedGeneralizedEffect)
    assert isinstance(nonidentified, NonIdentifiedGeneralizedEffect)
    assert isinstance(nonidentified.witness, GeneralizedNonIdentificationWitness)
    assert not hasattr(identified, "auxiliary_interventions")
    assert not hasattr(nonidentified, "auxiliary_interventions")
    assert not hasattr(nonidentified.witness, "auxiliary_interventions")
    assert nonidentified == same
    assert hash(nonidentified) == hash(same)
    values = (
        identified,
        nonidentified,
        nonidentified.witness,
        *nonidentified.witness.source_hedges,
    )
    for value in values:
        assert copy.copy(value) is value
        assert copy.deepcopy(value) is value
        assert not hasattr(value, "__dict__")
        assert not hasattr(value, "__setstate__")
        original_hash = hash(value)
        with pytest.raises(TypeError, match="pickle is unsupported"):
            pickle.dumps(value)
        assert hash(value) == original_hash

    with pytest.raises((AttributeError, TypeError)):
        identified.outcomes = ("changed",)  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        nonidentified.witness = nonidentified.witness  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        nonidentified.witness.source_hedges = ()  # type: ignore[misc]


def test_generalized_result_classes_are_factory_only_final_and_slotted() -> None:
    graph = _adbg(
        nodes=("X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    identified = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=(("X",),),
    )
    nonidentified = identify_generalized_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        experiment_interventions=((),),
    )
    assert isinstance(identified, IdentifiedGeneralizedEffect)
    assert isinstance(nonidentified, NonIdentifiedGeneralizedEffect)

    with pytest.raises(TypeError):
        IdentifiedGeneralizedEffect(  # type: ignore[call-arg]
            graph=identified.graph,
            outcomes=identified.outcomes,
            interventions=identified.interventions,
            experiment_interventions=identified.experiment_interventions,
            formula=identified.formula,
        )
    with pytest.raises(TypeError):
        NonIdentifiedGeneralizedEffect(  # type: ignore[call-arg]
            witness=nonidentified.witness
        )
    with pytest.raises(TypeError):
        GeneralizedNonIdentificationWitness(  # type: ignore[call-arg]
            graph=nonidentified.witness.graph,
            outcomes=nonidentified.witness.outcomes,
            interventions=nonidentified.witness.interventions,
            experiment_interventions=(nonidentified.witness.experiment_interventions),
            target_vertices=nonidentified.witness.target_vertices,
            unresolved_district=nonidentified.witness.unresolved_district,
            source_hedges=nonidentified.witness.source_hedges,
        )

    expected_slots = {
        IdentifiedGeneralizedEffect: (
            "graph",
            "outcomes",
            "interventions",
            "experiment_interventions",
            "formula",
        ),
        NonIdentifiedGeneralizedEffect: ("witness",),
        GeneralizedNonIdentificationWitness: (
            "graph",
            "outcomes",
            "interventions",
            "experiment_interventions",
            "target_vertices",
            "unresolved_district",
            "source_hedges",
        ),
    }
    for factory in (
        IdentifiedGeneralizedEffect,
        NonIdentifiedGeneralizedEffect,
        GeneralizedNonIdentificationWitness,
    ):
        with pytest.raises(TypeError):
            factory()  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="cannot be subclassed"):
            type("MutableGeneralizedIdentificationSubclass", (factory,), {})
        assert factory.__slots__ == expected_slots[factory]


def test_observational_only_wrapper_matches_every_classical_n3_id_status_and_ast() -> (
    None
):
    labels = ("V0", "V1", "V2")
    comparisons = 0
    for tiny_graph in all_bow_admgs_n3():
        graph = _tiny_to_public(tiny_graph, labels)
        for interventions, outcomes in all_nonempty_outcome_queries_n3():
            normalized_outcomes = _ordered_labels(labels, outcomes)
            normalized_interventions = _ordered_labels(labels, interventions)
            generalized = identify_generalized_causal_effect(
                graph,
                outcomes=normalized_outcomes,
                interventions=normalized_interventions,
                experiment_interventions=((),),
            )
            classical = identify_causal_effect(
                graph,
                outcomes=normalized_outcomes,
                interventions=normalized_interventions,
            )
            assert isinstance(generalized, IdentifiedGeneralizedEffect) == isinstance(
                classical, IdentifiedEffect
            )
            if isinstance(generalized, IdentifiedGeneralizedEffect):
                assert isinstance(classical, IdentifiedEffect)
                assert generalized.formula == classical.formula
                _assert_formula_uses_only_exact_supplied_sources(generalized)
            else:
                assert isinstance(generalized, NonIdentifiedGeneralizedEffect)
                assert isinstance(classical, NonIdentifiedEffect)
                assert len(generalized.witness.source_hedges) == 1
                (source_hedge,) = generalized.witness.source_hedges
                assert source_hedge.graph == classical.witness.graph
                assert source_hedge.inner_vertices == classical.witness.inner_vertices
                assert source_hedge.outer_vertices == classical.witness.outer_vertices
                assert source_hedge.roots == classical.witness.roots
                assert source_hedge.fixing_sequence == classical.witness.fixing_sequence
            comparisons += 1
    assert comparisons == 3_800


def test_minimal_source_formulas_match_exact_positive_binary_scms() -> None:
    labels = ("V0", "V1", "V2")
    comparison_count = 0
    for case in minimal_finite_formula_cases_n3():
        graph = _tiny_to_public(case.graph, labels)
        outcomes = _ordered_labels(labels, case.outcomes)
        interventions = _ordered_labels(labels, case.interventions)
        experiments = tuple(
            _ordered_labels(labels, source) for source in case.experiment_interventions
        )
        result = identify_generalized_causal_effect(
            graph,
            outcomes=outcomes,
            interventions=interventions,
            experiment_interventions=experiments,
        )
        assert isinstance(result, IdentifiedGeneralizedEffect)
        _assert_formula_uses_only_exact_supplied_sources(result)
        assert result.formula.free_variables <= frozenset(
            _ordered_labels(labels, case.outcomes | case.interventions)
        )
        assignment_nodes = case.outcomes | case.interventions
        assignment_labels = _ordered_labels(labels, assignment_nodes)

        for variant in (0, 1):
            for assignment in itertools.product((0, 1), repeat=len(assignment_labels)):
                environment = dict(zip(assignment_labels, assignment, strict=True))
                expected = exact_binary_interventional_probability(
                    case.graph,
                    outcomes=tuple(sorted(case.outcomes)),
                    interventions=tuple(sorted(case.interventions)),
                    values={
                        index: environment[labels[index]]
                        for index in case.outcomes | case.interventions
                    },
                    variant=variant,
                )
                actual = evaluate_generalized_probability_expression(
                    result.formula,
                    graph=case.graph,
                    node_labels=labels,
                    experiment_interventions=case.experiment_interventions,
                    environment=environment,
                    variant=variant,
                )
                assert actual == expected
                comparison_count += 1

    assert comparison_count == 39_200


def test_public_status_first_district_and_every_source_hedge_match_all_n3_rows() -> (
    None
):
    labels = ("V0", "V1", "V2")
    sources = all_intervention_targets((0, 1, 2))
    analyses = n3_gid_query_analyses()
    row_count = 0

    for graph_index, tiny_graph in enumerate(all_bow_admgs_n3()):
        graph = _tiny_to_public(tiny_graph, labels)
        graph_analyses = analyses[graph_index * 19 : (graph_index + 1) * 19]
        for analysis in graph_analyses:
            assert analysis.graph is tiny_graph
            outcomes = _ordered_labels(labels, analysis.outcomes)
            interventions = _ordered_labels(labels, analysis.interventions)
            for mask in range(256):
                experiments = tuple(
                    _ordered_labels(labels, source)
                    for source_index, source in enumerate(sources)
                    if mask & (1 << source_index)
                )
                result = identify_generalized_causal_effect(
                    graph,
                    outcomes=outcomes,
                    interventions=interventions,
                    experiment_interventions=experiments,
                )
                expected_identified = analysis.identified_for_mask(mask)
                assert (
                    isinstance(result, IdentifiedGeneralizedEffect)
                    is expected_identified
                ), (graph_index, analysis, mask)

                if expected_identified:
                    assert isinstance(result, IdentifiedGeneralizedEffect)
                    assert result.graph is graph
                    assert result.outcomes == outcomes
                    assert result.interventions == interventions
                    assert result.experiment_interventions == experiments
                    row_count += 1
                    continue

                assert isinstance(result, NonIdentifiedGeneralizedEffect)
                witness = result.witness
                unresolved = analysis.first_unresolved_for_mask(mask)
                assert unresolved is not None
                assert witness.graph is graph
                assert witness.outcomes == outcomes
                assert witness.interventions == interventions
                assert witness.experiment_interventions == experiments
                assert witness.target_vertices == _ordered_labels(
                    labels, analysis.target_vertices
                )
                assert witness.unresolved_district == _ordered_labels(
                    labels, unresolved.district
                )

                expected_source_hedges = []
                for source_index, source in enumerate(sources):
                    if not mask & (1 << source_index):
                        continue
                    if not unresolved.covering_mask & (1 << source_index):
                        continue
                    problem = gid_local_problem(tiny_graph, unresolved.district, source)
                    if problem.identified:
                        raise AssertionError(
                            "unresolved district has a successful source"
                        )
                    expected_source_hedges.append(problem)
                assert len(witness.source_hedges) == len(expected_source_hedges)
                for actual_hedge, problem in zip(
                    witness.source_hedges, expected_source_hedges, strict=True
                ):
                    (expected_hedge,) = problem.hedges
                    _assert_source_hedge_matches_oracle(
                        actual_hedge,
                        full_labels=labels,
                        expected_graph=problem.source_graph,
                        expected_outcomes=problem.district,
                        expected_interventions=problem.local_interventions,
                        expected_hedge=expected_hedge,
                    )
                row_count += 1

    assert row_count == 972_800


def test_lee_figure1_public_formula_and_failures_match_adjudicated_fixture() -> None:
    fixture = _load_fixture("lee_correa_bareinboim_2019_gid_figures1_2.json")
    figure = fixture["figure_1"]
    expected_formula = Summation(
        "W",
        Product(
            (
                Probability(
                    "Y",
                    conditioned_on=("X1", "W"),
                    interventions="X2",
                ),
                Probability("W", interventions="X1"),
            )
        ),
    )

    for panel_name, panel in figure["panels"].items():
        graph = _adbg(
            nodes=figure["nodes"],
            directed=figure["common_directed"],
            bidirected=figure["common_bidirected"] + panel["added_bidirected"],
        )
        result = identify_generalized_causal_effect(
            graph,
            outcomes=figure["query"]["outcomes"],
            interventions=figure["query"]["interventions"],
            experiment_interventions=figure["experiment_interventions"],
        )
        expected_identified = panel["adjudicated_status"] == "identified"
        assert isinstance(result, IdentifiedGeneralizedEffect) is expected_identified
        if expected_identified:
            assert isinstance(result, IdentifiedGeneralizedEffect)
            assert result.formula == expected_formula
            assert figure["adjudicated_kernel_formula"] == (
                "sum_w P_x2(y|x1,w) P_x1(w)"
            )
            _assert_formula_uses_only_exact_supplied_sources(result)
        else:
            assert isinstance(result, NonIdentifiedGeneralizedEffect)
            witness = result.witness
            assert witness.target_vertices == tuple(panel["expected_target_vertices"])
            assert witness.unresolved_district == tuple(
                panel["expected_unresolved_district"]
            )
            assert tuple(
                tuple(
                    node
                    for node in figure["nodes"]
                    if node not in source_hedge.graph.topology.nodes
                )
                for source_hedge in witness.source_hedges
            ) == tuple(tuple(source) for source in panel["expected_failed_sources"])

        assert panel_name in ("a", "b", "c", "d")


def test_lee_figure2_public_status_matches_caption_and_independent_adjudication() -> (
    None
):
    fixture = _load_fixture("lee_correa_bareinboim_2019_gid_figures1_2.json")
    figure = fixture["figure_2"]
    for panel_name, panel in figure["panels"].items():
        graph = _adbg(
            nodes=figure["nodes"],
            directed=figure["common_directed"] + panel["added_directed"],
            bidirected=figure["common_bidirected"],
        )
        for query_name, query in figure["queries"].items():
            result = identify_generalized_causal_effect(
                graph,
                outcomes=query["outcomes"],
                interventions=query["interventions"],
                experiment_interventions=figure["experiment_interventions"],
            )
            expected = query["adjudicated_status"][panel_name]
            assert isinstance(result, IdentifiedGeneralizedEffect) is (
                expected == "identified"
            )
            assert expected == figure["reported_caption"][panel_name][query_name]

    assert figure["reported_prose"]["b"]["P_x1(Y1)"] == "not_identified"
    assert figure["queries"]["P_x1(Y1)"]["adjudicated_status"]["b"] == ("identified")
    assert fixture["official_erratum"]["figure_2_correction_found"] is False


def test_more_than_sixteen_vertices_have_no_hidden_size_limit() -> None:
    nodes = tuple(f"V{index:03d}" for index in range(80))
    graph = _dag(
        nodes=nodes,
        directed=tuple(zip(nodes[:-1], nodes[1:], strict=True)),
    )
    interventions = nodes[:-1]
    result = identify_generalized_causal_effect(
        graph,
        outcomes=nodes[-1],
        interventions=interventions,
        experiment_interventions=(interventions,),
    )

    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert result.graph is graph
    assert result.outcomes == (nodes[-1],)
    assert result.interventions == interventions
    assert result.experiment_interventions == (interventions,)
    assert result.formula == Probability(nodes[-1], interventions=interventions)


def test_generalized_local_id_ladder_does_not_depend_on_python_recursion() -> None:
    levels = 24
    ancestors = tuple(f"A{index:02d}" for index in range(levels))
    connectors = tuple(f"Z{index:02d}" for index in range(levels))
    outcomes = ("Y", "E")
    source_action = "R"
    base_nodes = (
        tuple(
            vertex
            for pair in zip(ancestors, connectors, strict=True)
            for vertex in pair
        )
        + outcomes
    )
    nodes = (source_action, *base_nodes)
    graph = _adbg(
        nodes=nodes,
        directed=tuple((ancestor, "Y") for ancestor in ancestors)
        + tuple(
            (connectors[index], ancestors[index - 1]) for index in range(1, levels)
        ),
        bidirected=tuple(
            (ancestors[index], connectors[index]) for index in range(levels)
        )
        + tuple(
            (connectors[index], ancestors[index + 1]) for index in range(levels - 1)
        )
        + ((connectors[-1], "Y"),),
    )
    interventions = tuple(vertex for vertex in base_nodes if vertex not in outcomes)

    previous_recursion_limit = sys.getrecursionlimit()
    current_depth = 0
    frame: Any = sys._getframe()
    while frame is not None:
        current_depth += 1
        frame = frame.f_back
    sys.setrecursionlimit(current_depth + 40)
    try:
        result = identify_generalized_causal_effect(
            graph,
            outcomes=outcomes,
            interventions=interventions,
            experiment_interventions=((source_action,),),
        )
    finally:
        sys.setrecursionlimit(previous_recursion_limit)

    assert isinstance(result, IdentifiedGeneralizedEffect)
    assert result.graph is graph
    assert result.outcomes == outcomes
    assert result.interventions == interventions
    _assert_formula_uses_only_exact_supplied_sources(result)
