from __future__ import annotations

import copy
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
    GraphValidationError,
    IdentificationDomainError,
    HedgeWitness,
    IdentifiedEffect,
    IdentificationResult,
    NonIdentifiedEffect,
    Probability,
    ProbabilityExpression,
    Product,
    Ratio,
    Summation,
    identify_causal_effect,
)


DOMAIN_MESSAGE = (
    "causal-effect identification requires an acyclic directed-bidirected or DAG view"
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
    pending = [expression]
    while pending:
        current = pending.pop()
        walked.append(current)
        if isinstance(current, Product):
            pending.extend(reversed(current.factors))
        elif isinstance(current, Summation):
            pending.append(current.body)
        elif isinstance(current, Ratio):
            pending.append(current.denominator)
            pending.append(current.numerator)
    return tuple(walked)


def _assert_formula_is_source_bound_and_observational(
    result: IdentifiedEffect,
) -> None:
    source_nodes = set(result.graph.topology.nodes)
    expression = result.formula
    free = expression.free_variables

    assert set(result.outcomes) <= free
    assert free <= set(result.outcomes) | set(result.interventions)

    for node in _walk_expression(expression):
        if isinstance(node, Probability):
            assert node.interventions == ()
            assert (
                set(node.variables) | set(node.conditioned_on) | set(node.interventions)
            ) <= source_nodes
        elif isinstance(node, Summation):
            assert set(node.variables) <= source_nodes


def _canonical_bidirected_edges(
    graph: DirectedBidirectedGraph,
) -> tuple[tuple[str, str], ...]:
    positions = {node: index for index, node in enumerate(graph.nodes)}
    return tuple(
        sorted(
            graph.bidirected,
            key=lambda edge: (positions[edge[0]], positions[edge[1]]),
        )
    )


def _node_kernel_probability(
    graph: DirectedBidirectedGraph,
    node: str,
    observed: dict[str, int],
    latent: tuple[int, ...],
    latent_edges: tuple[tuple[str, str], ...],
    variant: int,
) -> Fraction:
    positions = {vertex: index for index, vertex in enumerate(graph.nodes)}
    parents = tuple(sorted(graph.directed_parents(node), key=positions.__getitem__))
    incident_latents = tuple(
        index for index, edge in enumerate(latent_edges) if node in edge
    )
    input_bits = tuple(observed[parent] for parent in parents) + tuple(
        latent[index] for index in incident_latents
    )
    code = sum(bit << index for index, bit in enumerate(input_bits))
    offset = (positions[node] + 1) * (variant + 1)
    probability_one = Fraction(
        offset + code + 1,
        offset + (1 << len(input_bits)) + 1,
    )
    return probability_one if observed[node] else 1 - probability_one


def _latent_assignment_probability(latent: tuple[int, ...], variant: int) -> Fraction:
    probability = Fraction(1)
    for index, value in enumerate(latent):
        probability_one = Fraction(2 + ((index + variant) % 3), 7)
        probability *= probability_one if value else 1 - probability_one
    return probability


def _observational_distribution(
    graph: DirectedBidirectedGraph,
    variant: int,
) -> dict[tuple[int, ...], Fraction]:
    latent_edges = _canonical_bidirected_edges(graph)
    table: dict[tuple[int, ...], Fraction] = {}
    for observed_bits in itertools.product((0, 1), repeat=len(graph.nodes)):
        observed = dict(zip(graph.nodes, observed_bits, strict=True))
        probability = Fraction(0)
        for latent in itertools.product((0, 1), repeat=len(latent_edges)):
            term = _latent_assignment_probability(latent, variant)
            for node in graph.nodes:
                term *= _node_kernel_probability(
                    graph,
                    node,
                    observed,
                    latent,
                    latent_edges,
                    variant,
                )
            probability += term
        table[observed_bits] = probability
    assert sum(table.values(), Fraction(0)) == 1
    assert all(probability > 0 for probability in table.values())
    return table


def _direct_interventional_probability(
    graph: DirectedBidirectedGraph,
    *,
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
    values: dict[str, int],
    variant: int,
) -> Fraction:
    latent_edges = _canonical_bidirected_edges(graph)
    intervention_set = set(interventions)
    fixed = set(outcomes) | intervention_set
    probability = Fraction(0)
    for observed_bits in itertools.product((0, 1), repeat=len(graph.nodes)):
        observed = dict(zip(graph.nodes, observed_bits, strict=True))
        if any(observed[node] != values[node] for node in fixed):
            continue
        for latent in itertools.product((0, 1), repeat=len(latent_edges)):
            term = _latent_assignment_probability(latent, variant)
            for node in graph.nodes:
                if node not in intervention_set:
                    term *= _node_kernel_probability(
                        graph,
                        node,
                        observed,
                        latent,
                        latent_edges,
                        variant,
                    )
            probability += term
    return probability


def _evaluate_probability_atom(
    atom: Probability,
    table: dict[tuple[int, ...], Fraction],
    nodes: tuple[str, ...],
    environment: dict[str, int],
) -> Fraction:
    assert atom.interventions == ()
    positions = {node: index for index, node in enumerate(nodes)}

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

    numerator = event_probability(atom.variables + atom.conditioned_on)
    if not atom.conditioned_on:
        return numerator
    denominator = event_probability(atom.conditioned_on)
    assert denominator > 0
    return numerator / denominator


def _evaluate_expression(
    expression: ProbabilityExpression,
    table: dict[tuple[int, ...], Fraction],
    nodes: tuple[str, ...],
    environment: dict[str, int],
) -> Fraction:
    if isinstance(expression, Probability):
        return _evaluate_probability_atom(expression, table, nodes, environment)
    if isinstance(expression, Product):
        value = Fraction(1)
        for factor in expression.factors:
            value *= _evaluate_expression(factor, table, nodes, environment)
        return value
    if isinstance(expression, Summation):
        value = Fraction(0)
        for assignment in itertools.product((0, 1), repeat=len(expression.variables)):
            bound_environment = dict(environment)
            bound_environment.update(zip(expression.variables, assignment, strict=True))
            value += _evaluate_expression(
                expression.body, table, nodes, bound_environment
            )
        return value
    assert isinstance(expression, Ratio)
    denominator = _evaluate_expression(
        expression.denominator, table, nodes, environment
    )
    assert denominator > 0
    return (
        _evaluate_expression(expression.numerator, table, nodes, environment)
        / denominator
    )


def _assert_expression_matches_exact_binary_scm(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    *,
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
    expression: ProbabilityExpression,
) -> None:
    topology = graph.topology
    query_labels = outcomes + interventions
    for variant in (0, 1):
        table = _observational_distribution(topology, variant)
        for assignment in itertools.product((0, 1), repeat=len(query_labels)):
            values = dict(zip(query_labels, assignment, strict=True))
            expected = _direct_interventional_probability(
                topology,
                outcomes=outcomes,
                interventions=interventions,
                values=values,
                variant=variant,
            )
            actual = _evaluate_expression(expression, table, topology.nodes, values)
            assert actual == expected


def _assert_formula_matches_exact_binary_scm(
    result: IdentifiedEffect,
) -> None:
    _assert_expression_matches_exact_binary_scm(
        result.graph,
        outcomes=result.outcomes,
        interventions=result.interventions,
        expression=result.formula,
    )


def test_identification_public_result_alias_and_domain_error_type() -> None:
    assert get_args(IdentificationResult) == (
        IdentifiedEffect,
        NonIdentifiedEffect,
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
def test_identification_rejects_raw_and_wrong_views_with_exact_message(
    invalid_graph: object,
) -> None:
    with pytest.raises(
        IdentificationDomainError,
        match=f"^{re.escape(DOMAIN_MESSAGE)}$",
    ):
        identify_causal_effect(  # type: ignore[arg-type]
            invalid_graph,
            outcomes="Y",
            interventions="X",
        )


def test_graph_domain_is_checked_before_selector_consumption() -> None:
    class ExplodingIterable:
        def __iter__(self) -> Any:
            raise AssertionError("selector must not be consumed")

    raw = DirectedBidirectedGraph(nodes=("X", "Y"))
    with pytest.raises(IdentificationDomainError, match="^causal-effect"):
        identify_causal_effect(
            raw,  # type: ignore[arg-type]
            outcomes=ExplodingIterable(),  # type: ignore[arg-type]
            interventions=ExplodingIterable(),  # type: ignore[arg-type]
        )


def test_identification_signature_is_keyword_only_after_graph() -> None:
    graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    with pytest.raises(TypeError):
        identify_causal_effect(graph, "Y", "X")  # type: ignore[misc]
    with pytest.raises(TypeError):
        identify_causal_effect(graph, interventions="X")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        identify_causal_effect(  # type: ignore[call-arg]
            graph,
            outcomes="Y",
            unsupported=True,
        )


def test_selectors_use_existing_set_normalization_and_source_order() -> None:
    graph = _dag(nodes=("B", "A", "C"))
    outcomes = ["C", "A", "C"]
    interventions = (node for node in ("B", "B"))

    result = identify_causal_effect(
        graph,
        outcomes=outcomes,
        interventions=interventions,
    )
    outcomes.clear()

    assert isinstance(result, IdentifiedEffect)
    assert result.graph is graph
    assert result.outcomes == ("A", "C")
    assert result.interventions == ("B",)
    _assert_formula_is_source_bound_and_observational(result)


def test_unknown_selectors_raise_graph_validation_error_in_argument_order() -> None:
    graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    with pytest.raises(GraphValidationError, match="unknown vertex: 'bad-outcome'"):
        identify_causal_effect(
            graph,
            outcomes="bad-outcome",
            interventions="bad-intervention",
        )
    with pytest.raises(
        GraphValidationError, match="unknown vertex: 'bad-intervention'"
    ):
        identify_causal_effect(
            graph,
            outcomes="Y",
            interventions="bad-intervention",
        )
    with pytest.raises(GraphValidationError, match="seed vertices must be iterable"):
        identify_causal_effect(
            graph,
            outcomes=None,  # type: ignore[arg-type]
        )


def test_overlap_is_a_domain_error_after_selector_normalization() -> None:
    graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    with pytest.raises(
        IdentificationDomainError,
        match=f"^{OVERLAP_MESSAGE}$",
    ):
        identify_causal_effect(
            graph,
            outcomes=("Y", "X", "Y"),
            interventions=("X", "X"),
        )


def test_empty_outcomes_are_the_identified_unit_after_full_validation() -> None:
    bow = _adbg(
        nodes=("X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    result = identify_causal_effect(
        bow,
        outcomes=(),
        interventions="X",
    )

    assert isinstance(result, IdentifiedEffect)
    assert result.graph is bow
    assert result.outcomes == ()
    assert result.interventions == ("X",)
    assert result.formula == Product(())
    assert result.formula.free_variables == frozenset()

    empty_graph = _dag(nodes=())
    empty_result = identify_causal_effect(empty_graph, outcomes=())
    assert isinstance(empty_result, IdentifiedEffect)
    assert empty_result.formula == Product(())

    with pytest.raises(GraphValidationError, match="unknown vertex"):
        identify_causal_effect(
            bow,
            outcomes=(),
            interventions="missing",
        )


def test_default_empty_intervention_and_irrelevant_intervention_semantics() -> None:
    graph = _dag(nodes=("X", "Y"))

    observational = identify_causal_effect(graph, outcomes="Y")
    assert isinstance(observational, IdentifiedEffect)
    assert observational.interventions == ()
    assert observational.formula.free_variables == frozenset({"Y"})
    _assert_formula_is_source_bound_and_observational(observational)
    _assert_formula_matches_exact_binary_scm(observational)

    intervened = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )
    assert isinstance(intervened, IdentifiedEffect)
    assert intervened.interventions == ("X",)
    assert intervened.formula.free_variables == frozenset({"Y"})
    _assert_formula_is_source_bound_and_observational(intervened)
    _assert_formula_matches_exact_binary_scm(intervened)


def test_line_three_auxiliary_action_is_bound_by_an_observational_kernel() -> None:
    graph = _dag(
        nodes=("Z", "X", "Y"),
        directed=(("Z", "X"), ("X", "Y")),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )

    assert isinstance(result, IdentifiedEffect)
    assert result.formula == Summation(
        "Z",
        Product(
            (
                Probability("Y", conditioned_on=("Z", "X")),
                Probability("Z", conditioned_on="X"),
            )
        ),
    )
    assert result.formula.free_variables == frozenset({"X", "Y"})
    _assert_formula_is_source_bound_and_observational(result)
    _assert_formula_matches_exact_binary_scm(result)


def test_auxiliary_weight_uses_topological_order_without_reordering_the_query() -> None:
    graph = _dag(
        nodes=("X2", "X1", "Z", "Y"),
        directed=(("Z", "X1"), ("X1", "X2"), ("X2", "Y")),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions=("X2", "X1"),
    )

    assert isinstance(result, IdentifiedEffect)
    assert result.interventions == ("X2", "X1")
    assert result.formula == Summation(
        "Z",
        Product(
            (
                Probability("Y", conditioned_on=("Z", "X1", "X2")),
                Probability("Z", conditioned_on=("X1", "X2")),
            )
        ),
    )
    _assert_formula_matches_exact_binary_scm(result)


def test_multiple_line_three_auxiliaries_use_one_joint_observational_kernel() -> None:
    graph = _dag(
        nodes=("X", "W2", "W1", "Y"),
        directed=(("W1", "X"), ("W2", "X"), ("X", "Y")),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )

    assert isinstance(result, IdentifiedEffect)
    assert result.formula == Summation(
        ("W2", "W1"),
        Product(
            (
                Probability("Y", conditioned_on=("W2", "W1", "X")),
                Probability(("W2", "W1"), conditioned_on="X"),
            )
        ),
    )
    _assert_formula_matches_exact_binary_scm(result)


def test_identified_results_are_source_bound_immutable_values() -> None:
    graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )
    same = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )

    assert isinstance(result, IdentifiedEffect)
    assert result.graph is graph
    assert result == same
    assert hash(result) == hash(same)
    assert copy.copy(result) is result
    assert copy.deepcopy(result) is result
    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "__setstate__")
    with pytest.raises(TypeError, match="pickle is unsupported"):
        pickle.dumps(result)
    with pytest.raises((AttributeError, TypeError)):
        result.outcomes = ("changed",)  # type: ignore[misc]


def test_nonidentified_result_and_witness_are_immutable_source_bound_values() -> None:
    graph = _adbg(
        nodes=("X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )
    same = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )

    assert isinstance(result, NonIdentifiedEffect)
    assert result == same
    assert hash(result) == hash(same)
    witness = result.witness
    assert isinstance(witness, HedgeWitness)
    assert witness == same.witness
    assert hash(witness) == hash(same.witness)
    assert witness.graph is graph
    assert witness.outcomes == ("Y",)
    assert witness.interventions == ("X",)

    for value in (result, witness):
        assert copy.copy(value) is value
        assert copy.deepcopy(value) is value
        assert not hasattr(value, "__dict__")
        assert not hasattr(value, "__setstate__")
        with pytest.raises(TypeError, match="pickle is unsupported"):
            pickle.dumps(value)
    with pytest.raises((AttributeError, TypeError)):
        witness.roots = ("X",)  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        result.witness = witness  # type: ignore[misc]


def test_result_and_witness_classes_are_factory_only_and_sealed() -> None:
    graph = _adbg(
        nodes=("X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    identified = identify_causal_effect(
        _dag(nodes=("X", "Y"), directed=(("X", "Y"),)),
        outcomes="Y",
        interventions="X",
    )
    nonidentified = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )
    assert isinstance(identified, IdentifiedEffect)
    assert isinstance(nonidentified, NonIdentifiedEffect)

    with pytest.raises(TypeError):
        IdentifiedEffect(  # type: ignore[call-arg]
            graph=identified.graph,
            outcomes=identified.outcomes,
            interventions=identified.interventions,
            formula=identified.formula,
        )
    with pytest.raises(TypeError):
        NonIdentifiedEffect(  # type: ignore[call-arg]
            witness=nonidentified.witness
        )
    with pytest.raises(TypeError):
        HedgeWitness(  # type: ignore[call-arg]
            graph=graph,
            outcomes=("Y",),
            interventions=("X",),
            inner_vertices=("Y",),
            outer_vertices=("X", "Y"),
            roots=("Y",),
            fixing_sequence=(),
        )

    for base in (IdentifiedEffect, NonIdentifiedEffect, HedgeWitness):
        with pytest.raises(TypeError, match="cannot be subclassed"):
            type("MutableIdentificationSubclass", (base,), {})


def test_dag_backdoor_query_returns_the_classical_g_formula() -> None:
    graph = _dag(
        nodes=("Z", "X", "Y"),
        directed=(("Z", "X"), ("Z", "Y"), ("X", "Y")),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )
    expected = Summation(
        "Z",
        Product(
            (
                Probability("Z"),
                Probability("Y", conditioned_on=("Z", "X")),
            )
        ),
    )

    assert isinstance(result, IdentifiedEffect)
    assert result.formula == expected
    _assert_formula_is_source_bound_and_observational(result)
    _assert_formula_matches_exact_binary_scm(result)


def test_topological_ties_use_declared_node_order_in_chain_factors() -> None:
    graph = _dag(
        nodes=("B", "A", "Y"),
        directed=(("A", "Y"), ("B", "Y")),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions=("A", "B"),
    )

    assert isinstance(result, IdentifiedEffect)
    assert result.interventions == ("B", "A")
    assert result.formula == Probability("Y", conditioned_on=("B", "A"))
    _assert_formula_matches_exact_binary_scm(result)


def test_frontdoor_query_has_the_classical_nested_sum_formula() -> None:
    graph = _adbg(
        nodes=("X", "M", "Y"),
        directed=(("X", "M"), ("M", "Y")),
        bidirected=(("X", "Y"),),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )
    expected = Summation(
        "M",
        Product(
            (
                Probability("M", conditioned_on="X"),
                Summation(
                    "X",
                    Product(
                        (
                            Probability("X"),
                            Probability("Y", conditioned_on=("X", "M")),
                        )
                    ),
                ),
            )
        ),
    )

    assert isinstance(result, IdentifiedEffect)
    assert result.formula == expected
    _assert_formula_is_source_bound_and_observational(result)
    _assert_formula_matches_exact_binary_scm(result)


def test_published_bow_fixture_returns_an_exact_query_hedge() -> None:
    fixture = _load_fixture("shpitser_pearl_2006_figure2a_bow.json")
    graph = _adbg(
        nodes=fixture["nodes"],
        directed=fixture["directed"],
        bidirected=fixture["bidirected"],
    )
    query = fixture["query"]
    result = identify_causal_effect(
        graph,
        outcomes=query["outcomes"],
        interventions=query["interventions"],
    )

    assert isinstance(result, NonIdentifiedEffect)
    witness = result.witness
    assert witness.graph is graph
    assert witness.outcomes == tuple(query["outcomes"])
    assert witness.interventions == tuple(query["interventions"])
    assert witness.inner_vertices == tuple(fixture["expected_inner_vertices"])
    assert witness.outer_vertices == tuple(fixture["expected_outer_vertices"])
    assert witness.roots == tuple(fixture["expected_roots"])
    assert witness.fixing_sequence == tuple(fixture["expected_fixing_sequence"])


def test_aaai_2006_figure1_identified_and_nonidentified_panels() -> None:
    fixture = _load_fixture("shpitser_pearl_2006_figure1_identification.json")
    query = fixture["query"]
    panel_a = fixture["panels"]["a"]
    graph_a = _adbg(
        nodes=fixture["nodes"],
        directed=panel_a["directed"],
        bidirected=fixture["bidirected"],
    )
    identified = identify_causal_effect(
        graph_a,
        outcomes=query["outcomes"],
        interventions=query["interventions"],
    )

    assert isinstance(identified, IdentifiedEffect)
    _assert_formula_is_source_bound_and_observational(identified)
    _assert_formula_matches_exact_binary_scm(identified)
    reported_formula = Product(
        (
            Summation(
                "W2",
                Product(
                    (
                        Probability("Y2", conditioned_on="W2"),
                        Probability("W2"),
                    )
                ),
            ),
            Summation(
                "W1",
                Product(
                    (
                        Probability("Y1", conditioned_on=("X", "W1")),
                        Probability("W1"),
                    )
                ),
            ),
        )
    )
    _assert_expression_matches_exact_binary_scm(
        graph_a,
        outcomes=tuple(query["outcomes"]),
        interventions=tuple(query["interventions"]),
        expression=reported_formula,
    )

    panel_b = fixture["panels"]["b"]
    graph_b = _adbg(
        nodes=fixture["nodes"],
        directed=panel_b["directed"],
        bidirected=fixture["bidirected"],
    )
    nonidentified = identify_causal_effect(
        graph_b,
        outcomes=query["outcomes"],
        interventions=query["interventions"],
    )

    assert isinstance(nonidentified, NonIdentifiedEffect)
    witness = nonidentified.witness
    assert witness.graph is graph_b
    assert witness.outcomes == tuple(query["outcomes"])
    assert witness.interventions == tuple(query["interventions"])
    assert witness.inner_vertices == tuple(panel_b["expected_inner_vertices"])
    assert witness.outer_vertices == tuple(panel_b["expected_outer_vertices"])
    assert witness.roots == tuple(panel_b["expected_roots"])
    assert witness.fixing_sequence == tuple(panel_b["expected_fixing_sequence"])


def test_2023_exact_query_counterexample_does_not_use_a_subquery_hedge() -> None:
    fixture = _load_fixture("shpitser_2023_figure1c_hedge_counterexample.json")
    assert fixture["full_query"]["reported_formula"] == (
        "sum_C P(y,a2|a1,C)P(C) / sum_C P(a2|a1,C)P(C)"
    )
    assert fixture["full_query"]["adjudicated_formula"] == (
        "sum_w P(y,a2|a1,w)P(w) / sum_w P(a2|a1,w)P(w)"
    )
    assert "prints C" in fixture["full_query"]["reported_formula_status"]
    graph = _adbg(
        nodes=fixture["nodes"],
        directed=fixture["directed"],
        bidirected=fixture["bidirected"],
    )
    full_query = fixture["full_query"]
    full_result = identify_causal_effect(
        graph,
        outcomes=full_query["outcomes"],
        interventions=full_query["interventions"],
    )

    assert isinstance(full_result, IdentifiedEffect)
    assert any(
        isinstance(node, Ratio) for node in _walk_expression(full_result.formula)
    )
    _assert_formula_is_source_bound_and_observational(full_result)
    _assert_formula_matches_exact_binary_scm(full_result)
    adjudicated_formula = Ratio(
        Summation(
            "W",
            Product(
                (
                    Probability(("Y", "A2"), conditioned_on=("A1", "W")),
                    Probability("W"),
                )
            ),
        ),
        Summation(
            "W",
            Product(
                (
                    Probability("A2", conditioned_on=("A1", "W")),
                    Probability("W"),
                )
            ),
        ),
    )
    _assert_expression_matches_exact_binary_scm(
        graph,
        outcomes=tuple(full_query["outcomes"]),
        interventions=tuple(full_query["interventions"]),
        expression=adjudicated_formula,
    )

    strict_subquery = fixture["strict_subquery"]
    subquery_result = identify_causal_effect(
        graph,
        outcomes=strict_subquery["outcomes"],
        interventions=strict_subquery["interventions"],
    )

    assert isinstance(subquery_result, NonIdentifiedEffect)
    witness = subquery_result.witness
    assert witness.graph is graph
    assert witness.outcomes == tuple(strict_subquery["outcomes"])
    assert witness.interventions == tuple(strict_subquery["interventions"])
    assert witness.inner_vertices == tuple(strict_subquery["expected_inner_vertices"])
    assert witness.outer_vertices == tuple(strict_subquery["expected_outer_vertices"])
    assert witness.roots == tuple(strict_subquery["expected_roots"])
    assert witness.fixing_sequence == tuple(strict_subquery["expected_fixing_sequence"])


def test_minimal_subset_hedge_false_positive_binds_the_full_query() -> None:
    graph = _adbg(
        nodes=("X0", "M", "Y"),
        directed=(("X0", "M"), ("M", "Y")),
        bidirected=(("X0", "M"),),
    )

    full_result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions=("X0", "M"),
    )
    assert isinstance(full_result, IdentifiedEffect)
    _assert_formula_is_source_bound_and_observational(full_result)
    _assert_formula_matches_exact_binary_scm(full_result)

    subquery_result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X0",
    )
    assert isinstance(subquery_result, NonIdentifiedEffect)
    witness = subquery_result.witness
    assert witness.outcomes == ("Y",)
    assert witness.interventions == ("X0",)
    assert witness.inner_vertices == ("M",)
    assert witness.outer_vertices == ("X0", "M")
    assert witness.roots == ("M",)
    assert witness.fixing_sequence == ("Y",)


def test_failure_local_outcome_is_not_substituted_for_the_original_query() -> None:
    graph = _adbg(
        nodes=("X", "M", "Y"),
        directed=(("X", "M"), ("M", "Y")),
        bidirected=(("X", "M"),),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )

    assert isinstance(result, NonIdentifiedEffect)
    witness = result.witness
    assert witness.outcomes == ("Y",)
    assert witness.interventions == ("X",)
    assert witness.inner_vertices == ("M",)
    assert witness.outer_vertices == ("X", "M")
    assert witness.roots == ("M",)
    assert witness.fixing_sequence == ("Y",)


def test_line_three_added_intervention_is_not_bound_as_the_requested_query() -> None:
    graph = _adbg(
        nodes=("X", "Z", "Y"),
        directed=(("X", "Y"), ("Z", "Y")),
        bidirected=(("X", "Z"), ("X", "Y")),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )

    assert isinstance(result, NonIdentifiedEffect)
    witness = result.witness
    assert witness.outcomes == ("Y",)
    assert witness.interventions == ("X",)
    assert witness.inner_vertices == ("Y",)
    assert witness.outer_vertices == ("X", "Z", "Y")
    assert witness.roots == ("Y",)
    assert witness.fixing_sequence == ()


def test_hedge_roots_are_sinks_of_the_inner_district_not_the_active_graph() -> None:
    graph = _adbg(
        nodes=("0", "2", "1"),
        directed=(("0", "2"), ("2", "1")),
        bidirected=(("0", "1"), ("0", "2")),
    )
    result = identify_causal_effect(
        graph,
        outcomes=("0", "1"),
        interventions="2",
    )

    assert isinstance(result, NonIdentifiedEffect)
    witness = result.witness
    assert witness.outcomes == ("0", "1")
    assert witness.interventions == ("2",)
    assert witness.inner_vertices == ("0", "1")
    assert witness.outer_vertices == ("0", "2", "1")
    assert witness.roots == ("0", "1")
    assert witness.fixing_sequence == ()


def test_fixing_sequence_uses_declared_order_outside_the_reachable_closure() -> None:
    graph = _adbg(
        nodes=("B", "X", "A", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    result = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )

    assert isinstance(result, NonIdentifiedEffect)
    witness = result.witness
    assert witness.inner_vertices == ("Y",)
    assert witness.outer_vertices == ("X", "Y")
    assert witness.roots == ("Y",)
    assert witness.fixing_sequence == ("B", "A")


def test_more_than_one_thousand_dag_vertices_have_no_legacy_or_recursion_cap() -> None:
    nodes = tuple(f"V{index:04d}" for index in range(1_001))
    graph = _dag(
        nodes=nodes,
        directed=tuple(zip(nodes[:-1], nodes[1:], strict=True)),
    )
    result = identify_causal_effect(
        graph,
        outcomes=nodes[-1],
        interventions=nodes[:-1],
    )

    assert isinstance(result, IdentifiedEffect)
    assert result.graph is graph
    assert result.outcomes == (nodes[-1],)
    assert result.interventions == nodes[:-1]
    assert result.formula == Probability(nodes[-1], conditioned_on=nodes[:-1])
    _assert_formula_is_source_bound_and_observational(result)

    acyclic = _adbg(
        nodes=nodes,
        directed=tuple(zip(nodes[:-1], nodes[1:], strict=True)),
    )
    acyclic_result = identify_causal_effect(
        acyclic,
        outcomes=nodes[-1],
        interventions=nodes[:-1],
    )
    assert isinstance(acyclic_result, IdentifiedEffect)
    assert acyclic_result.formula == result.formula


def test_id_district_continuation_and_line_seven_ladder_do_not_recurse() -> None:
    levels = 24
    ancestors = tuple(f"A{index:02d}" for index in range(levels))
    connectors = tuple(f"Z{index:02d}" for index in range(levels))
    outcomes = ("Y", "E")
    nodes = tuple(
        vertex
        for pair in zip(ancestors, connectors, strict=True)
        for vertex in pair
    ) + outcomes
    graph = _adbg(
        nodes=nodes,
        directed=tuple((ancestor, "Y") for ancestor in ancestors)
        + tuple(
            (connectors[index], ancestors[index - 1])
            for index in range(1, levels)
        ),
        bidirected=tuple(
            (ancestors[index], connectors[index]) for index in range(levels)
        )
        + tuple(
            (connectors[index], ancestors[index + 1])
            for index in range(levels - 1)
        )
        + ((connectors[-1], "Y"),),
    )
    interventions = tuple(vertex for vertex in nodes if vertex not in outcomes)

    # Figure 3 first removes Z00 at line 2, then leaves one line-4 product
    # continuation open.  Its Y child drops E at line 2, then follows 23
    # strict line-7 inclusions, each followed by another line-2 ancestor
    # restriction, before line 6.  A direct recursive transcription therefore
    # needs 50 simultaneous ID calls; the production work stack needs only
    # constant Python call depth.
    previous_recursion_limit = sys.getrecursionlimit()
    current_depth = 0
    frame: Any = sys._getframe()
    while frame is not None:
        current_depth += 1
        frame = frame.f_back
    sys.setrecursionlimit(current_depth + 40)
    try:
        result = identify_causal_effect(
            graph,
            outcomes=outcomes,
            interventions=interventions,
        )
    finally:
        sys.setrecursionlimit(previous_recursion_limit)

    assert isinstance(result, IdentifiedEffect)
    assert result.graph is graph
    assert result.outcomes == outcomes
    assert result.interventions == interventions
    assert set(outcomes) <= result.formula.free_variables
    assert result.formula.free_variables <= set(outcomes) | set(interventions)
