from __future__ import annotations

import copy
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, get_args

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    AncestralGraph,
    ClosedPAG,
    ConditionalIdentificationResult,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    HedgeWitness,
    IdentifiedConditionalEffect,
    IdentifiedEffect,
    IdentificationDomainError,
    MaximalAncestralGraph,
    NonIdentifiedConditionalEffect,
    NonIdentifiedEffect,
    Probability,
    ProbabilityExpression,
    Product,
    Ratio,
    Summation,
    identify_causal_effect,
    identify_conditional_causal_effect,
)


DOMAIN_MESSAGE = (
    "conditional causal-effect identification requires an acyclic "
    "directed-bidirected or DAG view"
)
OUTCOME_INTERVENTION_OVERLAP = "outcomes and interventions must be disjoint"
OUTCOME_CONDITIONING_OVERLAP = "outcomes and conditioning variables must be disjoint"
INTERVENTION_CONDITIONING_OVERLAP = (
    "interventions and conditioning variables must be disjoint"
)


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


def _assert_normalized_ratio(result: IdentifiedConditionalEffect) -> None:
    assert result.outcomes
    assert isinstance(result.formula, Ratio)
    assert isinstance(result.formula.denominator, Summation)
    assert result.formula.denominator.variables == result.outcomes
    assert result.formula.denominator.body is result.formula.numerator


def _assert_formula_is_source_bound_and_observational(
    result: IdentifiedConditionalEffect,
) -> None:
    source_nodes = set(result.graph.topology.nodes)
    moved = set(result.rule_two_interventions)
    remaining_conditions = set(result.conditioned_on) - moved
    free = result.formula.free_variables

    assert set(result.outcomes) | remaining_conditions <= free
    assert free <= (
        set(result.outcomes) | set(result.interventions) | set(result.conditioned_on)
    )
    for node in _walk_expression(result.formula):
        if isinstance(node, Probability):
            assert node.interventions == ()
            assert set(node.variables) | set(node.conditioned_on) <= source_nodes
        elif isinstance(node, Summation):
            assert set(node.variables) <= source_nodes


def test_conditional_public_result_alias_and_domain_error_type() -> None:
    assert get_args(ConditionalIdentificationResult) == (
        IdentifiedConditionalEffect,
        NonIdentifiedConditionalEffect,
    )
    assert issubclass(IdentificationDomainError, ValueError)


def test_conditional_identification_rejects_every_wrong_graph_family() -> None:
    raw_directed_bidirected = DirectedBidirectedGraph(nodes=("X", "Y"))
    raw_endpoint_marked = EndpointMarkedGraph(nodes=("X", "Y"))
    ancestral = AncestralGraph.from_graph(raw_endpoint_marked)
    maximal = MaximalAncestralGraph.from_graph(raw_endpoint_marked)
    closed_pag = ClosedPAG.from_mag(maximal)

    invalid_graphs: tuple[object, ...] = (
        raw_directed_bidirected,
        raw_endpoint_marked,
        ancestral,
        maximal,
        closed_pag,
        object(),
        None,
    )
    for invalid_graph in invalid_graphs:
        with pytest.raises(
            IdentificationDomainError,
            match=f"^{re.escape(DOMAIN_MESSAGE)}$",
        ):
            identify_conditional_causal_effect(  # type: ignore[arg-type]
                invalid_graph,
                outcomes="Y",
                interventions="X",
                conditioned_on=(),
            )


def test_graph_domain_is_checked_before_any_selector_is_consumed() -> None:
    class ExplodingIterable:
        def __iter__(self) -> Any:
            raise AssertionError("selector must not be consumed")

    raw = DirectedBidirectedGraph(nodes=("X", "Y"))
    with pytest.raises(
        IdentificationDomainError,
        match=f"^{re.escape(DOMAIN_MESSAGE)}$",
    ):
        identify_conditional_causal_effect(
            raw,  # type: ignore[arg-type]
            outcomes=ExplodingIterable(),  # type: ignore[arg-type]
            interventions=ExplodingIterable(),  # type: ignore[arg-type]
            conditioned_on=ExplodingIterable(),  # type: ignore[arg-type]
        )


def test_conditional_identification_signature_is_keyword_only_after_graph() -> None:
    graph = _dag(nodes=("X", "Z", "Y"), directed=(("X", "Y"),))
    with pytest.raises(TypeError):
        identify_conditional_causal_effect(  # type: ignore[misc]
            graph, "Y", "X", "Z"
        )
    with pytest.raises(TypeError):
        identify_conditional_causal_effect(  # type: ignore[call-arg]
            graph,
            interventions="X",
            conditioned_on="Z",
        )
    with pytest.raises(TypeError):
        identify_conditional_causal_effect(  # type: ignore[call-arg]
            graph,
            outcomes="Y",
            unsupported=True,
        )


def test_selectors_use_set_normalization_and_source_order() -> None:
    graph = _dag(nodes=("B", "D", "A", "C"))
    outcomes = ["C", "A", "C"]
    interventions = (node for node in ("D", "D"))
    conditioned_on = ["B", "B"]

    result = identify_conditional_causal_effect(
        graph,
        outcomes=outcomes,
        interventions=interventions,
        conditioned_on=conditioned_on,
    )
    outcomes.clear()
    conditioned_on.clear()

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.graph is graph
    assert result.outcomes == ("A", "C")
    assert result.interventions == ("D",)
    assert result.conditioned_on == ("B",)
    assert result.rule_two_interventions == ("B",)
    _assert_normalized_ratio(result)
    assert result.formula.free_variables == frozenset({"A", "C"})
    _assert_formula_is_source_bound_and_observational(result)

    reordered = identify_conditional_causal_effect(
        graph,
        outcomes=("A", "C"),
        interventions=("D",),
        conditioned_on=("B",),
    )
    assert reordered == result


def test_unknown_selectors_are_normalized_in_argument_order_before_overlap() -> None:
    graph = _dag(nodes=("X", "Z", "Y"))
    with pytest.raises(GraphValidationError, match="unknown vertex: 'bad-outcome'"):
        identify_conditional_causal_effect(
            graph,
            outcomes="bad-outcome",
            interventions="bad-intervention",
            conditioned_on="bad-condition",
        )
    with pytest.raises(
        GraphValidationError,
        match="unknown vertex: 'bad-intervention'",
    ):
        identify_conditional_causal_effect(
            graph,
            outcomes="Y",
            interventions="bad-intervention",
            conditioned_on="bad-condition",
        )
    with pytest.raises(
        GraphValidationError,
        match="unknown vertex: 'bad-condition'",
    ):
        identify_conditional_causal_effect(
            graph,
            outcomes="X",
            interventions="X",
            conditioned_on="bad-condition",
        )
    with pytest.raises(GraphValidationError, match="seed vertices must be iterable"):
        identify_conditional_causal_effect(
            graph,
            outcomes="X",
            interventions="X",
            conditioned_on=None,  # type: ignore[arg-type]
        )


def test_pairwise_overlap_messages_have_stable_precedence() -> None:
    graph = _dag(nodes=("X", "Y", "Z"))
    cases = (
        (
            ("X",),
            ("X",),
            ("X",),
            OUTCOME_INTERVENTION_OVERLAP,
        ),
        (
            ("Y",),
            ("X",),
            ("Y",),
            OUTCOME_CONDITIONING_OVERLAP,
        ),
        (
            ("Y",),
            ("X",),
            ("X",),
            INTERVENTION_CONDITIONING_OVERLAP,
        ),
        (
            ("Y",),
            ("X",),
            ("X", "Y"),
            OUTCOME_CONDITIONING_OVERLAP,
        ),
    )
    for outcomes, interventions, conditioned_on, message in cases:
        with pytest.raises(
            IdentificationDomainError,
            match=f"^{re.escape(message)}$",
        ):
            identify_conditional_causal_effect(
                graph,
                outcomes=outcomes,
                interventions=interventions,
                conditioned_on=conditioned_on,
            )


def test_empty_outcomes_return_the_unit_only_after_full_validation() -> None:
    graph = _adbg(
        nodes=("Z", "X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    result = identify_conditional_causal_effect(
        graph,
        outcomes=(),
        interventions="X",
        conditioned_on="Z",
    )

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.graph is graph
    assert result.outcomes == ()
    assert result.interventions == ("X",)
    assert result.conditioned_on == ("Z",)
    assert result.rule_two_interventions == ()
    assert result.formula == Product(())
    assert result.formula.free_variables == frozenset()

    empty_graph = _dag(nodes=())
    empty_result = identify_conditional_causal_effect(empty_graph, outcomes=())
    assert isinstance(empty_result, IdentifiedConditionalEffect)
    assert empty_result.formula == Product(())

    with pytest.raises(GraphValidationError, match="unknown vertex: 'missing'"):
        identify_conditional_causal_effect(
            graph,
            outcomes=(),
            interventions="X",
            conditioned_on="missing",
        )
    with pytest.raises(
        IdentificationDomainError,
        match=f"^{re.escape(INTERVENTION_CONDITIONING_OVERLAP)}$",
    ):
        identify_conditional_causal_effect(
            graph,
            outcomes=(),
            interventions="X",
            conditioned_on="X",
        )


def test_nonempty_identified_results_always_normalize_one_id_numerator() -> None:
    graph = _dag(nodes=("X", "Y"), directed=(("X", "Y"),))
    unconditional = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )

    assert isinstance(unconditional, IdentifiedEffect)
    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.conditioned_on == ()
    assert result.rule_two_interventions == ()
    _assert_normalized_ratio(result)
    assert result.formula.numerator == unconditional.formula
    _assert_formula_is_source_bound_and_observational(result)


def test_nonidentified_empty_conditioning_wraps_the_exact_id_witness() -> None:
    graph = _adbg(
        nodes=("X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    direct = identify_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
    )

    assert isinstance(direct, NonIdentifiedEffect)
    assert isinstance(result, NonIdentifiedConditionalEffect)
    assert result.graph is graph
    assert result.outcomes == ("Y",)
    assert result.interventions == ("X",)
    assert result.conditioned_on == ()
    assert result.rule_two_interventions == ()
    assert result.witness == direct.witness


def test_rule_two_deletes_outgoing_edges_from_the_candidate() -> None:
    graph = _dag(nodes=("Z", "Y"), directed=(("Z", "Y"),))
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        conditioned_on="Z",
    )

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.rule_two_interventions == ("Z",)
    _assert_normalized_ratio(result)
    assert result.formula.numerator == Probability("Y", conditioned_on="Z")


def test_rule_two_preserves_incoming_edges_to_the_candidate() -> None:
    graph = _dag(nodes=("Y", "Z"), directed=(("Y", "Z"),))
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        conditioned_on="Z",
    )

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.rule_two_interventions == ()
    _assert_normalized_ratio(result)
    assert result.formula.numerator == Product(
        (
            Probability("Y"),
            Probability("Z", conditioned_on="Y"),
        )
    )


def test_rule_two_preserves_bidirected_edges_at_the_candidate() -> None:
    graph = _adbg(nodes=("Y", "Z"), bidirected=(("Y", "Z"),))
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        conditioned_on="Z",
    )

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.rule_two_interventions == ()
    _assert_normalized_ratio(result)


def test_rule_two_conditions_on_current_interventions() -> None:
    graph = _dag(
        nodes=("X", "Z", "Y"),
        directed=(("X", "Z"), ("X", "Y")),
    )
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        conditioned_on="Z",
    )

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.rule_two_interventions == ("Z",)
    _assert_normalized_ratio(result)


def test_rule_two_conditions_on_other_observed_context() -> None:
    graph = _adbg(
        nodes=("Z", "K", "Y"),
        directed=(("K", "Z"), ("K", "Y")),
        bidirected=(("K", "Y"),),
    )
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        conditioned_on=("K", "Z"),
    )

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.conditioned_on == ("Z", "K")
    assert result.rule_two_interventions == ("Z",)
    _assert_normalized_ratio(result)


def test_rule_two_removes_directed_arrowheads_into_interventions() -> None:
    graph = _adbg(
        nodes=("Z", "A", "X", "B", "Y"),
        directed=(("A", "X"), ("B", "X"), ("B", "Y")),
        bidirected=(("Z", "A"),),
    )
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        conditioned_on="Z",
    )

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.rule_two_interventions == ("Z",)
    _assert_normalized_ratio(result)


def test_rule_two_removes_bidirected_arrowheads_into_interventions() -> None:
    graph = _adbg(
        nodes=("Z", "X", "Y"),
        bidirected=(("Z", "X"), ("X", "Y")),
    )
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        conditioned_on="Z",
    )

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.rule_two_interventions == ("Z",)
    _assert_normalized_ratio(result)


def test_multiple_conditions_reach_a_source_ordered_fixed_point() -> None:
    graph = _dag(
        nodes=("Z2", "Y", "Z1", "K", "X"),
        directed=(("Y", "K"),),
    )
    first = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        conditioned_on=("Z1", "K", "Z2", "Z1"),
    )
    reordered = identify_conditional_causal_effect(
        graph,
        outcomes=("Y", "Y"),
        interventions=("X", "X"),
        conditioned_on=("Z2", "Z1", "K"),
    )

    assert isinstance(first, IdentifiedConditionalEffect)
    assert first == reordered
    assert first.conditioned_on == ("Z2", "Z1", "K")
    assert first.rule_two_interventions == ("Z2", "Z1")
    _assert_normalized_ratio(first)
    _assert_formula_is_source_bound_and_observational(first)


def test_nonidentification_witness_is_bound_to_the_reduced_joint_query() -> None:
    graph = _adbg(
        nodes=("Z", "X", "Y"),
        directed=(("X", "Y"),),
        bidirected=(("X", "Y"),),
    )
    result = identify_conditional_causal_effect(
        graph,
        outcomes="Y",
        interventions="X",
        conditioned_on="Z",
    )

    assert isinstance(result, NonIdentifiedConditionalEffect)
    assert result.graph is graph
    assert result.outcomes == ("Y",)
    assert result.interventions == ("X",)
    assert result.conditioned_on == ("Z",)
    assert result.rule_two_interventions == ("Z",)
    assert isinstance(result.witness, HedgeWitness)
    assert result.witness.graph is graph
    assert result.witness.outcomes == ("Y",)
    assert result.witness.interventions == ("Z", "X")
    assert result.witness.inner_vertices == ("Y",)
    assert result.witness.outer_vertices == ("X", "Y")
    assert result.witness.roots == ("Y",)
    assert result.witness.fixing_sequence == ("Z",)


def test_uai_2006_figure_one_panels_preserve_report_and_adjudication() -> None:
    fixture = _load_fixture("shpitser_pearl_2006_idc_figure1.json")
    assert fixture["sources"]["ucla_technical_report"] == {
        "report": "R-329 UAI proceedings version",
        "url": "https://ftp.cs.ucla.edu/pub/stat_ser/r329-uai.pdf",
        "sha256": ("098b12e5792305495533b9b037ced23b87b429223a993ee9f6fd992213bf9caa"),
    }
    assert fixture["sources"]["author_arxiv_copy"]["sha256"] == (
        "c148ca71be8aab1b90c0f8088ce3d121d8d22ead90f770cd88736d1d6da8717c"
    )
    assert fixture["transcription_source"] == "ucla_technical_report"
    assert fixture["figure_pdf_page"] == 2
    assert fixture["figure_proceedings_page"] == 438
    assert "arbitrary-subquery" in fixture["paper_status_note"]
    assert fixture["panels"]["a"]["reported_claim"] == (
        "P_x(Y) is not identifiable, while P_x(Y|Z) is identifiable because "
        "Rule 2 exchanges observation of Z for intervention on Z"
    )
    assert fixture["panels"]["b"]["reported_claim"] == (
        "P_x(Y) is identifiable and equal to P(Y), while conditioning on Z makes "
        "P_x(Y|Z) nonidentifiable"
    )
    assert fixture["panels"]["c"]["reported_claim"] == (
        "P_x(W|Z) is nonidentifiable and is a counterexample to the soundness of "
        "Tian's conditional-effect algorithm"
    )

    for panel_name in ("a", "b", "c"):
        panel = fixture["panels"][panel_name]
        assert panel["reported_claim"] != panel["adjudicated_claim"]
        graph = _adbg(
            nodes=panel["nodes"],
            directed=panel["directed"],
            bidirected=panel["bidirected"],
        )
        query = panel["query"]
        result = identify_conditional_causal_effect(
            graph,
            outcomes=query["outcomes"],
            interventions=query["interventions"],
            conditioned_on=query["conditioned_on"],
        )

        assert result.graph is graph
        assert result.outcomes == tuple(query["outcomes"])
        assert result.interventions == tuple(query["interventions"])
        assert result.conditioned_on == tuple(query["conditioned_on"])
        assert result.rule_two_interventions == tuple(
            panel["expected_rule_two_interventions"]
        )
        if panel["expected_status"] == "identified":
            assert isinstance(result, IdentifiedConditionalEffect)
            _assert_normalized_ratio(result)
            moved = frozenset(result.rule_two_interventions)
            reduced_outcomes = tuple(
                node
                for node in graph.topology.nodes
                if node in result.outcomes
                or (node in result.conditioned_on and node not in moved)
            )
            reduced = identify_causal_effect(
                graph,
                outcomes=reduced_outcomes,
                interventions=(
                    *result.interventions,
                    *result.rule_two_interventions,
                ),
            )
            assert isinstance(reduced, IdentifiedEffect)
            assert result.formula.numerator == reduced.formula
            _assert_formula_is_source_bound_and_observational(result)
        else:
            assert isinstance(result, NonIdentifiedConditionalEffect)
            expected = panel["expected_joint_witness"]
            assert result.witness.outcomes == tuple(expected["outcomes"])
            assert result.witness.interventions == tuple(expected["interventions"])
            assert result.witness.inner_vertices == tuple(expected["inner_vertices"])
            assert result.witness.outer_vertices == tuple(expected["outer_vertices"])
            assert result.witness.roots == tuple(expected["roots"])
            assert result.witness.fixing_sequence == tuple(expected["fixing_sequence"])


def test_conditional_result_values_are_immutable_factory_values() -> None:
    identified = identify_conditional_causal_effect(
        _dag(nodes=("Z", "Y"), directed=(("Z", "Y"),)),
        outcomes="Y",
        conditioned_on="Z",
    )
    nonidentified = identify_conditional_causal_effect(
        _adbg(
            nodes=("X", "Y"),
            directed=(("X", "Y"),),
            bidirected=(("X", "Y"),),
        ),
        outcomes="Y",
        interventions="X",
    )
    assert isinstance(identified, IdentifiedConditionalEffect)
    assert isinstance(nonidentified, NonIdentifiedConditionalEffect)
    identified_again = identify_conditional_causal_effect(
        identified.graph,
        outcomes=identified.outcomes,
        interventions=identified.interventions,
        conditioned_on=identified.conditioned_on,
    )
    nonidentified_again = identify_conditional_causal_effect(
        nonidentified.graph,
        outcomes=nonidentified.outcomes,
        interventions=nonidentified.interventions,
        conditioned_on=nonidentified.conditioned_on,
    )
    assert identified_again == identified
    assert hash(identified_again) == hash(identified)
    assert nonidentified_again == nonidentified
    assert hash(nonidentified_again) == hash(nonidentified)

    for value in (identified, nonidentified):
        assert not hasattr(value, "__dict__")
        assert not hasattr(value, "__setstate__")
        assert hash(value) == hash(copy.copy(value))
        assert copy.copy(value) is value
        assert copy.deepcopy(value) is value
        with pytest.raises(AttributeError):
            value.outcomes = ()  # type: ignore[misc]
        with pytest.raises(
            TypeError,
            match="^pickle is unsupported for identification results$",
        ):
            pickle.dumps(value)

    with pytest.raises(TypeError):
        IdentifiedConditionalEffect(  # type: ignore[call-arg]
            graph=identified.graph,
            outcomes=identified.outcomes,
            interventions=identified.interventions,
            conditioned_on=identified.conditioned_on,
            rule_two_interventions=identified.rule_two_interventions,
            formula=identified.formula,
        )
    with pytest.raises(TypeError):
        NonIdentifiedConditionalEffect(  # type: ignore[call-arg]
            graph=nonidentified.graph,
            outcomes=nonidentified.outcomes,
            interventions=nonidentified.interventions,
            conditioned_on=nonidentified.conditioned_on,
            rule_two_interventions=nonidentified.rule_two_interventions,
            witness=nonidentified.witness,
        )
    for base in (IdentifiedConditionalEffect, NonIdentifiedConditionalEffect):
        with pytest.raises(TypeError, match="cannot be subclassed"):
            type("MutableConditionalIdentificationSubclass", (base,), {})


def test_one_thousand_conditions_do_not_use_python_recursion() -> None:
    conditions = tuple(f"Z{index:04d}" for index in range(1_000))
    graph = _dag(nodes=("Y", *conditions))

    previous_recursion_limit = sys.getrecursionlimit()
    current_depth = 0
    frame: Any = sys._getframe()
    while frame is not None:
        current_depth += 1
        frame = frame.f_back
    sys.setrecursionlimit(current_depth + 50)
    try:
        result = identify_conditional_causal_effect(
            graph,
            outcomes="Y",
            conditioned_on=reversed(conditions),
        )
    finally:
        sys.setrecursionlimit(previous_recursion_limit)

    assert isinstance(result, IdentifiedConditionalEffect)
    assert result.conditioned_on == conditions
    assert result.rule_two_interventions == conditions
    _assert_normalized_ratio(result)
    assert result.formula.numerator == Probability("Y")
    _assert_formula_is_source_bound_and_observational(result)
