from __future__ import annotations

import copy
import inspect
import itertools
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, get_args

import pytest

import causal_graph_algorithms as cga
from causal_graph_algorithms import (
    CausalPAG,
    ClosedPAG,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    IdentifiedPAGEffect,
    IdentificationDomainError,
    MaximalAncestralGraph,
    NonIdentifiedPAGEffect,
    PAGIdentificationResult,
    Probability,
    ProbabilityExpression,
    Product,
    Ratio,
    Summation,
    identify_pag_causal_effect,
    probability_expression_to_tex,
)

from .id_oracles import (
    TinyADMG,
    all_bow_admgs_n3,
    bow_admg_ci_signature_by_canonical_dag,
    evaluate_probability_expression,
    exact_binary_interventional_probability,
    exact_binary_observational_distribution,
)
from .oracles import (
    ConditionalIndependenceSignature,
    EndpointMarkAtom,
    MixedEdgeAtom,
    no_selection_census,
)
from .pag_idp_oracles import (
    OracleExpression,
    OracleProbability,
    OracleProduct,
    OracleRatio,
    OracleSummation,
    OracleUnit,
    TinyPAG,
    all_nonempty_outcome_queries,
    identify_pag_by_definition,
)


DOMAIN_MESSAGE = "PAG causal-effect identification requires a causal PAG proof view"
OVERLAP_MESSAGE = "outcomes and interventions must be disjoint"


def _load_fixture(name: str) -> dict[str, Any]:
    path = Path(__file__).parents[1] / "fixtures" / "papers" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _endpoint_atoms(raw: object) -> tuple[EndpointMarkAtom, ...]:
    assert isinstance(raw, list)
    atoms = tuple(tuple(edge) for edge in raw)
    assert all(
        len(edge) == 4
        and all(isinstance(value, str) for value in edge)
        and edge[1] in {"tail", "arrow", "circle"}
        and edge[3] in {"tail", "arrow", "circle"}
        for edge in atoms
    )
    return atoms  # type: ignore[return-value]


def _mag_from_endpoint_atoms(
    nodes: tuple[str, ...], edges: tuple[EndpointMarkAtom, ...]
) -> MaximalAncestralGraph:
    return MaximalAncestralGraph.from_graph(
        EndpointMarkedGraph(nodes=nodes, edges=edges)
    )


def _mag_from_mixed_atoms(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> MaximalAncestralGraph:
    return _mag_from_endpoint_atoms(
        nodes,
        tuple(
            (
                left,
                "arrow" if arrow_left else "tail",
                right,
                "arrow" if arrow_right else "tail",
            )
            for left, arrow_left, right, arrow_right in edges
        ),
    )


def _causal_pag_from_seed(
    nodes: tuple[str, ...], seed_edges: tuple[EndpointMarkAtom, ...]
) -> CausalPAG:
    return CausalPAG.from_mag(_mag_from_endpoint_atoms(nodes, seed_edges))


def _pag_from_fixture(raw: dict[str, Any]) -> CausalPAG:
    nodes = tuple(raw["nodes"])
    pag = _causal_pag_from_seed(nodes, _endpoint_atoms(raw["seed_mag"]))
    assert _pag_atoms(pag) == _endpoint_atoms(raw["edges"])
    return pag


def _pag_atoms(graph: CausalPAG) -> tuple[EndpointMarkAtom, ...]:
    return tuple(
        (left, left_mark.value, right, right_mark.value)
        for left, left_mark, right, right_mark in graph.topology.edges
    )


def _empty_pag(nodes: tuple[str, ...]) -> CausalPAG:
    return _causal_pag_from_seed(nodes, ())


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


def _public_ir_from_oracle_expression(
    expression: OracleExpression,
) -> ProbabilityExpression:
    """Translate only the independent test AST, preserving shared objects."""

    memo: dict[int, ProbabilityExpression] = {}

    def translate(current: OracleExpression) -> ProbabilityExpression:
        cached = memo.get(id(current))
        if cached is not None:
            return cached
        if isinstance(current, OracleProbability):
            translated: ProbabilityExpression = Probability(
                current.variables,
                conditioned_on=current.conditioned_on,
            )
        elif isinstance(current, OracleUnit):
            translated = Product(())
        elif isinstance(current, OracleProduct):
            translated = Product(tuple(translate(factor) for factor in current.factors))
        elif isinstance(current, OracleSummation):
            translated = Summation(current.variables, translate(current.body))
        else:
            assert isinstance(current, OracleRatio)
            translated = Ratio(
                translate(current.numerator),
                translate(current.denominator),
            )
        memo[id(current)] = translated
        return translated

    return translate(expression)


def _tiny_admg_from_mixed_atoms(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> TinyADMG:
    positions = {node: index for index, node in enumerate(nodes)}
    directed: set[tuple[int, int]] = set()
    bidirected: set[tuple[int, int]] = set()
    for left, arrow_left, right, arrow_right in edges:
        left_index = positions[left]
        right_index = positions[right]
        if not arrow_left and arrow_right:
            directed.add((left_index, right_index))
        elif arrow_left and not arrow_right:
            directed.add((right_index, left_index))
        elif arrow_left and arrow_right:
            bidirected.add((left_index, right_index))
        else:
            raise AssertionError("no-selection source MAG has an undirected edge")
    return TinyADMG(
        tuple(range(len(nodes))),
        frozenset(directed),
        frozenset(bidirected),
    )


def _compatible_bow_admgs_n3() -> dict[
    ConditionalIndependenceSignature, tuple[TinyADMG, ...]
]:
    labels = ("A", "B", "C")
    grouped: dict[ConditionalIndependenceSignature, list[TinyADMG]] = {}
    for graph in all_bow_admgs_n3():
        signature = bow_admg_ci_signature_by_canonical_dag(graph, labels)
        grouped.setdefault(signature, []).append(graph)
    return {signature: tuple(graphs) for signature, graphs in grouped.items()}


def _exact_binary_numeric_comparison_count(
    expression: ProbabilityExpression,
    graph: TinyADMG,
    *,
    node_labels: tuple[str, ...],
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
    assignment_labels: tuple[str, ...],
) -> int:
    positions = {label: index for index, label in enumerate(node_labels)}
    outcome_nodes = tuple(positions[label] for label in outcomes)
    intervention_nodes = tuple(positions[label] for label in interventions)
    comparisons = 0
    for variant in (0, 1):
        table = exact_binary_observational_distribution(graph, variant)
        for assignment in itertools.product((0, 1), repeat=len(assignment_labels)):
            environment = dict(zip(assignment_labels, assignment, strict=True))
            node_values = {
                positions[label]: value for label, value in environment.items()
            }
            expected = exact_binary_interventional_probability(
                graph,
                outcomes=outcome_nodes,
                interventions=intervention_nodes,
                values=node_values,
                variant=variant,
            )
            actual = evaluate_probability_expression(
                expression,
                table=table,
                node_labels=node_labels,
                environment=environment,
            )
            assert actual == expected
            comparisons += 1
    return comparisons


def _assert_exact_query_observational_formula(
    result: IdentifiedPAGEffect,
) -> None:
    source = set(result.graph.topology.nodes)
    free = result.formula.free_variables
    assert set(result.outcomes) <= free
    assert free <= set(result.outcomes) | set(result.interventions)
    for node in _walk_expression(result.formula):
        if isinstance(node, Probability):
            assert node.interventions == ()
            assert set(node.variables) | set(node.conditioned_on) <= source
        elif isinstance(node, Summation):
            assert set(node.variables) <= source


def test_pag_identification_public_alias_and_exact_signature() -> None:
    assert get_args(PAGIdentificationResult) == (
        IdentifiedPAGEffect,
        NonIdentifiedPAGEffect,
    )
    signature = inspect.signature(identify_pag_causal_effect)
    assert tuple(signature.parameters) == (
        "graph",
        "outcomes",
        "interventions",
    )
    assert signature.parameters["graph"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["outcomes"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["outcomes"].default is inspect.Parameter.empty
    assert signature.parameters["interventions"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["interventions"].default == ()
    assert not hasattr(cga, "PAGIdentificationResourceLimitError")

    graph = _empty_pag(("X", "Y"))
    with pytest.raises(TypeError):
        identify_pag_causal_effect(graph, "Y", "X")  # type: ignore[misc]
    with pytest.raises(TypeError):
        identify_pag_causal_effect(graph, interventions="X")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        identify_pag_causal_effect(  # type: ignore[call-arg]
            graph,
            outcomes="Y",
            max_frames=1,
        )


def test_pag_identification_rejects_every_wrong_proof_view_exactly() -> None:
    raw_endpoint = EndpointMarkedGraph(nodes=("X", "Y"))
    mag = MaximalAncestralGraph.from_graph(raw_endpoint)
    invalid_graphs: tuple[object, ...] = (
        raw_endpoint,
        mag,
        ClosedPAG.from_mag(mag),
        DirectedBidirectedGraph(nodes=("X", "Y")),
        DAG.from_graph(DirectedBidirectedGraph(nodes=("X", "Y"))),
        object(),
        None,
    )
    for invalid_graph in invalid_graphs:
        with pytest.raises(
            IdentificationDomainError,
            match=f"^{re.escape(DOMAIN_MESSAGE)}$",
        ):
            identify_pag_causal_effect(  # type: ignore[arg-type]
                invalid_graph,
                outcomes="Y",
                interventions="X",
            )


def test_pag_graph_domain_is_checked_before_selector_consumption() -> None:
    class ExplodingIterable:
        def __iter__(self) -> Any:
            raise AssertionError("selector must not be consumed")

    with pytest.raises(
        IdentificationDomainError,
        match=f"^{re.escape(DOMAIN_MESSAGE)}$",
    ):
        identify_pag_causal_effect(
            EndpointMarkedGraph(nodes=("X", "Y")),  # type: ignore[arg-type]
            outcomes=ExplodingIterable(),  # type: ignore[arg-type]
            interventions=ExplodingIterable(),  # type: ignore[arg-type]
        )


def test_pag_selectors_copy_set_normalize_and_follow_source_order() -> None:
    graph = _empty_pag(("B", "D", "A", "C"))
    outcomes = ["C", "A", "C"]
    interventions = (node for node in ("D", "D"))
    result = identify_pag_causal_effect(
        graph,
        outcomes=outcomes,
        interventions=interventions,
    )
    outcomes.clear()

    assert isinstance(result, IdentifiedPAGEffect)
    assert result.graph is graph
    assert result.outcomes == ("A", "C")
    assert result.interventions == ("D",)
    _assert_exact_query_observational_formula(result)
    assert (
        identify_pag_causal_effect(
            graph,
            outcomes=("A", "C"),
            interventions=("D",),
        )
        == result
    )


def test_unknown_pag_selectors_precede_overlap_in_argument_order() -> None:
    graph = _empty_pag(("X", "Y"))
    with pytest.raises(GraphValidationError, match="unknown vertex: 'bad-outcome'"):
        identify_pag_causal_effect(
            graph,
            outcomes="bad-outcome",
            interventions="bad-intervention",
        )
    with pytest.raises(
        GraphValidationError,
        match="unknown vertex: 'bad-intervention'",
    ):
        identify_pag_causal_effect(
            graph,
            outcomes="Y",
            interventions="bad-intervention",
        )
    with pytest.raises(GraphValidationError, match="seed vertices must be iterable"):
        identify_pag_causal_effect(
            graph,
            outcomes=None,  # type: ignore[arg-type]
        )

    with pytest.raises(
        IdentificationDomainError,
        match=f"^{OVERLAP_MESSAGE}$",
    ):
        identify_pag_causal_effect(
            graph,
            outcomes=("Y", "X", "Y"),
            interventions=("X", "X"),
        )


def test_empty_outcomes_are_unit_only_after_full_validation() -> None:
    graph = _empty_pag(("X", "Y"))
    result = identify_pag_causal_effect(
        graph,
        outcomes=(),
        interventions="X",
    )
    assert isinstance(result, IdentifiedPAGEffect)
    assert result.graph is graph
    assert result.outcomes == ()
    assert result.interventions == ("X",)
    assert result.formula == Product(())
    assert result.formula.free_variables == frozenset()

    empty = identify_pag_causal_effect(_empty_pag(()), outcomes=())
    assert isinstance(empty, IdentifiedPAGEffect)
    assert empty.formula == Product(())

    with pytest.raises(GraphValidationError, match="unknown vertex"):
        identify_pag_causal_effect(
            graph,
            outcomes=(),
            interventions="missing",
        )


def test_empty_intervention_is_exact_canonical_observational_marginal() -> None:
    graph = _empty_pag(("A", "B", "C"))
    result = identify_pag_causal_effect(graph, outcomes="B")
    q = Probability(("A", "B", "C"))
    expected = Summation(("A", "C"), q)

    assert isinstance(result, IdentifiedPAGEffect)
    assert result.interventions == ()
    assert result.formula == expected
    assert probability_expression_to_tex(result.formula) == (
        r"\sum_{\textnormal{A},\,\textnormal{C}}\left["
        r"P\!\left(\textnormal{A},\,\textnormal{B},\,\textnormal{C}\right)"
        r"\right]"
    )
    _assert_exact_query_observational_formula(result)

    joint = identify_pag_causal_effect(
        graph,
        outcomes=("A", "B", "C"),
    )
    assert isinstance(joint, IdentifiedPAGEffect)
    assert joint.formula == Probability(("A", "B", "C"))


def test_line_six_prop_two_ast_is_flattened_and_shares_q() -> None:
    figure = _load_fixture("jaber_2022_r86_idp_figures.json")["figures"]["4a"]
    graph = _pag_from_fixture(figure)
    result = identify_pag_causal_effect(
        graph,
        outcomes=("A", "C", "D"),
        interventions="B",
    )
    q = Probability(("A", "B", "C", "D"))
    expected = Ratio(
        Product((q, Summation(("B", "C"), q))),
        Summation("C", q),
    )

    assert isinstance(result, IdentifiedPAGEffect)
    assert result.formula == expected
    assert isinstance(result.formula, Ratio)
    assert isinstance(result.formula.numerator, Product)
    first, marginal = result.formula.numerator.factors
    assert isinstance(marginal, Summation)
    assert isinstance(result.formula.denominator, Summation)
    assert first is marginal.body
    assert first is result.formula.denominator.body
    _assert_exact_query_observational_formula(result)


def test_line_six_keeps_shared_q_denominator_when_possde_is_bucket() -> None:
    graph = _empty_pag(("A", "B"))
    result = identify_pag_causal_effect(
        graph,
        outcomes="B",
        interventions="A",
    )
    q = Probability(("A", "B"))
    expected = Ratio(Product((q, Summation("A", q))), q)

    assert isinstance(result, IdentifiedPAGEffect)
    assert result.formula == expected
    assert isinstance(result.formula, Ratio)
    assert isinstance(result.formula.numerator, Product)
    assert result.formula.denominator is result.formula.numerator.factors[0]


def test_r86_paper_panels_match_adjudicated_public_status_and_formula_shape() -> None:
    fixture = _load_fixture("jaber_2022_r86_idp_figures.json")
    status_types = {
        "identified": IdentifiedPAGEffect,
        "not_identified": NonIdentifiedPAGEffect,
    }
    for figure_name in ("2", "3a", "4a", "9"):
        figure = fixture["figures"][figure_name]
        graph = _pag_from_fixture(figure)
        for query in figure["queries"]:
            result = identify_pag_causal_effect(
                graph,
                outcomes=query["outcomes"],
                interventions=query["interventions"],
            )
            assert isinstance(result, status_types[query["independent_status"]])
            assert result.graph is graph
            if isinstance(result, IdentifiedPAGEffect):
                _assert_exact_query_observational_formula(result)

    figure9 = fixture["figures"]["9"]
    figure9_result = identify_pag_causal_effect(
        _pag_from_fixture(figure9),
        outcomes="Y",
        interventions="X",
    )
    assert isinstance(figure9_result, IdentifiedPAGEffect)
    assert isinstance(figure9_result.formula, Summation)
    assert figure9_result.formula.variables == ("W",)
    assert isinstance(figure9_result.formula.body, Product)
    assert figure9_result.formula.body.factors[-1] == Probability(
        "W", conditioned_on="X"
    )
    explicit_unit_ratios = [
        node
        for node in _walk_expression(figure9_result.formula)
        if isinstance(node, Ratio) and node.denominator == Product(())
    ]
    assert len(explicit_unit_ratios) == 1
    assert isinstance(explicit_unit_ratios[0].numerator, Product)
    assert len(explicit_unit_ratios[0].numerator.factors) == 2

    figure4 = fixture["figures"]["4a"]
    figure4_failure = identify_pag_causal_effect(
        _pag_from_fixture(figure4),
        outcomes=("B", "C", "D"),
        interventions="A",
    )
    assert isinstance(figure4_failure, NonIdentifiedPAGEffect)
    assert figure4_failure.terminal_target_vertices == ("B", "D")
    assert figure4_failure.terminal_kernel_vertices == ("A", "B", "D")


def test_line_nine_uses_p_c_buckets_and_preserves_terminal_c_t() -> None:
    # A->X->B is Markov equivalent to both reversed noncollider chains, so its
    # causal PAG is A o-o X o-o B. P_C has singleton buckets for C={A,B}; P_T
    # has one crossing bucket and reaches a different terminal frame despite
    # matching the coarse n3/n4 status totals.
    graph = _causal_pag_from_seed(
        ("A", "X", "B"),
        (
            ("A", "tail", "X", "arrow"),
            ("X", "tail", "B", "arrow"),
        ),
    )
    assert _pag_atoms(graph) == (
        ("A", "circle", "X", "circle"),
        ("X", "circle", "B", "circle"),
    )
    result = identify_pag_causal_effect(
        graph,
        outcomes=("A", "B"),
        interventions="X",
    )
    assert isinstance(result, NonIdentifiedPAGEffect)
    assert result.graph is graph
    assert result.outcomes == ("A", "B")
    assert result.interventions == ("X",)
    assert result.terminal_target_vertices == ("A",)
    assert result.terminal_kernel_vertices == ("A", "X", "B")
    assert not hasattr(result, "witness")
    assert not hasattr(result, "auxiliary_variables")


def test_icml_overlap_region_keeps_a_nonempty_line_nine_denominator() -> None:
    fixture = _load_fixture("jaber_2019_icml_idp_figure2.json")
    result = identify_pag_causal_effect(
        _pag_from_fixture(fixture),
        outcomes=fixture["query"]["outcomes"],
        interventions=fixture["query"]["interventions"],
    )
    assert isinstance(result, IdentifiedPAGEffect)
    assert isinstance(result.formula, Ratio)
    assert result.formula.denominator != Product(())
    assert {"Y4", "Y5"} <= result.formula.denominator.free_variables
    _assert_exact_query_observational_formula(result)


def test_nonempty_intervention_auxiliary_is_weighted_bound_and_numerically_exact() -> (
    None
):
    graph = _causal_pag_from_seed(
        ("A", "B", "C"),
        (
            ("A", "arrow", "B", "tail"),
            ("A", "arrow", "C", "tail"),
        ),
    )
    assert _pag_atoms(graph) == (
        ("A", "arrow", "B", "circle"),
        ("A", "arrow", "C", "circle"),
    )
    result = identify_pag_causal_effect(
        graph,
        outcomes="B",
        interventions="A",
    )
    assert isinstance(result, IdentifiedPAGEffect)
    assert isinstance(result.formula, Summation)
    assert result.formula.variables == ("C",)
    assert isinstance(result.formula.body, Product)
    raw, weight = result.formula.body.factors
    assert raw.free_variables == frozenset({"A", "B", "C"})
    assert weight == Probability("C", conditioned_on="A")
    assert result.formula.free_variables == frozenset({"A", "B"})
    assert not hasattr(result, "auxiliary_variables")

    # The finite SCM produces strictly positive P(V), the published domain.
    # Raw ratios are only promised almost everywhere on supported intervention
    # values; these exact checks intentionally make no off-support assertion.
    scm = TinyADMG(
        (0, 1, 2),
        frozenset({(1, 0), (2, 0)}),
        frozenset(),
    )
    for variant in (0, 1):
        table = exact_binary_observational_distribution(scm, variant)
        for a, b in itertools.product((0, 1), repeat=2):
            expected = exact_binary_interventional_probability(
                scm,
                outcomes=(1,),
                interventions=(0,),
                values={0: a, 1: b},
                variant=variant,
            )
            actual = evaluate_probability_expression(
                result.formula,
                table=table,
                node_labels=("A", "B", "C"),
                environment={"A": a, "B": b},
            )
            assert actual == expected


def test_pag_identification_values_are_factory_only_sealed_and_immutable() -> None:
    identified = identify_pag_causal_effect(
        _empty_pag(("A", "B")),
        outcomes="B",
        interventions="A",
    )
    nonidentified = identify_pag_causal_effect(
        _causal_pag_from_seed(
            ("A", "X", "B"),
            (
                ("A", "tail", "X", "arrow"),
                ("X", "tail", "B", "arrow"),
            ),
        ),
        outcomes=("A", "B"),
        interventions="X",
    )
    assert isinstance(identified, IdentifiedPAGEffect)
    assert isinstance(nonidentified, NonIdentifiedPAGEffect)

    identified_same = identify_pag_causal_effect(
        identified.graph,
        outcomes=identified.outcomes,
        interventions=identified.interventions,
    )
    nonidentified_same = identify_pag_causal_effect(
        nonidentified.graph,
        outcomes=nonidentified.outcomes,
        interventions=nonidentified.interventions,
    )
    for value, same in (
        (identified, identified_same),
        (nonidentified, nonidentified_same),
    ):
        assert value == same
        assert hash(value) == hash(same)
        assert copy.copy(value) is value
        assert copy.deepcopy(value) is value
        assert not hasattr(value, "__dict__")
        assert not hasattr(value, "__setstate__")
        with pytest.raises(
            TypeError,
            match="^pickle is unsupported for identification results$",
        ):
            pickle.dumps(value)
        with pytest.raises((AttributeError, TypeError)):
            value.outcomes = ("changed",)  # type: ignore[misc]

    with pytest.raises(
        TypeError,
        match="^IdentifiedPAGEffect values are factory-created$",
    ):
        IdentifiedPAGEffect(  # type: ignore[call-arg]
            object(),
            graph=identified.graph,
            outcomes=identified.outcomes,
            interventions=identified.interventions,
            formula=identified.formula,
        )
    with pytest.raises(
        TypeError,
        match="^NonIdentifiedPAGEffect values are factory-created$",
    ):
        NonIdentifiedPAGEffect(  # type: ignore[call-arg]
            object(),
            graph=nonidentified.graph,
            outcomes=nonidentified.outcomes,
            interventions=nonidentified.interventions,
            terminal_target_vertices=nonidentified.terminal_target_vertices,
            terminal_kernel_vertices=nonidentified.terminal_kernel_vertices,
        )
    for base in (IdentifiedPAGEffect, NonIdentifiedPAGEffect):
        with pytest.raises(TypeError, match="cannot be subclassed"):
            type("MutablePAGIdentificationSubclass", (base,), {})

    assert IdentifiedPAGEffect.__slots__ == (
        "graph",
        "outcomes",
        "interventions",
        "formula",
    )
    assert NonIdentifiedPAGEffect.__slots__ == (
        "graph",
        "outcomes",
        "interventions",
        "terminal_target_vertices",
        "terminal_kernel_vertices",
    )


def test_n3_all_209_public_queries_match_the_independent_oracle() -> None:
    nodes = ("A", "B", "C")
    census = no_selection_census(nodes)
    identified = 0
    structurally_equal_nonempty_intervention = 0
    nonidentified = 0
    for members, pag_edges in census.classes.values():
        graph = CausalPAG.from_mag(_mag_from_mixed_atoms(nodes, members[0]))
        assert _pag_atoms(graph) == pag_edges
        oracle_pag = TinyPAG(nodes, pag_edges)
        for outcomes, interventions in all_nonempty_outcome_queries(nodes):
            expected = identify_pag_by_definition(
                oracle_pag,
                outcomes=outcomes,
                interventions=interventions,
            )
            actual = identify_pag_causal_effect(
                graph,
                outcomes=outcomes,
                interventions=interventions,
            )
            assert isinstance(actual, IdentifiedPAGEffect) is expected.identified
            if isinstance(actual, IdentifiedPAGEffect):
                identified += 1
                _assert_exact_query_observational_formula(actual)
                assert expected.formula is not None
                if interventions:
                    assert actual.formula == _public_ir_from_oracle_expression(
                        expected.formula
                    )
                    structurally_equal_nonempty_intervention += 1
                if not interventions:
                    q = Probability(nodes)
                    removed = tuple(node for node in nodes if node not in outcomes)
                    expected_observational: ProbabilityExpression = q
                    if removed:
                        expected_observational = Summation(removed, q)
                    assert actual.formula == expected_observational
            else:
                nonidentified += 1
                assert isinstance(actual, NonIdentifiedPAGEffect)
                assert actual.terminal_target_vertices == (
                    expected.terminal_target_vertices
                )
                assert actual.terminal_kernel_vertices == (
                    expected.terminal_kernel_vertices
                )
    assert structurally_equal_nonempty_intervention == 51
    assert (identified, nonidentified) == (128, 81)


def test_n4_all_16120_public_queries_match_the_independent_oracle() -> None:
    nodes = ("A", "B", "C", "D")
    census = no_selection_census(nodes)
    queries = all_nonempty_outcome_queries(nodes)
    assert len(census.classes) == 248
    assert len(queries) == 65

    identified = 0
    structurally_equal = 0
    nonidentified = 0
    for members, pag_edges in census.classes.values():
        graph = CausalPAG.from_mag(_mag_from_mixed_atoms(nodes, members[0]))
        assert _pag_atoms(graph) == pag_edges
        oracle_pag = TinyPAG(nodes, pag_edges)
        for outcomes, interventions in queries:
            expected = identify_pag_by_definition(
                oracle_pag,
                outcomes=outcomes,
                interventions=interventions,
            )
            expected_outcomes = tuple(node for node in nodes if node in outcomes)
            expected_interventions = tuple(
                node for node in nodes if node in interventions
            )
            # Oppose caller order to the graph order on every nonsingleton role.
            actual = identify_pag_causal_effect(
                graph,
                outcomes=tuple(reversed(expected_outcomes)),
                interventions=tuple(reversed(expected_interventions)),
            )

            assert actual.graph is graph
            assert actual.outcomes == expected_outcomes
            assert actual.interventions == expected_interventions
            assert isinstance(actual, IdentifiedPAGEffect) is expected.identified
            if isinstance(actual, IdentifiedPAGEffect):
                identified += 1
                _assert_exact_query_observational_formula(actual)
                assert expected.formula is not None
                assert actual.formula == _public_ir_from_oracle_expression(
                    expected.formula
                )
                structurally_equal += 1
            else:
                nonidentified += 1
                assert isinstance(actual, NonIdentifiedPAGEffect)
                assert actual.terminal_target_vertices == (
                    expected.terminal_target_vertices
                )
                assert actual.terminal_kernel_vertices == (
                    expected.terminal_kernel_vertices
                )

    assert structurally_equal == identified == 8_906
    assert nonidentified == 7_214
    assert identified + nonidentified == 16_120


def test_n3_all_compatible_bow_admgs_give_13664_exact_query_values() -> None:
    """Replay all 128 ID rows over all 200 CI-compatible bow ADMGs."""

    nodes = ("A", "B", "C")
    census = no_selection_census(nodes)
    compatible_bow_admgs = _compatible_bow_admgs_n3()
    assert len(compatible_bow_admgs) == 11
    assert sum(map(len, compatible_bow_admgs.values())) == 200

    bow_exact_query_values = 0
    identified_rows = 0
    for signature, (members, pag_edges) in census.classes.items():
        graph = CausalPAG.from_mag(_mag_from_mixed_atoms(nodes, members[0]))
        oracle_pag = TinyPAG(nodes, pag_edges)
        for outcomes, interventions in all_nonempty_outcome_queries(nodes):
            oracle_result = identify_pag_by_definition(
                oracle_pag,
                outcomes=outcomes,
                interventions=interventions,
            )
            if not oracle_result.identified:
                continue
            identified_rows += 1
            actual = identify_pag_causal_effect(
                graph,
                outcomes=outcomes,
                interventions=interventions,
            )
            assert isinstance(actual, IdentifiedPAGEffect)
            assignment_labels = (*actual.outcomes, *actual.interventions)
            for bow_admg in compatible_bow_admgs[signature]:
                bow_exact_query_values += _exact_binary_numeric_comparison_count(
                    actual.formula,
                    bow_admg,
                    node_labels=nodes,
                    outcomes=actual.outcomes,
                    interventions=actual.interventions,
                    assignment_labels=assignment_labels,
                )

    assert identified_rows == 128
    assert bow_exact_query_values == 13_664


def test_n3_source_mag_members_give_8672_exact_full_cell_values() -> None:
    """Replay all 128 ID rows over 56 source MAGs and all binary cells."""

    nodes = ("A", "B", "C")
    census = no_selection_census(nodes)
    assert len(census.mags) == 56

    source_mag_exact_full_cell_values = 0
    identified_rows = 0
    for members, pag_edges in census.classes.values():
        graph = CausalPAG.from_mag(_mag_from_mixed_atoms(nodes, members[0]))
        oracle_pag = TinyPAG(nodes, pag_edges)
        source_admgs = tuple(
            _tiny_admg_from_mixed_atoms(nodes, member) for member in members
        )
        for outcomes, interventions in all_nonempty_outcome_queries(nodes):
            oracle_result = identify_pag_by_definition(
                oracle_pag,
                outcomes=outcomes,
                interventions=interventions,
            )
            if not oracle_result.identified:
                continue
            identified_rows += 1
            actual = identify_pag_causal_effect(
                graph,
                outcomes=outcomes,
                interventions=interventions,
            )
            assert isinstance(actual, IdentifiedPAGEffect)
            for source_admg in source_admgs:
                source_mag_exact_full_cell_values += (
                    _exact_binary_numeric_comparison_count(
                        actual.formula,
                        source_admg,
                        node_labels=nodes,
                        outcomes=actual.outcomes,
                        interventions=actual.interventions,
                        assignment_labels=nodes,
                    )
                )

    assert identified_rows == 128
    assert source_mag_exact_full_cell_values == 8_672


def test_pag_identification_uses_an_explicit_stack_under_low_recursion_limit() -> None:
    leaves = tuple(f"L{index:03d}" for index in range(128))
    sink = "S"
    nodes = (*leaves, sink)
    graph = _causal_pag_from_seed(
        nodes,
        tuple((leaf, "tail", sink, "arrow") for leaf in leaves),
    )

    previous_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        result = identify_pag_causal_effect(
            graph,
            outcomes=leaves[0],
            interventions=leaves[1],
        )
    finally:
        sys.setrecursionlimit(previous_limit)
    assert isinstance(result, IdentifiedPAGEffect)
    assert result.formula.free_variables <= frozenset({leaves[0], leaves[1]})
