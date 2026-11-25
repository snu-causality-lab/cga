"""Definition-level and exhaustive scientific adjudication for UAI-06 IDC."""

from __future__ import annotations

import itertools
from collections import Counter

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DirectedBidirectedGraph,
    IdentifiedConditionalEffect,
    NonIdentifiedConditionalEffect,
    Probability,
    Product,
    Ratio,
    Summation,
    identify_conditional_causal_effect,
)

from .id_oracles import (
    LiteralIDCReduction,
    TinyADMG,
    all_bow_admgs_n3,
    all_conditional_queries_n3,
    evaluate_probability_expression,
    exact_binary_conditional_interventional_probability,
    exact_binary_observational_distribution,
    exact_query_hedges,
    literal_exact_query_hedges,
    literal_idc_one_pass_reduction,
    literal_idc_reduction,
    obsolete_idc_corollary_two_flag,
    validate_literal_hedge_by_definition,
)


def _production_graph(graph: TinyADMG) -> AcyclicDirectedBidirectedGraph:
    return AcyclicDirectedBidirectedGraph.from_graph(
        DirectedBidirectedGraph(
            nodes=tuple(str(node) for node in graph.nodes),
            directed=tuple(
                (str(parent), str(child)) for parent, child in graph.directed
            ),
            bidirected=tuple(
                (str(left), str(right)) for left, right in graph.bidirected
            ),
        )
    )


def _ordered_nodes(graph: TinyADMG, vertices: frozenset[int]) -> tuple[int, ...]:
    return tuple(node for node in graph.nodes if node in vertices)


def _ordered_labels(graph: TinyADMG, vertices: frozenset[int]) -> tuple[str, ...]:
    return tuple(str(node) for node in graph.nodes if node in vertices)


def _assert_observational_source_bound_formula(
    result: IdentifiedConditionalEffect,
    *,
    source_nodes: frozenset[str],
    context: str,
) -> None:
    formula = result.formula
    assert isinstance(formula, Ratio), context
    assert isinstance(formula.denominator, Summation), context
    assert formula.denominator.variables == result.outcomes, context
    assert formula.denominator.body is formula.numerator, context
    residual_conditions = set(result.conditioned_on) - set(
        result.rule_two_interventions
    )
    assert set(result.outcomes) | residual_conditions <= formula.free_variables, context
    assert formula.free_variables <= (
        set(result.outcomes) | set(result.interventions) | set(result.conditioned_on)
    ), context

    pending = [formula]
    while pending:
        expression = pending.pop()
        if isinstance(expression, Probability):
            assert expression.interventions == (), context
            assert (
                set(expression.variables)
                | set(expression.conditioned_on)
                | set(expression.interventions)
            ) <= source_nodes, context
        elif isinstance(expression, Product):
            pending.extend(expression.factors)
        elif isinstance(expression, Summation):
            assert set(expression.variables) <= source_nodes, context
            pending.append(expression.body)
        elif isinstance(expression, Ratio):
            pending.extend((expression.numerator, expression.denominator))
        else:
            raise AssertionError(f"unknown probability-expression node: {expression!r}")


def _literal_reduction_census() -> tuple[Counter[str], Counter[tuple[int, int, int]]]:
    counts: Counter[str] = Counter()
    role_counts: Counter[tuple[int, int, int]] = Counter()

    for graph_index, graph in enumerate(all_bow_admgs_n3()):
        for interventions, outcomes, conditions in all_conditional_queries_n3():
            context = (
                f"graph_index={graph_index}, directed={sorted(graph.directed)}, "
                f"bidirected={sorted(graph.bidirected)}, "
                f"interventions={sorted(interventions)}, outcomes={sorted(outcomes)}, "
                f"conditions={sorted(conditions)}"
            )
            reduction = literal_idc_reduction(
                graph,
                interventions=interventions,
                outcomes=outcomes,
                conditions=conditions,
            )
            one_pass = literal_idc_one_pass_reduction(
                graph,
                interventions=interventions,
                outcomes=outcomes,
                conditions=conditions,
            )

            counts["rows"] += 1
            counts["moves"] += len(reduction.rule_two_interventions)
            counts["queries_moved"] += bool(reduction.rule_two_interventions)
            counts["all_nonempty_conditions_moved"] += bool(conditions) and not (
                reduction.residual_conditions
            )
            counts["residual_condition_rows"] += bool(reduction.residual_conditions)
            counts["identified"] += not reduction.id_failed
            counts["nonidentified"] += reduction.id_failed
            counts["fixed_point_one_pass_divergence"] += reduction != one_pass
            role_counts[
                (
                    len(conditions),
                    len(reduction.rule_two_interventions),
                    len(reduction.residual_conditions),
                )
            ] += 1

            if reduction.id_failed:
                outcome_key = (
                    "nonid_residual"
                    if reduction.residual_conditions
                    else "nonid_no_residual"
                )
                literal_witnesses = literal_exact_query_hedges(
                    graph,
                    interventions=reduction.interventions,
                    outcomes=reduction.joint_outcomes,
                )
                assert literal_witnesses, context
                assert all(
                    validate_literal_hedge_by_definition(
                        graph,
                        interventions=reduction.interventions,
                        outcomes=reduction.joint_outcomes,
                        witness=witness,
                    )
                    for witness in literal_witnesses
                ), context
            else:
                outcome_key = (
                    "id_ratio" if reduction.residual_conditions else "id_no_residual"
                )
            counts[outcome_key] += 1

            obsolete_flag = obsolete_idc_corollary_two_flag(graph, reduction)
            counts["obsolete_flags"] += obsolete_flag
            counts["obsolete_false_positives"] += (
                obsolete_flag and not reduction.id_failed
            )

    return counts, role_counts


def test_literal_idc_n3_census_matches_independent_definitions() -> None:
    assert len(all_bow_admgs_n3()) == 200
    assert len(all_conditional_queries_n3()) == 37

    counts, role_counts = _literal_reduction_census()

    assert counts == Counter(
        {
            "rows": 7_400,
            "moves": 1_020,
            "queries_moved": 948,
            "all_nonempty_conditions_moved": 816,
            "residual_condition_rows": 2_784,
            "identified": 6_398,
            "nonidentified": 1_002,
            "id_no_residual": 3_902,
            "id_ratio": 2_496,
            "nonid_no_residual": 714,
            "nonid_residual": 288,
            "obsolete_flags": 1_050,
            "obsolete_false_positives": 48,
        }
    )
    assert role_counts == Counter(
        {
            (0, 0, 0): 3_800,
            (1, 0, 1): 2_256,
            (1, 1, 0): 744,
            (2, 0, 2): 396,
            (2, 1, 1): 132,
            (2, 2, 0): 72,
        }
    )
    assert counts["fixed_point_one_pass_divergence"] == 0


def test_uai_2006_corollary_two_subset_quantifier_has_a_false_positive() -> None:
    # X0 -> M -> Y with X0 <-> M; query P_x0(Y | M).
    graph = TinyADMG(
        nodes=(0, 1, 2),
        directed=frozenset({(0, 1), (1, 2)}),
        bidirected=frozenset({(0, 1)}),
    )
    reduction = literal_idc_reduction(
        graph,
        interventions=frozenset({0}),
        outcomes=frozenset({2}),
        conditions=frozenset({1}),
    )

    assert reduction.rule_two_interventions == (1,)
    assert reduction.interventions == frozenset({0, 1})
    assert reduction.residual_conditions == frozenset()
    assert not reduction.id_failed
    assert not literal_exact_query_hedges(
        graph,
        interventions=reduction.interventions,
        outcomes=reduction.joint_outcomes,
    )
    assert literal_exact_query_hedges(
        graph,
        interventions=frozenset({0}),
        outcomes=frozenset({2}),
    )
    assert obsolete_idc_corollary_two_flag(graph, reduction)


def _assert_production_nonidentification_witness(
    graph: TinyADMG,
    reduction: LiteralIDCReduction,
    result: NonIdentifiedConditionalEffect,
    *,
    context: str,
) -> None:
    witness = result.witness
    label_to_node = {str(node): node for node in graph.nodes}
    inner = frozenset(label_to_node[label] for label in witness.inner_vertices)
    outer = frozenset(label_to_node[label] for label in witness.outer_vertices)
    roots = frozenset(label_to_node[label] for label in witness.roots)
    fixing_order = tuple(label_to_node[label] for label in witness.fixing_sequence)

    assert witness.outcomes == _ordered_labels(graph, reduction.joint_outcomes), context
    assert witness.interventions == _ordered_labels(graph, reduction.interventions), (
        context
    )
    modern_witnesses = exact_query_hedges(
        graph,
        interventions=reduction.interventions,
        outcomes=reduction.joint_outcomes,
    )
    assert any(
        candidate.inner.nodes == inner
        and candidate.outer.nodes == outer
        and candidate.inner.roots == roots
        and candidate.outer.roots == roots
        and candidate.fixing_order == fixing_order
        for candidate in modern_witnesses
    ), context

    literal_witnesses = literal_exact_query_hedges(
        graph,
        interventions=reduction.interventions,
        outcomes=reduction.joint_outcomes,
    )
    matching_literal = tuple(
        candidate
        for candidate in literal_witnesses
        if candidate.inner.nodes == inner
        and candidate.outer.nodes == outer
        and candidate.inner.roots == roots
        and candidate.outer.roots == roots
    )
    assert matching_literal, context
    assert all(
        validate_literal_hedge_by_definition(
            graph,
            interventions=reduction.interventions,
            outcomes=reduction.joint_outcomes,
            witness=candidate,
        )
        for candidate in matching_literal
    ), context


def test_production_idc_matches_all_n3_literal_and_exact_scm_oracles() -> None:
    counts: Counter[str] = Counter()

    for graph_index, graph in enumerate(all_bow_admgs_n3()):
        production_graph = _production_graph(graph)
        source_nodes = frozenset(production_graph.topology.nodes)
        for interventions, outcomes, conditions in all_conditional_queries_n3():
            reduction = literal_idc_reduction(
                graph,
                interventions=interventions,
                outcomes=outcomes,
                conditions=conditions,
            )
            expected_outcomes = _ordered_labels(graph, outcomes)
            expected_interventions = _ordered_labels(graph, interventions)
            expected_conditions = _ordered_labels(graph, conditions)
            expected_rule_two = _ordered_labels(
                graph, frozenset(reduction.rule_two_interventions)
            )
            context = (
                f"graph_index={graph_index}, directed={sorted(graph.directed)}, "
                f"bidirected={sorted(graph.bidirected)}, "
                f"interventions={sorted(interventions)}, outcomes={sorted(outcomes)}, "
                f"conditions={sorted(conditions)}"
            )

            result = identify_conditional_causal_effect(
                production_graph,
                outcomes=expected_outcomes,
                interventions=expected_interventions,
                conditioned_on=expected_conditions,
            )
            counts["rows"] += 1
            assert result.graph is production_graph, context
            assert result.outcomes == expected_outcomes, context
            assert result.interventions == expected_interventions, context
            assert result.conditioned_on == expected_conditions, context
            assert result.rule_two_interventions == expected_rule_two, context

            if reduction.id_failed:
                counts["nonidentified"] += 1
                assert isinstance(result, NonIdentifiedConditionalEffect), context
                _assert_production_nonidentification_witness(
                    graph,
                    reduction,
                    result,
                    context=context,
                )
                continue

            counts["identified"] += 1
            assert isinstance(result, IdentifiedConditionalEffect), context
            _assert_observational_source_bound_formula(
                result,
                source_nodes=source_nodes,
                context=context,
            )
            ordered_outcomes = _ordered_nodes(graph, outcomes)
            ordered_interventions = _ordered_nodes(graph, interventions)
            ordered_conditions = _ordered_nodes(graph, conditions)
            query_nodes = (
                *ordered_outcomes,
                *ordered_interventions,
                *ordered_conditions,
            )
            for assignment in itertools.product((0, 1), repeat=len(query_nodes)):
                counts["assignment_rows"] += 1
                values = dict(zip(query_nodes, assignment, strict=True))
                environment = {str(node): value for node, value in values.items()}
                for variant in (0, 1):
                    table = exact_binary_observational_distribution(graph, variant)
                    expected = exact_binary_conditional_interventional_probability(
                        graph,
                        outcomes=ordered_outcomes,
                        interventions=ordered_interventions,
                        conditioned_on=ordered_conditions,
                        values=values,
                        variant=variant,
                    )
                    actual = evaluate_probability_expression(
                        result.formula,
                        table=table,
                        node_labels=production_graph.topology.nodes,
                        environment=environment,
                    )
                    value_context = f"{context}, variant={variant}, values={values}"
                    assert actual == expected, value_context
                    counts["formula_values"] += 1

    assert counts == Counter(
        {
            "rows": 7_400,
            "identified": 6_398,
            "nonidentified": 1_002,
            "assignment_rows": 36_592,
            "formula_values": 73_184,
        }
    )
