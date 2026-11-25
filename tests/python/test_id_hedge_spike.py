"""SPIKE-0 scientific adjudication of exact-query hedge semantics."""

from __future__ import annotations

import itertools

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DirectedBidirectedGraph,
    IdentifiedEffect,
    NonIdentifiedEffect,
    Probability,
    Product,
    Ratio,
    Summation,
    identify_causal_effect,
)

from .id_oracles import (
    ReachableClosure,
    TinyADMG,
    all_bow_admgs_n3,
    all_literal_hedge_subgraphs,
    all_nonempty_outcome_queries_n3,
    evaluate_probability_expression,
    exact_query_hedges,
    exact_binary_interventional_probability,
    exact_binary_observational_distribution,
    greedy_reachable_closure,
    literal_exact_query_hedges,
    literal_id_fails,
    obsolete_subset_literal_hedge_criterion,
    validate_hedge_by_definition,
    validate_literal_hedge_by_definition,
    validate_reachable_closure_trace,
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


def _ordered_labels(graph: TinyADMG, vertices: frozenset[int]) -> tuple[str, ...]:
    return tuple(str(node) for node in graph.nodes if node in vertices)


def _assert_identified_formula_contract(
    result: IdentifiedEffect,
    *,
    source_nodes: frozenset[str],
    context: str,
) -> None:
    assert set(result.outcomes) <= result.formula.free_variables, context
    assert result.formula.free_variables <= (
        set(result.outcomes) | set(result.interventions)
    ), context

    pending = [result.formula]
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


def test_all_n3_literal_hedges_match_modern_closure_and_id_failure() -> None:
    graphs = all_bow_admgs_n3()
    queries = all_nonempty_outcome_queries_n3()

    assert len(graphs) == 200
    assert len({graph.directed for graph in graphs}) == 25
    assert len(queries) == 19
    assert sum(not interventions for interventions, _ in queries) == 7
    assert sum(len(all_literal_hedge_subgraphs(graph)) for graph in graphs) == 630

    row_count = 0
    literal_nonidentification_count = 0
    modern_nonidentification_count = 0
    literal_id_failure_count = 0
    obsolete_subset_count = 0
    false_positive_count = 0
    literal_witness_count = 0
    modern_witness_count = 0

    for graph in graphs:
        for interventions, outcomes in queries:
            row_count += 1
            literal_witnesses = literal_exact_query_hedges(
                graph,
                interventions=interventions,
                outcomes=outcomes,
            )
            modern_witnesses = exact_query_hedges(
                graph,
                interventions=interventions,
                outcomes=outcomes,
            )
            literal_nonidentified = bool(literal_witnesses)
            modern_nonidentified = bool(modern_witnesses)
            id_failed = literal_id_fails(
                graph,
                interventions=interventions,
                outcomes=outcomes,
            )
            obsolete_subset_flag = obsolete_subset_literal_hedge_criterion(
                graph,
                interventions=interventions,
                outcomes=outcomes,
            )

            assert literal_nonidentified == modern_nonidentified == id_failed
            if literal_nonidentified:
                assert obsolete_subset_flag

            literal_nonidentification_count += literal_nonidentified
            modern_nonidentification_count += modern_nonidentified
            literal_id_failure_count += id_failed
            obsolete_subset_count += obsolete_subset_flag
            false_positive_count += obsolete_subset_flag and not literal_nonidentified

            for witness in literal_witnesses:
                literal_witness_count += 1
                assert validate_literal_hedge_by_definition(
                    graph,
                    interventions=interventions,
                    outcomes=outcomes,
                    witness=witness,
                )

            for witness in modern_witnesses:
                modern_witness_count += 1
                closure = ReachableClosure(
                    target=witness.inner.nodes,
                    closure=witness.outer.nodes,
                    fixing_order=witness.fixing_order,
                )
                assert witness.inner.nodes < witness.outer.nodes
                assert validate_reachable_closure_trace(graph, closure)
                assert validate_hedge_by_definition(
                    graph,
                    interventions=interventions,
                    outcomes=outcomes,
                    witness=witness,
                )

    assert row_count == 3_800
    assert literal_nonidentification_count == 636
    assert modern_nonidentification_count == 636
    assert literal_id_failure_count == 636
    assert obsolete_subset_count == 648
    assert false_positive_count == 12
    assert literal_witness_count >= literal_nonidentification_count
    assert modern_witness_count >= modern_nonidentification_count


def test_production_id_matches_literal_oracles_on_all_n3_exact_queries() -> None:
    graphs = all_bow_admgs_n3()
    queries = all_nonempty_outcome_queries_n3()
    row_count = 0
    identified_count = 0
    nonidentified_count = 0
    formula_value_count = 0

    for graph_index, graph in enumerate(graphs):
        production_graph = _production_graph(graph)
        source_nodes = frozenset(production_graph.topology.nodes)
        label_to_node = {str(node): node for node in graph.nodes}

        for interventions, outcomes in queries:
            row_count += 1
            context = (
                f"graph_index={graph_index}, directed={sorted(graph.directed)}, "
                f"bidirected={sorted(graph.bidirected)}, "
                f"interventions={sorted(interventions)}, outcomes={sorted(outcomes)}"
            )
            literal_witnesses = literal_exact_query_hedges(
                graph,
                interventions=interventions,
                outcomes=outcomes,
            )
            oracle_nonidentified = literal_id_fails(
                graph,
                interventions=interventions,
                outcomes=outcomes,
            )
            assert oracle_nonidentified == bool(literal_witnesses), context

            expected_outcomes = _ordered_labels(graph, outcomes)
            expected_interventions = _ordered_labels(graph, interventions)
            result = identify_causal_effect(
                production_graph,
                outcomes=expected_outcomes,
                interventions=expected_interventions,
            )

            if not oracle_nonidentified:
                identified_count += 1
                assert isinstance(result, IdentifiedEffect), context
                assert result.graph is production_graph, context
                assert result.outcomes == expected_outcomes, context
                assert result.interventions == expected_interventions, context
                _assert_identified_formula_contract(
                    result,
                    source_nodes=source_nodes,
                    context=context,
                )
                ordered_outcomes = tuple(
                    node for node in graph.nodes if node in outcomes
                )
                ordered_interventions = tuple(
                    node for node in graph.nodes if node in interventions
                )
                query_nodes = ordered_outcomes + ordered_interventions
                for variant in (0, 1):
                    table = exact_binary_observational_distribution(graph, variant)
                    for assignment in itertools.product(
                        (0, 1), repeat=len(query_nodes)
                    ):
                        values = dict(zip(query_nodes, assignment, strict=True))
                        environment = {
                            str(node): value for node, value in values.items()
                        }
                        expected = exact_binary_interventional_probability(
                            graph,
                            outcomes=ordered_outcomes,
                            interventions=ordered_interventions,
                            values=values,
                            variant=variant,
                        )
                        actual = evaluate_probability_expression(
                            result.formula,
                            table=table,
                            node_labels=production_graph.topology.nodes,
                            environment=environment,
                        )
                        value_context = (
                            f"{context}, variant={variant}, values={values}"
                        )
                        assert actual == expected, value_context
                        formula_value_count += 1
                continue

            nonidentified_count += 1
            assert isinstance(result, NonIdentifiedEffect), context
            witness = result.witness
            assert witness.graph is production_graph, context
            assert witness.outcomes == expected_outcomes, context
            assert witness.interventions == expected_interventions, context

            witness_fields = (
                witness.inner_vertices,
                witness.outer_vertices,
                witness.roots,
                witness.fixing_sequence,
            )
            assert all(set(field) <= source_nodes for field in witness_fields), context
            assert all(len(field) == len(set(field)) for field in witness_fields), context

            inner = frozenset(label_to_node[label] for label in witness.inner_vertices)
            outer = frozenset(label_to_node[label] for label in witness.outer_vertices)
            roots = frozenset(label_to_node[label] for label in witness.roots)
            fixing_order = tuple(
                label_to_node[label] for label in witness.fixing_sequence
            )
            assert witness.inner_vertices == _ordered_labels(graph, inner), context
            assert witness.outer_vertices == _ordered_labels(graph, outer), context
            assert witness.roots == _ordered_labels(graph, roots), context
            assert inner < outer, context
            assert not inner & interventions, context
            assert (outer - inner) & interventions, context

            expected_roots = frozenset(
                node
                for node in inner
                if not any(
                    parent == node and child in inner
                    for parent, child in graph.directed
                )
            )
            assert roots == expected_roots, context

            closure = ReachableClosure(
                target=inner,
                closure=outer,
                fixing_order=fixing_order,
            )
            assert validate_reachable_closure_trace(graph, closure), context
            assert closure == greedy_reachable_closure(graph, inner), context

            matching_literal_witnesses = tuple(
                candidate
                for candidate in literal_witnesses
                if candidate.inner.nodes == inner
                and candidate.outer.nodes == outer
                and candidate.inner.roots == roots
                and candidate.outer.roots == roots
            )
            assert matching_literal_witnesses, context
            assert all(
                validate_literal_hedge_by_definition(
                    graph,
                    interventions=interventions,
                    outcomes=outcomes,
                    witness=candidate,
                )
                for candidate in matching_literal_witnesses
            ), context

    assert row_count == 3_800
    assert identified_count == 3_164
    assert nonidentified_count == 636
    assert formula_value_count == 31_040


def test_minimal_subset_criterion_false_positive() -> None:
    # X0 -> M -> Y with X0 <-> M.
    x0, mediator, outcome = 0, 1, 2
    graph = TinyADMG(
        nodes=(x0, mediator, outcome),
        directed=frozenset({(x0, mediator), (mediator, outcome)}),
        bidirected=frozenset({(x0, mediator)}),
    )
    full_interventions = frozenset({x0, mediator})
    outcomes = frozenset({outcome})

    assert not literal_id_fails(
        graph,
        interventions=full_interventions,
        outcomes=outcomes,
    )
    assert not exact_query_hedges(
        graph,
        interventions=full_interventions,
        outcomes=outcomes,
    )
    assert not literal_exact_query_hedges(
        graph,
        interventions=full_interventions,
        outcomes=outcomes,
    )
    assert obsolete_subset_literal_hedge_criterion(
        graph,
        interventions=full_interventions,
        outcomes=outcomes,
    )

    subquery_witnesses = exact_query_hedges(
        graph,
        interventions=frozenset({x0}),
        outcomes=outcomes,
    )
    assert len(subquery_witnesses) == 1
    (witness,) = subquery_witnesses
    assert witness.inner.nodes == frozenset({mediator})
    assert witness.outer.nodes == frozenset({x0, mediator})
    assert witness.inner.roots == frozenset({mediator})
    assert witness.inner.directed == frozenset()
    assert witness.inner.bidirected_tree == frozenset()
    assert witness.outer.directed == frozenset({(x0, mediator)})
    assert witness.outer.bidirected_tree == frozenset({(x0, mediator)})

    literal_subquery_witnesses = literal_exact_query_hedges(
        graph,
        interventions=frozenset({x0}),
        outcomes=outcomes,
    )
    assert len(literal_subquery_witnesses) == 1
    (literal_witness,) = literal_subquery_witnesses
    assert literal_witness.inner.nodes == frozenset({mediator})
    assert literal_witness.outer.nodes == frozenset({x0, mediator})
    assert literal_witness.inner.roots == frozenset({mediator})
    assert literal_witness.inner.directed == frozenset()
    assert literal_witness.inner.bidirected_tree == frozenset()
    assert literal_witness.outer.directed == frozenset({(x0, mediator)})
    assert literal_witness.outer.bidirected_tree == frozenset({(x0, mediator)})
