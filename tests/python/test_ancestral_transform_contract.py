from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    AncestralDomainError,
    AncestralGraph,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    Mark,
    MaximalAncestralGraph,
    ancestral_marginal_condition_transform,
    m_separated_sets,
    maximal_ancestral_extension,
)

from .oracles import (
    MixedEdgeAtom,
    ancestral_transform_by_all_conditioning_sets,
    ancestral_transform_by_inducing_paths,
    endpoint_graph_is_ancestral_by_definition,
    inducing_path_is_valid_by_definition,
)


NODES3 = ("A", "B", "C")
PAIRS3 = tuple(itertools.combinations(NODES3, 2))
ANCESTRAL_STATES: tuple[None | tuple[Mark, Mark], ...] = (
    None,
    (Mark.TAIL, Mark.TAIL),
    (Mark.TAIL, Mark.ARROW),
    (Mark.ARROW, Mark.TAIL),
    (Mark.ARROW, Mark.ARROW),
)


def endpoint_atoms(graph: EndpointMarkedGraph) -> tuple[MixedEdgeAtom, ...]:
    return tuple(
        (
            left,
            left_mark is Mark.ARROW,
            right,
            right_mark is Mark.ARROW,
        )
        for left, left_mark, right, right_mark in graph.edges
    )


def make_n3_choice_graph(choices: tuple[int, ...]) -> EndpointMarkedGraph:
    return EndpointMarkedGraph(
        nodes=NODES3,
        edges=[
            (left, state[0], right, state[1])
            for (left, right), choice in zip(PAIRS3, choices, strict=True)
            if (state := ANCESTRAL_STATES[choice]) is not None
        ],
    )


def graph_from_json_fields(
    nodes: object, edges: object
) -> EndpointMarkedGraph:
    return EndpointMarkedGraph(
        nodes=nodes,  # type: ignore[arg-type]
        edges=[
            (left, Mark(left_mark), right, Mark(right_mark))
            for left, left_mark, right, right_mark in edges  # type: ignore[union-attr]
        ],
    )


def set_queries(
    nodes: tuple[str, ...],
):
    for assignment in itertools.product(range(4), repeat=len(nodes)):
        left = tuple(
            node
            for node, status in zip(nodes, assignment, strict=True)
            if status == 1
        )
        right = tuple(
            node
            for node, status in zip(nodes, assignment, strict=True)
            if status == 2
        )
        given = tuple(
            node
            for node, status in zip(nodes, assignment, strict=True)
            if status == 3
        )
        if left and right:
            yield left, right, given


@pytest.mark.parametrize(
    ("source", "conditioned", "marginalized", "expected"),
    [
        (
            EndpointMarkedGraph(
                nodes=["a", "b", "c"],
                edges=[
                    ("a", Mark.TAIL, "b", Mark.TAIL),
                    ("b", Mark.TAIL, "c", Mark.ARROW),
                ],
            ),
            (),
            ("b",),
            EndpointMarkedGraph(
                nodes=["a", "c"],
                edges=[("a", Mark.TAIL, "c", Mark.ARROW)],
            ),
        ),
        (
            EndpointMarkedGraph(
                nodes=["a", "s", "b"],
                edges=[
                    ("a", Mark.TAIL, "s", Mark.ARROW),
                    ("b", Mark.TAIL, "s", Mark.ARROW),
                ],
            ),
            ("s",),
            (),
            EndpointMarkedGraph(
                nodes=["a", "b"],
                edges=[("a", Mark.TAIL, "b", Mark.TAIL)],
            ),
        ),
        (
            EndpointMarkedGraph(
                nodes=["a", "m", "b"],
                edges=[
                    ("a", Mark.TAIL, "m", Mark.ARROW),
                    ("b", Mark.TAIL, "m", Mark.ARROW),
                ],
            ),
            (),
            ("m",),
            EndpointMarkedGraph(nodes=["a", "b"]),
        ),
    ],
)
def test_endpoint_mark_and_collider_regressions(
    source: EndpointMarkedGraph,
    conditioned: tuple[str, ...],
    marginalized: tuple[str, ...],
    expected: EndpointMarkedGraph,
) -> None:
    view = AncestralGraph.from_graph(source)

    result = ancestral_marginal_condition_transform(
        view,
        conditioned=conditioned,
        marginalized=marginalized,
    )

    assert type(result) is MaximalAncestralGraph
    assert result.topology == expected
    assert ancestral_transform_by_all_conditioning_sets(
        source.nodes,
        endpoint_atoms(source),
        conditioned,
        marginalized,
    ) == (expected.nodes, endpoint_atoms(expected))
    assert ancestral_transform_by_inducing_paths(
        source.nodes,
        endpoint_atoms(source),
        conditioned,
        marginalized,
    ) == (expected.nodes, endpoint_atoms(expected))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "richardson_spirtes_2002_figure10.json",
        "richardson_spirtes_2002_figure11.json",
    ],
)
def test_richardson_spirtes_figures_10_and_11(
    fixture_name: str,
) -> None:
    fixture_path = (
        Path(__file__).parents[1] / "fixtures" / "papers" / fixture_name
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["erratum_status"] == "not amended by the official errata"
    assert fixture["adjudicated_claim"].startswith(
        "both panels"
        if fixture["figure"] == "10"
        else "all three panels"
    )
    source = graph_from_json_fields(fixture["nodes"], fixture["edges"])
    source_view = AncestralGraph.from_graph(source)
    source_atoms = endpoint_atoms(source)

    for transform in fixture["transforms"]:
        expected = graph_from_json_fields(
            transform["nodes"], transform["edges"]
        )
        conditioned = tuple(transform["conditioned"])
        marginalized = tuple(transform["marginalized"])

        all_sets = ancestral_transform_by_all_conditioning_sets(
            source.nodes,
            source_atoms,
            conditioned,
            marginalized,
        )
        inducing_paths = ancestral_transform_by_inducing_paths(
            source.nodes,
            source_atoms,
            conditioned,
            marginalized,
        )
        assert all_sets == (expected.nodes, endpoint_atoms(expected))
        assert inducing_paths == all_sets

        actual = ancestral_marginal_condition_transform(
            source_view,
            conditioned=conditioned,
            marginalized=marginalized,
        )
        assert actual.topology == expected
        for left, right, given in set_queries(expected.nodes):
            assert m_separated_sets(actual, left, right, given) == (
                m_separated_sets(
                    source_view,
                    left,
                    right,
                    given + conditioned,
                )
            )

    if fixture["figure"] == "11":
        assert inducing_path_is_valid_by_definition(
            source.nodes,
            source_atoms,
            fixture["reported_inducing_path"],
            conditioned=("s",),
            marginalized=("l1", "l2"),
        )
        expected_joint = graph_from_json_fields(
            fixture["transforms"][2]["nodes"],
            fixture["transforms"][2]["edges"],
        )
        conditioned_first = ancestral_marginal_condition_transform(
            source_view,
            conditioned="s",
        )
        marginalized_first = ancestral_marginal_condition_transform(
            source_view,
            marginalized=("l1", "l2"),
        )
        assert ancestral_marginal_condition_transform(
            conditioned_first,
            marginalized=("l1", "l2"),
        ).topology == expected_joint
        assert ancestral_marginal_condition_transform(
            marginalized_first,
            conditioned="s",
        ).topology == expected_joint


def test_default_empty_and_normalized_selector_contracts() -> None:
    figure9_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "papers"
        / "richardson_spirtes_2002_figure9.json"
    )
    figure9 = json.loads(figure9_path.read_text(encoding="utf-8"))
    source = graph_from_json_fields(figure9["nodes"], figure9["edges"])
    ancestral = AncestralGraph.from_graph(source)
    expected_maximal = maximal_ancestral_extension(ancestral)

    default_result = ancestral_marginal_condition_transform(ancestral)

    assert default_result.topology == expected_maximal.topology
    assert ancestral_marginal_condition_transform(expected_maximal) is expected_maximal
    assert ancestral.topology == source

    ordered = AncestralGraph.from_graph(
        EndpointMarkedGraph(nodes=["Z", "A", "M", "B"])
    )
    duplicate_result = ancestral_marginal_condition_transform(
        ordered,
        conditioned=["M", "M"],
        marginalized=["A", "A"],
    )
    string_result = ancestral_marginal_condition_transform(
        ordered,
        conditioned="M",
        marginalized="A",
    )
    assert duplicate_result.topology.nodes == ("Z", "B")
    assert string_result.topology == duplicate_result.topology

    empty = ancestral_marginal_condition_transform(
        ordered,
        conditioned=["B", "M", "Z", "A", "B"],
    )
    assert type(empty) is MaximalAncestralGraph
    assert empty.topology == EndpointMarkedGraph(nodes=[])


def test_transform_domain_and_vertex_errors_fail_closed() -> None:
    raw = EndpointMarkedGraph(nodes=["A", "B"])
    ancestral = AncestralGraph.from_graph(raw)
    dag = DAG.from_graph(DirectedBidirectedGraph(nodes=["A", "B"]))

    with pytest.raises(AncestralDomainError, match="ancestral"):
        ancestral_marginal_condition_transform(raw)  # type: ignore[arg-type]
    with pytest.raises(AncestralDomainError, match="ancestral"):
        ancestral_marginal_condition_transform(dag)  # type: ignore[arg-type]
    with pytest.raises(AncestralDomainError, match="disjoint"):
        ancestral_marginal_condition_transform(
            ancestral,
            conditioned=["A", "A"],
            marginalized="A",
        )
    with pytest.raises(GraphValidationError, match="unknown vertex"):
        ancestral_marginal_condition_transform(
            ancestral, conditioned="missing"
        )
    with pytest.raises(GraphValidationError, match="unknown vertex"):
        ancestral_marginal_condition_transform(
            ancestral, marginalized=["missing", "missing"]
        )
    with pytest.raises(TypeError):
        ancestral_marginal_condition_transform(  # type: ignore[call-arg]
            ancestral, (), ()
        )


def test_n3_universe_matches_both_definitions_and_theorem_4_18() -> None:
    ancestral_count = 0
    transform_count = 0

    for choices in itertools.product(
        range(len(ANCESTRAL_STATES)), repeat=len(PAIRS3)
    ):
        topology = make_n3_choice_graph(choices)
        atoms = endpoint_atoms(topology)
        if not endpoint_graph_is_ancestral_by_definition(NODES3, atoms):
            continue
        ancestral_count += 1
        source = AncestralGraph.from_graph(topology)

        for assignment in itertools.product(range(3), repeat=len(NODES3)):
            conditioned = tuple(
                node
                for node, status in zip(NODES3, assignment, strict=True)
                if status == 1
            )
            marginalized = tuple(
                node
                for node, status in zip(NODES3, assignment, strict=True)
                if status == 2
            )
            expected_all_sets = ancestral_transform_by_all_conditioning_sets(
                NODES3, atoms, conditioned, marginalized
            )
            expected_paths = ancestral_transform_by_inducing_paths(
                NODES3, atoms, conditioned, marginalized
            )
            assert expected_paths == expected_all_sets

            actual = ancestral_marginal_condition_transform(
                source,
                conditioned=conditioned,
                marginalized=marginalized,
            )
            assert type(actual) is MaximalAncestralGraph
            assert (actual.topology.nodes, endpoint_atoms(actual.topology)) == (
                expected_all_sets
            )
            for left, right, given in set_queries(actual.topology.nodes):
                assert m_separated_sets(actual, left, right, given) == (
                    m_separated_sets(
                        source,
                        left,
                        right,
                        given + conditioned,
                    )
                )
            transform_count += 1

    assert ancestral_count == 72
    assert transform_count == 72 * 27


def test_n3_transform_composition_matches_theorem_4_20() -> None:
    ancestral_count = 0
    composition_count = 0

    for choices in itertools.product(
        range(len(ANCESTRAL_STATES)), repeat=len(PAIRS3)
    ):
        topology = make_n3_choice_graph(choices)
        if not endpoint_graph_is_ancestral_by_definition(
            NODES3, endpoint_atoms(topology)
        ):
            continue
        ancestral_count += 1
        source = AncestralGraph.from_graph(topology)

        for assignment in itertools.product(range(5), repeat=len(NODES3)):
            conditioned_first = tuple(
                node
                for node, status in zip(NODES3, assignment, strict=True)
                if status == 1
            )
            marginalized_first = tuple(
                node
                for node, status in zip(NODES3, assignment, strict=True)
                if status == 2
            )
            conditioned_second = tuple(
                node
                for node, status in zip(NODES3, assignment, strict=True)
                if status == 3
            )
            marginalized_second = tuple(
                node
                for node, status in zip(NODES3, assignment, strict=True)
                if status == 4
            )

            first = ancestral_marginal_condition_transform(
                source,
                conditioned=conditioned_first,
                marginalized=marginalized_first,
            )
            staged = ancestral_marginal_condition_transform(
                first,
                conditioned=conditioned_second,
                marginalized=marginalized_second,
            )
            direct = ancestral_marginal_condition_transform(
                source,
                conditioned=conditioned_first + conditioned_second,
                marginalized=marginalized_first + marginalized_second,
            )
            assert staged.topology == direct.topology
            composition_count += 1

    assert ancestral_count == 72
    assert composition_count == 72 * 125
