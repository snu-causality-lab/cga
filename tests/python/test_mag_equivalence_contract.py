from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    AncestralGraph,
    EndpointMarkedGraph,
    Mark,
    MarkovEquivalenceDomainError,
    MaximalAncestralGraph,
    mag_markov_equivalent,
    m_separated,
)

from .oracles import (
    MixedEdgeAtom,
    endpoint_graph_is_ancestral_by_definition,
    m_separation_signature_by_all_conditioning_subsets,
    nonmaximal_pair_by_all_conditioning_sets,
)


NODES4 = ("A", "B", "C", "D")
PAIRS4 = tuple(itertools.combinations(NODES4, 2))
MAG_STATES: tuple[None | tuple[Mark, Mark], ...] = (
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


def make_n4_choice_graph(choices: tuple[int, ...]) -> EndpointMarkedGraph:
    return EndpointMarkedGraph(
        nodes=NODES4,
        edges=[
            (left, state[0], right, state[1])
            for (left, right), choice in zip(PAIRS4, choices, strict=True)
            if (state := MAG_STATES[choice]) is not None
        ],
    )


def figure4_graphs() -> tuple[
    dict[str, object], dict[str, MaximalAncestralGraph]
]:
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "papers"
        / "ali_richardson_spirtes_2009_figure4.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    graphs = {
        name: MaximalAncestralGraph.from_graph(
            EndpointMarkedGraph(
                nodes=fixture["nodes"],
                edges=[
                    (left, Mark(left_mark), right, Mark(right_mark))
                    for left, left_mark, right, right_mark in edges
                ],
            )
        )
        for name, edges in fixture["graphs"].items()
    }
    return fixture, graphs


def figure6_graphs() -> tuple[
    dict[str, object], dict[str, dict[str, MaximalAncestralGraph]]
]:
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "papers"
        / "ali_richardson_spirtes_2009_figure6.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    panels = {
        panel: {
            name: MaximalAncestralGraph.from_graph(
                EndpointMarkedGraph(
                    nodes=fixture["nodes"],
                    edges=[
                        (left, Mark(left_mark), right, Mark(right_mark))
                        for left, left_mark, right, right_mark in edges
                    ],
                )
            )
            for name, edges in graphs.items()
        }
        for panel, graphs in fixture["panels"].items()
    }
    return fixture, panels


def order2_regression_graphs() -> tuple[
    dict[str, object], dict[str, MaximalAncestralGraph]
]:
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "papers"
        / "ali_richardson_spirtes_2009_order2_regression.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    shared_edges = fixture["shared_edges"]
    graphs = {
        name: MaximalAncestralGraph.from_graph(
            EndpointMarkedGraph(
                nodes=fixture["nodes"],
                edges=[
                    (left, Mark(left_mark), right, Mark(right_mark))
                    for left, left_mark, right, right_mark in (
                        *shared_edges,
                        last_edge,
                    )
                ],
            )
        )
        for name, last_edge in fixture["last_edges"].items()
    }
    return fixture, graphs


def skeleton(
    graph: EndpointMarkedGraph,
) -> frozenset[frozenset[str]]:
    return frozenset(
        frozenset((left, right))
        for left, _, right, _ in graph.edges
    )


def unshielded_colliders(
    graph: EndpointMarkedGraph,
) -> frozenset[tuple[str, str, str]]:
    positions = {node: index for index, node in enumerate(graph.nodes)}
    marks: dict[tuple[str, str], Mark] = {}
    neighbors: dict[str, set[str]] = {node: set() for node in graph.nodes}
    for left, left_mark, right, right_mark in graph.edges:
        marks[(left, right)] = left_mark
        marks[(right, left)] = right_mark
        neighbors[left].add(right)
        neighbors[right].add(left)

    result: set[tuple[str, str, str]] = set()
    for middle in graph.nodes:
        for first, second in itertools.combinations(neighbors[middle], 2):
            if second in neighbors[first]:
                continue
            if marks[(middle, first)] is not Mark.ARROW or (
                marks[(middle, second)] is not Mark.ARROW
            ):
                continue
            left, right = (
                (first, second)
                if positions[first] < positions[second]
                else (second, first)
            )
            result.add((left, middle, right))
    return frozenset(result)


def triple_is_collider(
    graph: EndpointMarkedGraph, first: str, middle: str, third: str
) -> bool:
    first_edge = graph.edge(middle, first)
    third_edge = graph.edge(middle, third)
    return first_edge is not None and third_edge is not None and (
        first_edge[0] is Mark.ARROW and third_edge[0] is Mark.ARROW
    )


def displayed_path_is_discriminating(
    graph: EndpointMarkedGraph,
) -> bool:
    return (
        graph.edge("x", "y") is None
        and triple_is_collider(graph, "x", "q", "b")
        and graph.edge("q", "y") == (Mark.TAIL, Mark.ARROW)
    )


def test_ali_richardson_spirtes_figure4_exact_equivalence_claims() -> None:
    fixture, graphs = figure4_graphs()
    assert fixture["journal_page"] == 2815
    assert fixture["reprint_page"] == 9
    assert fixture["figure"] == "4"
    assert fixture["erratum_status"] == "no figure-specific correction located"
    g1 = graphs["G1"]
    g2 = graphs["G2"]
    g3 = graphs["G3"]
    expected_skeleton = frozenset(
        frozenset(pair) for pair in fixture["shared_skeleton"]
    )
    expected_colliders = frozenset(
        tuple(triple) for triple in fixture["shared_unshielded_colliders"]
    )

    assert skeleton(g1.topology) == skeleton(g2.topology) == skeleton(g3.topology)
    assert skeleton(g1.topology) == expected_skeleton
    assert unshielded_colliders(g1.topology) == (
        unshielded_colliders(g2.topology)
    ) == unshielded_colliders(g3.topology)
    assert unshielded_colliders(g1.topology) == expected_colliders

    signatures = {
        name: m_separation_signature_by_all_conditioning_subsets(
            graph.topology.nodes, endpoint_atoms(graph.topology)
        )
        for name, graph in graphs.items()
    }
    assert signatures["G1"] == signatures["G2"]
    assert signatures["G1"] != signatures["G3"]
    assert m_separated(g1, "x", "y", given="q")
    assert m_separated(g2, "x", "y", given="q")
    assert not m_separated(g3, "x", "y", given="q")

    before = {name: graph.topology for name, graph in graphs.items()}
    assert mag_markov_equivalent(g1, g1)
    assert mag_markov_equivalent(g1, g2)
    assert mag_markov_equivalent(g2, g1)
    assert not mag_markov_equivalent(g1, g3)
    assert not mag_markov_equivalent(g3, g1)
    assert {name: graph.topology for name, graph in graphs.items()} == before


def test_ali_richardson_spirtes_figure6_discriminating_path_traps() -> None:
    fixture, panels = figure6_graphs()
    assert fixture["journal_page"] == 2820
    assert fixture["reprint_page"] == 14
    assert fixture["figure"] == "6"
    assert fixture["erratum_status"] == "no figure-specific correction located"

    for graphs in panels.values():
        g1 = graphs["G1"]
        g2 = graphs["G2"]
        signature1 = m_separation_signature_by_all_conditioning_subsets(
            g1.topology.nodes, endpoint_atoms(g1.topology)
        )
        signature2 = m_separation_signature_by_all_conditioning_subsets(
            g2.topology.nodes, endpoint_atoms(g2.topology)
        )
        assert signature1 == signature2
        assert displayed_path_is_discriminating(g1.topology)
        assert not displayed_path_is_discriminating(g2.topology)
        assert mag_markov_equivalent(g1, g2)
        assert mag_markov_equivalent(g2, g1)

    assert not triple_is_collider(
        panels["i"]["G1"].topology, "q", "b", "y"
    )
    assert triple_is_collider(
        panels["i"]["G2"].topology, "q", "b", "y"
    )
    assert triple_is_collider(
        panels["ii"]["G1"].topology, "q", "b", "y"
    )
    assert triple_is_collider(
        panels["ii"]["G2"].topology, "q", "b", "y"
    )


def test_order_two_discriminating_collider_requires_two_updates() -> None:
    fixture, graphs = order2_regression_graphs()
    assert fixture["status"] == (
        "independently constructed regression, not a graph reported in the paper"
    )
    order2 = graphs["G_order2"]
    noncollider = graphs["G_noncollider"]

    assert m_separated(order2, "x", "y", given=("q1", "q2"))
    assert not m_separated(
        noncollider, "x", "y", given=("q1", "q2")
    )
    assert not m_separated(
        order2, "x", "y", given=("b", "q1", "q2")
    )
    assert m_separated(
        noncollider, "x", "y", given=("b", "q1", "q2")
    )
    assert not mag_markov_equivalent(order2, noncollider)
    assert not mag_markov_equivalent(noncollider, order2)


def test_empty_singleton_and_permuted_declaration_order_are_label_based() -> None:
    empty = MaximalAncestralGraph.from_graph(EndpointMarkedGraph(nodes=[]))
    singleton = MaximalAncestralGraph.from_graph(
        EndpointMarkedGraph(nodes=["only"])
    )
    singleton_again = MaximalAncestralGraph.from_graph(
        EndpointMarkedGraph(nodes=["only"])
    )
    assert mag_markov_equivalent(empty, empty)
    assert mag_markov_equivalent(singleton, singleton_again)

    first = MaximalAncestralGraph.from_graph(
        EndpointMarkedGraph(
            nodes=["C", "A", "B"],
            edges=[
                ("A", Mark.TAIL, "B", Mark.ARROW),
                ("B", Mark.TAIL, "C", Mark.ARROW),
            ],
        )
    )
    permuted = MaximalAncestralGraph.from_graph(
        EndpointMarkedGraph(
            nodes=["B", "C", "A"],
            edges=[
                ("B", Mark.TAIL, "C", Mark.ARROW),
                ("A", Mark.TAIL, "B", Mark.ARROW),
            ],
        )
    )
    assert first.topology.nodes != permuted.topology.nodes
    assert m_separation_signature_by_all_conditioning_subsets(
        first.topology.nodes, endpoint_atoms(first.topology)
    ) == m_separation_signature_by_all_conditioning_subsets(
        permuted.topology.nodes, endpoint_atoms(permuted.topology)
    )
    assert mag_markov_equivalent(first, permuted)
    assert mag_markov_equivalent(permuted, first)


def test_equivalence_domain_requires_matching_mag_proof_views() -> None:
    raw = EndpointMarkedGraph(nodes=["A", "B"])
    ancestral = AncestralGraph.from_graph(raw)
    mag = MaximalAncestralGraph.from_graph(raw)
    mismatched = MaximalAncestralGraph.from_graph(
        EndpointMarkedGraph(nodes=["A", "C"])
    )

    with pytest.raises(MarkovEquivalenceDomainError, match="MAG"):
        mag_markov_equivalent(raw, mag)  # type: ignore[arg-type]
    with pytest.raises(MarkovEquivalenceDomainError, match="MAG"):
        mag_markov_equivalent(mag, raw)  # type: ignore[arg-type]
    with pytest.raises(MarkovEquivalenceDomainError, match="MAG"):
        mag_markov_equivalent(ancestral, mag)  # type: ignore[arg-type]
    with pytest.raises(MarkovEquivalenceDomainError, match="vertex labels"):
        mag_markov_equivalent(mag, mismatched)


def test_n4_mag_partition_matches_complete_ci_signatures() -> None:
    candidate_count = 0
    mags: list[
        tuple[
            tuple[int, ...],
            MaximalAncestralGraph,
            tuple[tuple[str, str, tuple[str, ...], bool], ...],
        ]
    ] = []
    oracle_groups: dict[
        tuple[tuple[str, str, tuple[str, ...], bool], ...],
        list[tuple[int, ...]],
    ] = defaultdict(list)

    for choices in itertools.product(range(len(MAG_STATES)), repeat=len(PAIRS4)):
        candidate_count += 1
        topology = make_n4_choice_graph(choices)
        atoms = endpoint_atoms(topology)
        if not endpoint_graph_is_ancestral_by_definition(NODES4, atoms):
            continue
        if nonmaximal_pair_by_all_conditioning_sets(NODES4, atoms) is not None:
            continue
        signature = m_separation_signature_by_all_conditioning_subsets(
            NODES4, atoms
        )
        graph = MaximalAncestralGraph.from_graph(topology)
        mags.append((choices, graph, signature))
        oracle_groups[signature].append(choices)

    assert candidate_count == 15_625
    assert len(mags) == 3_045
    assert len(oracle_groups) == 251
    assert max(len(group) for group in oracle_groups.values()) == 242
    assert all(len(signature) == 24 for signature in oracle_groups)

    production_representatives: list[MaximalAncestralGraph] = []
    production_groups: list[list[tuple[int, ...]]] = []
    for graph_id, graph, _ in mags:
        matching_groups = [
            index
            for index, representative in enumerate(production_representatives)
            if mag_markov_equivalent(graph, representative)
        ]
        assert len(matching_groups) <= 1
        if matching_groups:
            production_groups[matching_groups[0]].append(graph_id)
        else:
            production_representatives.append(graph)
            production_groups.append([graph_id])

    oracle_partition = {
        frozenset(group) for group in oracle_groups.values()
    }
    production_partition = {
        frozenset(group) for group in production_groups
    }
    assert production_partition == oracle_partition
