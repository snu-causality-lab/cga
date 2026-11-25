from __future__ import annotations

import copy
import itertools
import json
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    AncestralDomainError,
    AncestralGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    Mark,
    MaximalAncestralGraph,
    NonMaximalityWitness,
    find_nonmaximality_witness,
    find_primitive_inducing_path,
    induced_subgraph,
    m_separated,
    maximal_ancestral_extension,
)

from .oracles import (
    EndpointMarkAtom,
    MixedEdgeAtom,
    endpoint_graph_is_ancestral_by_definition,
    maximal_extension_by_all_conditioning_sets,
    m_separated_by_simple_edge_paths,
    nonmaximal_pair_by_all_conditioning_sets,
    nonmaximality_path_by_simple_paths,
    primitive_inducing_paths_by_simple_paths,
    strict_ancestors_by_simple_paths,
    strict_anterior_vertices_by_simple_paths,
    strict_descendants_by_simple_paths,
)


NODES4 = ("A", "B", "C", "D")
PAIRS4 = tuple(itertools.combinations(NODES4, 2))
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


def endpoint_mark_atoms(
    graph: EndpointMarkedGraph,
) -> tuple[EndpointMarkAtom, ...]:
    return tuple(
        (left, left_mark.value, right, right_mark.value)
        for left, left_mark, right, right_mark in graph.edges
    )


def make_ancestral_choice_graph(
    choices: tuple[int, ...],
    *,
    nodes: tuple[str, ...] = NODES4,
    pairs: tuple[tuple[str, str], ...] = PAIRS4,
) -> EndpointMarkedGraph:
    return EndpointMarkedGraph(
        nodes=nodes,
        edges=[
            (left, state[0], right, state[1])
            for (left, right), choice in zip(pairs, choices, strict=True)
            if (state := ANCESTRAL_STATES[choice]) is not None
        ],
    )


def pair_queries(
    nodes: tuple[str, ...],
):
    for left_index, right_index in itertools.combinations(range(len(nodes)), 2):
        remaining = tuple(
            node
            for index, node in enumerate(nodes)
            if index not in (left_index, right_index)
        )
        for size in range(len(remaining) + 1):
            for given in itertools.combinations(remaining, size):
                yield nodes[left_index], nodes[right_index], given


def figure9_view() -> tuple[dict[str, object], AncestralGraph]:
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "papers"
        / "richardson_spirtes_2002_figure9.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    graph = EndpointMarkedGraph(
        nodes=fixture["nodes"],
        edges=[
            (left, Mark(left_mark), right, Mark(right_mark))
            for left, left_mark, right, right_mark in fixture["edges"]
        ],
    )
    return fixture, AncestralGraph.from_graph(graph)


def test_endpoint_relations_distinguish_directed_anterior_and_possible_paths() -> None:
    raw = EndpointMarkedGraph(
        nodes=["U", "V", "W", "X", "Y", "C", "D"],
        edges=[
            ("U", Mark.TAIL, "V", Mark.TAIL),
            ("V", Mark.TAIL, "W", Mark.ARROW),
            ("W", Mark.TAIL, "X", Mark.ARROW),
            ("X", Mark.ARROW, "Y", Mark.ARROW),
            ("C", Mark.CIRCLE, "D", Mark.ARROW),
        ],
    )

    assert raw.directed_ancestors_strict("X") == frozenset({"V", "W"})
    assert raw.directed_ancestors_reflexive("X") == frozenset({"V", "W", "X"})
    assert raw.directed_descendants_strict("V") == frozenset({"W", "X"})
    assert raw.directed_descendants_reflexive("V") == frozenset({"V", "W", "X"})
    assert raw.anterior_strict("X") == frozenset({"U", "V", "W"})
    assert raw.anterior_reflexive("X") == frozenset({"U", "V", "W", "X"})
    assert raw.possible_ancestors_strict("D") == frozenset({"C"})
    assert raw.anterior_strict("D") == frozenset()
    assert raw.bidirected_spouses("X") == frozenset({"Y"})
    assert raw.undirected_neighbors("U") == frozenset({"V"})
    assert raw.undirected_neighbors("V") == frozenset({"U"})

    nonancestral_order = EndpointMarkedGraph(
        nodes=["A", "B", "C"],
        edges=[
            ("A", Mark.TAIL, "B", Mark.ARROW),
            ("B", Mark.TAIL, "C", Mark.TAIL),
        ],
    )
    assert nonancestral_order.anterior_strict("C") == frozenset({"A", "B"})


def test_relation_set_semantics_forward_through_proof_views() -> None:
    raw = EndpointMarkedGraph(
        nodes=["U", "V", "W", "X", "Y"],
        edges=[
            ("U", Mark.TAIL, "V", Mark.TAIL),
            ("V", Mark.TAIL, "W", Mark.ARROW),
            ("W", Mark.TAIL, "X", Mark.ARROW),
            ("X", Mark.ARROW, "Y", Mark.ARROW),
        ],
    )
    ancestral = AncestralGraph.from_graph(raw)
    maximal = maximal_ancestral_extension(ancestral)

    for view in (ancestral, maximal):
        assert view.directed_ancestors_strict({"W", "X"}) == frozenset({"V"})
        assert view.directed_ancestors_reflexive({"W", "X"}) == frozenset(
            {"V", "W", "X"}
        )
        assert view.directed_descendants_strict({"V", "W"}) == frozenset({"X"})
        assert view.directed_descendants_reflexive({"V", "W"}) == frozenset(
            {"V", "W", "X"}
        )
        assert view.anterior_strict({"W", "X"}) == frozenset({"U", "V"})
        assert view.anterior_reflexive({"W", "X"}) == frozenset(
            {"U", "V", "W", "X"}
        )
        assert view.undirected_neighbors("U") == frozenset({"V"})
        assert "Y" in view.bidirected_spouses("X")

    assert ancestral.bidirected_spouses("X") == frozenset({"Y"})

    with pytest.raises(GraphValidationError, match="unknown vertex"):
        raw.anterior_strict("missing")


def test_endpoint_relations_match_literal_path_oracles_exhaustively_at_n3() -> None:
    nodes = ("A", "B", "C")
    pairs = tuple(itertools.combinations(nodes, 2))
    states: tuple[None | tuple[Mark, Mark], ...] = (None,) + tuple(
        itertools.product(tuple(Mark), repeat=2)
    )
    seed_sets = tuple(
        seeds
        for size in range(1, len(nodes) + 1)
        for seeds in itertools.combinations(nodes, size)
    )

    for choices in itertools.product(range(len(states)), repeat=len(pairs)):
        raw = EndpointMarkedGraph(
            nodes=nodes,
            edges=[
                (left, marks[0], right, marks[1])
                for (left, right), choice in zip(pairs, choices, strict=True)
                if (marks := states[choice]) is not None
            ],
        )
        atoms = endpoint_mark_atoms(raw)
        directed = frozenset(
            (first, second)
            if first_mark == "tail" and second_mark == "arrow"
            else (second, first)
            for first, first_mark, second, second_mark in atoms
            if (first_mark, second_mark)
            in (("tail", "arrow"), ("arrow", "tail"))
        )
        for seeds in seed_sets:
            expected_ancestors = strict_ancestors_by_simple_paths(
                nodes, directed, seeds
            )
            expected_descendants = strict_descendants_by_simple_paths(
                nodes, directed, seeds
            )
            expected_anteriors = strict_anterior_vertices_by_simple_paths(
                nodes, atoms, seeds
            )
            assert raw.directed_ancestors_strict(seeds) == expected_ancestors
            assert raw.directed_ancestors_reflexive(seeds) == (
                expected_ancestors | frozenset(seeds)
            )
            assert raw.directed_descendants_strict(seeds) == expected_descendants
            assert raw.directed_descendants_reflexive(seeds) == (
                expected_descendants | frozenset(seeds)
            )
            assert raw.anterior_strict(seeds) == expected_anteriors
            assert raw.anterior_reflexive(seeds) == (
                expected_anteriors | frozenset(seeds)
            )

        for vertex in nodes:
            assert raw.bidirected_spouses(vertex) == frozenset(
                neighbor
                for first, first_mark, second, second_mark in atoms
                if first_mark == second_mark == "arrow"
                for neighbor in (
                    second if first == vertex else first if second == vertex else "",
                )
                if neighbor
            )
            assert raw.undirected_neighbors(vertex) == frozenset(
                neighbor
                for first, first_mark, second, second_mark in atoms
                if first_mark == second_mark == "tail"
                for neighbor in (
                    second if first == vertex else first if second == vertex else "",
                )
                if neighbor
            )


def test_richardson_spirtes_figure9_witness_and_extension() -> None:
    fixture, ancestral = figure9_view()
    expected_path = tuple(fixture["primitive_inducing_path"])
    assert fixture["primitive_inducing_path_source_page"] == 984

    assert find_primitive_inducing_path(ancestral, "γ", "δ") == expected_path
    assert find_primitive_inducing_path(ancestral, "δ", "γ") == tuple(
        reversed(expected_path)
    )
    witness = find_nonmaximality_witness(ancestral)
    assert isinstance(witness, NonMaximalityWitness)
    assert witness.endpoints == ("γ", "δ")
    assert witness.path == expected_path

    with pytest.raises(GraphValidationError, match="maximal"):
        MaximalAncestralGraph.from_graph(ancestral.topology)
    with pytest.raises(GraphValidationError, match="maximal"):
        MaximalAncestralGraph.from_ancestral(ancestral)

    maximal = maximal_ancestral_extension(ancestral)
    extension = fixture["maximal_extension"]
    assert extension["reported_claim"] == (
        "an extension with the same (trivial) independence model"
    )
    assert extension["source_page"] == 979
    assert "caption alone" in extension["transcription_status"]
    assert extension["adjudicated_claim"] == (
        "adding gamma bidirected delta is the unique maximal ancestral extension"
    )
    reported_added = {
        (left, Mark(left_mark), right, Mark(right_mark))
        for left, left_mark, right, right_mark in extension["added_edges"]
    }
    assert maximal.topology.edge("γ", "δ") == (Mark.ARROW, Mark.ARROW)
    assert ancestral.topology.edge("γ", "δ") is None
    assert (
        set(maximal.topology.edges) - set(ancestral.topology.edges)
        == reported_added
    )
    assert find_nonmaximality_witness(maximal) is None
    for query in fixture["queries"]:
        assert m_separated(
            ancestral, query["left"], query["right"], query["given"]
        ) == m_separated(
            maximal, query["left"], query["right"], query["given"]
        )


def test_richardson_spirtes_figure13_maximal_extension() -> None:
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "papers"
        / "richardson_spirtes_2002_figure13.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["page"] == 994
    assert fixture["figure"] == "13"
    assert fixture["erratum_status"] == "not amended by the official errata"
    source = EndpointMarkedGraph(
        nodes=fixture["nodes"],
        edges=[
            (left, Mark(left_mark), right, Mark(right_mark))
            for left, left_mark, right, right_mark in fixture["edges"]
        ],
    )
    expected_added = {
        (left, Mark(left_mark), right, Mark(right_mark))
        for left, left_mark, right, right_mark in fixture[
            "maximal_extension_added_edges"
        ]
    }
    ancestral = AncestralGraph.from_graph(source)
    atoms = endpoint_atoms(source)

    assert find_nonmaximality_witness(ancestral) is not None
    maximal = maximal_ancestral_extension(ancestral)

    assert endpoint_atoms(maximal.topology) == (
        maximal_extension_by_all_conditioning_sets(source.nodes, atoms)
    )
    assert set(maximal.topology.edges) - set(source.edges) == expected_added
    assert find_nonmaximality_witness(maximal) is None
    for left, right, given in pair_queries(source.nodes):
        assert m_separated(ancestral, left, right, given) == m_separated(
            maximal, left, right, given
        )


def test_primitive_path_search_is_shortest_and_declared_order_deterministic() -> None:
    raw = EndpointMarkedGraph(
        nodes=["X", "Y", "C", "D", "A", "B"],
        edges=[
            ("X", Mark.ARROW, "A", Mark.ARROW),
            ("A", Mark.ARROW, "B", Mark.ARROW),
            ("B", Mark.ARROW, "Y", Mark.ARROW),
            ("A", Mark.TAIL, "Y", Mark.ARROW),
            ("B", Mark.TAIL, "X", Mark.ARROW),
            ("X", Mark.ARROW, "C", Mark.ARROW),
            ("C", Mark.ARROW, "D", Mark.ARROW),
            ("D", Mark.ARROW, "Y", Mark.ARROW),
            ("C", Mark.TAIL, "Y", Mark.ARROW),
            ("D", Mark.TAIL, "X", Mark.ARROW),
        ],
    )
    view = AncestralGraph.from_graph(raw)

    assert find_primitive_inducing_path(view, "X", "Y") == (
        "X",
        "C",
        "D",
        "Y",
    )
    assert find_primitive_inducing_path(view, "Y", "X") == (
        "Y",
        "D",
        "C",
        "X",
    )
    assert find_primitive_inducing_path(view, "X", "C") == ("X", "C")
    witness = find_nonmaximality_witness(view)
    assert witness is not None
    assert witness.endpoints == ("X", "Y")
    assert witness.path == ("X", "C", "D", "Y")


def test_mag_is_a_sealed_immutable_proof_value_and_witness_is_frozen() -> None:
    topology = EndpointMarkedGraph(
        nodes=["A", "B"],
        edges=[("A", Mark.TAIL, "B", Mark.ARROW)],
    )
    ancestral = AncestralGraph.from_graph(topology)
    from_graph = MaximalAncestralGraph.from_graph(topology)
    from_ancestral = MaximalAncestralGraph.from_ancestral(ancestral)

    assert from_graph.topology == topology
    assert from_ancestral.topology == topology
    assert copy.copy(from_graph) is from_graph
    assert copy.deepcopy(from_graph) is from_graph
    assert not hasattr(from_graph, "__dict__")
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("ForgedMAG", (MaximalAncestralGraph,), {})
    with pytest.raises((AttributeError, TypeError)):
        from_graph.__setstate__([EndpointMarkedGraph(nodes=[])])  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="canonical JSON"):
        pickle.dumps(from_graph)

    _, nonmaximal = figure9_view()
    witness = find_nonmaximality_witness(nonmaximal)
    assert witness is not None
    assert hash(witness) == hash(witness)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        witness.path = ()  # type: ignore[misc]


def test_nonmaximality_witness_normalizes_mutable_inputs() -> None:
    endpoints = ["A", "B"]
    path = ["A", "M", "B"]
    witness = NonMaximalityWitness(  # type: ignore[arg-type]
        endpoints=endpoints,
        path=path,
    )
    original_hash = hash(witness)

    endpoints[0] = "changed"
    path.clear()

    assert witness.endpoints == ("A", "B")
    assert witness.path == ("A", "M", "B")
    assert hash(witness) == original_hash


@pytest.mark.parametrize(
    ("endpoints", "path"),
    [
        (("A",), ("A", "B")),
        (("A", "A"), ("A", "A")),
        (("A", "B"), ("A",)),
        (("A", "B"), ("B", "A")),
        (("A", "B"), ("A", "M", "M", "B")),
        (("A", 1), ("A", 1)),
        ("AB", ("A", "B")),
        (("A", "B"), "AB"),
    ],
)
def test_nonmaximality_witness_rejects_invalid_structure(
    endpoints: object, path: object
) -> None:
    with pytest.raises(ValueError, match="witness"):
        NonMaximalityWitness(  # type: ignore[arg-type]
            endpoints=endpoints,
            path=path,
        )


def test_mag_domain_errors_are_distinct_from_unknown_vertex_errors() -> None:
    _, view = figure9_view()
    raw = view.topology

    with pytest.raises(AncestralDomainError, match="ancestral"):
        find_primitive_inducing_path(raw, "γ", "δ")  # type: ignore[arg-type]
    with pytest.raises(AncestralDomainError, match="ancestral"):
        find_nonmaximality_witness(raw)  # type: ignore[arg-type]
    with pytest.raises(AncestralDomainError, match="ancestral"):
        maximal_ancestral_extension(raw)  # type: ignore[arg-type]
    with pytest.raises(AncestralDomainError, match="ancestral"):
        MaximalAncestralGraph.from_ancestral(raw)  # type: ignore[arg-type]
    with pytest.raises(AncestralDomainError, match="distinct"):
        find_primitive_inducing_path(view, "γ", "γ")
    with pytest.raises(GraphValidationError, match="unknown vertex"):
        find_primitive_inducing_path(view, "missing", "δ")

    circles = EndpointMarkedGraph(
        nodes=["A", "B"],
        edges=[("A", Mark.CIRCLE, "B", Mark.ARROW)],
    )
    with pytest.raises(GraphValidationError):
        MaximalAncestralGraph.from_graph(circles)


def test_mag_is_closed_under_induced_subgraphs_and_accepted_by_m_separation() -> None:
    collider = MaximalAncestralGraph.from_graph(
        EndpointMarkedGraph(
            nodes=["A", "B", "C"],
            edges=[
                ("A", Mark.TAIL, "B", Mark.ARROW),
                ("C", Mark.TAIL, "B", Mark.ARROW),
            ],
        )
    )

    result = induced_subgraph(collider, ["C", "A"])

    assert type(result) is MaximalAncestralGraph
    assert result.topology == EndpointMarkedGraph(nodes=["A", "C"])
    assert m_separated(collider, "A", "C")
    assert m_separated(result, "A", "C")
    assert maximal_ancestral_extension(collider) is collider


def test_n4_endpoint_universe_matches_independent_maximality_definitions() -> None:
    ancestral_count = 0
    maximal_count = 0
    nonmaximal_count = 0

    for choices in itertools.product(
        range(len(ANCESTRAL_STATES)), repeat=len(PAIRS4)
    ):
        topology = make_ancestral_choice_graph(choices)
        atoms = endpoint_atoms(topology)
        if not endpoint_graph_is_ancestral_by_definition(NODES4, atoms):
            continue
        ancestral_count += 1
        ancestral = AncestralGraph.from_graph(topology)

        expected_pair = nonmaximal_pair_by_all_conditioning_sets(
            NODES4, atoms
        )
        expected_path = nonmaximality_path_by_simple_paths(NODES4, atoms)
        assert (expected_pair is None) == (expected_path is None)
        if expected_path is not None:
            assert expected_pair == (expected_path[0], expected_path[-1])

        for left, right in PAIRS4:
            paths = primitive_inducing_paths_by_simple_paths(
                NODES4, atoms, left, right
            )
            assert find_primitive_inducing_path(
                ancestral, left, right
            ) == (paths[0] if paths else None)

        witness = find_nonmaximality_witness(ancestral)
        if expected_pair is None:
            maximal_count += 1
            assert witness is None
            MaximalAncestralGraph.from_ancestral(ancestral)
        else:
            nonmaximal_count += 1
            assert witness is not None
            assert witness.endpoints == expected_pair
            assert witness.path == expected_path
            with pytest.raises(GraphValidationError, match="maximal"):
                MaximalAncestralGraph.from_ancestral(ancestral)

        maximal = maximal_ancestral_extension(ancestral)
        expected_extension = maximal_extension_by_all_conditioning_sets(
            NODES4, atoms
        )
        assert endpoint_atoms(maximal.topology) == expected_extension
        assert find_nonmaximality_witness(maximal) is None
        for left, right, given in pair_queries(NODES4):
            expected = m_separated_by_simple_edge_paths(
                NODES4, atoms, (left,), (right,), given
            )
            assert m_separated(ancestral, left, right, given) == expected
            assert m_separated(maximal, left, right, given) == expected

    assert ancestral_count == 3_057
    assert maximal_count == 3_045
    assert nonmaximal_count == 12
