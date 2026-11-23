from __future__ import annotations

import copy
import itertools
import json
import pickle
import random
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    AncestralGraph,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    Mark,
    SeparationDomainError,
    d_separated,
    d_separated_sets,
    m_separated,
    m_separated_sets,
)

from .oracles import (
    MixedEdgeAtom,
    endpoint_graph_is_ancestral_by_definition,
    m_separated_by_canonical_dag,
    m_separated_by_simple_edge_paths,
)


NODES3 = ("A", "B", "C")
PAIRS3 = (("A", "B"), ("A", "C"), ("B", "C"))
NODES4 = ("A", "B", "C", "D")
PAIRS4 = tuple(itertools.combinations(NODES4, 2))


def directed_bidirected_atoms(graph: DirectedBidirectedGraph) -> tuple[MixedEdgeAtom, ...]:
    atoms: list[MixedEdgeAtom] = [
        (source, False, target, True) for source, target in graph.directed
    ]
    atoms.extend((left, True, right, True) for left, right in graph.bidirected)
    return tuple(atoms)


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


def pair_queries(nodes: tuple[str, ...]):
    for left_index, right_index in itertools.combinations(range(len(nodes)), 2):
        remaining = tuple(
            node
            for index, node in enumerate(nodes)
            if index not in (left_index, right_index)
        )
        for size in range(len(remaining) + 1):
            for given in itertools.combinations(remaining, size):
                yield nodes[left_index], nodes[right_index], given


def set_queries(nodes: tuple[str, ...]):
    for assignments in itertools.product(range(4), repeat=len(nodes)):
        left = tuple(
            node for node, assignment in zip(nodes, assignments, strict=True)
            if assignment == 1
        )
        right = tuple(
            node for node, assignment in zip(nodes, assignments, strict=True)
            if assignment == 2
        )
        given = tuple(
            node for node, assignment in zip(nodes, assignments, strict=True)
            if assignment == 3
        )
        if left and right:
            yield left, right, given


def make_directed_bidirected_choice_graph(
    choices: tuple[int, ...],
    *,
    nodes: tuple[str, ...] = NODES3,
    pairs: tuple[tuple[str, str], ...] = PAIRS3,
) -> DirectedBidirectedGraph:
    directed: list[tuple[str, str]] = []
    bidirected: list[tuple[str, str]] = []
    for (left, right), choice in zip(pairs, choices, strict=True):
        if choice in (1, 4):
            directed.append((left, right))
        if choice in (2, 5):
            directed.append((right, left))
        if choice in (3, 4, 5):
            bidirected.append((left, right))
    return DirectedBidirectedGraph(
        nodes=nodes, directed=directed, bidirected=bidirected
    )


def make_endpoint_choice_graph(
    choices: tuple[int, ...],
    *,
    nodes: tuple[str, ...] = NODES3,
    pairs: tuple[tuple[str, str], ...] = PAIRS3,
) -> EndpointMarkedGraph:
    states: tuple[None | tuple[Mark, Mark], ...] = (
        None,
        (Mark.TAIL, Mark.TAIL),
        (Mark.TAIL, Mark.ARROW),
        (Mark.ARROW, Mark.TAIL),
        (Mark.ARROW, Mark.ARROW),
    )
    edges = [
        (left, state[0], right, state[1])
        for (left, right), choice in zip(pairs, choices, strict=True)
        if (state := states[choice]) is not None
    ]
    return EndpointMarkedGraph(nodes=nodes, edges=edges)


def decode_choices(value: int, base: int, count: int) -> tuple[int, ...]:
    return tuple((value // (base**index)) % base for index in range(count))


def test_ancestral_view_enforces_the_published_domain() -> None:
    valid = EndpointMarkedGraph(
        nodes=["A", "B", "C", "D"],
        edges=[
            ("A", Mark.TAIL, "B", Mark.TAIL),
            ("A", Mark.TAIL, "C", Mark.ARROW),
            ("C", Mark.ARROW, "D", Mark.ARROW),
        ],
    )
    assert AncestralGraph.from_graph(valid).topology == valid

    invalid = (
        EndpointMarkedGraph(
            nodes=["A", "B"],
            edges=[("A", Mark.CIRCLE, "B", Mark.ARROW)],
        ),
        EndpointMarkedGraph(
            nodes=["A", "B", "C"],
            edges=[
                ("A", Mark.TAIL, "B", Mark.ARROW),
                ("B", Mark.TAIL, "C", Mark.ARROW),
                ("C", Mark.TAIL, "A", Mark.ARROW),
            ],
        ),
        EndpointMarkedGraph(
            nodes=["A", "B", "C"],
            edges=[
                ("A", Mark.ARROW, "B", Mark.ARROW),
                ("A", Mark.TAIL, "C", Mark.ARROW),
                ("C", Mark.TAIL, "B", Mark.ARROW),
            ],
        ),
        EndpointMarkedGraph(
            nodes=["A", "B", "C"],
            edges=[
                ("A", Mark.TAIL, "B", Mark.TAIL),
                ("C", Mark.TAIL, "A", Mark.ARROW),
            ],
        ),
    )
    for graph in invalid:
        with pytest.raises(GraphValidationError):
            AncestralGraph.from_graph(graph)


def test_ancestral_view_is_a_sealed_immutable_proof_value() -> None:
    topology = EndpointMarkedGraph(
        nodes=["A", "B"], edges=[("A", Mark.TAIL, "B", Mark.ARROW)]
    )
    view = AncestralGraph.from_graph(topology)
    assert copy.copy(view) is view
    assert copy.deepcopy(view) is view
    assert not hasattr(view, "__dict__")
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("ForgedAncestral", (AncestralGraph,), {})
    with pytest.raises((AttributeError, TypeError)):
        view.__setstate__([EndpointMarkedGraph(nodes=[])])  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="canonical JSON"):
        pickle.dumps(view)


def test_bow_edge_choice_can_change_m_separation() -> None:
    bow = DirectedBidirectedGraph(
        nodes=["X", "M", "Y"],
        directed=[("M", "X"), ("Y", "M")],
        bidirected=[("X", "M")],
    )
    without_bidirected = DirectedBidirectedGraph(
        nodes=bow.nodes, directed=bow.directed
    )
    assert not m_separated(
        AcyclicDirectedBidirectedGraph.from_graph(bow), "X", "Y", given="M"
    )
    assert m_separated(
        AcyclicDirectedBidirectedGraph.from_graph(without_bidirected),
        "X",
        "Y",
        given="M",
    )


def test_chain_collider_and_descendant_activation() -> None:
    chain = DAG.from_graph(
        DirectedBidirectedGraph(
            nodes=["X", "M", "Y"], directed=[("X", "M"), ("M", "Y")]
        )
    )
    assert not d_separated(chain, "X", "Y")
    assert d_separated(chain, "X", "Y", given="M")

    collider = DAG.from_graph(
        DirectedBidirectedGraph(
            nodes=["X", "M", "Y", "D"],
            directed=[("X", "M"), ("Y", "M"), ("M", "D")],
        )
    )
    assert d_separated(collider, "X", "Y")
    assert not d_separated(collider, "X", "Y", given="M")
    assert not d_separated(collider, "X", "Y", given="D")
    assert m_separated(collider, "X", "Y")
    assert not m_separated(collider, "X", "Y", given="M")
    assert not m_separated(collider, "X", "Y", given="D")

    long_chain = DAG.from_graph(
        DirectedBidirectedGraph(
            nodes=NODES4,
            directed=[("A", "B"), ("B", "C"), ("C", "D")],
        )
    )
    assert not m_separated(long_chain, "A", "D")
    assert m_separated(long_chain, "A", "D", given="B")


def test_set_semantics_and_invalid_queries_fail_closed() -> None:
    view = AcyclicDirectedBidirectedGraph.from_graph(
        DirectedBidirectedGraph(
            nodes=["A", "B", "C"], directed=[("A", "B")]
        )
    )
    assert m_separated_sets(view, "A", ["B", "C"]) == (
        m_separated(view, "A", "B") and m_separated(view, "A", "C")
    )
    assert m_separated(view, "A", "C", given="B")

    with pytest.raises(SeparationDomainError):
        m_separated(view, "A", "A")
    with pytest.raises(SeparationDomainError):
        m_separated(view, "A", "B", given="A")
    with pytest.raises(SeparationDomainError):
        m_separated_sets(view, [], ["B"])
    with pytest.raises(SeparationDomainError):
        m_separated_sets(view, ["A"], [])
    with pytest.raises(SeparationDomainError):
        m_separated_sets(view, ["A"], ["A"])
    with pytest.raises(GraphValidationError):
        m_separated(view, "missing", "B")
    with pytest.raises(SeparationDomainError):
        m_separated(view.topology, "A", "B")  # type: ignore[arg-type]


def test_paper_fixture_richardson_spirtes_figure9() -> None:
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "papers"
        / "richardson_spirtes_2002_figure9.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["page"] == 979
    assert fixture["erratum_status"] == "not amended by the official errata"
    graph = EndpointMarkedGraph(
        nodes=fixture["nodes"],
        edges=[
            (left, Mark(left_mark), right, Mark(right_mark))
            for left, left_mark, right, right_mark in fixture["edges"]
        ],
    )
    view = AncestralGraph.from_graph(graph)
    atoms = endpoint_atoms(graph)
    for query in fixture["queries"]:
        assert not m_separated_by_simple_edge_paths(
            graph.nodes,
            atoms,
            [query["left"]],
            [query["right"]],
            query["given"],
        )
        assert not m_separated_by_canonical_dag(
            graph.nodes,
            atoms,
            [query["left"]],
            [query["right"]],
            query["given"],
        )
        assert not m_separated(
            view, query["left"], query["right"], given=query["given"]
        )


def test_bow_aware_universe_matches_two_independent_oracles() -> None:
    accepted = 0
    for choices in itertools.product(range(6), repeat=len(PAIRS3)):
        topology = make_directed_bidirected_choice_graph(choices)
        try:
            view = AcyclicDirectedBidirectedGraph.from_graph(topology)
        except GraphValidationError:
            continue
        accepted += 1
        atoms = directed_bidirected_atoms(topology)
        for left, right, given in pair_queries(NODES3):
            expected_path = m_separated_by_simple_edge_paths(
                NODES3, atoms, [left], [right], given
            )
            expected_dag = m_separated_by_canonical_dag(
                NODES3, atoms, [left], [right], given
            )
            assert expected_path == expected_dag
            assert m_separated(view, left, right, given=given) == expected_path
            assert m_separated(view, right, left, given=given) == expected_path
        for left, right, given in set_queries(NODES3):
            expected_path = m_separated_by_simple_edge_paths(
                NODES3, atoms, left, right, given
            )
            expected_dag = m_separated_by_canonical_dag(
                NODES3, atoms, left, right, given
            )
            assert expected_path == expected_dag
            assert m_separated_sets(
                view, left, right, given=given
            ) == expected_path
    assert accepted == 200


def test_endpoint_universe_validates_domain_and_matches_oracles() -> None:
    for choices in itertools.product(range(5), repeat=len(PAIRS3)):
        topology = make_endpoint_choice_graph(choices)
        atoms = endpoint_atoms(topology)
        expected_ancestral = endpoint_graph_is_ancestral_by_definition(NODES3, atoms)
        if not expected_ancestral:
            with pytest.raises(GraphValidationError):
                AncestralGraph.from_graph(topology)
            continue
        view = AncestralGraph.from_graph(topology)
        for left, right, given in pair_queries(NODES3):
            expected_path = m_separated_by_simple_edge_paths(
                NODES3, atoms, [left], [right], given
            )
            expected_dag = m_separated_by_canonical_dag(
                NODES3, atoms, [left], [right], given
            )
            assert expected_path == expected_dag
            assert m_separated(view, left, right, given=given) == expected_path
        for left, right, given in set_queries(NODES3):
            expected_path = m_separated_by_simple_edge_paths(
                NODES3, atoms, left, right, given
            )
            expected_dag = m_separated_by_canonical_dag(
                NODES3, atoms, left, right, given
            )
            assert expected_path == expected_dag
            assert m_separated_sets(
                view, left, right, given=given
            ) == expected_path


def test_d_separation_is_the_dag_special_case_exhaustively() -> None:
    possible_edges = tuple(
        (left, right) for left in NODES3 for right in NODES3 if left != right
    )
    for mask in range(1 << len(possible_edges)):
        topology = DirectedBidirectedGraph(
            nodes=NODES3,
            directed=[
                edge
                for index, edge in enumerate(possible_edges)
                if mask & (1 << index)
            ],
        )
        try:
            dag = DAG.from_graph(topology)
        except GraphValidationError:
            continue
        acyclic = AcyclicDirectedBidirectedGraph.from_graph(topology)
        atoms = directed_bidirected_atoms(topology)
        for left, right, given in pair_queries(NODES3):
            expected = m_separated_by_simple_edge_paths(
                NODES3, atoms, [left], [right], given
            )
            assert d_separated(dag, left, right, given=given) == expected
            assert m_separated(dag, left, right, given=given) == expected
            assert m_separated(acyclic, left, right, given=given) == expected
        for left, right, given in set_queries(NODES3):
            expected = m_separated_by_simple_edge_paths(
                NODES3, atoms, left, right, given
            )
            assert d_separated_sets(
                dag, left, right, given=given
            ) == expected
            assert m_separated_sets(
                dag, left, right, given=given
            ) == expected


def test_fixed_seed_four_vertex_sample_matches_two_independent_oracles() -> None:
    rng = random.Random(20260824)

    directed_bidirected_ids = rng.sample(
        range(6 ** len(PAIRS4)), 6 ** len(PAIRS4)
    )
    accepted_directed_bidirected = 0
    for graph_id in directed_bidirected_ids:
        choices = decode_choices(graph_id, 6, len(PAIRS4))
        topology = make_directed_bidirected_choice_graph(
            choices, nodes=NODES4, pairs=PAIRS4
        )
        try:
            view = AcyclicDirectedBidirectedGraph.from_graph(topology)
        except GraphValidationError:
            continue
        atoms = directed_bidirected_atoms(topology)
        for left, right, given in pair_queries(NODES4):
            expected_path = m_separated_by_simple_edge_paths(
                NODES4, atoms, [left], [right], given
            )
            expected_dag = m_separated_by_canonical_dag(
                NODES4, atoms, [left], [right], given
            )
            assert expected_path == expected_dag
            assert m_separated(view, left, right, given=given) == expected_path
        accepted_directed_bidirected += 1
        if accepted_directed_bidirected == 2_000:
            break
    assert accepted_directed_bidirected == 2_000

    endpoint_ids = rng.sample(range(5 ** len(PAIRS4)), 5 ** len(PAIRS4))
    accepted_endpoint = 0
    for graph_id in endpoint_ids:
        choices = decode_choices(graph_id, 5, len(PAIRS4))
        topology = make_endpoint_choice_graph(
            choices, nodes=NODES4, pairs=PAIRS4
        )
        atoms = endpoint_atoms(topology)
        expected_ancestral = endpoint_graph_is_ancestral_by_definition(
            NODES4, atoms
        )
        try:
            view = AncestralGraph.from_graph(topology)
        except GraphValidationError:
            assert not expected_ancestral
            continue
        assert expected_ancestral
        for left, right, given in pair_queries(NODES4):
            expected_path = m_separated_by_simple_edge_paths(
                NODES4, atoms, [left], [right], given
            )
            expected_dag = m_separated_by_canonical_dag(
                NODES4, atoms, [left], [right], given
            )
            assert expected_path == expected_dag
            assert m_separated(view, left, right, given=given) == expected_path
        accepted_endpoint += 1
        if accepted_endpoint == 2_000:
            break
    assert accepted_endpoint == 2_000
