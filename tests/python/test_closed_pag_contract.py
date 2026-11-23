from __future__ import annotations

import copy
import itertools
import json
import pickle
import sys
from dataclasses import FrozenInstanceError, dataclass
from functools import cache
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    AncestralGraph,
    ClosedPAG,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    MAGListingResourceLimitError,
    Mark,
    MaximalAncestralGraph,
    PAGDomainError,
    list_mag_equivalence_class,
)

from .oracles import (
    ConditionalIndependenceSignature,
    EndpointMarkAtom,
    FullPAGOracleClass,
    MixedEdgeAtom,
    endpoint_graph_is_ancestral_by_definition,
    full_pag_classes_by_all_conditioning_subsets,
    full_pag_mark_intersection_by_definition,
    m_separated_by_simple_edge_paths,
    m_separation_signature_by_all_conditioning_subsets,
    nonmaximal_pair_by_all_conditioning_sets,
    strict_anterior_vertices_by_simple_paths,
)


NODES4 = ("A", "B", "C", "D")
PAIRS4 = tuple(itertools.combinations(NODES4, 2))
CIRCLE_FREE_STATES: tuple[None | tuple[bool, bool], ...] = (
    None,
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)
PRESENT_STATES = CIRCLE_FREE_STATES[1:]


@dataclass(frozen=True)
class _N4OracleCensus:
    raw_count: int
    ancestral_count: int
    mags: tuple[tuple[MixedEdgeAtom, ...], ...]
    classes: dict[ConditionalIndependenceSignature, FullPAGOracleClass]


def _edges_from_choices(
    pairs: tuple[tuple[str, str], ...], choices: tuple[int, ...]
) -> tuple[MixedEdgeAtom, ...]:
    return tuple(
        (left, state[0], right, state[1])
        for (left, right), choice in zip(pairs, choices, strict=True)
        if (state := CIRCLE_FREE_STATES[choice]) is not None
    )


@cache
def _n4_oracle_census() -> _N4OracleCensus:
    raw_count = 0
    ancestral_count = 0
    mags: list[tuple[MixedEdgeAtom, ...]] = []
    for choices in itertools.product(
        range(len(CIRCLE_FREE_STATES)), repeat=len(PAIRS4)
    ):
        raw_count += 1
        edges = _edges_from_choices(PAIRS4, choices)
        if not endpoint_graph_is_ancestral_by_definition(NODES4, edges):
            continue
        ancestral_count += 1
        if nonmaximal_pair_by_all_conditioning_sets(NODES4, edges) is None:
            mags.append(edges)
    normalized = tuple(mags)
    return _N4OracleCensus(
        raw_count=raw_count,
        ancestral_count=ancestral_count,
        mags=normalized,
        classes=full_pag_classes_by_all_conditioning_subsets(
            NODES4, normalized
        ),
    )


def _mag_from_atoms(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> MaximalAncestralGraph:
    return MaximalAncestralGraph.from_graph(
        EndpointMarkedGraph(
            nodes=nodes,
            edges=[
                (
                    left,
                    Mark.ARROW if arrow_left else Mark.TAIL,
                    right,
                    Mark.ARROW if arrow_right else Mark.TAIL,
                )
                for left, arrow_left, right, arrow_right in edges
            ],
        )
    )


def _topology_mark_atoms(
    graph: EndpointMarkedGraph,
) -> tuple[EndpointMarkAtom, ...]:
    return tuple(
        (left, left_mark.value, right, right_mark.value)
        for left, left_mark, right, right_mark in graph.edges
    )


def _fixture_mark_atoms(raw_edges: list[list[str]]) -> tuple[EndpointMarkAtom, ...]:
    return tuple(tuple(edge) for edge in raw_edges)  # type: ignore[return-value]


def _mixed_atoms_from_fixture(
    nodes: tuple[str, ...],
    raw_edges: list[list[str]],
) -> tuple[MixedEdgeAtom, ...]:
    positions = {node: index for index, node in enumerate(nodes)}
    normalized: list[MixedEdgeAtom] = []
    for left, left_mark, right, right_mark in raw_edges:
        arrow_left = left_mark == "arrow"
        arrow_right = right_mark == "arrow"
        if positions[left] < positions[right]:
            normalized.append((left, arrow_left, right, arrow_right))
        else:
            normalized.append((right, arrow_right, left, arrow_left))
    normalized.sort(
        key=lambda edge: (positions[edge[0]], positions[edge[2]])
    )
    return tuple(normalized)


def _load_fixture(name: str) -> dict[str, object]:
    path = Path(__file__).parents[1] / "fixtures" / "papers" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_skeleton_class(
    nodes: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
    seed: tuple[MixedEdgeAtom, ...],
) -> tuple[int, tuple[tuple[MixedEdgeAtom, ...], ...]]:
    """Enumerate all circle-free orientations of one fixed skeleton."""

    positions = {node: index for index, node in enumerate(nodes)}
    canonical_pairs = tuple(
        sorted(
            (
                (left, right)
                if positions[left] < positions[right]
                else (right, left)
                for left, right in pairs
            ),
            key=lambda pair: (positions[pair[0]], positions[pair[1]]),
        )
    )
    seed_signature = m_separation_signature_by_all_conditioning_subsets(
        nodes, seed
    )
    candidate_count = 0
    members: list[tuple[MixedEdgeAtom, ...]] = []
    for states in itertools.product(PRESENT_STATES, repeat=len(canonical_pairs)):
        candidate_count += 1
        edges = tuple(
            (left, marks[0], right, marks[1])
            for (left, right), marks in zip(
                canonical_pairs, states, strict=True
            )
        )
        if not endpoint_graph_is_ancestral_by_definition(nodes, edges):
            continue
        if nonmaximal_pair_by_all_conditioning_sets(nodes, edges) is not None:
            continue
        if (
            m_separation_signature_by_all_conditioning_subsets(nodes, edges)
            == seed_signature
        ):
            members.append(edges)
    return candidate_count, tuple(members)


def _arrow_at(
    edges: tuple[MixedEdgeAtom, ...], left: str, right: str, endpoint: str
) -> bool:
    for first, arrow_first, second, arrow_second in edges:
        if first == left and second == right:
            return arrow_first if endpoint == left else arrow_second
        if first == right and second == left:
            return arrow_second if endpoint == left else arrow_first
    raise AssertionError(f"missing adjacency: {(left, right)!r}")


def _assert_endpoint_intersection_witnesses(
    members: tuple[tuple[MixedEdgeAtom, ...], ...],
    pag: tuple[EndpointMarkAtom, ...],
) -> None:
    for left, left_mark, right, right_mark in pag:
        for endpoint, mark in ((left, left_mark), (right, right_mark)):
            witnesses = {
                _arrow_at(member, left, right, endpoint)
                for member in members
            }
            if mark == "circle":
                assert witnesses == {False, True}
            elif mark == "tail":
                assert witnesses == {False}
            else:
                assert mark == "arrow"
                assert witnesses == {True}


def _labelwise_marks(
    topology: EndpointMarkedGraph,
) -> dict[frozenset[str], dict[str, Mark]]:
    return {
        frozenset((left, right)): {
            left: left_mark,
            right: right_mark,
        }
        for left, left_mark, right, right_mark in topology.edges
    }


def _contains_tail_circle(pag: tuple[EndpointMarkAtom, ...]) -> bool:
    return any(
        {left_mark, right_mark} == {"tail", "circle"}
        for _, left_mark, _, right_mark in pag
    )


def test_n4_mags_obey_richardson_spirtes_corollary_5_3() -> None:
    census = _n4_oracle_census()
    anterior_separator_checks = 0
    for edges in census.mags:
        endpoint_edges = tuple(
            (
                left,
                "arrow" if arrow_left else "tail",
                right,
                "arrow" if arrow_right else "tail",
            )
            for left, arrow_left, right, arrow_right in edges
        )
        adjacencies = {
            frozenset((left, right)) for left, _, right, _ in edges
        }
        for left, right in PAIRS4:
            if frozenset((left, right)) in adjacencies:
                continue
            given = strict_anterior_vertices_by_simple_paths(
                NODES4, endpoint_edges, (left, right)
            )
            assert m_separated_by_simple_edge_paths(
                NODES4, edges, (left,), (right,), given
            )
            anterior_separator_checks += 1
    assert anterior_separator_checks == 5_916


def test_n4_full_pag_matches_independent_complete_class_intersections() -> None:
    census = _n4_oracle_census()
    assert census.raw_count == 15_625
    assert census.ancestral_count == 3_057
    assert len(census.mags) == 3_045
    assert len(census.classes) == 251
    assert max(len(members) for members, _ in census.classes.values()) == 242
    assert len({pag for _, pag in census.classes.values()}) == 251
    assert sum(len(members) for members, _ in census.classes.values()) == 3_045

    for members, expected_pag in census.classes.values():
        _assert_endpoint_intersection_witnesses(members, expected_pag)
        for edges in members:
            source = _mag_from_atoms(NODES4, edges)
            before = source.topology
            actual = ClosedPAG.from_mag(source)
            assert actual.topology.nodes == NODES4
            assert _topology_mark_atoms(actual.topology) == expected_pag
            assert source.topology == before


def test_small_copy_seed_chain_and_collider_endpoint_intersections() -> None:
    cases = (
        (
            ("A", "B"),
            (("A", "B"),),
            (("A", False, "B", True),),
            (("A", "circle", "B", "circle"),),
        ),
        (
            ("A", "B", "C"),
            (("A", "B"), ("B", "C")),
            (("A", False, "B", True), ("B", False, "C", True)),
            (
                ("A", "circle", "B", "circle"),
                ("B", "circle", "C", "circle"),
            ),
        ),
        (
            ("A", "B", "C"),
            (("A", "B"), ("B", "C")),
            (("A", False, "B", True), ("B", True, "C", False)),
            (
                ("A", "circle", "B", "arrow"),
                ("B", "arrow", "C", "circle"),
            ),
        ),
    )
    for nodes, pairs, seed, expected in cases:
        _, members = _fixed_skeleton_class(nodes, pairs, seed)
        assert full_pag_mark_intersection_by_definition(
            nodes, members
        ) == expected
        _assert_endpoint_intersection_witnesses(members, expected)
        assert _topology_mark_atoms(
            ClosedPAG.from_mag(_mag_from_atoms(nodes, seed)).topology
        ) == expected


def test_directed_chain_distinguishes_full_from_no_undirected_universe() -> None:
    nodes = ("A", "B", "C")
    pairs = (("A", "B"), ("B", "C"))
    seed = (("A", False, "B", True), ("B", False, "C", True))
    candidate_count, members = _fixed_skeleton_class(nodes, pairs, seed)
    no_undirected_members = tuple(
        member
        for member in members
        if all(arrow_left or arrow_right for _, arrow_left, _, arrow_right in member)
    )

    assert candidate_count == 4**2
    assert len(members) == 8
    assert len(no_undirected_members) == 5
    pag = ClosedPAG.from_mag(_mag_from_atoms(nodes, seed))
    assert _topology_mark_atoms(pag.topology) == (
        ("A", "circle", "B", "circle"),
        ("B", "circle", "C", "circle"),
    )
    listed = list_mag_equivalence_class(pag, max_members=8)
    assert len(listed) == len(members)
    assert {member.topology for member in listed} == {
        _mag_from_atoms(nodes, edges).topology for edges in members
    }
    with pytest.raises(MAGListingResourceLimitError):
        list_mag_equivalence_class(pag, max_members=7)


def test_ali_figure4_full_pag_reuses_equivalence_fixture() -> None:
    fixture = _load_fixture("ali_richardson_spirtes_2009_figure4.json")
    nodes = tuple(fixture["nodes"])
    graphs = fixture["graphs"]
    g1 = _mixed_atoms_from_fixture(nodes, graphs["G1"])
    g2 = _mixed_atoms_from_fixture(nodes, graphs["G2"])
    g3 = _mixed_atoms_from_fixture(nodes, graphs["G3"])
    pairs = tuple((left, right) for left, _, right, _ in g1)
    _, equivalent_members = _fixed_skeleton_class(nodes, pairs, g1)
    expected = (
        ("x", "circle", "q", "arrow"),
        ("q", "arrow", "b", "arrow"),
        ("q", "tail", "y", "arrow"),
        ("b", "arrow", "y", "arrow"),
    )

    assert len(equivalent_members) == 2
    assert set(equivalent_members) == {g1, g2}
    assert full_pag_mark_intersection_by_definition(
        nodes, equivalent_members
    ) == expected
    for seed in (g1, g2):
        assert _topology_mark_atoms(
            ClosedPAG.from_mag(_mag_from_atoms(nodes, seed)).topology
        ) == expected

    _, g3_members = _fixed_skeleton_class(nodes, pairs, g3)
    g3_expected = full_pag_mark_intersection_by_definition(nodes, g3_members)
    assert g3_expected != expected
    assert _topology_mark_atoms(
        ClosedPAG.from_mag(_mag_from_atoms(nodes, g3)).topology
    ) == g3_expected


def test_ali_figure6_panels_protect_raw_discriminating_path_traps() -> None:
    fixture = _load_fixture("ali_richardson_spirtes_2009_figure6.json")
    nodes = tuple(fixture["nodes"])
    expected_by_panel = {
        "i": (
            ("x", "circle", "q", "circle"),
            ("x", "circle", "b", "circle"),
            ("q", "circle", "b", "circle"),
            ("q", "circle", "y", "circle"),
            ("b", "circle", "y", "circle"),
        ),
        "ii": (
            ("x", "circle", "q", "circle"),
            ("x", "circle", "b", "arrow"),
            ("q", "circle", "b", "arrow"),
            ("q", "circle", "y", "circle"),
            ("b", "arrow", "y", "circle"),
        ),
    }
    expected_sizes = {"i": 43, "ii": 30}

    for panel, raw_graphs in fixture["panels"].items():
        g1 = _mixed_atoms_from_fixture(nodes, raw_graphs["G1"])
        g2 = _mixed_atoms_from_fixture(nodes, raw_graphs["G2"])
        pairs = tuple((left, right) for left, _, right, _ in g1)
        candidate_count, members = _fixed_skeleton_class(nodes, pairs, g1)
        expected = expected_by_panel[panel]

        assert candidate_count == 4**5
        assert len(members) == expected_sizes[panel]
        assert g1 in members and g2 in members
        assert full_pag_mark_intersection_by_definition(
            nodes, members
        ) == expected
        _assert_endpoint_intersection_witnesses(members, expected)
        for seed in (g1, g2):
            assert _topology_mark_atoms(
                ClosedPAG.from_mag(_mag_from_atoms(nodes, seed)).topology
            ) == expected


def test_zhang_figure6_exact_full_pag_and_independent_class_size() -> None:
    fixture = _load_fixture("zhang_2008_figure6.json")
    assert fixture["journal_page"] == 1882
    assert fixture["pdf_page"] == 10
    assert fixture["figure"] == "6"
    assert fixture["reported_class_size_status"] == (
        "the paper does not report the equivalence-class size"
    )
    nodes = tuple(fixture["nodes"])
    seed = _mixed_atoms_from_fixture(nodes, fixture["seed_mag"])
    pairs = tuple((left, right) for left, _, right, _ in seed)
    candidate_count, members = _fixed_skeleton_class(nodes, pairs, seed)
    expected = _fixture_mark_atoms(fixture["expected_full_pag"])

    assert candidate_count == 4**4
    assert len(members) == 8
    assert seed in members
    assert full_pag_mark_intersection_by_definition(nodes, members) == expected
    _assert_endpoint_intersection_witnesses(members, expected)
    assert _topology_mark_atoms(
        ClosedPAG.from_mag(_mag_from_atoms(nodes, seed)).topology
    ) == expected

    negative = tuple(
        (left, True, right, False)
        if (left, right) == ("A", "C")
        else edge
        for edge in seed
        for left, _, right, _ in (edge,)
    )
    negative_mag = _mag_from_atoms(nodes, negative)
    assert m_separation_signature_by_all_conditioning_subsets(
        nodes, negative
    ) != m_separation_signature_by_all_conditioning_subsets(nodes, seed)
    assert _topology_mark_atoms(
        ClosedPAG.from_mag(negative_mag).topology
    ) != expected


def test_full_selection_undirected_c4_is_singleton_all_tail_pag() -> None:
    fixture = _load_fixture("zhang_2008_r5_r6_regressions.json")
    assert fixture["journal_page"] == 1881
    assert fixture["pdf_page"] == 9
    assert fixture["figure"] == "4"
    assert fixture["fixture_status"] == (
        "the two MAG examples below are independently constructed regressions "
        "tied to the R5-R6 schematics, not literal MAGs reported in Figure 4"
    )
    case = fixture["undirected_c4"]
    nodes = tuple(case["nodes"])
    pairs = tuple(tuple(pair) for pair in case["skeleton"])
    seed = _mixed_atoms_from_fixture(nodes, case["seed_mag"])
    candidate_count, members = _fixed_skeleton_class(nodes, pairs, seed)
    expected = _fixture_mark_atoms(case["expected_full_pag"])

    assert candidate_count == 4**4
    assert len(members) == case["adjudicated_class_size"] == 1
    assert members == (seed,)
    assert full_pag_mark_intersection_by_definition(nodes, members) == expected
    assert all(mark == "tail" for edge in expected for mark in (edge[1], edge[3]))
    assert _topology_mark_atoms(
        ClosedPAG.from_mag(_mag_from_atoms(nodes, seed)).topology
    ) == expected


def test_r5_then_r6_fixed_skeleton_protects_tail_circle_endpoint() -> None:
    fixture = _load_fixture("zhang_2008_r5_r6_regressions.json")
    case = fixture["r5_then_r6_tail_circle"]
    nodes = tuple(case["nodes"])
    pairs = tuple(tuple(pair) for pair in case["skeleton"])
    seed = _mixed_atoms_from_fixture(nodes, case["seed_mag"])
    candidate_count, members = _fixed_skeleton_class(nodes, pairs, seed)
    expected = _fixture_mark_atoms(case["expected_full_pag"])
    expected_members = {
        _fixture_mark_atoms(raw_member)
        for raw_member in case["adjudicated_members"]
    }

    assert candidate_count == case["adjudicated_fixed_skeleton_candidates"] == 1_024
    assert len(members) == case["adjudicated_class_size"] == 2
    assert {
        tuple(
            (
                left,
                "arrow" if arrow_left else "tail",
                right,
                "arrow" if arrow_right else "tail",
            )
            for left, arrow_left, right, arrow_right in member
        )
        for member in members
    } == expected_members
    assert full_pag_mark_intersection_by_definition(nodes, members) == expected
    assert _contains_tail_circle(expected)
    _assert_endpoint_intersection_witnesses(members, expected)
    for member in members:
        assert _topology_mark_atoms(
            ClosedPAG.from_mag(_mag_from_atoms(nodes, member)).topology
        ) == expected

    n4_pags = {pag for _, pag in _n4_oracle_census().classes.values()}
    assert all(not _contains_tail_circle(pag) for pag in n4_pags)
    for size in range(4):
        smaller_nodes = NODES4[:size]
        smaller_pairs = tuple(itertools.combinations(smaller_nodes, 2))
        smaller_mags = []
        for choices in itertools.product(
            range(len(CIRCLE_FREE_STATES)), repeat=len(smaller_pairs)
        ):
            edges = _edges_from_choices(smaller_pairs, choices)
            if endpoint_graph_is_ancestral_by_definition(smaller_nodes, edges) and (
                nonmaximal_pair_by_all_conditioning_sets(smaller_nodes, edges)
                is None
            ):
                smaller_mags.append(edges)
        assert all(
            not _contains_tail_circle(pag)
            for _, pag in full_pag_classes_by_all_conditioning_subsets(
                smaller_nodes, smaller_mags
            ).values()
        )


def test_closed_pag_is_factory_only_sealed_frozen_and_unforgeable_normally() -> None:
    source = _mag_from_atoms(("A", "B"), (("A", False, "B", True),))
    before = source.topology
    closed = ClosedPAG.from_mag(source)

    assert isinstance(closed.topology, EndpointMarkedGraph)
    assert closed.topology.nodes == source.topology.nodes
    assert copy.copy(closed) is closed
    assert copy.deepcopy(closed) is closed
    assert hash(closed) == hash(closed)
    assert not hasattr(closed, "__dict__")
    assert source.topology == before
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        closed.topology = EndpointMarkedGraph(nodes=[])  # type: ignore[misc]
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("ForgedClosedPAG", (ClosedPAG,), {})
    with pytest.raises((AttributeError, TypeError)):
        closed.__setstate__(  # type: ignore[attr-defined]
            [EndpointMarkedGraph(nodes=[])]
        )
    with pytest.raises(TypeError, match="pickle is unsupported"):
        pickle.dumps(closed)
    with pytest.raises(TypeError):
        ClosedPAG(source.topology)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ClosedPAG()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ClosedPAG.from_mag(source, background_knowledge=())  # type: ignore[call-arg]


def test_closed_pag_domain_requires_exact_mag_proof_view_message() -> None:
    raw_endpoint = EndpointMarkedGraph(nodes=["A"])
    ancestral = AncestralGraph.from_graph(raw_endpoint)
    raw_directed = DirectedBidirectedGraph(nodes=["A"])
    dag = DAG.from_graph(raw_directed)
    for wrong in (raw_endpoint, ancestral, raw_directed, dag):
        with pytest.raises(PAGDomainError) as error:
            ClosedPAG.from_mag(wrong)  # type: ignore[arg-type]
        assert str(error.value) == "closed PAG construction requires a MAG proof view"


def test_closed_pag_accepts_empty_singleton_and_more_than_sixteen_vertices() -> None:
    node_sets = (
        (),
        ("only",),
        tuple(f"V{index}" for index in range(40)),
    )
    for nodes in node_sets:
        source = MaximalAncestralGraph.from_graph(
            EndpointMarkedGraph(nodes=nodes)
        )
        closed = ClosedPAG.from_mag(source)
        assert closed.topology.nodes == nodes
        assert closed.topology.edges == ()

    path_nodes = tuple(f"P{index}" for index in range(20))
    directed_path = tuple(
        (parent, False, child, True)
        for parent, child in itertools.pairwise(path_nodes)
    )
    path_pag = ClosedPAG.from_mag(
        _mag_from_atoms(path_nodes, directed_path)
    )
    assert path_pag.topology.nodes == path_nodes
    assert all(
        left_mark is Mark.CIRCLE and right_mark is Mark.CIRCLE
        for _, left_mark, _, right_mark in path_pag.topology.edges
    )


def test_closed_pag_path_searches_do_not_depend_on_python_recursion_limit() -> None:
    nodes = tuple(f"C{index}" for index in range(80))
    cycle = EndpointMarkedGraph(
        nodes=nodes,
        edges=(
            (nodes[index], Mark.TAIL, nodes[(index + 1) % len(nodes)], Mark.TAIL)
            for index in range(len(nodes))
        ),
    )
    source = MaximalAncestralGraph.from_graph(cycle)
    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(75)
        closed = ClosedPAG.from_mag(source)
    finally:
        sys.setrecursionlimit(original_limit)

    assert closed.topology.nodes == nodes
    assert all(
        left_mark is Mark.TAIL and right_mark is Mark.TAIL
        for _, left_mark, _, right_mark in closed.topology.edges
    )


def test_closed_pag_preserves_source_order_but_is_labelwise_order_independent() -> None:
    fixture = _load_fixture("zhang_2008_figure6.json")
    original_nodes = tuple(fixture["nodes"])
    permuted_nodes = ("D", "B", "A", "C")
    seed = _mixed_atoms_from_fixture(original_nodes, fixture["seed_mag"])
    original = _mag_from_atoms(original_nodes, seed)
    permuted = _mag_from_atoms(permuted_nodes, seed)

    original_pag = ClosedPAG.from_mag(original)
    permuted_pag = ClosedPAG.from_mag(permuted)
    assert original_pag.topology.nodes == original_nodes
    assert permuted_pag.topology.nodes == permuted_nodes
    assert _labelwise_marks(original_pag.topology) == _labelwise_marks(
        permuted_pag.topology
    )
    assert original.topology.nodes == original_nodes
    assert permuted.topology.nodes == permuted_nodes
