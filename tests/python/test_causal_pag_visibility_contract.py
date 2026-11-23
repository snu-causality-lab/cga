from __future__ import annotations

import copy
import inspect
import itertools
import json
import pickle
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from causal_graph_algorithms import (
    AncestralGraph,
    CausalPAG,
    ClosedPAG,
    DAG,
    DirectedBidirectedGraph,
    DirectedEdgeVisibilityWitness,
    EndpointMarkedGraph,
    GraphValidationError,
    MAGListingDomainError,
    Mark,
    MaximalAncestralGraph,
    PAGDomainError,
    find_directed_edge_visibility_witness,
    list_mag_equivalence_class,
)

from .oracles import (
    EndpointMarkAtom,
    MixedEdgeAtom,
    NO_SELECTION_STATES,
    directed_edge_visibility_paths_by_simple_paths,
    directed_edge_visibility_witness_by_simple_paths,
    endpoint_graph_is_ancestral_by_definition,
    full_pag_mark_intersection_by_definition,
    m_separation_signature_by_all_conditioning_subsets,
    no_selection_census,
    nonmaximal_pair_by_all_conditioning_sets,
)


NODES4 = ("A", "B", "C", "D")
PRESENT_NO_SELECTION_STATES = NO_SELECTION_STATES[1:]
FIXTURE_NAME = "zhang_2008_visibility_jaber_2022_figure4a.json"


def _endpoint_atoms_from_raw(raw: object) -> tuple[EndpointMarkAtom, ...]:
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


def _mixed_atoms_from_raw(raw: object) -> tuple[MixedEdgeAtom, ...]:
    endpoint_atoms = _endpoint_atoms_from_raw(raw)
    assert all(
        left_mark != "circle" and right_mark != "circle"
        for _, left_mark, _, right_mark in endpoint_atoms
    )
    return tuple(
        (
            left,
            left_mark == "arrow",
            right,
            right_mark == "arrow",
        )
        for left, left_mark, right, right_mark in endpoint_atoms
    )


def _endpoint_atoms_from_mixed(
    edges: tuple[MixedEdgeAtom, ...],
) -> tuple[EndpointMarkAtom, ...]:
    return tuple(
        (
            left,
            "arrow" if arrow_left else "tail",
            right,
            "arrow" if arrow_right else "tail",
        )
        for left, arrow_left, right, arrow_right in edges
    )


def _mag_from_atoms(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> MaximalAncestralGraph:
    return MaximalAncestralGraph.from_graph(
        EndpointMarkedGraph(
            nodes=nodes,
            edges=(
                (
                    left,
                    Mark.ARROW if arrow_left else Mark.TAIL,
                    right,
                    Mark.ARROW if arrow_right else Mark.TAIL,
                )
                for left, arrow_left, right, arrow_right in edges
            ),
        )
    )


def _pag_atoms(pag: CausalPAG) -> tuple[EndpointMarkAtom, ...]:
    return tuple(
        (left, left_mark.value, right, right_mark.value)
        for left, left_mark, right, right_mark in pag.topology.edges
    )


def _directed_edges(
    edges: tuple[EndpointMarkAtom, ...],
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for left, left_mark, right, right_mark in edges:
        if (left_mark, right_mark) == ("tail", "arrow"):
            result.append((left, right))
        elif (left_mark, right_mark) == ("arrow", "tail"):
            result.append((right, left))
    return tuple(result)


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


def _arrow_at(
    edges: tuple[MixedEdgeAtom, ...],
    left: str,
    right: str,
    endpoint: str,
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
            member_marks = {
                _arrow_at(member, left, right, endpoint) for member in members
            }
            if mark == "circle":
                assert member_marks == {False, True}
            elif mark == "tail":
                assert member_marks == {False}
            else:
                assert mark == "arrow"
                assert member_marks == {True}


def _load_fixture() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "fixtures" / "papers" / FIXTURE_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_skeleton_class(
    nodes: tuple[str, ...], seed: tuple[MixedEdgeAtom, ...]
) -> tuple[tuple[MixedEdgeAtom, ...], ...]:
    positions = {node: index for index, node in enumerate(nodes)}
    pairs = tuple((left, right) for left, _, right, _ in seed)
    assert all(positions[left] < positions[right] for left, right in pairs)
    assert pairs == tuple(
        sorted(pairs, key=lambda pair: (positions[pair[0]], positions[pair[1]]))
    )
    seed_signature = m_separation_signature_by_all_conditioning_subsets(nodes, seed)
    members: list[tuple[MixedEdgeAtom, ...]] = []
    for states in itertools.product(PRESENT_NO_SELECTION_STATES, repeat=len(pairs)):
        edges = tuple(
            (left, marks[0], right, marks[1])
            for (left, right), marks in zip(pairs, states, strict=True)
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
    return tuple(members)


def _member_visibility_counts(
    nodes: tuple[str, ...],
    members: tuple[tuple[MixedEdgeAtom, ...], ...],
    source: str,
    target: str,
) -> tuple[int, int]:
    paths = tuple(
        directed_edge_visibility_witness_by_simple_paths(
            nodes,
            _endpoint_atoms_from_mixed(member),
            source,
            target,
        )
        for member in members
    )
    visible = sum(path is not None for path in paths)
    return visible, len(paths) - visible


def _assert_case(
    case: dict[str, Any],
) -> tuple[CausalPAG, DirectedEdgeVisibilityWitness | None]:
    nodes = tuple(case["nodes"])
    seed = _mixed_atoms_from_raw(case["seed_mag"])
    expected = _endpoint_atoms_from_raw(case["expected_causal_pag"])
    members = _fixed_skeleton_class(nodes, seed)
    expected_paths = tuple(tuple(path) for path in case["candidate_paths"])
    expected_path = (
        None if case["canonical_path"] is None else tuple(case["canonical_path"])
    )
    source = case["source"]
    target = case["target"]
    source_mag = _mag_from_atoms(nodes, seed)
    before = source_mag.topology

    assert len(members) == case["class_size"]
    assert full_pag_mark_intersection_by_definition(nodes, members) == expected
    _assert_endpoint_intersection_witnesses(members, expected)
    assert (
        directed_edge_visibility_paths_by_simple_paths(nodes, expected, source, target)
        == expected_paths
    )
    assert _member_visibility_counts(nodes, members, source, target) == (
        case["visible_member_count"],
        case["invisible_member_count"],
    )

    pag = CausalPAG.from_mag(source_mag)
    assert _pag_atoms(pag) == expected
    assert source_mag.topology == before
    witness = find_directed_edge_visibility_witness(pag, source, target)
    assert (None if witness is None else witness.path) == expected_path
    if witness is not None:
        assert witness.graph is pag
        assert witness.source == source
        assert witness.target == target
    return pag, witness


def test_n4_no_selection_census_intersections_and_visibility_are_exact() -> None:
    census = no_selection_census(NODES4)
    assert census.raw_count == 4_096
    assert census.ancestral_count == 2_504
    assert len(census.mags) == 2_492
    assert len(census.classes) == 248
    assert max(len(members) for members, _ in census.classes.values()) == 219
    assert len({pag for _, pag in census.classes.values()}) == 248
    assert sum(len(members) for members, _ in census.classes.values()) == 2_492
    assert all(
        arrow_left or arrow_right
        for edges in census.mags
        for _, arrow_left, _, arrow_right in edges
    )

    directed_count = 0
    visible_count = 0
    invisible_count = 0
    for members, expected_pag in census.classes.values():
        _assert_endpoint_intersection_witnesses(members, expected_pag)
        pag: CausalPAG | None = None
        for member in members:
            source_mag = _mag_from_atoms(NODES4, member)
            before = source_mag.topology
            actual_pag = CausalPAG.from_mag(source_mag)
            assert actual_pag.topology.nodes == NODES4
            assert _pag_atoms(actual_pag) == expected_pag
            assert source_mag.topology == before
            if pag is None:
                pag = actual_pag
        assert pag is not None

        for source, target in _directed_edges(expected_pag):
            directed_count += 1
            expected_path = directed_edge_visibility_witness_by_simple_paths(
                NODES4, expected_pag, source, target
            )
            member_paths = tuple(
                directed_edge_visibility_witness_by_simple_paths(
                    NODES4,
                    _endpoint_atoms_from_mixed(member),
                    source,
                    target,
                )
                for member in members
            )
            actual = find_directed_edge_visibility_witness(pag, source, target)
            assert (None if actual is None else actual.path) == expected_path
            if expected_path is None:
                invisible_count += 1
                assert any(path is None for path in member_paths)
                assert any(path is not None for path in member_paths)
            else:
                visible_count += 1
                assert all(path is not None for path in member_paths)
                assert actual is not None
                assert actual.graph is pag
                assert actual.source == source
                assert actual.target == target

    assert directed_count == 108
    assert visible_count == 60
    assert invisible_count == 48


def test_jaber_r86_figure4a_visibility_is_source_separated_from_oracle() -> None:
    fixture = _load_fixture()
    source = fixture["visibility_source"]
    assert source["definition"] == "Definition 8, journal page 1448"
    assert "non-endpoint vertex" in source["endpoint_clarification"]
    assert "does not assert that none exists" in source["erratum_check"]
    orientation = fixture["orientation_source"]
    assert orientation["adjudicated_no_selection_scope"] == (
        "R0 followed by R1-R4 and R8-R10; R5-R7 are not used in the "
        "no-selection construction."
    )

    paper = fixture["paper_fixture"]
    assert paper["figure"] == "4(a)"
    assert paper["source_reported"] == {
        "visible_directed_edge": ["B", "C"],
        "plain_directed_edge": ["D", "C"],
        "class_size_or_canonical_witness": "not reported",
    }
    adjudicated = paper["independent_adjudication"]
    for query in adjudicated["queries"]:
        case = {
            **adjudicated,
            **query,
        }
        _assert_case(case)


@pytest.mark.parametrize(
    "case_name",
    (
        "direct_declared_order_tie",
        "circle_is_not_an_arrowhead",
        "indirect_collider_parent_path",
        "shortest_path_precedes_declared_order",
        "shielded_chorded_only_witness",
    ),
)
def test_constructed_visibility_cases_are_independently_adjudicated(
    case_name: str,
) -> None:
    cases = {case["name"]: case for case in _load_fixture()["constructed_cases"]}
    case = cases[case_name]
    pag, witness = _assert_case(case)

    if case_name == "direct_declared_order_tie":
        assert witness is not None and witness.path == ("C", "A")
    if case_name == "shortest_path_precedes_declared_order":
        assert witness is not None and witness.path == ("D", "A")
    if case_name == "circle_is_not_an_arrowhead":
        assert witness is None
        assert pag.topology.edge("C", "A") == (Mark.CIRCLE, Mark.ARROW)
    if case_name == "shielded_chorded_only_witness":
        assert witness is not None
        assert witness.path == ("E", "C", "D", "A")
        chord = tuple(case["required_chord"])
        assert pag.topology.edge(*chord) is not None
        positions = {node: index for index, node in enumerate(witness.path)}
        assert abs(positions[chord[0]] - positions[chord[1]]) > 1


@pytest.mark.parametrize(
    "case",
    (
        pytest.param(
            {
                "name": "internal_collider_must_be_exact_parent",
                "nodes": ["V0", "V1", "V2", "V3", "V4", "V5"],
                "seed_mag": [
                    ["V0", "tail", "V4", "arrow"],
                    ["V0", "arrow", "V5", "arrow"],
                    ["V1", "arrow", "V2", "arrow"],
                    ["V1", "arrow", "V3", "arrow"],
                    ["V2", "tail", "V3", "arrow"],
                    ["V2", "arrow", "V4", "arrow"],
                    ["V2", "tail", "V5", "arrow"],
                    ["V3", "arrow", "V4", "arrow"],
                    ["V3", "arrow", "V5", "tail"],
                ],
                "expected_causal_pag": [
                    ["V0", "circle", "V4", "arrow"],
                    ["V0", "circle", "V5", "arrow"],
                    ["V1", "circle", "V2", "arrow"],
                    ["V1", "circle", "V3", "arrow"],
                    ["V2", "tail", "V3", "arrow"],
                    ["V2", "arrow", "V4", "arrow"],
                    ["V2", "tail", "V5", "arrow"],
                    ["V3", "arrow", "V4", "arrow"],
                    ["V3", "arrow", "V5", "tail"],
                ],
                "class_size": 9,
                "source": "V2",
                "target": "V3",
                "candidate_paths": [],
                "canonical_path": None,
                "visible_member_count": 0,
                "invisible_member_count": 9,
            },
            id="internal-exact-parent",
        ),
        pytest.param(
            {
                "name": "external_far_endpoint_tail_is_allowed",
                "nodes": ["X0", "X1", "X2", "X3", "X4"],
                "seed_mag": [
                    ["X0", "tail", "X2", "arrow"],
                    ["X1", "arrow", "X2", "arrow"],
                    ["X1", "tail", "X4", "arrow"],
                    ["X2", "tail", "X3", "arrow"],
                    ["X3", "tail", "X4", "arrow"],
                ],
                "expected_causal_pag": [
                    ["X0", "circle", "X2", "arrow"],
                    ["X1", "circle", "X2", "arrow"],
                    ["X1", "tail", "X4", "arrow"],
                    ["X2", "tail", "X3", "arrow"],
                    ["X3", "tail", "X4", "arrow"],
                ],
                "class_size": 4,
                "source": "X3",
                "target": "X4",
                "candidate_paths": [["X2", "X3"]],
                "canonical_path": ["X2", "X3"],
                "visible_member_count": 4,
                "invisible_member_count": 0,
            },
            id="external-far-tail",
        ),
        pytest.param(
            {
                "name": "shortest_path_precedes_external_declaration_order",
                "nodes": ["V0", "V1", "V2", "V3", "V4", "V5"],
                "seed_mag": [
                    ["V0", "arrow", "V1", "arrow"],
                    ["V0", "arrow", "V2", "tail"],
                    ["V0", "arrow", "V3", "tail"],
                    ["V0", "tail", "V5", "arrow"],
                    ["V1", "arrow", "V2", "tail"],
                    ["V1", "arrow", "V4", "tail"],
                    ["V1", "tail", "V5", "arrow"],
                    ["V2", "tail", "V4", "arrow"],
                    ["V2", "tail", "V5", "arrow"],
                ],
                "expected_causal_pag": [
                    ["V0", "arrow", "V1", "arrow"],
                    ["V0", "arrow", "V2", "circle"],
                    ["V0", "arrow", "V3", "circle"],
                    ["V0", "tail", "V5", "arrow"],
                    ["V1", "arrow", "V2", "circle"],
                    ["V1", "arrow", "V4", "circle"],
                    ["V1", "tail", "V5", "arrow"],
                    ["V2", "circle", "V4", "circle"],
                    ["V2", "tail", "V5", "arrow"],
                ],
                "class_size": 26,
                "source": "V1",
                "target": "V5",
                "candidate_paths": [
                    ["V4", "V1"],
                    ["V3", "V0", "V1"],
                ],
                "canonical_path": ["V4", "V1"],
                "visible_member_count": 26,
                "invisible_member_count": 0,
            },
            id="shortest-before-declared-order",
        ),
        pytest.param(
            {
                "name": "same_external_path_uses_declared_order",
                "nodes": ["V0", "V1", "V2", "V3", "V4"],
                "seed_mag": [
                    ["V0", "tail", "V1", "arrow"],
                    ["V0", "arrow", "V2", "tail"],
                    ["V0", "arrow", "V4", "arrow"],
                    ["V1", "arrow", "V3", "tail"],
                    ["V1", "arrow", "V4", "tail"],
                    ["V2", "tail", "V3", "arrow"],
                    ["V3", "arrow", "V4", "arrow"],
                ],
                "expected_causal_pag": [
                    ["V0", "tail", "V1", "arrow"],
                    ["V0", "arrow", "V2", "circle"],
                    ["V0", "arrow", "V4", "arrow"],
                    ["V1", "arrow", "V3", "tail"],
                    ["V1", "arrow", "V4", "tail"],
                    ["V2", "circle", "V3", "arrow"],
                    ["V3", "arrow", "V4", "arrow"],
                ],
                "class_size": 3,
                "source": "V4",
                "target": "V1",
                "candidate_paths": [
                    ["V2", "V0", "V4"],
                    ["V2", "V3", "V4"],
                ],
                "canonical_path": ["V2", "V0", "V4"],
                "visible_member_count": 3,
                "invisible_member_count": 0,
            },
            id="same-external-declared-tie",
        ),
        pytest.param(
            {
                "name": "shared_attachment_predecessor_uses_declared_order",
                "nodes": ["A", "B", "C", "U", "X", "Y"],
                "seed_mag": [
                    ["A", "tail", "B", "arrow"],
                    ["A", "arrow", "X", "arrow"],
                    ["A", "arrow", "Y", "arrow"],
                    ["B", "arrow", "U", "tail"],
                    ["B", "arrow", "X", "tail"],
                    ["B", "arrow", "Y", "tail"],
                    ["C", "tail", "U", "arrow"],
                    ["U", "arrow", "X", "arrow"],
                    ["U", "arrow", "Y", "arrow"],
                ],
                "expected_causal_pag": [
                    ["A", "tail", "B", "arrow"],
                    ["A", "arrow", "X", "arrow"],
                    ["A", "arrow", "Y", "arrow"],
                    ["B", "arrow", "U", "tail"],
                    ["B", "arrow", "X", "tail"],
                    ["B", "arrow", "Y", "tail"],
                    ["C", "circle", "U", "arrow"],
                    ["U", "arrow", "X", "arrow"],
                    ["U", "arrow", "Y", "arrow"],
                ],
                "class_size": 2,
                "source": "A",
                "target": "B",
                "candidate_paths": [
                    ["C", "U", "X", "A"],
                    ["C", "U", "Y", "A"],
                ],
                "canonical_path": ["C", "U", "X", "A"],
                "visible_member_count": 2,
                "invisible_member_count": 0,
            },
            id="shared-attachment-predecessor-tie",
        ),
    ),
)
def test_public_visibility_search_conditions_are_independently_adjudicated(
    case: dict[str, Any],
) -> None:
    pag, witness = _assert_case(case)
    name = case["name"]

    if name == "internal_collider_must_be_exact_parent":
        assert witness is None
        assert pag.topology.edge("V4", "V3") == (Mark.ARROW, Mark.ARROW)
        assert "V4" not in pag.topology.directed_parents("V3")
    elif name == "external_far_endpoint_tail_is_allowed":
        assert witness is not None and witness.path == ("X2", "X3")
        assert pag.topology.edge("X2", "X3") == (Mark.TAIL, Mark.ARROW)
    elif name == "shortest_path_precedes_external_declaration_order":
        assert witness is not None and witness.path == ("V4", "V1")
        assert pag.topology.nodes.index("V3") < pag.topology.nodes.index("V4")
    elif name == "same_external_path_uses_declared_order":
        assert witness is not None and witness.path == ("V2", "V0", "V4")
    else:
        assert name == "shared_attachment_predecessor_uses_declared_order"
        assert witness is not None
        assert witness.path == ("C", "U", "X", "A")


def test_literal_oracle_rejects_each_visibility_near_miss() -> None:
    cases = (
        (
            ("A", "B", "C"),
            (
                ("A", "tail", "B", "arrow"),
                ("A", "arrow", "C", "tail"),
                ("B", "arrow", "C", "tail"),
            ),
            "external vertex is adjacent to the target",
        ),
        (
            ("A", "B", "C"),
            (
                ("A", "tail", "B", "arrow"),
                ("A", "tail", "C", "arrow"),
            ),
            "candidate edge is out of the source endpoint",
        ),
        (
            ("A", "B", "C", "D"),
            (
                ("A", "tail", "B", "arrow"),
                ("A", "arrow", "D", "arrow"),
                ("B", "arrow", "D", "arrow"),
                ("C", "tail", "D", "arrow"),
            ),
            "internal collider is not an exact directed parent of target",
        ),
        (
            ("A", "B", "C", "D"),
            (
                ("A", "tail", "B", "arrow"),
                ("A", "arrow", "D", "tail"),
                ("B", "arrow", "D", "tail"),
                ("C", "tail", "D", "arrow"),
            ),
            "internal vertex is a noncollider",
        ),
    )
    for nodes, edges, reason in cases:
        assert (
            directed_edge_visibility_witness_by_simple_paths(nodes, edges, "A", "B")
            is None
        ), reason


def test_causal_pag_factory_domain_and_no_selection_boundary() -> None:
    raw = EndpointMarkedGraph(nodes=("A",))
    raw_undirected = EndpointMarkedGraph(
        nodes=("A", "B"),
        edges=(("A", Mark.TAIL, "B", Mark.TAIL),),
    )
    ancestral = AncestralGraph.from_graph(raw)
    directed = DirectedBidirectedGraph(nodes=("A",))
    dag = DAG.from_graph(directed)
    for wrong in (raw, raw_undirected, ancestral, directed, dag, object()):
        with pytest.raises(PAGDomainError) as error:
            CausalPAG.from_mag(wrong)  # type: ignore[arg-type]
        assert str(error.value) == ("causal PAG construction requires a MAG proof view")

    undirected = MaximalAncestralGraph.from_graph(raw_undirected)
    with pytest.raises(PAGDomainError) as error:
        CausalPAG.from_mag(undirected)
    assert str(error.value) == (
        "causal PAG construction requires a MAG without undirected edges"
    )

    valid = MaximalAncestralGraph.from_graph(raw)
    with pytest.raises(TypeError):
        CausalPAG.from_mag(  # type: ignore[call-arg]
            valid, background_knowledge=()
        )


def test_causal_pag_is_factory_only_sealed_frozen_and_unforgeable_normally() -> None:
    source = _mag_from_atoms(("A", "B"), (("A", False, "B", True),))
    before = source.topology
    pag = CausalPAG.from_mag(source)
    closed = ClosedPAG.from_mag(source)

    assert CausalPAG.__slots__ == ("topology",)
    assert not hasattr(CausalPAG, "from_graph")
    assert not hasattr(CausalPAG, "from_closed_pag")
    assert isinstance(pag.topology, EndpointMarkedGraph)
    assert pag.topology.nodes == source.topology.nodes
    assert pag.topology == closed.topology
    assert pag != closed
    assert closed != pag
    assert copy.copy(pag) is pag
    assert copy.deepcopy(pag) is pag
    assert hash(pag) == hash(pag)
    assert not hasattr(pag, "__dict__")
    assert source.topology == before
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        pag.topology = EndpointMarkedGraph(nodes=[])  # type: ignore[misc]
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("ForgedCausalPAG", (CausalPAG,), {})
    with pytest.raises((AttributeError, TypeError)):
        pag.__setstate__([EndpointMarkedGraph(nodes=[])])  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="pickle is unsupported"):
        pickle.dumps(pag)
    for arguments in ((), (source.topology,)):
        with pytest.raises(TypeError) as error:
            CausalPAG(*arguments)  # type: ignore[call-arg]
        assert str(error.value) == ("CausalPAG values are factory-only; use from_mag")


def test_causal_pag_preserves_order_and_visibility_ties_use_declared_order() -> None:
    case = _load_fixture()["constructed_cases"][0]
    original_nodes = tuple(case["nodes"])
    permuted_nodes = ("A", "B", "D", "C")
    seed = _mixed_atoms_from_raw(case["seed_mag"])
    original_source = _mag_from_atoms(original_nodes, seed)
    permuted_source = _mag_from_atoms(permuted_nodes, seed)

    original = CausalPAG.from_mag(original_source)
    permuted = CausalPAG.from_mag(permuted_source)
    original_witness = find_directed_edge_visibility_witness(original, "A", "B")
    permuted_witness = find_directed_edge_visibility_witness(permuted, "A", "B")

    assert original.topology.nodes == original_nodes
    assert permuted.topology.nodes == permuted_nodes
    assert _labelwise_marks(original.topology) == _labelwise_marks(permuted.topology)
    assert original_witness is not None
    assert original_witness.path == ("C", "A")
    assert permuted_witness is not None
    assert permuted_witness.path == ("D", "A")
    assert original_source.topology.nodes == original_nodes
    assert permuted_source.topology.nodes == permuted_nodes


def test_causal_pag_accepts_small_large_and_low_recursion_inputs() -> None:
    for nodes in ((), ("only",), tuple(f"V{index}" for index in range(40))):
        source = MaximalAncestralGraph.from_graph(EndpointMarkedGraph(nodes=nodes))
        pag = CausalPAG.from_mag(source)
        assert pag.topology.nodes == nodes
        assert pag.topology.edges == ()

    nodes = tuple(f"P{index}" for index in range(80))
    source = _mag_from_atoms(
        nodes,
        tuple(
            (parent, False, child, True) for parent, child in itertools.pairwise(nodes)
        ),
    )
    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(75)
        pag = CausalPAG.from_mag(source)
    finally:
        sys.setrecursionlimit(original_limit)
    assert pag.topology.nodes == nodes
    assert len(pag.topology.edges) == len(nodes) - 1
    assert all(
        left_mark is Mark.CIRCLE and right_mark is Mark.CIRCLE
        for _, left_mark, _, right_mark in pag.topology.edges
    )


def test_visibility_finds_long_public_witness_under_low_recursion_limit() -> None:
    colliders = tuple(f"C{index:03d}" for index in range(80))
    nodes = ("A", "B", "D", "E", *colliders)
    source = _mag_from_atoms(
        nodes,
        (
            ("A", False, "B", True),
            ("D", False, "A", True),
            ("D", False, "B", True),
            ("A", True, colliders[0], True),
            *(
                (left, True, right, True)
                for left, right in itertools.pairwise(colliders)
            ),
            *((collider, False, "B", True) for collider in colliders),
            ("E", False, colliders[-1], True),
        ),
    )
    before = source.topology
    pag = CausalPAG.from_mag(source)

    assert pag.topology.edge("E", "B") is None
    assert all(
        pag.topology.edge(collider, "B") == (Mark.TAIL, Mark.ARROW)
        for collider in colliders
    )

    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(75)
        witness = find_directed_edge_visibility_witness(pag, "A", "B")
    finally:
        sys.setrecursionlimit(original_limit)

    assert witness is not None
    assert witness.graph is pag
    assert witness.source == "A"
    assert witness.target == "B"
    assert witness.path == ("E", *reversed(colliders), "A")
    assert source.topology == before


def test_visibility_validation_order_and_exact_domain_errors() -> None:
    fixture = _load_fixture()["constructed_cases"][0]
    nodes = tuple(fixture["nodes"])
    source_mag = _mag_from_atoms(nodes, _mixed_atoms_from_raw(fixture["seed_mag"]))
    pag = CausalPAG.from_mag(source_mag)
    closed = ClosedPAG.from_mag(source_mag)
    raw = source_mag.topology
    ancestral = AncestralGraph.from_graph(raw)

    for wrong in (raw, ancestral, source_mag, closed, object()):
        with pytest.raises(PAGDomainError) as error:
            find_directed_edge_visibility_witness(  # type: ignore[arg-type]
                wrong, "missing", "missing"
            )
        assert str(error.value) == (
            "directed-edge visibility requires a causal PAG proof view"
        )

    for source, target, message in (
        ("missing", "B", "unknown vertex: 'missing'"),
        ("A", "missing", "unknown vertex: 'missing'"),
        (None, "B", "unknown vertex: None"),
    ):
        with pytest.raises(GraphValidationError) as error:
            find_directed_edge_visibility_witness(  # type: ignore[arg-type]
                pag, source, target
            )
        assert str(error.value) == message

    with pytest.raises(PAGDomainError) as error:
        find_directed_edge_visibility_witness(pag, "A", "A")
    assert str(error.value) == ("directed-edge visibility endpoints must be distinct")

    for source, target in (("B", "A"), ("B", "D"), ("C", "A")):
        with pytest.raises(PAGDomainError) as error:
            find_directed_edge_visibility_witness(pag, source, target)
        assert str(error.value) == (
            "directed-edge visibility requires a directed edge from source to target"
        )


def test_visibility_api_signature_and_rejects_extra_options() -> None:
    signature = inspect.signature(find_directed_edge_visibility_witness)
    assert tuple(signature.parameters) == ("pag", "source", "target")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )
    fixture = _load_fixture()["constructed_cases"][0]
    pag = CausalPAG.from_mag(
        _mag_from_atoms(
            tuple(fixture["nodes"]),
            _mixed_atoms_from_raw(fixture["seed_mag"]),
        )
    )
    with pytest.raises(TypeError):
        find_directed_edge_visibility_witness(  # type: ignore[call-arg]
            pag, "A", "B", max_length=2
        )
    with pytest.raises(TypeError):
        find_directed_edge_visibility_witness(  # type: ignore[call-arg]
            pag, "A", "B", require_uncovered=True
        )


def test_visibility_witness_is_factory_bound_frozen_and_exactly_shaped() -> None:
    case = _load_fixture()["constructed_cases"][0]
    pag, witness = _assert_case(case)
    assert isinstance(witness, DirectedEdgeVisibilityWitness)
    assert DirectedEdgeVisibilityWitness.__slots__ == (
        "graph",
        "source",
        "target",
        "path",
    )
    assert witness.graph is pag
    assert witness.source == "A"
    assert witness.target == "B"
    assert witness.path == ("C", "A")
    assert copy.copy(witness) is witness
    assert copy.deepcopy(witness) is witness
    assert hash(witness) == hash(witness)
    assert not hasattr(witness, "__dict__")
    for field, replacement in (
        ("graph", pag),
        ("source", "B"),
        ("target", "A"),
        ("path", ("D", "A")),
    ):
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(witness, field, replacement)
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type(
            "ForgedDirectedEdgeVisibilityWitness",
            (DirectedEdgeVisibilityWitness,),
            {},
        )
    with pytest.raises((AttributeError, TypeError)):
        witness.__setstate__((pag, "A", "B", ("D", "A")))  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="pickle is unsupported"):
        pickle.dumps(witness)
    with pytest.raises(TypeError):
        DirectedEdgeVisibilityWitness(  # type: ignore[call-arg]
            graph=pag,
            source="A",
            target="B",
            path=("C", "A"),
        )


def test_full_mag_listing_rejects_causal_pag_proof_view() -> None:
    source = _mag_from_atoms(("A", "B"), (("A", False, "B", True),))
    pag = CausalPAG.from_mag(source)
    with pytest.raises(MAGListingDomainError) as error:
        list_mag_equivalence_class(  # type: ignore[arg-type]
            pag, max_members=3
        )
    assert str(error.value) == "MAG listing requires a closed PAG proof view"
