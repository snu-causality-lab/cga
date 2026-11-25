from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from causal_graph_algorithms import (
    AncestralGraph,
    ClosedPAG,
    EndpointMarkedGraph,
    MAGListingDomainError,
    MAGListingResourceLimitError,
    Mark,
    MaximalAncestralGraph,
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
    m_separation_signature_by_all_conditioning_subsets,
    nonmaximal_pair_by_all_conditioning_sets,
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
class _N4ListingOracleCensus:
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
def _n4_listing_oracle_census() -> _N4ListingOracleCensus:
    """Enumerate the FULL four-node universe without production traversal."""

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
    return _N4ListingOracleCensus(
        raw_count=raw_count,
        ancestral_count=ancestral_count,
        mags=normalized,
        classes=full_pag_classes_by_all_conditioning_subsets(NODES4, normalized),
    )


def _canonical_pairs(
    nodes: tuple[str, ...], pairs: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    positions = {node: index for index, node in enumerate(nodes)}
    return tuple(
        sorted(
            (
                (left, right) if positions[left] < positions[right] else (right, left)
                for left, right in pairs
            ),
            key=lambda pair: (positions[pair[0]], positions[pair[1]]),
        )
    )


def _literal_fixed_skeleton_class(
    nodes: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
    seed: tuple[MixedEdgeAtom, ...],
) -> tuple[int, tuple[tuple[MixedEdgeAtom, ...], ...]]:
    """Use only ancestral, maximal, and literal all-Z CI definitions."""

    canonical_pairs = _canonical_pairs(nodes, pairs)
    seed_signature = m_separation_signature_by_all_conditioning_subsets(nodes, seed)
    candidate_count = 0
    members: list[tuple[MixedEdgeAtom, ...]] = []
    for states in itertools.product(PRESENT_STATES, repeat=len(canonical_pairs)):
        candidate_count += 1
        edges = tuple(
            (left, marks[0], right, marks[1])
            for (left, right), marks in zip(canonical_pairs, states, strict=True)
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


def _mag_atoms(graph: MaximalAncestralGraph) -> tuple[MixedEdgeAtom, ...]:
    atoms: list[MixedEdgeAtom] = []
    for left, left_mark, right, right_mark in graph.topology.edges:
        assert left_mark is not Mark.CIRCLE
        assert right_mark is not Mark.CIRCLE
        atoms.append(
            (
                left,
                left_mark is Mark.ARROW,
                right,
                right_mark is Mark.ARROW,
            )
        )
    return tuple(atoms)


def _pag_atoms(graph: ClosedPAG) -> tuple[EndpointMarkAtom, ...]:
    return tuple(
        (left, left_mark.value, right, right_mark.value)
        for left, left_mark, right, right_mark in graph.topology.edges
    )


def _listing_key(edges: tuple[MixedEdgeAtom, ...]) -> tuple[int, ...]:
    return tuple(
        int(arrow)
        for _, arrow_left, _, arrow_right in edges
        for arrow in (arrow_left, arrow_right)
    )


def _sorted_members(
    members: tuple[tuple[MixedEdgeAtom, ...], ...],
) -> tuple[tuple[MixedEdgeAtom, ...], ...]:
    return tuple(sorted(members, key=_listing_key))


def _listed_atoms(
    listed: tuple[MaximalAncestralGraph, ...],
) -> tuple[tuple[MixedEdgeAtom, ...], ...]:
    return tuple(_mag_atoms(member) for member in listed)


def _labelwise_member(
    edges: tuple[MixedEdgeAtom, ...],
) -> frozenset[tuple[tuple[str, bool], tuple[str, bool]]]:
    return frozenset(
        tuple(sorted(((left, arrow_left), (right, arrow_right))))
        for left, arrow_left, right, arrow_right in edges
    )


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


def _load_fixture(name: str) -> dict[str, Any]:
    path = Path(__file__).parents[1] / "fixtures" / "papers" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _mixed_atoms_from_fixture(
    nodes: tuple[str, ...], raw_edges: list[list[str]]
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
    normalized.sort(key=lambda edge: (positions[edge[0]], positions[edge[2]]))
    return tuple(normalized)


def _private_rule_conclusions(
    rule_number: int,
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, Mark, str, Mark], ...],
) -> set[tuple[str, str, Mark]]:
    """Delay private imports so RED collection fails only at the public API."""

    from causal_graph_algorithms.mag_listing import (
        _ListingState,
        _r11,
        _r12,
        _r13,
        _r14,
        _r15,
        _r16,
    )

    rules = {
        11: _r11,
        12: _r12,
        13: _r13,
        14: _r14,
        15: _r15,
        16: _r16,
    }
    state = _ListingState.from_topology(
        EndpointMarkedGraph(nodes=nodes, edges=edges), separators={}
    )
    return set(rules[rule_number](state))


def test_empty_singleton_and_two_node_full_class_counts_and_order() -> None:
    cases = (
        ((), (), (), 1),
        (("only",), (), (), 1),
        (("A", "B"), (), (), 1),
        (
            ("A", "B"),
            (("A", "B"),),
            (("A", False, "B", True),),
            4,
        ),
    )
    for nodes, pairs, seed, expected_count in cases:
        _, oracle_members = _literal_fixed_skeleton_class(nodes, pairs, seed)
        source = _mag_from_atoms(nodes, seed)
        pag = ClosedPAG.from_mag(source)
        listed = list_mag_equivalence_class(pag, max_members=expected_count)

        assert len(oracle_members) == expected_count
        assert _listed_atoms(listed) == _sorted_members(oracle_members)
        assert all(member.topology.nodes == nodes for member in listed)

    adjacent = ClosedPAG.from_mag(
        _mag_from_atoms(("A", "B"), (("A", False, "B", True),))
    )
    assert _listed_atoms(list_mag_equivalence_class(adjacent, max_members=4)) == (
        (("A", False, "B", False),),
        (("A", False, "B", True),),
        (("A", True, "B", False),),
        (("A", True, "B", True),),
    )


def test_directed_chain_lists_full_eight_not_only_five_dirbi_members() -> None:
    nodes = ("A", "B", "C")
    pairs = (("A", "B"), ("B", "C"))
    seed = (("A", False, "B", True), ("B", False, "C", True))
    candidate_count, oracle_members = _literal_fixed_skeleton_class(nodes, pairs, seed)
    no_undirected_members = tuple(
        member
        for member in oracle_members
        if all(arrow_left or arrow_right for _, arrow_left, _, arrow_right in member)
    )
    source = _mag_from_atoms(nodes, seed)
    before = source.topology
    pag = ClosedPAG.from_mag(source)
    pag_before = pag.topology

    listed = list_mag_equivalence_class(pag, max_members=8)

    assert candidate_count == 4**2
    assert len(oracle_members) == 8
    assert len(no_undirected_members) == 5
    assert _listed_atoms(listed) == _sorted_members(oracle_members)
    assert source.topology == before
    assert pag.topology == pag_before


def test_resource_limit_is_atomic_and_exact_cap_succeeds() -> None:
    nodes = ("A", "B", "C")
    seed = (("A", False, "B", True), ("B", False, "C", True))
    pag = ClosedPAG.from_mag(_mag_from_atoms(nodes, seed))
    before = pag.topology

    assert len(list_mag_equivalence_class(pag, max_members=8)) == 8
    for insufficient in (0, 1, 7):
        with pytest.raises(MAGListingResourceLimitError) as error:
            list_mag_equivalence_class(pag, max_members=insufficient)
        assert str(error.value) == (
            "MAG listing exceeded max_members before class completion"
        )
        assert not hasattr(error.value, "members")
        assert pag.topology == before


def test_domain_budget_and_keyword_contracts_fail_closed() -> None:
    raw = EndpointMarkedGraph(nodes=("A",))
    ancestral = AncestralGraph.from_graph(raw)
    maximal = MaximalAncestralGraph.from_graph(raw)
    pag = ClosedPAG.from_mag(maximal)

    for wrong in (raw, ancestral, maximal, object()):
        with pytest.raises(MAGListingDomainError) as error:
            list_mag_equivalence_class(  # type: ignore[arg-type]
                wrong, max_members=1
            )
        assert str(error.value) == "MAG listing requires a closed PAG proof view"

    for invalid in (-1, True, False, 1.0, "1", None):
        with pytest.raises(MAGListingDomainError) as error:
            list_mag_equivalence_class(  # type: ignore[arg-type]
                pag, max_members=invalid
            )
        assert str(error.value) == "max_members must be a nonnegative integer"

    with pytest.raises(TypeError):
        list_mag_equivalence_class(pag, 1)  # type: ignore[misc]
    with pytest.raises(TypeError):
        list_mag_equivalence_class(
            pag,
            max_members=1,
            representative=maximal,  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        list_mag_equivalence_class(
            pag,
            max_members=1,
            background_knowledge=(),  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        list_mag_equivalence_class(
            pag,
            max_members=1,
            universe="dirbi",  # type: ignore[call-arg]
        )


def test_n4_all_251_classes_match_literal_all_z_oracle_exactly() -> None:
    census = _n4_listing_oracle_census()
    assert census.raw_count == 15_625
    assert census.ancestral_count == 3_057
    assert len(census.mags) == 3_045
    assert len(census.classes) == 251
    assert sum(len(members) for members, _ in census.classes.values()) == 3_045

    observed_pags: set[tuple[EndpointMarkAtom, ...]] = set()
    listed_total = 0
    for members, expected_pag in census.classes.values():
        representative = _mag_from_atoms(NODES4, members[0])
        pag = ClosedPAG.from_mag(representative)
        expected = _sorted_members(members)
        listed = list_mag_equivalence_class(pag, max_members=len(expected))
        actual = _listed_atoms(listed)

        assert _pag_atoms(pag) == expected_pag
        assert actual == expected
        assert len(actual) == len(set(actual))
        assert tuple(sorted(actual, key=_listing_key)) == actual
        assert all(member.topology.nodes == NODES4 for member in listed)
        observed_pags.add(_pag_atoms(pag))
        listed_total += len(listed)

    assert len(observed_pags) == 251
    assert listed_total == 3_045


def test_constructive_member_is_deterministic_and_in_every_n4_class() -> None:
    from causal_graph_algorithms.mag_listing import _constructive_mag

    census = _n4_listing_oracle_census()
    for members, expected_pag in census.classes.values():
        pag = ClosedPAG.from_mag(_mag_from_atoms(NODES4, members[0]))
        first = _constructive_mag(pag)
        second = _constructive_mag(pag)

        assert isinstance(first, MaximalAncestralGraph)
        assert first.topology.nodes == NODES4
        assert first.topology == second.topology
        assert _pag_atoms(pag) == expected_pag
        assert _mag_atoms(first) in set(members)


def test_equivalent_seeds_cannot_supply_hidden_listing_evidence() -> None:
    nodes = ("A", "B", "C")
    pairs = (("A", "B"), ("B", "C"))
    directed_chain = (
        ("A", False, "B", True),
        ("B", False, "C", True),
    )
    _, members = _literal_fixed_skeleton_class(nodes, pairs, directed_chain)
    alternative = next(member for member in members if member != directed_chain)
    first_source = _mag_from_atoms(nodes, directed_chain)
    second_source = _mag_from_atoms(nodes, alternative)
    first_before = first_source.topology
    second_before = second_source.topology
    first_pag = ClosedPAG.from_mag(first_source)
    second_pag = ClosedPAG.from_mag(second_source)

    first_listing = list_mag_equivalence_class(first_pag, max_members=8)
    second_listing = list_mag_equivalence_class(second_pag, max_members=8)

    assert first_pag.topology == second_pag.topology
    assert _listed_atoms(first_listing) == _listed_atoms(second_listing)
    assert first_source.topology == first_before
    assert second_source.topology == second_before
    assert first_pag.__slots__ == ("topology",)


def test_node_permutation_preserves_labelwise_class_and_resorts_endpoints() -> None:
    original_nodes = ("A", "B", "C")
    permuted_nodes = ("C", "A", "B")
    pairs = (("A", "B"), ("B", "C"))
    original_seed = (
        ("A", False, "B", True),
        ("B", False, "C", True),
    )
    permuted_seed = tuple(
        sorted(
            original_seed,
            key=lambda edge: (
                permuted_nodes.index(edge[0]),
                permuted_nodes.index(edge[2]),
            ),
        )
    )
    _, original_oracle = _literal_fixed_skeleton_class(
        original_nodes, pairs, original_seed
    )
    _, permuted_oracle = _literal_fixed_skeleton_class(
        permuted_nodes, pairs, permuted_seed
    )
    original_pag = ClosedPAG.from_mag(_mag_from_atoms(original_nodes, original_seed))
    permuted_pag = ClosedPAG.from_mag(_mag_from_atoms(permuted_nodes, permuted_seed))

    original = list_mag_equivalence_class(original_pag, max_members=8)
    permuted = list_mag_equivalence_class(permuted_pag, max_members=8)
    original_atoms = _listed_atoms(original)
    permuted_atoms = _listed_atoms(permuted)

    assert original_atoms == _sorted_members(original_oracle)
    assert permuted_atoms == _sorted_members(permuted_oracle)
    assert all(member.topology.nodes == original_nodes for member in original)
    assert all(member.topology.nodes == permuted_nodes for member in permuted)
    assert {_labelwise_member(member) for member in original_atoms} == {
        _labelwise_member(member) for member in permuted_atoms
    }


def test_wang_du_zhou_figure1_lists_30_and_adjudicates_bk_implication() -> None:
    fixture = _load_fixture("wang_du_zhou_2025_figure1.json")
    assert fixture["figure"] == "1(b)-1(d)"
    assert fixture["pdf_page"] == 1
    assert fixture["reported_class_size_status"] == (
        "The paper does not report the equivalence-class size or the number "
        "of members satisfying this singleton background knowledge."
    )
    nodes = tuple(fixture["nodes"])
    seed = _mixed_atoms_from_fixture(nodes, fixture["seed_mag"])
    pairs = tuple((left, right) for left, _, right, _ in seed)
    candidate_count, oracle_members = _literal_fixed_skeleton_class(nodes, pairs, seed)
    expected_pag = tuple(tuple(edge) for edge in fixture["expected_full_pag"])
    bk_members = tuple(
        member for member in oracle_members if _arrow_at(member, "A", "B", "B")
    )
    pag = ClosedPAG.from_mag(_mag_from_atoms(nodes, seed))
    listed = list_mag_equivalence_class(pag, max_members=30)

    assert candidate_count == fixture["adjudicated_fixed_skeleton_candidates"] == 1_024
    assert len(oracle_members) == fixture["adjudicated_class_size"] == 30
    assert (
        full_pag_mark_intersection_by_definition(nodes, oracle_members) == expected_pag
    )
    assert _pag_atoms(pag) == expected_pag
    assert _listed_atoms(listed) == _sorted_members(oracle_members)
    assert len(bk_members) == fixture["adjudicated_bk_member_count"] == 10
    assert all(
        not _arrow_at(member, "B", "C", "B") and _arrow_at(member, "B", "C", "C")
        for member in bk_members
    )


def test_listing_accepts_more_than_sixteen_vertices_without_recursion() -> None:
    nodes = tuple(f"C{index}" for index in range(80))
    cycle = EndpointMarkedGraph(
        nodes=nodes,
        edges=(
            (
                nodes[index],
                Mark.TAIL,
                nodes[(index + 1) % len(nodes)],
                Mark.TAIL,
            )
            for index in range(len(nodes))
        ),
    )
    source = MaximalAncestralGraph.from_graph(cycle)
    pag = ClosedPAG.from_mag(source)
    before = pag.topology
    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(75)
        listed = list_mag_equivalence_class(pag, max_members=1)
    finally:
        sys.setrecursionlimit(original_limit)

    assert len(listed) == 1
    assert listed[0].topology == cycle
    assert listed[0].topology.nodes == nodes
    assert pag.topology == before

    # The undirected cycle above has no variant endpoints.  Exercise the
    # actual nested listing search as well: this directed path closes to a
    # circle-rich PAG, and recursive endpoint-by-endpoint traversal would
    # exceed the deliberately low recursion limit before a second member.
    path_nodes = tuple(f"P{index}" for index in range(60))
    path = EndpointMarkedGraph(
        nodes=path_nodes,
        edges=(
            (
                path_nodes[index],
                Mark.TAIL,
                path_nodes[index + 1],
                Mark.ARROW,
            )
            for index in range(len(path_nodes) - 1)
        ),
    )
    path_pag = ClosedPAG.from_mag(MaximalAncestralGraph.from_graph(path))
    path_before = path_pag.topology
    assert sum(
        mark is Mark.CIRCLE
        for _, left_mark, _, right_mark in path_pag.topology.edges
        for mark in (left_mark, right_mark)
    ) == 2 * (len(path_nodes) - 1)

    recursion_limit = 100
    assert 2 * (len(path_nodes) - 1) > recursion_limit
    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(recursion_limit)
        with pytest.raises(MAGListingResourceLimitError):
            list_mag_equivalence_class(path_pag, max_members=1)
    finally:
        sys.setrecursionlimit(original_limit)

    assert path_pag.topology == path_before


def test_r11_literal_premise_and_shielded_near_miss() -> None:
    nodes = ("A", "B", "K", "C")
    firing = (
        ("A", Mark.CIRCLE, "B", Mark.CIRCLE),
        ("B", Mark.CIRCLE, "K", Mark.CIRCLE),
        ("K", Mark.TAIL, "C", Mark.ARROW),
        ("C", Mark.TAIL, "A", Mark.ARROW),
        # Shield C -> A o-* B without creating the R2 chain B -> C -> A.
        ("C", Mark.TAIL, "B", Mark.ARROW),
    )
    shielded = (*firing, ("A", Mark.CIRCLE, "K", Mark.CIRCLE))

    assert ("A", "B", Mark.ARROW) in _private_rule_conclusions(11, nodes, firing)
    assert ("A", "B", Mark.ARROW) not in _private_rule_conclusions(11, nodes, shielded)


def test_r12_literal_unbridged_premise_and_endpoint_near_miss() -> None:
    nodes = ("A", "B", "K1", "K2", "C1", "C2")
    firing = (
        ("A", Mark.CIRCLE, "B", Mark.CIRCLE),
        ("B", Mark.CIRCLE, "K1", Mark.CIRCLE),
        ("K1", Mark.CIRCLE, "K2", Mark.CIRCLE),
        ("K1", Mark.CIRCLE, "C1", Mark.ARROW),
        ("K2", Mark.CIRCLE, "C2", Mark.ARROW),
        ("C1", Mark.TAIL, "A", Mark.ARROW),
        ("C2", Mark.TAIL, "A", Mark.ARROW),
        # Shield both C_i -> A o-* B triples without R2 chains B -> C_i -> A.
        ("C1", Mark.TAIL, "B", Mark.ARROW),
        ("C2", Mark.TAIL, "B", Mark.ARROW),
        # Definition 1 permits this sole nonconsecutive anchor adjacency.
        ("C1", Mark.ARROW, "C2", Mark.ARROW),
    )
    near_miss = tuple(
        (left, left_mark, right, Mark.TAIL)
        if (left, right) == ("K2", "C2")
        else (left, left_mark, right, right_mark)
        for left, left_mark, right, right_mark in firing
    )

    assert ("A", "B", Mark.ARROW) in _private_rule_conclusions(12, nodes, firing)
    assert ("A", "B", Mark.ARROW) not in _private_rule_conclusions(12, nodes, near_miss)


def test_unbridged_missing_anchor_attachment_skips_path_search() -> None:
    from causal_graph_algorithms.mag_listing import (
        _ListingState,
        _has_unbridged_path,
    )

    class _CountingListingState(_ListingState):
        ordered_neighbor_calls: int

        def ordered_neighbors(self, node: str) -> tuple[str, ...]:
            self.ordered_neighbor_calls += 1
            return super().ordered_neighbors(node)

    layers = tuple(
        tuple(f"L{layer}_{side}" for side in range(2))
        for layer in range(12)
    )
    body = tuple(node for layer in layers for node in layer)
    nodes = ("A", "Z", *body)
    edges: list[tuple[str, Mark, str, Mark]] = [
        ("A", Mark.ARROW, node, Mark.CIRCLE) for node in layers[0]
    ]
    for current, following in itertools.pairwise(layers):
        edges.extend(
            (left, Mark.CIRCLE, right, Mark.CIRCLE)
            for left in current
            for right in following
        )

    state = _CountingListingState.from_topology(
        EndpointMarkedGraph(nodes=nodes, edges=edges), separators={}
    )
    state.ordered_neighbor_calls = 0

    assert not _has_unbridged_path(
        state,
        anchors=("A", "Z"),
        eligible_body=body,
    )
    # Z has no eligible endpoint, so the cheap attachment precheck must avoid
    # path traversal.  This is not a global complexity bound on exact search.
    assert state.ordered_neighbor_calls == 0


def test_unbridged_transient_state_rejects_a_non_circle_chord() -> None:
    from causal_graph_algorithms.mag_listing import (
        _ListingState,
        _has_unbridged_path,
    )

    # A shortest path in the o-o subgraph is not necessarily minimal in the
    # full transient PMG.  This private-state regression deliberately does not
    # assume that every intermediate rule/branching state already has the
    # Definition-4 compatibility properties of a completed local transform.
    state = _ListingState.from_topology(
        EndpointMarkedGraph(
            nodes=("A", "Z", "s", "u", "t"),
            edges=(
                ("A", Mark.ARROW, "s", Mark.CIRCLE),
                ("s", Mark.CIRCLE, "u", Mark.CIRCLE),
                ("u", Mark.CIRCLE, "t", Mark.CIRCLE),
                ("t", Mark.CIRCLE, "Z", Mark.ARROW),
                ("s", Mark.TAIL, "t", Mark.CIRCLE),
            ),
        ),
        separators={},
    )

    assert not _has_unbridged_path(
        state,
        anchors=("A", "Z"),
        eligible_body=("s", "u", "t"),
    )


def test_r13_literal_premise_and_adjacent_d_b_near_miss() -> None:
    nodes = ("A", "B", "C", "D")
    firing = (
        ("A", Mark.CIRCLE, "B", Mark.ARROW),
        ("A", Mark.CIRCLE, "C", Mark.CIRCLE),
        ("A", Mark.CIRCLE, "D", Mark.CIRCLE),
        ("C", Mark.TAIL, "B", Mark.ARROW),
        ("D", Mark.TAIL, "C", Mark.ARROW),
    )
    adjacent_d_b = (*firing, ("D", Mark.TAIL, "B", Mark.ARROW))

    assert _private_rule_conclusions(13, nodes, firing) == {("A", "B", Mark.TAIL)}
    assert ("A", "B", Mark.TAIL) not in _private_rule_conclusions(
        13, nodes, adjacent_d_b
    )


def test_r14_nonadjacent_prior_vertices_and_near_miss() -> None:
    nodes = ("X", "B", "T1", "T2")
    firing = (
        ("X", Mark.CIRCLE, "B", Mark.ARROW),
        # T1 -o X and T2 -o X deliberately are not possible-directed
        # first steps from X.  Definition 3(1) still makes each Ti prior
        # to B directly through Ti -> B.
        ("X", Mark.CIRCLE, "T1", Mark.TAIL),
        ("X", Mark.CIRCLE, "T2", Mark.TAIL),
        ("T1", Mark.TAIL, "B", Mark.ARROW),
        ("T2", Mark.TAIL, "B", Mark.ARROW),
    )
    adjacent_tails = (*firing, ("T1", Mark.TAIL, "T2", Mark.TAIL))

    assert _private_rule_conclusions(14, nodes, firing) == {("X", "B", Mark.TAIL)}
    assert _private_rule_conclusions(14, nodes, adjacent_tails) == set()


def test_r14_unbridged_path_alternative_and_broken_endpoint_near_miss() -> None:
    nodes = ("X", "B", "T1", "T2", "F1", "F2")
    firing = (
        ("X", Mark.CIRCLE, "B", Mark.ARROW),
        ("X", Mark.CIRCLE, "T1", Mark.TAIL),
        ("X", Mark.CIRCLE, "T2", Mark.TAIL),
        ("X", Mark.CIRCLE, "F1", Mark.CIRCLE),
        ("X", Mark.CIRCLE, "F2", Mark.CIRCLE),
        ("T1", Mark.TAIL, "B", Mark.ARROW),
        ("T2", Mark.TAIL, "B", Mark.ARROW),
        ("T1", Mark.ARROW, "T2", Mark.ARROW),
        ("F1", Mark.CIRCLE, "F2", Mark.CIRCLE),
        ("F1", Mark.CIRCLE, "T1", Mark.ARROW),
        ("F2", Mark.CIRCLE, "T2", Mark.ARROW),
    )
    near_miss = tuple(
        (left, left_mark, right, Mark.TAIL)
        if (left, right) == ("F2", "T2")
        else (left, left_mark, right, right_mark)
        for left, left_mark, right, right_mark in firing
    )

    assert _private_rule_conclusions(14, nodes, firing) == {("X", "B", Mark.TAIL)}
    assert _private_rule_conclusions(14, nodes, near_miss) == set()


def test_r15_literal_premise_and_missing_tail_near_miss() -> None:
    nodes = ("A", "B", "R")
    firing = (
        ("A", Mark.TAIL, "B", Mark.TAIL),
        ("B", Mark.TAIL, "R", Mark.ARROW),
        ("A", Mark.CIRCLE, "R", Mark.ARROW),
    )
    near_miss = tuple(
        (left, Mark.CIRCLE, right, right_mark)
        if (left, right) == ("A", "B")
        else (left, left_mark, right, right_mark)
        for left, left_mark, right, right_mark in firing
    )

    assert _private_rule_conclusions(15, nodes, firing) == {("A", "R", Mark.TAIL)}
    assert _private_rule_conclusions(15, nodes, near_miss) == set()


def test_r16_both_undirected_endpoint_cases_and_no_arrowhead_near_miss() -> None:
    nodes = ("A", "B", "R")
    tail_circle = (
        ("A", Mark.TAIL, "B", Mark.ARROW),
        ("B", Mark.TAIL, "R", Mark.CIRCLE),
    )
    no_arrowhead = tuple(
        (left, Mark.CIRCLE, right, Mark.CIRCLE)
        if (left, right) == ("A", "B")
        else (left, left_mark, right, right_mark)
        for left, left_mark, right, right_mark in tail_circle
    )
    circle_tail = (
        ("A", Mark.TAIL, "B", Mark.ARROW),
        ("B", Mark.CIRCLE, "R", Mark.TAIL),
        # Shield A *-> B o-* R and preserve Definition 4 balance.
        ("A", Mark.TAIL, "R", Mark.ARROW),
    )
    no_tail = tuple(
        (left, left_mark, right, Mark.CIRCLE)
        if (left, right) == ("B", "R")
        else (left, left_mark, right, right_mark)
        for left, left_mark, right, right_mark in circle_tail
    )

    assert _private_rule_conclusions(16, nodes, tail_circle) == {
        ("R", "B", Mark.ARROW)
    }
    assert _private_rule_conclusions(16, nodes, no_arrowhead) == set()
    assert _private_rule_conclusions(16, nodes, circle_tail) == {
        ("B", "R", Mark.ARROW)
    }
    assert _private_rule_conclusions(16, nodes, no_tail) == set()
