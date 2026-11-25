"""Independent finite-universe checks for unrestricted MAG enumeration."""

from __future__ import annotations

import itertools
import os
import random

import pytest

import causal_graph_algorithms as cga

from .mag_enumeration_oracles import (
    MARKS,
    canonical_word,
    is_mag_by_definition,
    naive_mag_words,
    orbit_keys,
    pairs,
)
from .oracles import (
    endpoint_graph_is_ancestral_by_definition,
    nonmaximal_pair_by_all_conditioning_sets,
)


def _word(graph: cga.MaximalAncestralGraph) -> tuple[int, ...]:
    topology = graph.topology
    positions = {node: index for index, node in enumerate(topology.nodes)}
    pair_index = {pair: index for index, pair in enumerate(pairs(len(topology.nodes)))}
    word = [0] * len(pair_index)
    codes = {(False, False): 1, (False, True): 2, (True, False): 3, (True, True): 4}
    for left, left_mark, right, right_mark in topology.edges:
        word[pair_index[(positions[left], positions[right])]] = codes[
            left_mark == cga.Mark.ARROW, right_mark == cga.Mark.ARROW
        ]
    return tuple(word)


def test_new_leaf_oracle_agrees_with_existing_literal_oracles_through_n4() -> None:
    # This checks the fast test-only leaf evaluator against independently written
    # topological-order and m-separation routines, not production validators.
    for n in range(5):
        nodes = tuple(str(node) for node in range(n))
        for word in itertools.product(range(5), repeat=len(pairs(n))):
            atoms = tuple(
                (nodes[left], MARKS[state][0], nodes[right], MARKS[state][1])
                for (left, right), state in zip(pairs(n), word, strict=True)
                if state
            )
            expected = endpoint_graph_is_ancestral_by_definition(nodes, atoms) and (
                nonmaximal_pair_by_all_conditioning_sets(nodes, atoms) is None
            )
            assert is_mag_by_definition(n, word) is expected


def test_new_leaf_oracle_agrees_with_literal_oracles_on_seeded_n5_n6_words() -> None:
    rng = random.Random(20260923)
    for n in (5, 6):
        nodes = tuple(str(node) for node in range(n))
        for states in ((0, 1, 2, 3, 4), (0, 2, 3, 4)):
            for _ in range(256):
                word = tuple(rng.choice(states) for _ in pairs(n))
                atoms = tuple(
                    (nodes[left], MARKS[state][0], nodes[right], MARKS[state][1])
                    for (left, right), state in zip(pairs(n), word, strict=True)
                    if state
                )
                expected = endpoint_graph_is_ancestral_by_definition(nodes, atoms) and (
                    nonmaximal_pair_by_all_conditioning_sets(nodes, atoms) is None
                )
                assert is_mag_by_definition(n, word) is expected


def test_naive_census_records_labeled_graphs_not_markov_equivalence_classes() -> None:
    # Independently derived here from all 4^(n choose 2) / 5^(n choose 2)
    # assignments. The second count quotients vertex names, NOT independences.
    for include_undirected, expected in (
        (False, ((1, 1), (1, 1), (4, 3), (56, 14), (2492, 142))),
        (True, ((1, 1), (1, 1), (5, 4), (72, 19), (3045, 183))),
    ):
        actual = tuple(
            (
                len(naive_mag_words(n, include_undirected)),
                len(orbit_keys(naive_mag_words(n, include_undirected), n)),
            )
            for n in range(5)
        )
        assert actual == expected


def test_ancestral_nonmaximal_four_vertex_witness_is_excluded() -> None:
    # X <-> A <-> B <-> Y, B -> X, A -> Y. Every conditioning set
    # leaves X and Y connected despite their missing edge.
    word = (4, 3, 0, 4, 2, 4)
    nodes = ("X", "A", "B", "Y")
    atoms = tuple(
        (nodes[left], MARKS[state][0], nodes[right], MARKS[state][1])
        for (left, right), state in zip(pairs(4), word, strict=True)
        if state
    )
    assert endpoint_graph_is_ancestral_by_definition(nodes, atoms)
    assert nonmaximal_pair_by_all_conditioning_sets(nodes, atoms) == ("X", "Y")
    assert not is_mag_by_definition(4, word)


@pytest.mark.parametrize("n", range(5))
@pytest.mark.parametrize("include_undirected", (False, True))
def test_labeled_enumeration_equals_cartesian_definition_universe(
    n: int, include_undirected: bool
) -> None:
    graphs = list(
        cga.enumerate_mags(
            n, up_to_isomorphism=False, include_undirected=include_undirected
        )
    )
    words = [_word(graph) for graph in graphs]
    assert len(words) == len(set(words))
    assert set(words) == naive_mag_words(n, include_undirected)


@pytest.mark.parametrize("n", range(5))
@pytest.mark.parametrize("include_undirected", (False, True))
def test_every_pinned_subgroup_has_exactly_one_representative_per_orbit(
    n: int, include_undirected: bool
) -> None:
    expected_words = naive_mag_words(n, include_undirected)
    for pinned_count in range(n + 1):
        pinned = tuple(f"P{index}" for index in range(pinned_count))
        graphs = list(
            cga.enumerate_mags(n, pinned=pinned, include_undirected=include_undirected)
        )
        actual = [canonical_word(_word(graph), n, pinned_count) for graph in graphs]
        assert len(actual) == len(set(actual))
        assert set(actual) == orbit_keys(expected_words, n, pinned_count)


@pytest.mark.parametrize("include_undirected", (False, True))
@pytest.mark.parametrize("pinned", ((), ("X",), ("X", "Y")))
def test_n6_bounded_prefix_is_legal_and_orbit_unique_not_a_completeness_claim(
    include_undirected: bool, pinned: tuple[str, ...]
) -> None:
    stream = cga.enumerate_mags(
        6, pinned=pinned, include_undirected=include_undirected, max_output=32
    )
    words = [_word(graph) for graph in itertools.islice(stream, 32)]
    with pytest.raises(cga.MAGListingResourceLimitError):
        next(stream)
    assert len(words) == 32
    assert all(is_mag_by_definition(6, word) for word in words)
    assert len({canonical_word(word, 6, len(pinned)) for word in words}) == 32


@pytest.mark.skipif(
    os.environ.get("CGA_EXHAUSTIVE_MAG_N5") != "1",
    reason="complete 5^10/4^10 leaf-oracle replay: set CGA_EXHAUSTIVE_MAG_N5=1",
)
@pytest.mark.parametrize("include_undirected", (False, True))
def test_n5_complete_definition_universe_and_pinned_orbits(
    include_undirected: bool,
) -> None:
    expected_words = naive_mag_words(5, include_undirected)
    labeled = [
        _word(graph)
        for graph in cga.enumerate_mags(
            5, up_to_isomorphism=False, include_undirected=include_undirected
        )
    ]
    assert len(labeled) == len(set(labeled))
    assert set(labeled) == expected_words
    for pinned in ((), ("X",), ("X", "Y")):
        actual = [
            canonical_word(_word(graph), 5, len(pinned))
            for graph in cga.enumerate_mags(
                5, pinned=pinned, include_undirected=include_undirected
            )
        ]
        assert len(actual) == len(set(actual))
        assert set(actual) == orbit_keys(expected_words, 5, len(pinned))
