from __future__ import annotations

from collections.abc import Iterator

import pytest

from causal_graph_algorithms import (
    MAGListingDomainError,
    MAGListingResourceLimitError,
    Mark,
    MaximalAncestralGraph,
    enumerate_mags,
)


@pytest.mark.parametrize("n", [0, 1])
def test_empty_and_singleton_universes_have_one_mag(n: int) -> None:
    result = enumerate_mags(n)
    assert isinstance(result, Iterator)
    members = list(result)
    assert len(members) == 1
    assert isinstance(members[0], MaximalAncestralGraph)
    assert members[0].topology.nodes == (() if n == 0 else ("V0",))
    assert members[0].topology.edges == ()


@pytest.mark.parametrize("include_undirected, expected", [(True, 5), (False, 4)])
def test_two_node_labelled_universe_keeps_distinct_markov_equivalent_graphs(
    include_undirected: bool, expected: int
) -> None:
    members = list(
        enumerate_mags(2, up_to_isomorphism=False, include_undirected=include_undirected)
    )
    assert len(members) == expected
    assert len({member.topology for member in members}) == expected
    assert [member.topology.edge("V0", "V1") for member in members] == [
        None,
        (Mark.TAIL, Mark.ARROW),
        (Mark.ARROW, Mark.TAIL),
        (Mark.ARROW, Mark.ARROW),
        *([(Mark.TAIL, Mark.TAIL)] if include_undirected else []),
    ]


@pytest.mark.parametrize("include_undirected, expected", [(True, 4), (False, 3)])
def test_isomorphism_fixes_each_pinned_label_individually(
    include_undirected: bool, expected: int
) -> None:
    assert len(list(enumerate_mags(2, include_undirected=include_undirected))) == expected
    for pinned in (("X",), ("X", "Y")):
        members = list(enumerate_mags(2, pinned=pinned, include_undirected=include_undirected))
        assert len(members) == expected + 1
        assert all(member.topology.nodes[: len(pinned)] == pinned for member in members)


def test_pinned_order_collisions_and_snapshot_are_deterministic() -> None:
    pinned = ["V0", "X", "V2"]
    result = enumerate_mags(5, pinned=pinned, max_output=1)
    pinned[:] = ["changed"]
    member = next(result)
    assert member.topology.nodes == ("V0", "X", "V2", "V1", "V3")
    assert member.topology.edges == ()
    with pytest.raises(MAGListingResourceLimitError):
        next(result)


@pytest.mark.parametrize("n", [-1, True, False, 1.0, "2", None])
def test_invalid_node_count_is_rejected_at_call_time(n: object) -> None:
    with pytest.raises(MAGListingDomainError):
        enumerate_mags(n)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "pinned",
    [
        "X", b"X", None, 3, {"X"}, {"X": 1}, ("",), (None,), (1,),
        ("X", "X"), ("X", "Y", "Z"),
    ],
)
def test_invalid_pinned_nodes_are_rejected_at_call_time(pinned: object) -> None:
    with pytest.raises(MAGListingDomainError):
        enumerate_mags(2, pinned=pinned)  # type: ignore[arg-type]


@pytest.mark.parametrize("keyword", ["up_to_isomorphism", "include_undirected"])
@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_boolean_options_are_strict(keyword: str, value: object) -> None:
    with pytest.raises(MAGListingDomainError):
        enumerate_mags(2, **{keyword: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, True, False, 1.0, "2"])
def test_invalid_output_budget_is_rejected_at_call_time(value: object) -> None:
    with pytest.raises(MAGListingDomainError):
        enumerate_mags(2, max_output=value)  # type: ignore[arg-type]


def test_output_limit_raises_only_when_an_extra_valid_output_is_discovered() -> None:
    unbounded = list(enumerate_mags(2))
    assert list(enumerate_mags(2, max_output=len(unbounded))) == unbounded
    for cap in range(len(unbounded)):
        result = enumerate_mags(2, max_output=cap)
        assert [next(result) for _ in range(cap)] == unbounded[:cap]
        with pytest.raises(MAGListingResourceLimitError, match="max_output"):
            next(result)
        with pytest.raises(StopIteration):
            next(result)
    assert len(list(enumerate_mags(0, max_output=1))) == 1
    with pytest.raises(MAGListingResourceLimitError):
        next(enumerate_mags(0, max_output=0))


def test_large_symmetry_group_can_stream_a_short_prefix() -> None:
    # Twelve interchangeable vertices have 12! permutations. A short stream
    # must not construct that group before yielding its first graphs.
    result = enumerate_mags(12, max_output=2)
    assert next(result).topology.edges == ()
    assert len(next(result).topology.edges) == 1
    with pytest.raises(MAGListingResourceLimitError):
        next(result)


def test_repeated_calls_and_interleaved_iterators_have_the_same_order() -> None:
    expected = list(enumerate_mags(3, pinned=("X",)))
    first = enumerate_mags(3, pinned=("X",))
    second = enumerate_mags(3, pinned=("X",))
    actual = []
    for member in second:
        assert next(first) == member
        actual.append(member)
    assert actual == expected
    with pytest.raises(StopIteration):
        next(first)
