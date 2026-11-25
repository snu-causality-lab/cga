from __future__ import annotations

import itertools

import pytest

from causal_graph_algorithms import (
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    LossyConversionError,
    Mark,
    directed_bidirected_to_endpoint_marked,
    endpoint_marked_to_directed_bidirected,
)


def test_all_two_node_directed_bidirected_states_convert_only_when_lossless() -> None:
    edge_options = (("A", "B"), ("B", "A"), ("A", "B"))
    for forward, backward, bidirected in itertools.product((False, True), repeat=3):
        graph = DirectedBidirectedGraph(
            nodes=["A", "B"],
            directed=[
                edge
                for present, edge in zip(
                    (forward, backward), edge_options[:2], strict=True
                )
                if present
            ],
            bidirected=[edge_options[2]] if bidirected else [],
        )
        edge_count = int(forward) + int(backward) + int(bidirected)
        if edge_count <= 1:
            endpoint = directed_bidirected_to_endpoint_marked(graph)
            assert endpoint_marked_to_directed_bidirected(endpoint) == graph
        else:
            with pytest.raises(LossyConversionError):
                directed_bidirected_to_endpoint_marked(graph)


def test_all_endpoint_mark_pairs_convert_only_when_expressible() -> None:
    convertible = {
        (Mark.TAIL, Mark.ARROW),
        (Mark.ARROW, Mark.TAIL),
        (Mark.ARROW, Mark.ARROW),
    }
    for left_mark, right_mark in itertools.product(Mark, repeat=2):
        graph = EndpointMarkedGraph(
            nodes=["A", "B"],
            edges=[("A", left_mark, "B", right_mark)],
        )
        if (left_mark, right_mark) in convertible:
            directed_bidirected = endpoint_marked_to_directed_bidirected(graph)
            assert directed_bidirected_to_endpoint_marked(directed_bidirected) == graph
        else:
            with pytest.raises(LossyConversionError):
                endpoint_marked_to_directed_bidirected(graph)


def test_empty_graph_conversion_is_explicit_and_lossless() -> None:
    directed_bidirected = DirectedBidirectedGraph(nodes=["A", "B"])
    endpoint_marked = directed_bidirected_to_endpoint_marked(directed_bidirected)
    assert isinstance(endpoint_marked, EndpointMarkedGraph)
    assert endpoint_marked_to_directed_bidirected(endpoint_marked) == directed_bidirected


def test_conversion_preserves_declared_node_order() -> None:
    graph = DirectedBidirectedGraph(
        nodes=["B", "α", "A"],
        directed=[("A", "B")],
        bidirected=[("B", "α")],
    )
    endpoint = directed_bidirected_to_endpoint_marked(graph)
    assert endpoint.nodes == ("B", "α", "A")
    restored = endpoint_marked_to_directed_bidirected(endpoint)
    assert restored.nodes == ("B", "α", "A")
    assert restored == graph


def test_conversion_type_guards_reject_the_other_family() -> None:
    directed_bidirected = DirectedBidirectedGraph(nodes=[])
    endpoint_marked = EndpointMarkedGraph(nodes=[])
    with pytest.raises(TypeError):
        directed_bidirected_to_endpoint_marked(endpoint_marked)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        endpoint_marked_to_directed_bidirected(directed_bidirected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("directed", "bidirected", "expected_marks"),
    [
        ([("A", "B")], [], (Mark.TAIL, Mark.ARROW)),
        ([("B", "A")], [], (Mark.ARROW, Mark.TAIL)),
        ([], [("A", "B")], (Mark.ARROW, Mark.ARROW)),
    ],
)
def test_conversion_images_are_semantically_pinned(
    directed: list[tuple[str, str]],
    bidirected: list[tuple[str, str]],
    expected_marks: tuple[Mark, Mark],
) -> None:
    graph = DirectedBidirectedGraph(
        nodes=["A", "B"], directed=directed, bidirected=bidirected
    )
    endpoint = directed_bidirected_to_endpoint_marked(graph)
    assert endpoint.edges == (
        ("A", expected_marks[0], "B", expected_marks[1]),
    )

    direct_endpoint = EndpointMarkedGraph(
        nodes=["A", "B"],
        edges=[("A", expected_marks[0], "B", expected_marks[1])],
    )
    converted = endpoint_marked_to_directed_bidirected(direct_endpoint)
    assert converted.directed == frozenset(directed)
    assert converted.bidirected == frozenset({("A", "B")} if bidirected else set())
