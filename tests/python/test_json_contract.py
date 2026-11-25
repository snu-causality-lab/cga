from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    AncestralGraph,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    Mark,
    SchemaError,
    graph_from_json,
    graph_to_canonical_json,
)


CANONICAL_BOW = (
    '{"schema":"cga.graph/1","family":"directed-bidirected",'
    '"nodes":["B","α","A"],"directed":[["B","α"],["A","B"]],'
    '"bidirected":[["B","A"]]}'
)
CANONICAL_BOW_SHA256 = "9917995df56c1643110fdce5eaae1b991557cbaa9f817cb207b7da2b6c30ea70"


def make_bow() -> DirectedBidirectedGraph:
    return DirectedBidirectedGraph(
        nodes=["B", "α", "A"],
        directed=[("A", "B"), ("B", "α")],
        bidirected=[("A", "B")],
    )


def test_canonical_json_has_pinned_bytes_and_digest() -> None:
    encoded = graph_to_canonical_json(make_bow())
    assert encoded == CANONICAL_BOW
    assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == CANONICAL_BOW_SHA256
    assert graph_from_json(encoded) == make_bow()


def test_canonical_json_preserves_node_order_but_ignores_edge_input_order() -> None:
    reversed_input = DirectedBidirectedGraph(
        nodes=["B", "α", "A"],
        directed=[("B", "α"), ("A", "B")],
        bidirected=[("B", "A")],
    )
    assert graph_to_canonical_json(reversed_input) == CANONICAL_BOW
    reordered_nodes = DirectedBidirectedGraph(
        nodes=["A", "B", "α"],
        directed=[("A", "B"), ("B", "α")],
        bidirected=[("A", "B")],
    )
    assert graph_to_canonical_json(reordered_nodes) != CANONICAL_BOW


def test_canonical_json_is_stable_across_hash_seeds() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    source = root / "python" / "src"
    code = """
from causal_graph_algorithms import DirectedBidirectedGraph, graph_to_canonical_json
graph = DirectedBidirectedGraph(
    nodes=["B", "α", "A"],
    directed={("A", "B"), ("B", "α")},
    bidirected={("A", "B")},
)
print(graph_to_canonical_json(graph), end="")
"""
    results = []
    for seed in ("1", "8675309"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        environment["PYTHONPATH"] = str(source)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        results.append(completed.stdout)
    assert results == [CANONICAL_BOW, CANONICAL_BOW]


def test_endpoint_marked_canonical_shape_and_roundtrip() -> None:
    graph = EndpointMarkedGraph(
        nodes=["A", "B"],
        edges=[("B", Mark.ARROW, "A", Mark.CIRCLE)],
    )
    expected = (
        '{"schema":"cga.graph/1","family":"endpoint-marked",'
        '"nodes":["A","B"],"edges":['
        '{"u":"A","u_mark":"circle","v":"B","v_mark":"arrow"}]}'
    )
    assert graph_to_canonical_json(graph) == expected
    assert graph_from_json(expected) == graph


def test_canonical_escaping_and_complete_edge_sort_order() -> None:
    graph = DirectedBidirectedGraph(
        nodes=['A"x', "B\\y", "C\nz", "D"],
        directed=[
            ("D", 'A"x'),
            ('A"x', "C\nz"),
            ('A"x', "B\\y"),
        ],
        bidirected=[("C\nz", "B\\y"), ("D", 'A"x')],
    )
    expected = (
        '{"schema":"cga.graph/1","family":"directed-bidirected",'
        '"nodes":["A\\"x","B\\\\y","C\\nz","D"],'
        '"directed":[["A\\"x","B\\\\y"],["A\\"x","C\\nz"],'
        '["D","A\\"x"]],"bidirected":['
        '["A\\"x","D"],["B\\\\y","C\\nz"]]}'
    )
    assert graph_to_canonical_json(graph) == expected
    assert graph_from_json(expected) == graph


def test_endpoint_edges_sort_by_both_declared_positions() -> None:
    graph = EndpointMarkedGraph(
        nodes=["B", "A", "D", "C"],
        edges=[
            ("D", Mark.ARROW, "C", Mark.ARROW),
            ("B", Mark.TAIL, "D", Mark.CIRCLE),
            ("B", Mark.CIRCLE, "A", Mark.TAIL),
        ],
    )
    expected = (
        '{"schema":"cga.graph/1","family":"endpoint-marked",'
        '"nodes":["B","A","D","C"],"edges":['
        '{"u":"B","u_mark":"circle","v":"A","v_mark":"tail"},'
        '{"u":"B","u_mark":"tail","v":"D","v_mark":"circle"},'
        '{"u":"D","u_mark":"arrow","v":"C","v_mark":"arrow"}]}'
    )
    assert graph_to_canonical_json(graph) == expected


@pytest.mark.parametrize(
    "text",
    [
        '{"schema":"cga.graph/1","schema":"cga.graph/1","family":"directed-bidirected","nodes":[],"directed":[],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":[],"directed":[],"bidirected":[],"extra":0}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":["A","A"],"directed":[],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":["A","B"],"directed":[["A","B"],["A","B"]],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":["A","B"],"directed":[],"bidirected":[["A","B"],["B","A"]]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":["A"],"directed":[["A","A"]],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":["A"],"directed":[["A","B"]],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"endpoint-marked","nodes":["A","B"],"edges":[{"u":"A","u_mark":"none","v":"B","v_mark":"arrow"}]}',
        '{"schema":"cga.graph/1","family":"endpoint-marked","nodes":["A","B"],"edges":[{"u":"A","u_mark":"tail","v":"B","v_mark":"arrow"},{"u":"B","u_mark":"arrow","v":"A","v_mark":"tail"}]}',
        '{"schema":"cga.graph/1","family":"unknown","nodes":[]}',
        '{"schema":"cga.graph/2","family":"directed-bidirected","nodes":[],"directed":[],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":[],"directed":[]}',
        '{"schema":"cga.graph/1","family":"endpoint-marked","nodes":["A","B"],"edges":[{"u":"A","u":"B","u_mark":"tail","v":"B","v_mark":"arrow"}]}',
        'not JSON',
        '[]',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":{},"directed":[],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":[1],"directed":[],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":[],"directed":{},"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":["A","B"],"directed":[["A"]],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"endpoint-marked","nodes":["A","B"],"edges":[["A","tail","B","arrow"]]}',
        '{"schema":"cga.graph/1","family":"endpoint-marked","nodes":["A","B"],"edges":[{"u":"A","u_mark":"tail","v":"B"}]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":[],"directed":[],"bidirected":[],"unused":NaN}',
        '\ufeff{"schema":"cga.graph/1","family":"directed-bidirected","nodes":[],"directed":[],"bidirected":[]}',
        '{"schema":"cga.graph/1","family":"directed-bidirected","nodes":["\ud800"],"directed":[],"bidirected":[]}',
    ],
)
def test_json_parser_fails_closed(text: str) -> None:
    with pytest.raises(SchemaError):
        graph_from_json(text)


def test_json_parser_accepts_noncanonical_edge_order_then_canonicalizes() -> None:
    text = (
        '{"schema":"cga.graph/1","family":"directed-bidirected",'
        '"nodes":["B","α","A"],"directed":[["A","B"],["B","α"]],'
        '"bidirected":[["A","B"]]}'
    )
    assert graph_to_canonical_json(graph_from_json(text)) == CANONICAL_BOW


def test_public_type_guards_fail_closed() -> None:
    with pytest.raises(SchemaError):
        graph_from_json(b"{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        graph_to_canonical_json(object())  # type: ignore[arg-type]

    directed_bidirected = DirectedBidirectedGraph(nodes=["A"])
    endpoint_marked = EndpointMarkedGraph(nodes=["A"])
    views = (
        AcyclicDirectedBidirectedGraph.from_graph(directed_bidirected),
        DAG.from_graph(directed_bidirected),
        AncestralGraph.from_graph(endpoint_marked),
    )
    for view in views:
        with pytest.raises(TypeError):
            graph_to_canonical_json(view)  # type: ignore[arg-type]


def test_nonstandard_nonfinite_json_constants_are_rejected_at_parse_time() -> None:
    text = (
        '{"schema":"cga.graph/1","family":"directed-bidirected",'
        '"nodes":[],"directed":[],"bidirected":[],"unused":NaN}'
    )
    with pytest.raises(SchemaError, match="non-standard JSON constant"):
        graph_from_json(text)
