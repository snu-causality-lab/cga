from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .directed_bidirected import DirectedBidirectedGraph
from .endpoint_marked import EndpointMarkedGraph
from .errors import GraphValidationError, SchemaError


SCHEMA = "cga.graph/1"


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise SchemaError(f"non-standard JSON constant: {value}")


def _require_exact_keys(
    value: dict[str, Any], expected: tuple[str, ...], *, context: str
) -> None:
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        unknown = sorted(actual - expected_set)
        raise SchemaError(
            f"{context} fields differ: missing={missing!r}, unknown={unknown!r}"
        )


def _require_list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{context} must be a JSON array")
    return value


def graph_from_json(
    text: str,
) -> DirectedBidirectedGraph | EndpointMarkedGraph:
    if not isinstance(text, str):
        raise SchemaError("graph JSON must be text")
    if text.startswith("\ufeff"):
        raise SchemaError("graph JSON cannot begin with a byte-order mark")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except SchemaError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise SchemaError(f"malformed graph JSON: {error}") from error
    if not isinstance(value, dict):
        raise SchemaError("graph JSON must be an object")
    if value.get("schema") != SCHEMA:
        raise SchemaError(f"unsupported graph schema: {value.get('schema')!r}")

    family = value.get("family")
    try:
        if family == "directed-bidirected":
            _require_exact_keys(
                value,
                ("schema", "family", "nodes", "directed", "bidirected"),
                context="directed-bidirected graph",
            )
            nodes = _require_list(value["nodes"], context="nodes")
            directed_values = _require_list(
                value["directed"], context="directed"
            )
            bidirected_values = _require_list(
                value["bidirected"], context="bidirected"
            )
            for edge in directed_values:
                if not isinstance(edge, list) or len(edge) != 2:
                    raise SchemaError("directed edges must be two-element arrays")
            for edge in bidirected_values:
                if not isinstance(edge, list) or len(edge) != 2:
                    raise SchemaError("bidirected edges must be two-element arrays")
            return DirectedBidirectedGraph(
                nodes=nodes,
                directed=directed_values,
                bidirected=bidirected_values,
            )

        if family == "endpoint-marked":
            _require_exact_keys(
                value,
                ("schema", "family", "nodes", "edges"),
                context="endpoint-marked graph",
            )
            nodes = _require_list(value["nodes"], context="nodes")
            edge_values = _require_list(value["edges"], context="edges")
            edges: list[tuple[object, object, object, object]] = []
            for edge in edge_values:
                if not isinstance(edge, dict):
                    raise SchemaError("endpoint-marked edges must be objects")
                _require_exact_keys(
                    edge,
                    ("u", "u_mark", "v", "v_mark"),
                    context="endpoint-marked edge",
                )
                edges.append(
                    (edge["u"], edge["u_mark"], edge["v"], edge["v_mark"])
                )
            return EndpointMarkedGraph(nodes=nodes, edges=edges)

        raise SchemaError(f"unknown graph family: {family!r}")
    except GraphValidationError as error:
        raise SchemaError(str(error)) from error


def graph_to_canonical_json(
    graph: DirectedBidirectedGraph | EndpointMarkedGraph,
) -> str:
    if isinstance(graph, DirectedBidirectedGraph):
        positions = {node: index for index, node in enumerate(graph.nodes)}
        directed = sorted(
            graph.directed,
            key=lambda edge: (positions[edge[0]], positions[edge[1]]),
        )
        bidirected = sorted(
            graph.bidirected,
            key=lambda edge: (positions[edge[0]], positions[edge[1]]),
        )
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "family": "directed-bidirected",
            "nodes": list(graph.nodes),
            "directed": [list(edge) for edge in directed],
            "bidirected": [list(edge) for edge in bidirected],
        }
    elif isinstance(graph, EndpointMarkedGraph):
        value = {
            "schema": SCHEMA,
            "family": "endpoint-marked",
            "nodes": list(graph.nodes),
            "edges": [
                {
                    "u": left,
                    "u_mark": left_mark.value,
                    "v": right,
                    "v_mark": right_mark.value,
                }
                for left, left_mark, right, right_mark in graph.edges
            ],
        }
    else:
        raise TypeError("expected a supported graph topology")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
