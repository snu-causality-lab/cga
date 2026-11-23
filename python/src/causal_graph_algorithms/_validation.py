from __future__ import annotations

from collections.abc import Iterable

from .errors import GraphValidationError


def normalize_nodes(nodes: Iterable[str]) -> tuple[str, ...]:
    try:
        normalized = tuple(nodes)
    except TypeError as error:
        raise GraphValidationError("nodes must be iterable") from error

    seen: set[str] = set()
    for node in normalized:
        if not isinstance(node, str):
            raise GraphValidationError("every node label must be a string")
        if not node:
            raise GraphValidationError("node labels must be non-empty")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in node):
            raise GraphValidationError("node labels cannot contain unpaired surrogates")
        if node in seen:
            raise GraphValidationError(f"duplicate node label: {node!r}")
        seen.add(node)
    return normalized


def node_positions(nodes: tuple[str, ...]) -> dict[str, int]:
    return {node: index for index, node in enumerate(nodes)}


def require_vertex(positions: dict[str, int], vertex: object) -> str:
    if not isinstance(vertex, str) or vertex not in positions:
        raise GraphValidationError(f"unknown vertex: {vertex!r}")
    return vertex


def normalize_seeds(
    positions: dict[str, int], seeds: str | Iterable[str]
) -> frozenset[str]:
    values: tuple[str, ...]
    if isinstance(seeds, str):
        values = (seeds,)
    else:
        try:
            values = tuple(seeds)
        except TypeError as error:
            raise GraphValidationError("seed vertices must be iterable") from error
    for vertex in values:
        require_vertex(positions, vertex)
    return frozenset(values)


def normalize_pair(
    raw: object,
    positions: dict[str, int],
    *,
    relation: str,
) -> tuple[str, str]:
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise GraphValidationError(f"{relation} edges must be endpoint pairs")
    raw_left, raw_right = raw
    left = require_vertex(positions, raw_left)
    right = require_vertex(positions, raw_right)
    if left == right:
        raise GraphValidationError(f"{relation} self-loops are not allowed")
    return left, right
