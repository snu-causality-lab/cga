from __future__ import annotations

import html
import json
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import TypeAlias

from .ancestral import AncestralGraph
from .directed_bidirected import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
)
from .endpoint_marked import EndpointMarkedGraph, Mark
from .maximality import MaximalAncestralGraph
from .pag import CausalPAG, ClosedPAG


GraphForSVG: TypeAlias = (
    DirectedBidirectedGraph
    | AcyclicDirectedBidirectedGraph
    | DAG
    | EndpointMarkedGraph
    | AncestralGraph
    | MaximalAncestralGraph
    | CausalPAG
    | ClosedPAG
)


_Point: TypeAlias = tuple[float, float]


@dataclass(frozen=True)
class _SVGEdge:
    u: str
    u_mark: Mark
    v: str
    v_mark: Mark


def _raw_topology(
    graph: GraphForSVG,
) -> DirectedBidirectedGraph | EndpointMarkedGraph:
    if isinstance(graph, (DirectedBidirectedGraph, EndpointMarkedGraph)):
        return graph
    if isinstance(
        graph,
        (
            AcyclicDirectedBidirectedGraph,
            DAG,
            AncestralGraph,
            MaximalAncestralGraph,
            CausalPAG,
            ClosedPAG,
        ),
    ):
        return graph.topology
    raise TypeError("expected a supported graph topology or view")


def _svg_edges(
    topology: DirectedBidirectedGraph | EndpointMarkedGraph,
) -> tuple[_SVGEdge, ...]:
    positions = {node: index for index, node in enumerate(topology.nodes)}
    edges: list[_SVGEdge] = []

    if isinstance(topology, DirectedBidirectedGraph):
        for source, target in topology.directed:
            if positions[source] < positions[target]:
                edges.append(_SVGEdge(source, Mark.TAIL, target, Mark.ARROW))
            else:
                edges.append(_SVGEdge(target, Mark.ARROW, source, Mark.TAIL))
        for left, right in topology.bidirected:
            edges.append(_SVGEdge(left, Mark.ARROW, right, Mark.ARROW))
    else:
        edges.extend(
            _SVGEdge(left, left_mark, right, right_mark)
            for left, left_mark, right, right_mark in topology.edges
        )

    mark_order = {Mark.TAIL: 0, Mark.ARROW: 1, Mark.CIRCLE: 2}
    edges.sort(
        key=lambda edge: (
            positions[edge.u],
            positions[edge.v],
            mark_order[edge.u_mark],
            mark_order[edge.v_mark],
        )
    )
    return tuple(edges)


def _directed_edges(
    topology: DirectedBidirectedGraph | EndpointMarkedGraph,
) -> tuple[tuple[str, str], ...]:
    if isinstance(topology, DirectedBidirectedGraph):
        return tuple(topology.directed)

    directed: list[tuple[str, str]] = []
    for left, left_mark, right, right_mark in topology.edges:
        if (left_mark, right_mark) == (Mark.TAIL, Mark.ARROW):
            directed.append((left, right))
        elif (left_mark, right_mark) == (Mark.ARROW, Mark.TAIL):
            directed.append((right, left))
    return tuple(directed)


def _topological_levels(
    nodes: tuple[str, ...],
    directed: tuple[tuple[str, str], ...],
) -> dict[str, int] | None:
    positions = {node: index for index, node in enumerate(nodes)}
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in directed:
        if target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1

    pending = sorted(
        (node for node in nodes if indegree[node] == 0),
        key=positions.__getitem__,
    )
    levels = {node: 0 for node in nodes}
    visited = 0
    while pending:
        current = pending.pop(0)
        visited += 1
        for child in sorted(outgoing[current], key=positions.__getitem__):
            levels[child] = max(levels[child], levels[current] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
                pending.sort(key=positions.__getitem__)
    if visited != len(nodes):
        return None
    return levels


def _layered_positions(
    nodes: tuple[str, ...], levels: dict[str, int], sizes: dict[str, _Point]
) -> dict[str, _Point]:
    maximum = max(levels.values(), default=0)
    grouped: dict[int, list[str]] = defaultdict(list)
    for node in nodes:
        grouped[levels[node]].append(node)

    result: dict[str, _Point] = {}
    x = 72.0
    previous_radius = 0.0
    for level in range(maximum + 1):
        level_nodes = grouped[level]
        radius = max(sizes[node][0] for node in level_nodes)
        if level:
            # A midpoint control must lie outside both incident ellipses, even
            # when one label is much wider; otherwise an arrow can point back.
            x += max(496.0 / maximum, 2.0 * max(previous_radius, radius) + 60.0)
        previous_radius = radius
        spacing = max(64.0, 248.0 / max(1, len(level_nodes) - 1))
        for index, node in enumerate(level_nodes):
            y = 180.0 + (index - (len(level_nodes) - 1) / 2.0) * spacing
            result[node] = (x, y)
    return result


def _circular_positions(
    nodes: tuple[str, ...], sizes: dict[str, _Point]
) -> dict[str, _Point]:
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: (320.0, 180.0)}

    # Scale the minor radius so even the widest adjacent nodes stay apart.
    radius = max(
        125.0,
        (max(rx for rx, _ in sizes.values()) + 24.0) / math.sin(math.pi / len(nodes)),
    )
    result: dict[str, _Point] = {}
    for index, node in enumerate(nodes):
        angle = -math.pi / 2.0 + 2.0 * math.pi * index / len(nodes)
        result[node] = (
            320.0 + max(235.0, radius) * math.cos(angle),
            180.0 + radius * math.sin(angle),
        )
    return result


def _positions(
    topology: DirectedBidirectedGraph | EndpointMarkedGraph,
    sizes: dict[str, _Point],
) -> dict[str, _Point]:
    directed = _directed_edges(topology)
    if directed:
        levels = _topological_levels(topology.nodes, directed)
        if levels is not None:
            return _layered_positions(topology.nodes, levels, sizes)
    return _circular_positions(topology.nodes, sizes)


def _node_size(label: str) -> tuple[float, float]:
    # Conservative advances for the monospace label font and wide Unicode glyphs.
    width = sum(
        16.0 if unicodedata.east_asian_width(char) in "WF" else 9.0 for char in label
    )
    return max(25.0, 12.0 + width / 2.0), 22.0


def _ellipse_boundary(
    center: _Point,
    toward: _Point,
    radii: tuple[float, float],
) -> _Point:
    dx = toward[0] - center[0]
    dy = toward[1] - center[1]
    if dx == 0.0 and dy == 0.0:
        return center
    rx, ry = radii
    scale = 1.0 / math.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    return center[0] + dx * scale, center[1] + dy * scale


def _marker(mark: Mark, *, position: str) -> str:
    if mark is Mark.ARROW:
        return f' marker-{position}="url(#cga-arrow)"'
    if mark is Mark.CIRCLE:
        return f' marker-{position}="url(#cga-circle)"'
    return ""


def _edge_color(edge: _SVGEdge) -> str:
    marks = {edge.u_mark, edge.v_mark}
    if Mark.CIRCLE in marks:
        return "#b45309"
    if edge.u_mark is Mark.ARROW and edge.v_mark is Mark.ARROW:
        return "#7c3aed"
    if edge.u_mark is Mark.TAIL and edge.v_mark is Mark.TAIL:
        return "#64748b"
    return "#334155"


def _escape(value: str) -> str:
    # Literal attribute whitespace is normalized by XML parsers; references are not.
    return (
        html.escape(value, quote=True)
        .replace("\t", "&#9;")
        .replace("\n", "&#10;")
        .replace("\r", "&#13;")
    )


def _xml_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        codepoint in (0x9, 0xA, 0xD)
        or 0x20 <= codepoint <= 0xD7FF
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )


def _label_attributes(name: str, value: str) -> str:
    """Preserve XML labels exactly, encoding unsupported characters as JSON.

    The per-attribute encoding flag distinguishes an encoded label from a literal
    label with the same spelling. XML consumers decode only flagged values.
    """
    encoding = ""
    if not all(_xml_character(character) for character in value):
        value = json.dumps(value, ensure_ascii=True)
        encoding = f' data-{name}-encoding="json"'
    return f'data-{name}="{_escape(value)}"{encoding}'


def _display_label(value: str) -> str:
    """Make whitespace and controls visible without conflating literal escapes."""
    if not any(
        character.isspace()
        or character in '\\"'
        or unicodedata.category(character) in ("Cc", "Cf")
        or not _xml_character(character)
        for character in value
    ):
        return value

    quoted = json.dumps(value, ensure_ascii=False)
    return "".join(
        json.dumps(character, ensure_ascii=True)[1:-1]
        if (
            (character.isspace() and character != " ")
            or unicodedata.category(character) in ("Cc", "Cf")
            or not _xml_character(character)
        )
        else character
        for character in quoted
    )


def _format_number(value: float) -> str:
    return f"{value:.2f}"


def _curve_hits_box(
    start: _Point,
    control: _Point,
    end: _Point,
    box: tuple[float, float, float, float],
) -> bool:
    """Conservatively intersect a quadratic curve with a padded node box."""
    left, top, right, bottom = box
    pending = [(start, control, end, 0)]
    while pending:
        a, b, c, depth = pending.pop()
        if (
            max(a[0], b[0], c[0]) < left
            or min(a[0], b[0], c[0]) > right
            or max(a[1], b[1], c[1]) < top
            or min(a[1], b[1], c[1]) > bottom
        ):
            continue
        if depth == 8:
            return True
        ab = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        bc = ((b[0] + c[0]) / 2.0, (b[1] + c[1]) / 2.0)
        mid = ((ab[0] + bc[0]) / 2.0, (ab[1] + bc[1]) / 2.0)
        pending.extend(((a, ab, mid, depth + 1), (mid, bc, c, depth + 1)))
    return False


def _edge_curve(
    edge: _SVGEdge,
    curvature: float,
    positions: dict[str, _Point],
    sizes: dict[str, _Point],
    used: set[float],
) -> tuple[_Point, _Point, _Point]:
    u, v = positions[edge.u], positions[edge.v]
    dx, dy = v[0] - u[0], v[1] - u[1]
    distance = math.hypot(dx, dy)
    perpendicular = (-dy / distance, dx / distance)
    boxes = [
        (
            x - sizes[node][0] - 8,
            y - sizes[node][1] - 8,
            x + sizes[node][0] + 8,
            y + sizes[node][1] + 8,
        )
        for node, (x, y) in positions.items()
        if node not in (edge.u, edge.v)
    ]
    signs = (-1.0, 1.0) if curvature == 0 else (math.copysign(1.0, curvature),)
    candidates = [curvature] + [
        sign * (abs(curvature) + 64.0 * step) for step in range(1, 17) for sign in signs
    ]
    best: tuple[_Point, _Point, _Point] | None = None
    best_score = len(boxes) + 1
    best_bend = curvature
    for bend in candidates:
        if bend in used:
            continue
        control = (
            (u[0] + v[0]) / 2.0 + perpendicular[0] * bend,
            (u[1] + v[1]) / 2.0 + perpendicular[1] * bend,
        )
        start = _ellipse_boundary(u, control, sizes[edge.u])
        end = _ellipse_boundary(v, control, sizes[edge.v])
        score = sum(_curve_hits_box(start, control, end, box) for box in boxes)
        if score < best_score:
            best, best_score, best_bend = (start, control, end), score, bend
        if score == 0:
            break
    assert best is not None
    used.add(best_bend)
    return best


def graph_to_svg(graph: GraphForSVG) -> str:
    """Render a deterministic, dependency-free SVG for a small graph."""

    topology = _raw_topology(graph)
    sizes = {node: _node_size(_display_label(node)) for node in topology.nodes}
    positions = _positions(topology, sizes)
    edges = _svg_edges(topology)

    grouped_edges: dict[tuple[str, str], list[_SVGEdge]] = defaultdict(list)
    for edge in edges:
        grouped_edges[(edge.u, edge.v)].append(edge)

    path_lines: list[str] = []
    bounds = [(0.0, 0.0), (640.0, 360.0)]
    for node, (x, y) in positions.items():
        rx, ry = sizes[node]
        bounds.extend(((x - rx - 24, y - ry - 24), (x + rx + 24, y + ry + 24)))
    for pair in sorted(
        grouped_edges,
        key=lambda value: (
            topology.nodes.index(value[0]),
            topology.nodes.index(value[1]),
        ),
    ):
        pair_edges = grouped_edges[pair]
        curvatures: tuple[float, ...]
        if len(pair_edges) == 1:
            curvatures = (0.0,)
        elif len(pair_edges) == 2:
            curvatures = (-28.0, 28.0)
        else:
            curvatures = (-38.0, 0.0, 38.0)

        used: set[float] = set()
        for edge, curvature in zip(pair_edges, curvatures, strict=True):
            start, control, end = _edge_curve(edge, curvature, positions, sizes, used)
            for px, py in (start, control, end):
                bounds.extend(((px - 16, py - 16), (px + 16, py + 16)))
            path = (
                f"M {_format_number(start[0])} {_format_number(start[1])} "
                f"Q {_format_number(control[0])} {_format_number(control[1])} "
                f"{_format_number(end[0])} {_format_number(end[1])}"
            )
            path_lines.append(
                "    <path "
                f'd="{path}" fill="none" stroke="{_edge_color(edge)}" '
                'stroke-width="2.2" '
                f"{_label_attributes('u', edge.u)} "
                f'data-u-mark="{edge.u_mark.value}" '
                f"{_label_attributes('v', edge.v)} "
                f'data-v-mark="{edge.v_mark.value}"'
                f"{_marker(edge.u_mark, position='start')}"
                f"{_marker(edge.v_mark, position='end')} />"
            )

    node_lines: list[str] = []
    for node in topology.nodes:
        x, y = positions[node]
        rx, ry = sizes[node]
        escaped = _escape(_display_label(node))
        node_lines.extend(
            (
                f"    <g {_label_attributes('node', node)}>",
                "      <ellipse "
                f'cx="{_format_number(x)}" cy="{_format_number(y)}" '
                f'rx="{_format_number(rx)}" ry="{_format_number(ry)}" '
                'fill="#ffffff" stroke="#0f172a" stroke-width="1.8" />',
                "      <text "
                f'x="{_format_number(x)}" y="{_format_number(y + 0.5)}" '
                'text-anchor="middle" dominant-baseline="middle" xml:space="preserve" '
                'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
                f'font-size="14" fill="#0f172a">{escaped}</text>',
                "    </g>",
            )
        )

    content = "\n".join((*path_lines, *node_lines))
    if content:
        content += "\n"
    left = math.floor(min(x for x, _ in bounds))
    top = math.floor(min(y for _, y in bounds))
    width = math.ceil(max(x for x, _ in bounds)) - left
    height = math.ceil(max(y for _, y in bounds)) - top
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{left} {top} {width} {height}" width="640" height="360" '
        'role="img" aria-label="Causal graph">\n'
        "  <title>Causal graph</title>\n"
        "  <defs>\n"
        '    <marker id="cga-arrow" viewBox="0 0 10 10" refX="8" '
        'refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse">\n'
        '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155" />\n'
        "    </marker>\n"
        '    <marker id="cga-circle" viewBox="0 0 10 10" refX="5" '
        'refY="5" markerWidth="7" markerHeight="7">\n'
        '      <circle cx="5" cy="5" r="3.2" fill="#ffffff" '
        'stroke="#b45309" stroke-width="1.7" />\n'
        "    </marker>\n"
        "  </defs>\n"
        f"{content}"
        "</svg>"
    )
