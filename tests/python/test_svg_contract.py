from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from itertools import pairwise

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    CausalPAG,
    ClosedPAG,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    Mark,
    MaximalAncestralGraph,
    graph_to_svg,
)


SVG_NAMESPACE = {"svg": "http://www.w3.org/2000/svg"}


def _edge_semantics(svg: str) -> set[tuple[str, str, str, str]]:
    root = ET.fromstring(svg)
    return {
        (
            path.attrib["data-u"],
            path.attrib["data-u-mark"],
            path.attrib["data-v"],
            path.attrib["data-v-mark"],
        )
        for path in root.findall(".//svg:path[@data-u]", SVG_NAMESPACE)
    }


def test_svg_is_deterministic_well_formed_and_semantically_annotated() -> None:
    graph = DirectedBidirectedGraph.from_text("X --> M --> Y; X <-> Y")
    first = graph_to_svg(graph)
    second = graph_to_svg(graph)
    root = ET.fromstring(first)

    assert first == second
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib["viewBox"] == "0 0 640 360"
    assert len(first.encode("utf-8")) < 20_000
    assert _edge_semantics(first) == {
        ("X", "tail", "M", "arrow"),
        ("M", "tail", "Y", "arrow"),
        ("X", "arrow", "Y", "arrow"),
    }


def test_svg_preserves_every_endpoint_mark_pair() -> None:
    graph = EndpointMarkedGraph(
        nodes=tuple(f"V{i}" for i in range(18)),
        edges=tuple(
            (f"V{2 * index}", left, f"V{2 * index + 1}", right)
            for index, (left, right) in enumerate(
                (
                    (Mark.TAIL, Mark.TAIL),
                    (Mark.TAIL, Mark.ARROW),
                    (Mark.ARROW, Mark.TAIL),
                    (Mark.ARROW, Mark.ARROW),
                    (Mark.CIRCLE, Mark.TAIL),
                    (Mark.TAIL, Mark.CIRCLE),
                    (Mark.CIRCLE, Mark.ARROW),
                    (Mark.ARROW, Mark.CIRCLE),
                    (Mark.CIRCLE, Mark.CIRCLE),
                )
            )
        ),
    )
    semantics = _edge_semantics(graph_to_svg(graph))
    assert {(left, right) for _, left, _, right in semantics} == {
        (left.value, right.value) for left in Mark for right in Mark
    }


def test_svg_accepts_validated_views_without_erasing_their_topology() -> None:
    dag = DAG.from_text("A --> B --> C")
    acyclic = AcyclicDirectedBidirectedGraph.from_text("X --> M --> Y; X <-> Y")
    mag = MaximalAncestralGraph.from_text("A --> B; B <-> C")
    views = (
        dag,
        acyclic,
        mag,
        CausalPAG.from_mag(mag),
        ClosedPAG.from_mag(mag),
    )

    for view in views:
        svg = graph_to_svg(view)
        assert "<svg" in svg
        assert set(view.topology.nodes) <= {
            text.text
            for text in ET.fromstring(svg).findall(".//svg:text", SVG_NAMESPACE)
            if text.text is not None
        }


def test_svg_escapes_labels_and_handles_empty_and_singleton_graphs() -> None:
    escaped = graph_to_svg(DirectedBidirectedGraph(nodes=('A<&"',)))
    assert "A&lt;&amp;&quot;" in escaped
    ET.fromstring(escaped)

    empty = graph_to_svg(EndpointMarkedGraph(nodes=()))
    root = ET.fromstring(empty)
    assert root.findall(".//svg:ellipse", SVG_NAMESPACE) == []


def test_svg_rejects_non_graph_values() -> None:
    with pytest.raises(TypeError, match="supported graph topology or view"):
        graph_to_svg(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    (
        "X --> M --> Y; X <-> Y",
        "X --> M --> Y; X --> Y",
        "X --> M --> N --> Y; X <-> Y; M <-> Y",
        "X --> M --> Y; X --> Y; X <-> Y",
    ),
)
def test_svg_routes_long_edges_around_nonincident_nodes(text: str) -> None:
    """Check visible geometry, independently of the endpoint metadata test."""
    root = ET.fromstring(graph_to_svg(DirectedBidirectedGraph.from_text(text)))
    nodes = {}
    for group in root.findall("svg:g[@data-node]", SVG_NAMESPACE):
        ellipse = group.find("svg:ellipse", SVG_NAMESPACE)
        assert ellipse is not None
        nodes[group.attrib["data-node"]] = tuple(
            float(ellipse.attrib[key]) for key in ("cx", "cy", "rx", "ry")
        )
    for path in root.findall("svg:path[@data-u]", SVG_NAMESPACE):
        coordinates = tuple(
            map(float, re.findall(r"-?\d+(?:\.\d+)?", path.attrib["d"]))
        )
        assert len(coordinates) == 6, "expected one quadratic edge"
        x0, y0, xc, yc, x1, y1 = coordinates
        for label, (cx, cy, rx, ry) in nodes.items():
            if label in (path.attrib["data-u"], path.attrib["data-v"]):
                continue
            for step in range(201):
                t = step / 200
                x = (1 - t) ** 2 * x0 + 2 * t * (1 - t) * xc + t**2 * x1
                y = (1 - t) ** 2 * y0 + 2 * t * (1 - t) * yc + t**2 * y1
                assert ((x - cx) / (rx + 4)) ** 2 + ((y - cy) / (ry + 4)) ** 2 > 1, (
                    path.attrib,
                    label,
                    t,
                )


def test_svg_fits_long_labels_and_keeps_nodes_apart() -> None:
    labels = ("baseline blood pressure", "post-treatment blood pressure")
    root = ET.fromstring(
        graph_to_svg(DirectedBidirectedGraph(nodes=labels, directed=(labels,)))
    )
    left, top, width, height = map(float, root.attrib["viewBox"].split())
    bounds = []
    for ellipse in root.findall(".//svg:ellipse", SVG_NAMESPACE):
        cx, cy, rx, ry = (
            float(ellipse.attrib[key]) for key in ("cx", "cy", "rx", "ry")
        )
        assert left + 8 <= cx - rx < cx + rx <= left + width - 8
        assert top + 8 <= cy - ry < cy + ry <= top + height - 8
        bounds.append((cx - rx, cx + rx))
    assert bounds[0][1] + 40 <= bounds[1][0]


def test_svg_long_chain_does_not_overlap_node_ellipses() -> None:
    graph = DAG.from_text(" --> ".join(f"V{i}" for i in range(14)))
    root = ET.fromstring(graph_to_svg(graph))
    previous_right = float("-inf")
    for ellipse in root.findall(".//svg:ellipse", SVG_NAMESPACE):
        cx, rx = (float(ellipse.attrib[key]) for key in ("cx", "rx"))
        assert previous_right + 40 <= cx - rx
        previous_right = cx + rx


@pytest.mark.parametrize(
    "labels", (("X", "Y" * 60), ("X" * 60, "Y"), ("X", "post-treatment blood pressure", "Y"))
)
@pytest.mark.parametrize("bow", (False, True))
def test_svg_asymmetric_labels_do_not_reverse_endpoint_tangents(
    labels: tuple[str, ...], bow: bool,
) -> None:
    graph = DirectedBidirectedGraph(
        nodes=labels, directed=tuple(pairwise(labels)),
        bidirected=tuple(pairwise(labels)) if bow else (),
    )
    root = ET.fromstring(graph_to_svg(graph))
    centers = {}
    for group in root.findall("svg:g[@data-node]", SVG_NAMESPACE):
        ellipse = group.find("svg:ellipse", SVG_NAMESPACE)
        assert ellipse is not None
        centers[group.attrib["data-node"]] = tuple(
            float(ellipse.attrib[key]) for key in ("cx", "cy")
        )
    for path in root.findall("svg:path[@data-u]", SVG_NAMESPACE):
        x0, y0, xc, yc, x1, y1 = map(
            float, re.findall(r"-?\d+(?:\.\d+)?", path.attrib["d"])
        )
        ux, uy = centers[path.attrib["data-u"]]
        vx, vy = centers[path.attrib["data-v"]]
        assert (xc - x0) * (x0 - ux) + (yc - y0) * (y0 - uy) > 0
        assert (x1 - xc) * (vx - x1) + (y1 - yc) * (vy - y1) > 0
