from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from causal_graph_algorithms import (
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    Mark,
    graph_to_svg,
)


SVG_NAMESPACE = {"svg": "http://www.w3.org/2000/svg"}


def _decoded_label(element: ET.Element, name: str) -> str:
    value = element.attrib[f"data-{name}"]
    encoding = element.get(f"data-{name}-encoding")
    if encoding is None:
        return value
    assert encoding == "json"
    decoded = json.loads(value)
    assert isinstance(decoded, str)
    return decoded


@pytest.mark.parametrize("family", ("directed-bidirected", "endpoint-marked"))
def test_svg_reversibly_encodes_labels_outside_the_xml_character_set(
    family: str,
) -> None:
    nodes = ("A\x00B", "A\x01B", "A\ufffeB", "A\uffffB", '"A\\u0000B"')
    if family == "directed-bidirected":
        graph = DirectedBidirectedGraph(
            nodes=nodes,
            directed=tuple(zip(nodes, nodes[1:])),
        )
    else:
        graph = EndpointMarkedGraph(
            nodes=nodes,
            edges=tuple(
                (left, Mark.CIRCLE, right, Mark.ARROW)
                for left, right in zip(nodes, nodes[1:])
            ),
        )

    svg = graph_to_svg(graph)
    root = ET.fromstring(svg)
    groups = root.findall(".//svg:g[@data-node]", SVG_NAMESPACE)
    assert tuple(_decoded_label(group, "node") for group in groups) == nodes
    assert all(group.get("data-node-encoding") == "json" for group in groups[:4])
    assert groups[4].get("data-node-encoding") is None
    # The encoded value may equal a literal label; encoding metadata disambiguates.
    assert groups[0].attrib["data-node"] == groups[4].attrib["data-node"]
    paths = root.findall(".//svg:path[@data-u]", SVG_NAMESPACE)
    assert tuple(
        (_decoded_label(path, "u"), _decoded_label(path, "v")) for path in paths
    ) == tuple(zip(nodes, nodes[1:]))
    rendered = root.findall(".//svg:text", SVG_NAMESPACE)
    assert tuple(json.loads(text.text or "") for text in rendered) == nodes
    assert graph_to_svg(graph) == svg


def test_svg_preserves_exact_whitespace_in_all_label_attributes() -> None:
    nodes = ("A\nB", "A\tB", "A\rB", "A B")
    graph = DirectedBidirectedGraph(
        nodes=nodes,
        directed=tuple(zip(nodes, nodes[1:])),
    )
    root = ET.fromstring(graph_to_svg(graph))
    groups = root.findall(".//svg:g[@data-node]", SVG_NAMESPACE)
    assert tuple(group.attrib["data-node"] for group in groups) == nodes
    assert all(group.get("data-node-encoding") is None for group in groups)
    paths = root.findall(".//svg:path[@data-u]", SVG_NAMESPACE)
    assert tuple(
        (path.attrib["data-u"], path.attrib["data-v"]) for path in paths
    ) == tuple(zip(nodes, nodes[1:]))


def test_svg_visible_whitespace_controls_quotes_and_backslashes_are_unambiguous() -> (
    None
):
    nodes = (
        "A\nB",
        "A\tB",
        "A\rB",
        "A B",
        "A\\nB",
        '"A\\nB"',
        " A ",
        "A  B",
        "A\u0085B",
        "A\u00a0B",
        "A\u2028B",
        "A\u200bB",
    )
    root = ET.fromstring(graph_to_svg(EndpointMarkedGraph(nodes=nodes)))
    texts = root.findall(".//svg:text", SVG_NAMESPACE)
    visible = tuple(text.text or "" for text in texts)
    assert tuple(json.loads(label) for label in visible) == nodes
    assert len(set(visible)) == len(nodes)
    assert all(
        text.get("{http://www.w3.org/XML/1998/namespace}space") == "preserve"
        for text in texts
    )
    assert visible[:4] == ('"A\\nB"', '"A\\tB"', '"A\\rB"', '"A B"')
    assert all(
        character == " " or not character.isspace()
        for label in visible
        for character in label
    )
    assert all("\u200b" not in label and "\u0085" not in label for label in visible)


def test_svg_keeps_xml_legal_astral_and_readable_unicode_labels_literal() -> None:
    nodes = (
        "서울😀",
        "\U00010000",
        "\U0001fffe",
        "\U0010fffd",
        "\U0010ffff",
        "Café",
        "A<&",
    )
    root = ET.fromstring(graph_to_svg(EndpointMarkedGraph(nodes=nodes)))
    groups = root.findall(".//svg:g[@data-node]", SVG_NAMESPACE)
    assert tuple(group.attrib["data-node"] for group in groups) == nodes
    assert all(group.get("data-node-encoding") is None for group in groups)
    texts = root.findall(".//svg:text", SVG_NAMESPACE)
    assert tuple(text.text for text in texts) == nodes
