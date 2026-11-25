from __future__ import annotations

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    AncestralGraph,
    CausalPAG,
    ClosedPAG,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphTextSyntaxError,
    GraphValidationError,
    Mark,
    MaximalAncestralGraph,
    graph_from_text,
    graph_to_text,
)


@pytest.mark.parametrize(
    ("operator", "left_mark", "right_mark"),
    (
        ("---", Mark.TAIL, Mark.TAIL),
        ("-->", Mark.TAIL, Mark.ARROW),
        ("<--", Mark.ARROW, Mark.TAIL),
        ("<->", Mark.ARROW, Mark.ARROW),
        ("o--", Mark.CIRCLE, Mark.TAIL),
        ("--o", Mark.TAIL, Mark.CIRCLE),
        ("o->", Mark.CIRCLE, Mark.ARROW),
        ("<-o", Mark.ARROW, Mark.CIRCLE),
        ("o-o", Mark.CIRCLE, Mark.CIRCLE),
    ),
)
def test_endpoint_algebra_has_one_canonical_spelling_per_mark_pair(
    operator: str,
    left_mark: Mark,
    right_mark: Mark,
) -> None:
    graph = EndpointMarkedGraph.from_text(f"A {operator} B")
    assert graph.nodes == ("A", "B")
    assert graph.edge("A", "B") == (left_mark, right_mark)
    assert graph_to_text(graph).splitlines()[-1] == f"A {operator} B"


@pytest.mark.parametrize(("alias", "canonical"), (("->", "-->"), ("<-", "<--")))
def test_common_directed_aliases_parse_but_canonicalize(
    alias: str, canonical: str
) -> None:
    graph = EndpointMarkedGraph.from_text(f"A {alias} B")
    assert graph_to_text(graph).splitlines()[-1] == f"A {canonical} B"


def test_directed_bidirected_factory_preserves_chains_bows_and_isolates() -> None:
    graph = DirectedBidirectedGraph.from_text(
        "A <-> BB --> C_0 <-- D9 <-- E <-> F --> G, "
        "H <-- I <-> J; K; L --> M"
    )

    assert graph.nodes == (
        "A",
        "BB",
        "C_0",
        "D9",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
    )
    assert graph.directed == frozenset(
        {
            ("BB", "C_0"),
            ("D9", "C_0"),
            ("E", "D9"),
            ("F", "G"),
            ("I", "H"),
            ("L", "M"),
        }
    )
    assert graph.bidirected == frozenset(
        {("A", "BB"), ("E", "F"), ("I", "J")}
    )

    bow = DirectedBidirectedGraph.from_text("X --> Y; X <-> Y")
    assert bow.has_directed("X", "Y")
    assert bow.has_bidirected("X", "Y")


def test_class_factories_parse_raw_topology_then_apply_existing_validation() -> None:
    dag = DAG.from_text("A -> B -> C")
    acyclic = AcyclicDirectedBidirectedGraph.from_text(
        "X --> M --> Y; X <-> Y"
    )
    ancestral = AncestralGraph.from_text("A --> B; B <-> C")
    maximal = MaximalAncestralGraph.from_text("A --> B; B <-> C")

    assert dag.topology.directed == frozenset({("A", "B"), ("B", "C")})
    assert acyclic.topology.has_bidirected("X", "Y")
    assert ancestral.topology.edge("B", "C") == (Mark.ARROW, Mark.ARROW)
    assert maximal.topology == ancestral.topology

    with pytest.raises(GraphValidationError, match="DAG cannot contain bidirected"):
        DAG.from_text("X --> Y; X <-> Y")
    with pytest.raises(GraphValidationError, match="cycle"):
        AcyclicDirectedBidirectedGraph.from_text("A --> B --> A")
    with pytest.raises(GraphValidationError, match="circle"):
        MaximalAncestralGraph.from_text("A o-> B")


def test_pag_proof_views_remain_factory_only_from_a_validated_mag() -> None:
    assert "from_text" not in CausalPAG.__dict__
    assert "from_text" not in ClosedPAG.__dict__

    raw_marks = EndpointMarkedGraph.from_text("A o-> B; B o-o C")
    assert raw_marks.edge("A", "B") == (Mark.CIRCLE, Mark.ARROW)
    assert raw_marks.edge("B", "C") == (Mark.CIRCLE, Mark.CIRCLE)

    mag = MaximalAncestralGraph.from_text("A --> B; B <-> C")
    assert isinstance(CausalPAG.from_mag(mag), CausalPAG)
    assert isinstance(ClosedPAG.from_mag(mag), ClosedPAG)


@pytest.mark.parametrize("operator", ("---", "o--", "--o", "o->", "<-o", "o-o"))
def test_directed_bidirected_text_rejects_endpoint_only_operators(
    operator: str,
) -> None:
    with pytest.raises(GraphValidationError, match="endpoint-marked"):
        DirectedBidirectedGraph.from_text(f"A {operator} B")


def test_same_pair_semantics_differ_between_topology_families() -> None:
    bow = DirectedBidirectedGraph.from_text("A --> B; A <-> B")
    assert bow.directed == frozenset({("A", "B")})
    assert bow.bidirected == frozenset({("A", "B")})

    with pytest.raises(GraphValidationError, match="multiple endpoint-marked edges"):
        EndpointMarkedGraph.from_text("A --> B; A <-> B")


def test_quoted_unicode_names_comments_and_first_appearance_order() -> None:
    graph = EndpointMarkedGraph.from_text(
        'α o-> "treatment age"  # a comment\n'
        '"treatment age" --> Y; "D^9"; α'
    )
    assert graph.nodes == ("α", "treatment age", "Y", "D^9")
    assert graph.edge("α", "treatment age") == (Mark.CIRCLE, Mark.ARROW)
    assert graph.edge("treatment age", "Y") == (Mark.TAIL, Mark.ARROW)


@pytest.mark.parametrize("graph_type", (DirectedBidirectedGraph, EndpointMarkedGraph))
@pytest.mark.parametrize(
    "text",
    (
        "# heading\nA --> B\n# next statement\nC\n# ending\n",
        "\n \t\nA --> B\n \t\nC\n \t\n",
        "; ,\n# heading\n; A --> B ; \t, # between\n ; C , ;\n",
        "A --> B\r\nC\r\n",
        "# heading\r\n\t\r\nA --> B\r\n# next\r\nC\r\n",
    ),
)
def test_comments_blank_lines_and_repeated_separators_preserve_statements(
    graph_type: type[DirectedBidirectedGraph] | type[EndpointMarkedGraph],
    text: str,
) -> None:
    graph = graph_type.from_text(text)
    assert graph.nodes == ("A", "B", "C")
    if isinstance(graph, DirectedBidirectedGraph):
        assert graph.directed == frozenset({("A", "B")})
        assert graph.bidirected == frozenset()
    else:
        assert graph.edges == (("A", Mark.TAIL, "B", Mark.ARROW),)


@pytest.mark.parametrize("graph_type", (DirectedBidirectedGraph, EndpointMarkedGraph))
@pytest.mark.parametrize(
    "text",
    (
        "# first\n# second\n",
        "\n \t\n# only comments\n , ; \n# ending",
        "# first\r\n \t\r\n# second\r\n",
    ),
)
def test_comments_and_separators_alone_build_an_empty_graph(
    graph_type: type[DirectedBidirectedGraph] | type[EndpointMarkedGraph],
    text: str,
) -> None:
    assert graph_type.from_text(text) == graph_type(nodes=())


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize("bad_line", ("  B => C", r'  "bad\q" --> C'))
def test_syntax_positions_survive_comments_and_platform_newlines(
    newline: str,
    bad_line: str,
) -> None:
    text = newline.join(("A", "# comment", bad_line, "D"))
    with pytest.raises(GraphTextSyntaxError) as caught:
        DirectedBidirectedGraph.from_text(text)
    message = str(caught.value)
    expected_column = 5 if "=>" in bad_line else 7
    assert f"line 3, column {expected_column}" in message
    assert f"{bad_line}\n{' ' * (expected_column - 1)}^" in message
    assert "\r" not in message


@pytest.mark.parametrize("graph_type", (DirectedBidirectedGraph, EndpointMarkedGraph))
@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_quoted_label_error_caret_aligns_after_tab_indentation(
    graph_type: type[DirectedBidirectedGraph] | type[EndpointMarkedGraph],
    newline: str,
) -> None:
    bad_label = r'"bad\q" --> C'
    text = newline.join(("A", "\t" + bad_label))
    with pytest.raises(GraphTextSyntaxError) as caught:
        graph_type.from_text(text)
    assert str(caught.value).splitlines() == [
        "malformed graph text at line 2, column 6:",
        " " * 8 + bad_label,
        " " * 12 + "^",
    ]


@pytest.mark.parametrize("graph_type", (DirectedBidirectedGraph, EndpointMarkedGraph))
@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_quoted_hashes_and_json_escapes_preserve_labels(
    graph_type: type[DirectedBidirectedGraph] | type[EndpointMarkedGraph],
    newline: str,
) -> None:
    text = newline.join((
        r'"# carriage\rreturn" --> "both\r\nlines" # comment',
        r'"quoted\"#literal\\end"',
    ))
    graph = graph_type.from_text(text)
    assert graph.nodes == (
        "# carriage\rreturn", "both\r\nlines", 'quoted"#literal\\end',
    )
    if isinstance(graph, DirectedBidirectedGraph):
        assert graph.directed == frozenset({graph.nodes[:2]})
    else:
        assert graph.edges == (
            ("# carriage\rreturn", Mark.TAIL, "both\r\nlines", Mark.ARROW),
        )
    assert graph_type.from_text(graph_to_text(graph)) == graph


@pytest.mark.parametrize("text", ("A B", "A \tB", "A -->\nB", "A\rB"))
def test_ignored_horizontal_whitespace_does_not_separate_statements(text: str) -> None:
    with pytest.raises(GraphTextSyntaxError):
        DirectedBidirectedGraph.from_text(text)


@pytest.mark.parametrize("graph_type", (DirectedBidirectedGraph, EndpointMarkedGraph))
@pytest.mark.parametrize(
    ("text", "line", "column"),
    (
        ("\r", 1, 1),
        ("A\r", 1, 2),
        ("# comment\r", 1, 10),
        ("# comment\rB", 1, 10),
        ("A # comment\rB", 1, 12),
        ("A # comment\r\r\nB", 1, 12),
        ("A\r\n# comment\rB", 2, 10),
        ('"A#B" # comment\rC', 1, 16),
    ),
)
def test_bare_carriage_returns_are_rejected_including_inside_comments(
    graph_type: type[DirectedBidirectedGraph] | type[EndpointMarkedGraph],
    text: str,
    line: int,
    column: int,
) -> None:
    with pytest.raises(GraphTextSyntaxError) as caught:
        graph_type.from_text(text)
    assert f"line {line}, column {column}" in str(caught.value)


@pytest.mark.parametrize("graph_type", (DirectedBidirectedGraph, EndpointMarkedGraph))
@pytest.mark.parametrize(
    ("text", "nodes"),
    (
        ("# comment at EOF", ()),
        ("A# comment at EOF", ("A",)),
        ("A# comment\nB#", ("A", "B")),
        ("A# comment\r\nB#", ("A", "B")),
    ),
)
def test_comments_end_at_platform_newlines_or_eof(
    graph_type: type[DirectedBidirectedGraph] | type[EndpointMarkedGraph],
    text: str,
    nodes: tuple[str, ...],
) -> None:
    assert graph_type.from_text(text) == graph_type(nodes=nodes)


@pytest.mark.parametrize("graph_type", (DirectedBidirectedGraph, EndpointMarkedGraph))
@pytest.mark.parametrize("newline", ("\r", "\r\n"))
def test_quoted_labels_require_escaped_carriage_returns(
    graph_type: type[DirectedBidirectedGraph] | type[EndpointMarkedGraph],
    newline: str,
) -> None:
    with pytest.raises(GraphTextSyntaxError):
        graph_type.from_text(f'"A{newline}#B"')


def test_graph_from_text_requires_an_explicit_supported_family() -> None:
    directed = graph_from_text("A --> B")
    endpoint = graph_from_text("A o-> B", family="endpoint-marked")
    assert isinstance(directed, DirectedBidirectedGraph)
    assert isinstance(endpoint, EndpointMarkedGraph)

    with pytest.raises(ValueError, match="unknown graph text family"):
        graph_from_text("A", family="pag")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    (
        "A => B",
        "A -->",
        "A B",
        'A --> "unterminated',
        "A + B",
        r'"bad\q" --> A',
        r'"bad\u12X4" --> A',
    ),
)
def test_syntax_errors_include_line_column_and_source_context(text: str) -> None:
    with pytest.raises(GraphTextSyntaxError) as caught:
        DirectedBidirectedGraph.from_text(text)
    message = str(caught.value)
    assert "line 1" in message
    assert "column" in message
    assert "^" in message


def test_graph_text_rejects_nontext_and_byte_order_mark() -> None:
    with pytest.raises(GraphTextSyntaxError, match="must be text"):
        DirectedBidirectedGraph.from_text(b"A --> B")  # type: ignore[arg-type]
    with pytest.raises(GraphTextSyntaxError, match="byte-order mark"):
        DirectedBidirectedGraph.from_text("\ufeffA --> B")


def test_canonical_text_roundtrips_both_raw_topology_families() -> None:
    directed = DirectedBidirectedGraph(
        nodes=("B", "α", "A"),
        directed=(("A", "B"), ("B", "α")),
        bidirected=(("A", "B"),),
    )
    directed_text = graph_to_text(directed)
    assert directed_text == (
        "B\nα\nA\nB --> α\nA --> B\nB <-> A"
    )
    assert DirectedBidirectedGraph.from_text(directed_text) == directed

    endpoint = EndpointMarkedGraph(
        nodes=("A", "B", "C", "long label"),
        edges=(
            ("A", Mark.CIRCLE, "B", Mark.TAIL),
            ("B", Mark.ARROW, "C", Mark.CIRCLE),
        ),
    )
    endpoint_text = graph_to_text(endpoint)
    assert endpoint_text == (
        'A\nB\nC\n"long label"\nA o-- B\nB <-o C'
    )
    assert EndpointMarkedGraph.from_text(endpoint_text) == endpoint


def test_empty_text_builds_the_supported_empty_graphs() -> None:
    assert DirectedBidirectedGraph.from_text("") == DirectedBidirectedGraph(nodes=())
    assert EndpointMarkedGraph.from_text("\n,;") == EndpointMarkedGraph(nodes=())


def test_graph_to_text_accepts_only_raw_topology() -> None:
    with pytest.raises(TypeError, match="raw graph topology"):
        graph_to_text(DAG.from_text("A"))  # type: ignore[arg-type]
