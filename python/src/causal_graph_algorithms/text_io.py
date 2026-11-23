from __future__ import annotations

import json
import re
from typing import Literal, NoReturn, TypeAlias, overload

from lark import Lark, Token, Tree
from lark.exceptions import UnexpectedInput

from .directed_bidirected import DirectedBidirectedGraph
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import GraphTextSyntaxError, GraphValidationError


GraphTextFamily: TypeAlias = Literal[
    "directed-bidirected", "endpoint-marked"
]


_GRAPH_TEXT_GRAMMAR = r"""
    start: _SEP* statement_list?
    statement_list: statement (_SEP+ statement)* _SEP*
    statement: node (EDGE node)*

    ?node: BARE_NAME | ESCAPED_STRING

    EDGE: /<->|o->|<-o|o-o|o--|--o|-->|<--|---|->|<-/
    BARE_NAME: /[^\W\d]\w*/
    COMMENT: /#[^\r\n]*/
    _SEP: /(?:[,;\n]|\r\n)+/

    %import common.ESCAPED_STRING
    %import common.WS_INLINE
    %ignore WS_INLINE
    %ignore COMMENT
"""


_PARSER = Lark(
    _GRAPH_TEXT_GRAMMAR,
    parser="lalr",
    lexer="contextual",
    maybe_placeholders=False,
    propagate_positions=True,
)


_OPERATOR_MARKS: dict[str, tuple[Mark, Mark]] = {
    "---": (Mark.TAIL, Mark.TAIL),
    "-->": (Mark.TAIL, Mark.ARROW),
    "<--": (Mark.ARROW, Mark.TAIL),
    "<->": (Mark.ARROW, Mark.ARROW),
    "o--": (Mark.CIRCLE, Mark.TAIL),
    "--o": (Mark.TAIL, Mark.CIRCLE),
    "o->": (Mark.CIRCLE, Mark.ARROW),
    "<-o": (Mark.ARROW, Mark.CIRCLE),
    "o-o": (Mark.CIRCLE, Mark.CIRCLE),
    "->": (Mark.TAIL, Mark.ARROW),
    "<-": (Mark.ARROW, Mark.TAIL),
}


_CANONICAL_OPERATOR: dict[tuple[Mark, Mark], str] = {
    (Mark.TAIL, Mark.TAIL): "---",
    (Mark.TAIL, Mark.ARROW): "-->",
    (Mark.ARROW, Mark.TAIL): "<--",
    (Mark.ARROW, Mark.ARROW): "<->",
    (Mark.CIRCLE, Mark.TAIL): "o--",
    (Mark.TAIL, Mark.CIRCLE): "--o",
    (Mark.CIRCLE, Mark.ARROW): "o->",
    (Mark.ARROW, Mark.CIRCLE): "<-o",
    (Mark.CIRCLE, Mark.CIRCLE): "o-o",
}


_BARE_NAME = re.compile(r"[^\W\d]\w*", flags=re.UNICODE)


def _require_text(text: object) -> str:
    if not isinstance(text, str):
        raise GraphTextSyntaxError("graph input must be text")
    if text.startswith("\ufeff"):
        raise GraphTextSyntaxError(
            "graph text cannot begin with a byte-order mark"
        )
    return text


def _raise_syntax_error(text: str, error: UnexpectedInput) -> NoReturn:
    line = error.line if error.line is not None else 1
    column = error.column if error.column is not None else 1
    context = error.get_context(text, span=40).replace("\r\n", "\n").rstrip()
    raise GraphTextSyntaxError(
        f"malformed graph text at line {line}, column {column}:\n{context}"
    ) from None


def _raise_quoted_label_error(
    text: str,
    token: Token,
    error: json.JSONDecodeError,
) -> NoReturn:
    token_line = token.line if token.line is not None else 1
    token_column = token.column if token.column is not None else 1
    line = token_line + error.lineno - 1
    column = (
        token_column + error.colno - 1
        if error.lineno == 1
        else error.colno
    )
    lines = text.split("\n")
    source_line = lines[line - 1].removesuffix("\r") if 1 <= line <= len(lines) else ""
    display_offset = len(source_line[:column - 1].expandtabs())
    context = f"{source_line.expandtabs()}\n{' ' * display_offset}^"
    raise GraphTextSyntaxError(
        f"malformed graph text at line {line}, column {column}:\n{context}"
    ) from None


def _node_value(token: Token, *, text: str) -> str:
    if token.type == "BARE_NAME":
        return str(token)
    if token.type == "ESCAPED_STRING":
        try:
            value = json.loads(str(token))
        except json.JSONDecodeError as error:
            _raise_quoted_label_error(text, token, error)
        if not isinstance(value, str):
            raise RuntimeError("graph text string token did not decode to text")
        return value
    raise RuntimeError(f"unexpected graph text node token: {token.type!r}")


def _parse_statements(text: str) -> tuple[tuple[Token, ...], ...]:
    try:
        tree = _PARSER.parse(text)
    except UnexpectedInput as error:
        _raise_syntax_error(text, error)

    if not isinstance(tree, Tree) or tree.data != "start":
        raise RuntimeError("graph text parser returned an unexpected root")
    if not tree.children:
        return ()
    statement_list = tree.children[0]
    if not isinstance(statement_list, Tree) or statement_list.data != "statement_list":
        raise RuntimeError("graph text parser returned an unexpected statement list")

    statements: list[tuple[Token, ...]] = []
    for statement in statement_list.children:
        if not isinstance(statement, Tree) or statement.data != "statement":
            raise RuntimeError("graph text parser returned an unexpected statement")
        tokens: list[Token] = []
        for child in statement.children:
            if not isinstance(child, Token):
                raise RuntimeError("graph text statement contains a non-token value")
            tokens.append(child)
        if not tokens or len(tokens) % 2 == 0:
            raise RuntimeError("graph text statement has an invalid token shape")
        statements.append(tuple(tokens))
    return tuple(statements)


def _parse_graph(
    text: str,
    *,
    family: GraphTextFamily,
) -> DirectedBidirectedGraph | EndpointMarkedGraph:
    required_text = _require_text(text)
    statements = _parse_statements(required_text)

    nodes: list[str] = []
    seen_nodes: set[str] = set()
    directed: list[tuple[str, str]] = []
    bidirected: list[tuple[str, str]] = []
    endpoint_edges: list[tuple[str, Mark, str, Mark]] = []

    def add_node(node: str) -> None:
        if node not in seen_nodes:
            nodes.append(node)
            seen_nodes.add(node)

    for statement in statements:
        previous = _node_value(statement[0], text=required_text)
        add_node(previous)
        for index in range(1, len(statement), 2):
            operator = str(statement[index])
            current = _node_value(statement[index + 1], text=required_text)
            add_node(current)
            left_mark, right_mark = _OPERATOR_MARKS[operator]

            if family == "directed-bidirected":
                if (left_mark, right_mark) == (Mark.TAIL, Mark.ARROW):
                    directed.append((previous, current))
                elif (left_mark, right_mark) == (Mark.ARROW, Mark.TAIL):
                    directed.append((current, previous))
                elif (left_mark, right_mark) == (Mark.ARROW, Mark.ARROW):
                    bidirected.append((previous, current))
                else:
                    raise GraphValidationError(
                        f"operator {operator!r} requires endpoint-marked topology"
                    )
            else:
                endpoint_edges.append(
                    (previous, left_mark, current, right_mark)
                )
            previous = current

    if family == "directed-bidirected":
        return DirectedBidirectedGraph(
            nodes=nodes,
            directed=directed,
            bidirected=bidirected,
        )
    return EndpointMarkedGraph(nodes=nodes, edges=endpoint_edges)


def _directed_bidirected_graph_from_text(
    text: str,
) -> DirectedBidirectedGraph:
    value = _parse_graph(text, family="directed-bidirected")
    if not isinstance(value, DirectedBidirectedGraph):
        raise RuntimeError("directed-bidirected parser returned the wrong family")
    return value


def _endpoint_marked_graph_from_text(text: str) -> EndpointMarkedGraph:
    value = _parse_graph(text, family="endpoint-marked")
    if not isinstance(value, EndpointMarkedGraph):
        raise RuntimeError("endpoint-marked parser returned the wrong family")
    return value


@overload
def graph_from_text(
    text: str,
    *,
    family: Literal["directed-bidirected"] = "directed-bidirected",
) -> DirectedBidirectedGraph: ...


@overload
def graph_from_text(
    text: str,
    *,
    family: Literal["endpoint-marked"],
) -> EndpointMarkedGraph: ...


def graph_from_text(
    text: str,
    *,
    family: str = "directed-bidirected",
) -> DirectedBidirectedGraph | EndpointMarkedGraph:
    """Parse strict human-readable graph text into one raw topology family."""

    if family == "directed-bidirected":
        return _directed_bidirected_graph_from_text(text)
    if family == "endpoint-marked":
        return _endpoint_marked_graph_from_text(text)
    raise ValueError(f"unknown graph text family: {family!r}")


def _format_node(node: str) -> str:
    if _BARE_NAME.fullmatch(node) is not None:
        return node
    return json.dumps(node, ensure_ascii=False)


def graph_to_text(
    graph: DirectedBidirectedGraph | EndpointMarkedGraph,
) -> str:
    """Render deterministic human-readable text for one raw topology."""

    if isinstance(graph, DirectedBidirectedGraph):
        positions = {node: index for index, node in enumerate(graph.nodes)}
        lines = [_format_node(node) for node in graph.nodes]
        for source, target in sorted(
            graph.directed,
            key=lambda edge: (positions[edge[0]], positions[edge[1]]),
        ):
            lines.append(f"{_format_node(source)} --> {_format_node(target)}")
        for left, right in sorted(
            graph.bidirected,
            key=lambda edge: (positions[edge[0]], positions[edge[1]]),
        ):
            lines.append(f"{_format_node(left)} <-> {_format_node(right)}")
        return "\n".join(lines)

    if isinstance(graph, EndpointMarkedGraph):
        lines = [_format_node(node) for node in graph.nodes]
        for left, left_mark, right, right_mark in graph.edges:
            operator = _CANONICAL_OPERATOR[(left_mark, right_mark)]
            lines.append(
                f"{_format_node(left)} {operator} {_format_node(right)}"
            )
        return "\n".join(lines)

    raise TypeError("expected a supported raw graph topology")
