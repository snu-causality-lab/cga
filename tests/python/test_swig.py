from itertools import combinations, permutations

import pytest

from causal_graph_algorithms import (
    DAG,
    DirectedBidirectedGraph,
    GraphValidationError,
    SWIGTemplate,
    SWIGVertex,
    swig_template,
)


def test_richardson_robins_figure_2_template():
    # Figure 2(a,b), Richardson and Robins (2013), https://csss.uw.edu/Papers/2013/wp128.pdf#page=10
    template = swig_template(DAG.from_text("X --> Y"), interventions="X")
    assert isinstance(template, SWIGTemplate)
    assert template.random_vertices == (SWIGVertex("X"), SWIGVertex("Y"))
    assert template.fixed_vertices == (SWIGVertex("X", is_fixed=True),)
    assert template.directed == ((SWIGVertex("X", is_fixed=True), SWIGVertex("Y")),)
    assert template.fixed_ancestors("X") == ()
    assert template.fixed_ancestors("Y") == ("X",)


def test_richardson_robins_figure_10_successive_interventions():
    # Figure 10(iv), https://csss.uw.edu/Papers/2013/wp128.pdf#page=30
    template = swig_template(DAG.from_text("Z --> M --> Y"), interventions=("Z", "M"))
    assert template.directed == (
        (SWIGVertex("Z", is_fixed=True), SWIGVertex("M")),
        (SWIGVertex("M", is_fixed=True), SWIGVertex("Y")),
    )
    assert template.fixed_ancestors("Z") == ()
    assert template.fixed_ancestors("M") == ("Z",)
    assert template.fixed_ancestors("Y") == ("M",)


def test_vertex_kinds_distinguish_names_and_label_order_follows_input_graph():
    graph = DAG(
        DirectedBidirectedGraph(
            nodes=("Y", "B", "A", "A(fixed)", "isolated"),
            directed=(("A", "Y"), ("B", "Y"), ("A(fixed)", "A")),
        )
    )
    template = swig_template(graph, interventions=("isolated", "A", "B", "B"))
    reordered = swig_template(graph, interventions=("A", "B", "isolated"))
    assert template == reordered
    assert template.fixed_ancestors("Y") == ("B", "A")
    assert template.fixed_vertices == (
        SWIGVertex("B", is_fixed=True),
        SWIGVertex("A", is_fixed=True),
        SWIGVertex("isolated", is_fixed=True),
    )
    assert SWIGVertex("A(fixed)") != SWIGVertex("A", is_fixed=True)
    assert (SWIGVertex("A(fixed)"), SWIGVertex("A")) in template.directed
    assert all(vertex.variable != "isolated" for edge in template.directed for vertex in edge)


def test_empty_intervention_retains_random_topology():
    graph = DAG.from_text("B --> A; C")
    template = swig_template(graph, interventions=())
    assert template.fixed_vertices == ()
    assert template.random_vertices == tuple(SWIGVertex(v) for v in graph.topology.nodes)
    assert template.directed == ((SWIGVertex("B"), SWIGVertex("A")),)
    assert all(template.fixed_ancestors(v) == () for v in graph.topology.nodes)
    empty = swig_template(DAG(DirectedBidirectedGraph(nodes=())), interventions=())
    assert empty.random_vertices == empty.fixed_vertices == empty.directed == ()


def test_requires_validated_dag_and_known_targets():
    graph = DAG.from_text("A --> B")
    with pytest.raises(GraphValidationError, match="DAG"):
        swig_template(graph.topology, interventions="A")
    with pytest.raises(GraphValidationError, match="missing"):
        swig_template(graph, interventions="missing")
    template = swig_template(graph, interventions="A")
    with pytest.raises(GraphValidationError, match="missing"):
        template.fixed_ancestors("missing")


def _subsets(items):
    items = tuple(items)
    for size in range(len(items) + 1):
        yield from combinations(items, size)


def _has_label(edges, treatments, start, end, nodes):
    if start == end:
        return False
    middle = tuple(v for v in nodes if v not in treatments and v != end)
    for length in range(len(middle) + 1):
        for candidates in permutations(middle, length):
            path = (start, *candidates, end)
            if all(edge in edges for edge in zip(path, path[1:])):
                return True
    return False


def test_every_three_vertex_dag_agrees_with_directed_path_oracle():
    nodes = ("Z", "X", "Y")
    dags = {
        frozenset(edges)
        for order in permutations(nodes)
        for edges in _subsets(combinations(order, 2))
    }
    for edges in dags:
        graph = DAG(DirectedBidirectedGraph(nodes=nodes, directed=edges))
        for interventions in _subsets(nodes):
            result = swig_template(graph, interventions=interventions)
            for vertex in nodes:
                assert result.fixed_ancestors(vertex) == tuple(
                    start
                    for start in interventions
                    if _has_label(edges, interventions, start, vertex, nodes)
                )
