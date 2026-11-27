from itertools import combinations

import pytest

from causal_graph_algorithms import (
    ADMGHeadTail,
    AcyclicDirectedBidirectedGraph,
    CIStatement,
    DAG,
    DirectedBidirectedGraph,
    GraphValidationError,
    admg_head_tail_partition,
    admg_ordered_local_ci,
)

from .id_oracles import _directed_path_exists, _path_districts
from .oracles import m_separated_by_canonical_dag


def test_richardson_figure_1_all_ancestral_margins():
    # Richardson (2009), p. 470, https://proceedings.mlr.press/r7/richardson09a.html
    nodes = ("X1", "X2", "X3", "X4")
    graph = AcyclicDirectedBidirectedGraph(
        DirectedBidirectedGraph(
            nodes=nodes,
            directed=(("X1", "X3"), ("X2", "X4")),
            bidirected=(("X3", "X4"),),
        )
    )
    x1 = ADMGHeadTail(("X1",), ())
    x2 = ADMGHeadTail(("X2",), ())
    x3 = ADMGHeadTail(("X3",), ("X1",))
    x4 = ADMGHeadTail(("X4",), ("X2",))
    joint = ADMGHeadTail(("X3", "X4"), ("X1", "X2"))
    expected = {
        (): (),
        ("X1",): (x1,),
        ("X2",): (x2,),
        ("X1", "X2"): (x1, x2),
        ("X1", "X3"): (x1, x3),
        ("X2", "X4"): (x2, x4),
        ("X1", "X2", "X3"): (x1, x2, x3),
        ("X1", "X2", "X4"): (x1, x2, x4),
        nodes: (x1, x2, joint),
    }
    for margin, factors in expected.items():
        assert admg_head_tail_partition(graph, ancestral_vertices=reversed(margin)) == factors
    assert admg_head_tail_partition(graph) == expected[nodes]
    assert admg_ordered_local_ci(
        graph, "X3", ancestral_vertices=("X1", "X2", "X3")
    ) == CIStatement("X3", "X2", "X1")


def test_richardson_figure_2_tails_condition_on_other_heads():
    graph = AcyclicDirectedBidirectedGraph(
        DirectedBidirectedGraph(
            nodes=("X1", "X2", "X3", "X4"),
            directed=(("X1", "X3"), ("X2", "X4")),
            bidirected=(("X1", "X4"), ("X2", "X3")),
        )
    )
    assert admg_head_tail_partition(graph) == (
        ADMGHeadTail(("X1", "X4"), ("X2",)),
        ADMGHeadTail(("X2", "X3"), ("X1",)),
    )


def test_barrenness_uses_full_ancestry_outside_the_current_district():
    graph = AcyclicDirectedBidirectedGraph.from_text("X --> M --> Y; X <-> Y")
    assert admg_head_tail_partition(graph) == (
        ADMGHeadTail(("Y",), ("X", "M")),
        ADMGHeadTail(("M",), ("X",)),
        ADMGHeadTail(("X",), ()),
    )
    graph = AcyclicDirectedBidirectedGraph.from_text(
        "X --> M --> Y --> Z; X <-> Y; Y <-> Z"
    )
    assert admg_head_tail_partition(graph) == (
        ADMGHeadTail(("Z",), ("X", "M", "Y")),
        ADMGHeadTail(("M",), ("X",)),
        ADMGHeadTail(("Y",), ("X", "M")),
        ADMGHeadTail(("X",), ()),
    )


def test_markov_blanket_includes_the_whole_induced_district_and_its_parents():
    graph = AcyclicDirectedBidirectedGraph.from_text(
        "X <-> Y <-> Z; P --> Y; Q"
    )
    assert admg_ordered_local_ci(
        graph, "X", ancestral_vertices=("X", "Y", "Z", "P", "Q")
    ) == CIStatement("X", "Q", ("Y", "Z", "P"))
    assert admg_ordered_local_ci(
        graph, "X", ancestral_vertices=("X", "Z")
    ) == CIStatement("X", "Z")
    assert admg_ordered_local_ci(graph, "X", ancestral_vertices="X").trivial


def test_dag_head_tail_partition_reduces_to_singletons_and_parents():
    graph = DAG.from_text("X --> M --> Y; X --> Y")
    assert admg_head_tail_partition(graph) == (
        ADMGHeadTail(("X",), ()),
        ADMGHeadTail(("M",), ("X",)),
        ADMGHeadTail(("Y",), ("X", "M")),
    )


def test_validates_the_graph_ancestral_set_and_local_vertex():
    graph = DAG.from_text("X --> Y; Z")
    with pytest.raises(GraphValidationError, match="validated"):
        admg_head_tail_partition(graph.topology)
    with pytest.raises(GraphValidationError, match="validated"):
        admg_ordered_local_ci(graph.topology, "X", ancestral_vertices="X")
    with pytest.raises(GraphValidationError, match="ancestor-closed"):
        admg_head_tail_partition(graph, ancestral_vertices="Y")
    with pytest.raises(GraphValidationError, match="ancestor-closed"):
        admg_ordered_local_ci(graph, "Y", ancestral_vertices="Y")
    with pytest.raises(GraphValidationError, match="child"):
        admg_ordered_local_ci(graph, "X", ancestral_vertices=("X", "Y"))
    with pytest.raises(GraphValidationError, match="belong"):
        admg_ordered_local_ci(graph, "Z", ancestral_vertices="X")
    with pytest.raises(GraphValidationError, match="unknown vertex"):
        admg_head_tail_partition(graph, ancestral_vertices="unknown")
    with pytest.raises(GraphValidationError, match="unknown vertex"):
        admg_ordered_local_ci(graph, "unknown", ancestral_vertices="X")


def _subsets(nodes):
    for size in range(len(nodes) + 1):
        yield from (frozenset(subset) for subset in combinations(nodes, size))


def _literal_head_tail_pairs(nodes, directed, bidirected, margin):
    reach = {
        node: {other for other in nodes if _directed_path_exists(directed, node, other, nodes)}
        for node in nodes
    }

    def ancestors(targets):
        return frozenset(node for node in nodes if reach[node] & targets)

    def parents(targets):
        return {parent for parent, child in directed if child in targets}

    result = []
    remaining = margin
    while remaining:
        heads = set()
        for district in _path_districts(remaining, bidirected):
            # Richardson's original Phi operation takes barren(an(district)).
            closure = ancestors(district)
            head = frozenset(node for node in closure if reach[node] & closure == {node})
            component = next(
                group for group in _path_districts(ancestors(head), bidirected)
                if head <= group
            )
            result.append((head, (component - head) | parents(component)))
            heads.update(head)
        remaining -= heads
    return result


def test_ordered_three_node_admgs_against_independent_oracles():
    labels = ("b", "c", "a")
    nodes = frozenset(range(len(labels)))
    pairs = tuple(combinations(range(len(labels)), 2))

    def names(indices):
        return tuple(labels[index] for index in sorted(indices))

    for directed in _subsets(pairs):
        for bidirected in _subsets(pairs):
            graph = AcyclicDirectedBidirectedGraph(
                DirectedBidirectedGraph(
                    nodes=labels,
                    directed=((labels[a], labels[b]) for a, b in directed),
                    bidirected=((labels[a], labels[b]) for a, b in bidirected),
                )
            )
            atoms = tuple((labels[a], False, labels[b], True) for a, b in directed) + tuple(
                (labels[a], True, labels[b], True) for a, b in bidirected
            )
            for margin in _subsets(range(len(labels))):
                if any(b in margin and a not in margin for a, b in directed):
                    continue
                expected = tuple(
                    ADMGHeadTail(names(head), names(tail))
                    for head, tail in _literal_head_tail_pairs(
                        nodes, directed, bidirected, margin
                    )
                )
                margin_names = names(margin)
                assert admg_head_tail_partition(graph, ancestral_vertices=margin_names) == expected
                for vertex in margin:
                    if any(a == vertex and b in margin for a, b in directed):
                        continue
                    statement = admg_ordered_local_ci(
                        graph, labels[vertex], ancestral_vertices=margin_names
                    )
                    district = next(group for group in _path_districts(margin, bidirected) if vertex in group)
                    blanket = (district - {vertex}) | {
                        parent for parent, child in directed if child in district
                    }
                    assert statement == CIStatement(
                        labels[vertex], names(margin - blanket - {vertex}), names(blanket)
                    )
                    # The oracle expands latent roots and moralizes the full ancestral DAG.
                    assert m_separated_by_canonical_dag(
                        labels, atoms, statement.left, statement.right, statement.given
                    )
