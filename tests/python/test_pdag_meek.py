from itertools import product

import pytest

from causal_graph_algorithms import DAG, DirectedBidirectedGraph, EndpointMarkedGraph, Mark
from causal_graph_algorithms.pdag import (
    CPDAG, MPDAG, PDAG, NoConsistentExtension, PDAGDomainError,
    find_consistent_dag_extension,
)
from causal_graph_algorithms.meek import BackgroundKnowledge, meek_closure
from causal_graph_algorithms.equivalence import dag_markov_equivalent
from tests.python.pdag_oracles import (
    consistent_members, dag_classes, invariant_edges,
)


def directed(raw):
    return frozenset((a, b) if x is Mark.TAIL else (b, a)
                     for a, x, b, y in raw.edges if x is not y)


def test_proof_views_reject_ambiguous_raw_marks_and_nonessential_graph():
    for text in ("A <-> B", "A o-o B", "A --> B --> C --> A"):
        with pytest.raises(PDAGDomainError):
            PDAG.from_graph(EndpointMarkedGraph.from_text(text))
    with pytest.raises(PDAGDomainError):
        CPDAG.from_graph(EndpointMarkedGraph.from_text("A --> B"))
    cpdag = CPDAG.from_dag(DAG.from_text("A --> B"))
    assert cpdag.topology.edge("A", "B") == (Mark.TAIL, Mark.TAIL)
    assert CPDAG.from_graph(cpdag.topology) == cpdag


def test_consistent_extension_and_background_preserve_original_colliders():
    cycle = PDAG.from_graph(EndpointMarkedGraph.from_text("A --- B --- C --- D --- A"))
    assert isinstance(find_consistent_dag_extension(cycle), NoConsistentExtension)
    assert isinstance(meek_closure(cycle), NoConsistentExtension)
    triangle = PDAG.from_graph(EndpointMarkedGraph.from_text("A --> B; B --- C --- A"))
    assert isinstance(find_consistent_dag_extension(triangle), DAG)
    chain = CPDAG.from_dag(DAG.from_text("A --> B --> C"))
    result = meek_closure(chain, knowledge=BackgroundKnowledge(
        required=(("A", "B"), ("C", "B"))))
    assert isinstance(result, NoConsistentExtension)
    assert result.reason == "background_inconsistent"
    assert result.knowledge == BackgroundKnowledge(required=(("A", "B"), ("C", "B")))
    result = meek_closure(chain, knowledge=BackgroundKnowledge(forbidden=(("B", "A"),)))
    assert isinstance(result, MPDAG)
    assert directed(result.topology) == {("A", "B"), ("B", "C")}
    assert isinstance(meek_closure(chain, knowledge=BackgroundKnowledge(
        required=(("A", "C"),))), NoConsistentExtension)
    assert isinstance(meek_closure(chain, knowledge=BackgroundKnowledge(
        forbidden=(("A", "B"), ("B", "A")))), NoConsistentExtension)
    with pytest.raises(PDAGDomainError):
        meek_closure(cycle, knowledge=BackgroundKnowledge(required=(("missing", "A"),)))


def test_all_four_vertex_dag_classes_have_exact_compelled_orientations():
    nodes = ("A", "B", "C", "D")
    for (skeleton, _), members in dag_classes(nodes).items():
        cpdag = CPDAG.from_dag(DAG.from_graph(DirectedBidirectedGraph(
            nodes=nodes, directed=members[0])))
        assert {frozenset((a, b)) for a, _, b, _ in cpdag.topology.edges} == skeleton
        assert directed(cpdag.topology) == invariant_edges(members)
        assert CPDAG.from_graph(cpdag.topology) == cpdag
        extension = find_consistent_dag_extension(cpdag)
        assert isinstance(extension, DAG) and extension.topology.directed in members
        for member in members:
            assert dag_markov_equivalent(extension, DAG.from_graph(
                DirectedBidirectedGraph(nodes=nodes, directed=member)))
        for a, b in tuple(tuple(s) for s in skeleton):
            for required in (((a, b),), ((b, a),)):
                knowledge = BackgroundKnowledge(required=required)
                expected = consistent_members(cpdag.topology, knowledge)
                result = meek_closure(cpdag, knowledge=knowledge)
                assert isinstance(result, NoConsistentExtension) == (not expected)
                if expected:
                    assert isinstance(result, MPDAG)
                    assert directed(result.topology) == invariant_edges(expected)


@pytest.mark.parametrize("nodes", [("A", "B", "C"), ("A", "B", "C", "D")])
def test_arbitrary_extendable_small_pdags(nodes):
    from itertools import combinations
    pairs = tuple(combinations(nodes, 2))
    for states in product(range(4), repeat=len(pairs)):
        edges = [(a, Mark.ARROW if s == 3 else Mark.TAIL,
                  b, Mark.ARROW if s == 2 else Mark.TAIL)
                 for (a, b), s in zip(pairs, states) if s]
        raw = EndpointMarkedGraph(nodes=nodes, edges=edges)
        try:
            graph = PDAG.from_graph(raw)
        except PDAGDomainError:
            continue
        expected = consistent_members(raw)
        result = meek_closure(graph)
        assert isinstance(result, NoConsistentExtension) == (not expected)
        if expected:
            assert directed(result.topology) == invariant_edges(expected)


def test_shield_includes_undirected_edges():
    raw = EndpointMarkedGraph.from_text("A --- B; A --> C <-- B")
    expected = consistent_members(raw)
    assert expected and len(expected) == 2
    result = meek_closure(PDAG.from_graph(raw))
    assert isinstance(result, MPDAG)
    assert result.topology == raw
