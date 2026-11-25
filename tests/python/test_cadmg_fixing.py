from __future__ import annotations

import copy
import itertools
import json
import pickle
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph,
    CADMG, CADMGKernel, FixingDomainError, FixingBudgetExceeded,
    ValidFixingSequence, UnreachableCADMG, fixable_vertices, fix_vertices,
    fixing_sequence_is_valid, find_fixing_sequence, intrinsic_sets,
    intrinsic_set_is_valid, fix_kernel, Probability,
)
from .published_fixing_oracles import literal_reachable_states
from .gid_oracles import evaluate_generalized_probability_expression
from .id_oracles import TinyADMG, exact_binary_interventional_probability


def _verma():
    return CADMG.from_graph(AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
        nodes=("1","2","3","4"),
        directed=(("1","2"),("2","3"),("3","4"),("1","3")),
        bidirected=(("2","4"),))))


def test_cadmg_partition_and_no_arrowhead_at_fixed_validation():
    graph = DAG.from_text("W --> A --> B")
    value = CADMG.from_graph(graph, random_vertices=("A", "B"))
    assert value.random_vertices == ("A", "B") and value.fixed_vertices == ("W",)
    with pytest.raises(FixingDomainError):
        CADMG.from_graph(graph, random_vertices=("W", "B"))
    with pytest.raises(FixingDomainError):
        CADMG.from_graph(AcyclicDirectedBidirectedGraph.from_text("A <-> W"), random_vertices="A")
    with pytest.raises(FixingDomainError):
        CADMG.from_graph(DirectedBidirectedGraph(nodes=("A",)))


def test_richardson_figure3_fixability_and_each_printed_sequence():
    graph = _verma()
    assert fixable_vertices(graph) == ("1", "3", "4")
    assert not fixing_sequence_is_valid(graph, ("2",))
    for sequence in (("1","3","2"),("3","1","2"),("3","2","1")):
        assert fixing_sequence_is_valid(graph, sequence)
        result = fix_vertices(graph, sequence)
        assert result.random_vertices == ("4",)
        assert result.fixed_vertices == ("1","2","3")
        assert result.topology.directed == frozenset({("3","4")})
        assert not result.topology.bidirected
    assert fix_vertices(graph, ("3","1")) == fix_vertices(graph, ("1","3"))


def test_sequences_preserve_order_reject_repeated_fixed_and_unknown_nodes():
    graph = _verma()
    assert not fixing_sequence_is_valid(graph, ("3", "3"))
    with pytest.raises(FixingDomainError):
        fix_vertices(graph, ("2",))
    with pytest.raises(FixingDomainError):
        fixing_sequence_is_valid(graph, ("absent",))
    assert fix_vertices(graph, ()) is graph
    assert fix_vertices(graph, "3").fixed_vertices == ("3",)
    assert copy.copy(graph) is graph and copy.deepcopy(graph) is graph
    with pytest.raises(TypeError):
        pickle.dumps(graph)
    with pytest.raises(TypeError):
        CADMG(object())


@pytest.mark.parametrize("sequence", [{"1","3"}, frozenset({"1","3"}), {"1":None,"3":None}])
def test_unordered_fixing_sequences_are_rejected(sequence):
    graph = _verma()
    with pytest.raises(FixingDomainError):
        fixing_sequence_is_valid(graph,sequence)
    with pytest.raises(FixingDomainError):
        fix_vertices(graph,sequence)
    kernel = CADMGKernel.from_expression(graph,Probability(graph.topology.nodes))
    with pytest.raises(FixingDomainError):
        fix_kernel(kernel,sequence)


def test_reachability_and_intrinsic_membership_are_distinct():
    graph = _verma()
    result = find_fixing_sequence(graph, random_vertices="4")
    assert isinstance(result, ValidFixingSequence)
    assert fix_vertices(graph, result.sequence) == result.result
    blocked = CADMG.from_graph(AcyclicDirectedBidirectedGraph.from_text("X --> Y; X <-> Y"))
    assert isinstance(find_fixing_sequence(blocked, random_vertices="Y"), UnreachableCADMG)
    assert intrinsic_set_is_valid(blocked, "Y") is False
    assert intrinsic_set_is_valid(blocked, ("X", "Y")) is True
    assert intrinsic_set_is_valid(graph, ()) is False
    assert not intrinsic_set_is_valid(graph, ("1", "3"))


def test_atomic_intrinsic_enumeration_budget_and_input_errors():
    graph = _verma()
    with pytest.raises(FixingBudgetExceeded):
        intrinsic_sets(graph, max_states=1)
    for bad in (True, 0, -1, 1.5):
        with pytest.raises(FixingDomainError):
            intrinsic_sets(graph, max_states=bad)
    with pytest.raises(FixingDomainError):
        intrinsic_sets(DAG.from_text("X --> Y"))
    with pytest.raises(FixingDomainError):
        find_fixing_sequence(fix_vertices(graph,"1"), random_vertices="1")


def test_fixing_all_small_admgs_matches_literal_sequence_enumeration():
    from .id_oracles import all_bow_admgs_n3
    labels = ("a", "b", "c")
    for tiny in all_bow_admgs_n3():
        graph = CADMG.from_graph(AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
            nodes=labels, directed=((labels[a],labels[b]) for a,b in tiny.directed),
            bidirected=((labels[a],labels[b]) for a,b in tiny.bidirected))))
        states, districts = literal_reachable_states(tiny)
        expected = {tuple(labels[i] for i in sorted(d)) for d in districts}
        assert set(intrinsic_sets(graph)) == expected
        for mask in range(8):
            target = frozenset(i for i in range(3) if mask & (1 << i))
            result = find_fixing_sequence(graph, random_vertices=tuple(labels[i] for i in target))
            assert isinstance(result, ValidFixingSequence) == (target in states)


def test_richardson_figure3_kernel_fixing_equals_interventional_kernel():
    graph = _verma()
    labels = graph.topology.nodes
    positions = {v:i for i,v in enumerate(labels)}
    tiny = TinyADMG(tuple(range(4)), frozenset((positions[a],positions[b]) for a,b in graph.topology.directed),
        frozenset(tuple(sorted((positions[a],positions[b]))) for a,b in graph.topology.bidirected))
    kernel = CADMGKernel.from_expression(graph, Probability(labels))
    result = fix_kernel(kernel, ("3","1","2"))
    assert result.graph.random_vertices == ("4",)
    for bits in itertools.product((0,1), repeat=4):
        actual = evaluate_generalized_probability_expression(result.formula, graph=tiny,
            node_labels=labels, experiment_interventions=(frozenset(),),
            environment=dict(zip(labels,bits)), variant=0)
        expected = exact_binary_interventional_probability(tiny, outcomes=(3,), interventions=(0,1,2),
            values=dict(enumerate(bits)), variant=0)
        assert actual == expected
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/papers/richardson_2023_fixing.json").read_text())
    assert fixture["figure_3"]["valid_sequences"] == [["1","3","2"],["3","1","2"],["3","2","1"]]


def test_kernel_never_sums_fixed_parameters_and_checks_formula_scope():
    graph = CADMG.from_graph(DAG.from_text("W --> A --> B"), random_vertices=("A","B"))
    source = CADMGKernel.from_expression(graph, Probability(("A","B"), interventions="W"))
    fixed = fix_kernel(source, "A")
    assert fixed.graph.fixed_vertices == ("W","A")
    assert fixed.formula.free_variables <= {"W","A","B"}
    with pytest.raises(FixingDomainError):
        CADMGKernel.from_expression(graph, Probability("outside"))
    with pytest.raises(FixingDomainError):
        CADMGKernel.from_expression(graph, Probability("A", interventions="W"))


def test_fixed_root_kernel_formula_matches_direct_intervention_exactly():
    labels = ("W","A","B")
    graph = CADMG.from_graph(DAG.from_text("W --> A --> B"), random_vertices=("A","B"))
    source = CADMGKernel.from_expression(graph, Probability(("A","B"), interventions="W"))
    fixed = fix_kernel(source,"A")
    tiny = TinyADMG((0,1,2),frozenset({(0,1),(1,2)}),frozenset())
    for bits in itertools.product((0,1),repeat=3):
        actual = evaluate_generalized_probability_expression(fixed.formula,graph=tiny,
            node_labels=labels,experiment_interventions=(frozenset({0}),),
            environment=dict(zip(labels,bits)),variant=1)
        expected = exact_binary_interventional_probability(tiny,outcomes=(2,),
            interventions=(0,1),values=dict(enumerate(bits)),variant=1)
        assert actual == expected
