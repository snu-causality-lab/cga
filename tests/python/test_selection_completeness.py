"""Published completeness is fallible; RC failure is an unresolved API outcome."""
import json
from pathlib import Path

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph, DirectedBidirectedGraph, Probability,
    IdentifiedEffect, NonIdentifiedEffect, RecoveryNotEstablished,
    SelectionAdjustmentPair, find_selection_adjustment_pair, identify_causal_effect,
    recover_causal_effect, selection_adjustment_pair_is_valid,
)
from tests.python.selection_recovery_oracles import closure, separated


def test_six_vertex_rc_completeness_counterexample():
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/papers/"
                          "correa_tian_bareinboim_2019_rc.json").read_text())
    raw = DirectedBidirectedGraph(nodes=fixture["nodes"], directed=fixture["directed"],
                                 bidirected=fixture["bidirected"])
    g = AcyclicDirectedBidirectedGraph.from_graph(raw)
    data = Probability(("W", "X", "Y", "A", "B"), conditioned_on="S",
                       conditioned_values={"S": 1})
    reverse = tuple((b, a) for a, b in raw.directed)
    nodes = set(raw.nodes)
    assert closure({"Y"}, reverse, nodes - {"X"}) == {"Y"}
    assert closure({"Y"}, reverse, nodes) == {"W", "X", "Y"}
    assert closure({"S"}, reverse, nodes) == {"A", "B", "S"}
    # These literal bidirected components all meet An(S), so printed RC line 3 fails.
    assert {frozenset(e) for e in raw.bidirected} == {
        frozenset(("W", "Y")), frozenset(("W", "A")), frozenset(("X", "B"))}
    result = recover_causal_effect(g, data, interventions="X", outcomes="Y")
    assert isinstance(result, RecoveryNotEstablished) and result.reason == "rc_failed"
    assert isinstance(identify_causal_effect(g, interventions="X", outcomes="Y"),
                      IdentifiedEffect)
    assert selection_adjustment_pair_is_valid(g, data, interventions="X", outcomes="Y",
                                             adjustment=("W", "A"))
    pair = find_selection_adjustment_pair(g, data, interventions="X", outcomes="Y")
    assert isinstance(pair, SelectionAdjustmentPair) and pair.population_adjustment == ()
    # Independent literal paths establish the two do-calculus premises.
    bar_x = DirectedBidirectedGraph(nodes=raw.nodes,
        directed=tuple((a, b) for a, b in raw.directed if b != "X"),
        bidirected=tuple(e for e in raw.bidirected if "X" not in e))
    under_x = tuple(e for e in raw.directed if e != ("X", "Y"))
    assert separated(bar_x, {"Y"}, {"S"}, {"X"})
    assert separated(raw, {"X"}, {"Y"}, {"W", "A", "S"}, under_x)
    assert separated(raw, {"Y"}, {"S"}, set(), under_x)
    assert not separated(raw, {"X"}, {"Y"}, {"W", "S"}, under_x)


def test_selection_stop_does_not_imply_proper_backdoor_connection():
    raw = DirectedBidirectedGraph(nodes=("X", "Y", "W", "S"),
        directed=(("X", "Y"), ("W", "S")), bidirected=(("X", "Y"), ("X", "W")))
    g = AcyclicDirectedBidirectedGraph.from_graph(raw)
    data = Probability(("X", "Y", "W"), conditioned_on="S", conditioned_values={"S": 1})
    result = recover_causal_effect(g, data, interventions="X", outcomes="Y")
    assert isinstance(result, RecoveryNotEstablished) and result.reason == "rc_failed"
    # This smaller graph only refutes a proof implication: ordinary ID already fails.
    assert isinstance(identify_causal_effect(g, interventions="X", outcomes="Y"),
                      NonIdentifiedEffect)
    assert separated(raw, {"Y"}, {"S"}, set(), (("W", "S"),))
