from fractions import Fraction
from itertools import combinations

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DirectedBidirectedGraph,
    GraphValidationError,
    MissingnessGraph,
    find_colluding_path,
    m_separated,
    m_separated_sets,
    missing_data_full_law_obstructions,
    mohan_binary_independence_interval,
    mohan_direct_test_witnesses,
    mohan_joint_recovery_factors,
)


def _model(nodes, directed=(), bidirected=(), responses=None):
    graph = AcyclicDirectedBidirectedGraph(
        DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=bidirected)
    )
    return MissingnessGraph(graph, {"X": "RX"} if responses is None else responses)


def test_cross_censoring_recovers_joint_under_mohan_2013():
    model = _model(
        ("X", "Y", "RX", "RY"), (("X", "Y"), ("Y", "RX"), ("X", "RY")),
        responses={"Y": "RY", "X": "RX"},
    )
    plan = mohan_joint_recovery_factors(model)
    assert plan is not None
    assert [
        (term.indicator, term.parents, term.observed_parent_indicators) for term in plan
    ] == [("RX", ("Y",), ("RY",)), ("RY", ("X",), ("RX",))]
    assert missing_data_full_law_obstructions(model) == {}

    self_censoring = _model(("X", "RX"), (("X", "RX"),))
    assert mohan_joint_recovery_factors(self_censoring) is None
    assert missing_data_full_law_obstructions(self_censoring) == {"X": ("X", "RX")}


def test_colluder_and_latent_collider_chain():
    model = _model(
        ("X", "Y", "RX", "RY"), (("X", "RY"), ("RX", "RY")),
        responses={"X": "RX", "Y": "RY"},
    )
    assert find_colluding_path(model, "X") == ("X", "RY", "RX")
    assert find_colluding_path(model, "Y") is None
    with pytest.raises(GraphValidationError, match="edges between response"):
        mohan_joint_recovery_factors(model)

    mixed = _model(("X", "A", "B", "RX"), (("X", "A"),),
                   (("A", "B"), ("B", "RX")))
    assert find_colluding_path(mixed, "X") == ("X", "A", "B", "RX")
    with pytest.raises(GraphValidationError, match="latent parents"):
        mohan_joint_recovery_factors(mixed)

    no_collider_chain = _model(
        ("X", "A", "B", "RX"), (("X", "A"), ("A", "B")), (("B", "RX"),)
    )
    assert find_colluding_path(no_collider_chain, "X") is None


def test_direct_testability_examples_from_mohan_pearl_2014():
    model = _model(
        ("X", "Y", "Z", "W", "RX", "RY", "RZ", "RW"),
        responses={"X": "RX", "Y": "RY", "Z": "RZ", "W": "RW"},
    )
    assert mohan_direct_test_witnesses(model, ("X", "RX"), ("Y", "RY"), ("Z", "RZ")) == ("Y",)
    assert mohan_direct_test_witnesses(model, ("X", "RX"), ("RW", "RY"), "Y") == ("RW",)
    assert mohan_direct_test_witnesses(
        model, ("X", "RY", "RZ"), ("Y", "RX", "RW"), ("Z", "W")
    ) == ()
    assert mohan_direct_test_witnesses(model, "X", "RX") == ()


@pytest.mark.parametrize("responses", [{"X": "Z"}, {"X": "X"}, {"X": "RX", "Y": "RX"}])
def test_rejects_invalid_response_mapping(responses):
    with pytest.raises(GraphValidationError):
        _model(("X", "Y", "RX"), responses=responses)


def test_rejects_response_pointing_to_a_true_variable():
    with pytest.raises(GraphValidationError, match="substantive"):
        _model(("X", "RX"), (("RX", "X"),))


def _signature(graph, a, b):
    """All endpoint-mark choices for a step, using original edge tests."""
    if (a, b) in graph.directed:
        yield False, True
    if (b, a) in graph.directed:
        yield True, False
    if (a, b) in graph.bidirected or (b, a) in graph.bidirected:
        yield True, True


def _definition_has_colluding_path(graph, x, r):
    def walk(path, incoming):
        if path[-1] == r:
            return True
        for node in graph.nodes:
            if node in path:
                continue
            for tail, head in _signature(graph, path[-1], node):
                if (len(path) == 1 or (incoming and tail)) and walk(path + (node,), head):
                    return True
        return False

    return walk((x,), False)


def test_colluding_path_agrees_with_marked_simple_path_definition():
    nodes = ("X", "A", "B", "RX")
    ordered = tuple(combinations(nodes, 2))
    for arrow_mask in range(1 << len(ordered)):
        arrows = tuple(e for i, e in enumerate(ordered) if arrow_mask >> i & 1)
        for arc_mask in range(1 << len(ordered)):
            arcs = tuple(e for i, e in enumerate(ordered) if arc_mask >> i & 1)
            model = _model(nodes, arrows, arcs)
            found = find_colluding_path(model, "X")
            assert (found is not None) == _definition_has_colluding_path(model.topology, "X", "RX")
            if found is not None:
                assert found[0] == "X" and found[-1] == "RX"
                assert len(found) == len(set(found))
                for i, (a, b) in enumerate(zip(found, found[1:])):
                    assert any(
                        (left or i == 0) and (right or i == len(found) - 2)
                        for left, right in _signature(model.topology, a, b)
                    )


def test_mohan_pearl_2014_table_2_binary_completion():
    assert mohan_binary_independence_interval((8, 6, 20), (4, 3, 40)) == (
        Fraction(4, 17), Fraction(14, 17)
    )
    assert mohan_binary_independence_interval((13, 11, 3), (7, 5, 2)) == (
        Fraction(1, 2), Fraction(16, 27)
    )
    assert mohan_binary_independence_interval((100, 5, 5), (7, 3, 5)) is None
    with pytest.raises(GraphValidationError, match="positive mass"):
        mohan_binary_independence_interval((0, 0, 0), (1, 1, 1))


def test_mohan_pearl_2018_figure_1b_response_collider():
    graph = _model(
        ("Y", "X", "RX", "RY"),
        (("Y", "X"), ("Y", "RX"), ("X", "RY"), ("RX", "RY")),
        responses={"X": "RX", "Y": "RY"},
    )
    view = AcyclicDirectedBidirectedGraph(graph.topology)
    assert m_separated(view, "RX", "X", given="Y")
    assert not m_separated_sets(view, "RX", "X", given=("Y", "RY"))
    assert m_separated_sets(view, "RY", "Y", given=("X", "RX"))
