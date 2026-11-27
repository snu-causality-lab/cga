from itertools import combinations, permutations

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    DAG,
    DirectedBidirectedGraph,
    RecantingDistrictWitness,
    find_recanting_district,
)


def _admg(nodes, directed=(), bidirected=()):
    return AcyclicDirectedBidirectedGraph(
        DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=bidirected)
    )


def _longitudinal_graph(*, extra_directed=(), extra_bidirected=()):
    """Shpitser (2013), Figures 4 and 5, https://arxiv.org/pdf/1205.0241#page=41."""
    directed = (
        ("A0", "M1"), ("M1", "M2"), ("M1", "Y"), ("A1", "M2"), ("M2", "Y"),
        ("A0", "L1"), ("A0", "Y"), ("L1", "M1"), ("L1", "L2"), ("L1", "Y"),
        ("M1", "A1"), ("A1", "L2"), ("A1", "Y"), ("L2", "M2"), ("L2", "Y"),
    )
    bidirected = (("L1", "L2"), ("L1", "Y"), ("L2", "Y"))
    return _admg(
        ("A0", "L1", "M1", "A1", "L2", "M2", "Y"),
        directed + extra_directed,
        bidirected + extra_bidirected,
    )


def test_shpitser_longitudinal_figures_4_and_5():
    selected = (("A0", "M1", "Y"), ("A0", "M1", "M2", "Y"), ("A1", "M2", "Y"))
    query = dict(interventions=("A0", "A1"), outcomes=("Y",), selected_paths=selected)
    assert find_recanting_district(_longitudinal_graph(), **query) is None

    figure_5a = _longitudinal_graph(extra_bidirected=(("L2", "M2"),))
    witness = find_recanting_district(figure_5a, **query)
    assert isinstance(witness, RecantingDistrictWitness)
    assert witness.treatment == "A1"
    assert witness.district == ("L1", "L2", "M2", "Y")
    assert witness.selected_path == ("A1", "M2", "Y")
    assert witness.excluded_path in (("A1", "Y"), ("A1", "L2", "Y"), ("A1", "L2", "M2", "Y"))

    figure_5b = _longitudinal_graph(extra_directed=(("M1", "L2"),))
    witness = find_recanting_district(figure_5b, **query)
    assert isinstance(witness, RecantingDistrictWitness)
    assert witness.treatment == "A0"
    assert witness.district == ("M1",)
    assert witness.selected_path in selected[:2]
    assert witness.excluded_path in (("A0", "M1", "L2", "Y"), ("A0", "M1", "L2", "M2", "Y"))


def test_district_members_must_be_ancestors_via_an_untreated_path():
    graph = _admg(
        ("A", "M", "N", "B", "T", "Y"),
        (("A", "M"), ("A", "N"), ("M", "Y"), ("N", "Y"), ("B", "T"), ("T", "Y")),
        (("M", "B"), ("B", "N")),
    )
    selected = (("A", "M", "Y"),)
    assert find_recanting_district(
        graph, interventions=("A", "T"), outcomes=("Y",), selected_paths=selected
    ) is None
    witness = find_recanting_district(
        graph, interventions=("A",), outcomes=("Y",), selected_paths=selected
    )
    assert witness is not None
    assert witness.district == ("M", "N", "B")
    assert witness.excluded_path == ("A", "N", "Y")


def test_two_witness_paths_must_start_at_the_same_treatment():
    graph = _admg(
        ("A", "T", "M", "N", "Y"),
        (("A", "M"), ("T", "N"), ("M", "Y"), ("N", "Y")),
        (("M", "N"),),
    )
    assert find_recanting_district(
        graph, interventions=("A", "T"), outcomes=("Y",), selected_paths=(("A", "M", "Y"),)
    ) is None


@pytest.mark.parametrize(
    ("selected", "excluded"),
    [(("A", "B"), ("A", "B", "Y")), (("A", "B", "Y"), ("A", "B"))],
)
def test_paths_can_stop_at_or_continue_through_an_outcome(selected, excluded):
    graph = DAG.from_text("A -> B -> Y")
    witness = find_recanting_district(
        graph, interventions=("A",), outcomes=("B", "Y"), selected_paths=(selected,)
    )
    assert witness is not None
    assert witness.treatment == "A"
    assert witness.district == ("B",)
    assert witness.selected_path == selected
    assert witness.excluded_path == excluded


@pytest.mark.parametrize(
    "path",
    [(), ("A",), ("A", "T", "Y"), ("A", "B"), ("A", "Y"), ("B", "Y"), ("A", "missing", "Y")],
)
def test_rejects_selected_paths_outside_the_proper_directed_paths(path):
    graph = _admg(
        ("A", "T", "B", "Y"),
        (("A", "T"), ("T", "Y"), ("A", "B"), ("B", "Y")),
        (("A", "Y"),),
    )
    with pytest.raises(ValueError):
        find_recanting_district(
            graph, interventions=("A", "T"), outcomes=("Y",), selected_paths=(path,)
        )


@pytest.mark.parametrize(
    ("interventions", "outcomes"), [((), ("Y",)), (("A",), ()), (("A",), ("A",)), (("Z",), ("Y",)),
                                    (("A",), ("Z",))],
)
def test_rejects_invalid_query_even_with_empty_selection(interventions, outcomes):
    graph = DAG.from_text("A -> Y")
    with pytest.raises(ValueError):
        find_recanting_district(
            graph, interventions=interventions, outcomes=outcomes, selected_paths=()
        )


def _subsets(items):
    for size in range(len(items) + 1):
        yield from combinations(items, size)


def _paths(alphabet, edges):
    """Check every sequence of distinct vertices, independently of graph traversals."""
    return tuple(
        path
        for size in range(1, len(alphabet) + 1)
        for path in permutations(alphabet, size)
        if all(edge in edges for edge in zip(path[:-1], path[1:]))
    )


def _oracle_queries(nodes, arrows, spouses, active, outcomes):
    untreated = tuple(node for node in nodes if node not in active)
    live = {route[0] for route in _paths(untreated, arrows) if route[-1] in outcomes}
    paths = tuple(
        route
        for route in _paths(nodes, arrows)
        if route[0] in active and route[1:] and route[-1] in outcomes and set(route[1:]) <= live
    )
    # Compute bidirected reachability by enumerating simple paths in the induced graph.
    connections = set(spouses) | {(v, u) for u, v in spouses}
    routes = _paths(tuple(node for node in nodes if node in live), connections)
    districts = {
        node: tuple(other for other in nodes if any(route[0] == node and route[-1] == other for route in routes))
        for node in live
    }
    for chosen in _subsets(paths):
        witnesses = {
            (green[0], districts[green[1]], green, blue)
            for green in chosen
            for blue in paths
            if blue not in chosen and blue[0] == green[0] and blue[1] in districts[green[1]]
        }
        yield chosen, witnesses


def test_matches_definition_on_every_three_vertex_admg():
    nodes = ("A", "B", "Y")
    pairs = tuple(combinations(nodes, 2))
    for arrows in _subsets(pairs):
        for spouses in _subsets(pairs):
            graph = _admg(nodes, arrows, spouses)
            for active, outcomes in ((('A',), ('Y',)), (('A',), ('B', 'Y')), (('A', 'B'), ('Y',))):
                for selected, witnesses in _oracle_queries(nodes, arrows, spouses, active, outcomes):
                    result = find_recanting_district(
                        graph, interventions=active, outcomes=outcomes, selected_paths=selected
                    )
                    if result is None:
                        assert not witnesses
                    else:
                        assert (result.treatment, result.district, result.selected_path, result.excluded_path) in witnesses
