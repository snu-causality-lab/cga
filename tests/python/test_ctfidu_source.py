from itertools import combinations

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    CTFIDUSource,
    CounterfactualDomainError,
    DAG,
    DirectedBidirectedGraph,
    find_ctfidu_source,
)

from .id_oracles import TinyADMG, _is_fixable
from .published_fixing_oracles import literal_reachable_states


def test_correa_lee_bareinboim_figure_5a():
    # Figure 5(a), equation (17), https://proceedings.neurips.cc/paper/2021/file/36bedb6eb7152f39b16328448942822b-Paper.pdf#page=7
    graph = AcyclicDirectedBidirectedGraph.from_text(
        "X --> Z; X <-> Z; X --> Y; Z --> Y"
    )
    assert find_ctfidu_source(graph, district="Z", experiment_interventions=[()]) is None
    regimes = (("Z",), (), ("X",))
    assert find_ctfidu_source(
        graph, district="Z", experiment_interventions=regimes
    ) == CTFIDUSource(source_index=2, interventions=("X",), fixing_sequence=("Y",))
    assert find_ctfidu_source(
        graph, district="Y", experiment_interventions=regimes[1:]
    ) == CTFIDUSource(source_index=0, interventions=(), fixing_sequence=("Z", "X"))


def test_connected_target_can_be_a_proper_subset_of_the_original_district():
    graph = AcyclicDirectedBidirectedGraph.from_text("A <-> B; B <-> C; A --> C")
    result = find_ctfidu_source(
        graph, district=("B", "C"), experiment_interventions=((), "A")
    )
    assert result == CTFIDUSource(source_index=1, interventions=("A",), fixing_sequence=())
    with pytest.raises(CounterfactualDomainError, match="connected"):
        find_ctfidu_source(graph, district=("A", "C"), experiment_interventions=[()])


def test_selects_first_regime_and_orders_its_interventions_by_graph_vertex_order():
    graph = DAG.from_text("B --> X; A --> X")
    regimes = (regime for regime in ({"A", "B"}, (), "A"))
    result = find_ctfidu_source(graph, district="X", experiment_interventions=regimes)
    assert result == CTFIDUSource(source_index=0, interventions=("B", "A"), fixing_sequence=())
    assert find_ctfidu_source(graph, district="X", experiment_interventions=[]) is None


def test_checks_every_regime_even_when_the_first_is_admissible():
    graph = DAG.from_text("A --> X")
    with pytest.raises(CounterfactualDomainError, match="unknown vertex"):
        find_ctfidu_source(graph, district="X", experiment_interventions=[(), "missing"])


@pytest.mark.parametrize("regimes", ["A", b"A", bytearray(b"A"), {"A"}, {"A": {}}, [3]])
def test_rejects_nonordered_or_malformed_regime_collections(regimes):
    graph = DAG.from_text("A --> X")
    with pytest.raises(CounterfactualDomainError):
        find_ctfidu_source(graph, district="X", experiment_interventions=regimes)


@pytest.mark.parametrize("district", [(), "missing"])
def test_requires_a_nonempty_known_district(district):
    graph = DAG.from_text("A --> X")
    with pytest.raises(CounterfactualDomainError):
        find_ctfidu_source(graph, district=district, experiment_interventions=[()])


def test_requires_validated_graph_view():
    graph = DirectedBidirectedGraph.from_text("X --> Y")
    with pytest.raises(CounterfactualDomainError):
        find_ctfidu_source(graph, district="Y", experiment_interventions=[()])


def _subsets(vertices):
    for size in range(len(vertices) + 1):
        yield from (frozenset(subset) for subset in combinations(vertices, size))


def test_matches_source_order_and_reachability_from_literal_fixing_orders():
    # The oracle uses integer edge atoms and enumerates all possible fixing orders.
    nodes = (0, 1, 2)
    pairs = tuple(combinations(nodes, 2))
    regimes = tuple(_subsets(nodes))
    labels = ("b", "c", "a")
    label_indices = {label: index for index, label in enumerate(labels)}
    interventions = tuple(
        tuple(labels[v] for v in nodes if v in regime) for regime in regimes
    )
    source_indices = tuple(range(len(regimes)))
    for directed in _subsets(pairs):
        for bidirected in _subsets(pairs):
            literal = TinyADMG(nodes, directed, bidirected)
            graph = AcyclicDirectedBidirectedGraph.from_graph(
                DirectedBidirectedGraph(
                    nodes=labels,
                    directed=((labels[a], labels[b]) for a, b in directed),
                    bidirected=((labels[a], labels[b]) for a, b in bidirected),
                )
            )
            reachable = []
            for regime in regimes:
                free = tuple(v for v in nodes if v not in regime)
                source = TinyADMG(
                    free,
                    frozenset((a, b) for a, b in directed if a in free and b in free),
                    frozenset((a, b) for a, b in bidirected if a in free and b in free),
                )
                states, _ = literal_reachable_states(source)
                reachable.append(states)
            _, connected = literal_reachable_states(TinyADMG(nodes, frozenset(), bidirected))
            for district in connected:
                target = tuple(labels[v] for v in nodes if v in district)
                for order in (source_indices, source_indices[::-1], (0,)):
                    sources = tuple(interventions[index] for index in order)
                    expected = next(
                        (i for i, index in enumerate(order) if district in reachable[index]),
                        None,
                    )
                    result = find_ctfidu_source(
                        graph, district=target, experiment_interventions=sources
                    )
                    if expected is None:
                        assert result is None
                    else:
                        assert result is not None
                        assert result.source_index == expected
                        assert result.interventions == sources[expected]
                        active = frozenset(nodes) - regimes[order[expected]]
                        for label in result.fixing_sequence:
                            vertex = label_indices[label]
                            assert vertex in active and _is_fixable(literal, vertex, active)
                            active -= {vertex}
                        assert active == district
