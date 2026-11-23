from itertools import combinations

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph, CausalPAG, DAG, DirectedBidirectedGraph,
    EndpointMarkedGraph, MaximalAncestralGraph,
)
from causal_graph_algorithms.adjustment import (
    AdjustmentDomainError, AdjustmentNotFound, AdjustmentSet,
    adjustment_set_is_valid, find_adjustment_set, find_minimal_adjustment_set,
)
from causal_graph_algorithms.pdag import CPDAG
from tests.python.pdag_oracles import dag_classes
from tests.python.test_separators import subsets
from tests.python.oracles import no_selection_census
from tests.python.gac_oracles import endpoint_atoms, latent_dag_atoms, literal_gac


def test_graph_interpretations_and_bows():
    dag = DAG.from_text("X --> Y")
    mag = MaximalAncestralGraph.from_text("X --> Y")
    assert adjustment_set_is_valid(dag, interventions="X", outcomes="Y", adjustment=())
    assert not adjustment_set_is_valid(mag, interventions="X", outcomes="Y", adjustment=())
    assert isinstance(find_adjustment_set(mag, interventions="X", outcomes="Y"),
                      AdjustmentNotFound)
    bow = AcyclicDirectedBidirectedGraph.from_text("X --> Y; X <-> Y")
    assert not adjustment_set_is_valid(bow, interventions="X", outcomes="Y", adjustment=())
    visible = MaximalAncestralGraph.from_text("Z --> X --> Y")
    assert adjustment_set_is_valid(visible, interventions="X", outcomes="Y", adjustment=())
    with pytest.raises(AdjustmentDomainError):
        adjustment_set_is_valid(MaximalAncestralGraph.from_text("X --- Y"),
                                interventions="X", outcomes="Y", adjustment=())
    with pytest.raises(AdjustmentDomainError):
        find_adjustment_set(EndpointMarkedGraph.from_text("X --> Y"),
                            interventions="X", outcomes="Y")


def test_constraints_total_sets_and_forbidden_required():
    graph = DAG.from_text("X <-- C --> Y; X --> M --> Y")
    result = find_minimal_adjustment_set(graph, interventions="X", outcomes="Y", required="C")
    assert isinstance(result, AdjustmentSet) and result.vertices == ("C",)
    assert result.required == ("C",)
    assert isinstance(find_adjustment_set(graph, interventions="X", outcomes="Y", required="M"),
                      AdjustmentNotFound)
    assert isinstance(find_adjustment_set(graph, interventions="X", outcomes="Y", allowed=()),
                      AdjustmentNotFound)
    with pytest.raises(AdjustmentDomainError):
        find_adjustment_set(graph, interventions="X", outcomes="Y", required="C", allowed=())
    assert not adjustment_set_is_valid(graph, interventions="X", outcomes="Y", adjustment="M")


def test_every_four_vertex_cpdag_query_against_all_member_dags():
    nodes = ("A", "B", "C", "D")
    for _, members in dag_classes(nodes).items():
        seed = DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=members[0]))
        graph = CPDAG.from_dag(seed)
        dag_edges = tuple(
            tuple((a, "tail", b, "arrow") for a, b in directed)
            for directed in members
        )
        for x, y in combinations(nodes, 2):
            for x, y in ((x, y), (y, x)):
                available = tuple(v for v in nodes if v not in (x, y))
                valid = []
                for z in subsets(available):
                    expected = all(
                        literal_gac(nodes, edges, {x}, {y}, z, "dag")
                        for edges in dag_edges
                    )
                    assert literal_gac(nodes, endpoint_atoms(graph, cpdag=True),
                                       {x}, {y}, z, "cpdag") == expected
                    assert adjustment_set_is_valid(graph, interventions=x, outcomes=y,
                                                   adjustment=z) == expected
                    if expected:
                        valid.append(z)
                result = find_minimal_adjustment_set(graph, interventions=x, outcomes=y)
                assert isinstance(result, AdjustmentNotFound) == (not valid)
                if valid:
                    assert frozenset(result.vertices) in valid
                    assert not any(z < frozenset(result.vertices) for z in valid)


def test_pag_amenability_and_possible_forbidden_vertices():
    pag = CausalPAG.from_mag(MaximalAncestralGraph.from_text("Z --> X --> Y; W --> X; W --> Y"))
    result = find_adjustment_set(pag, interventions="X", outcomes="Y")
    if isinstance(result, AdjustmentSet):
        assert adjustment_set_is_valid(pag, interventions="X", outcomes="Y",
                                       adjustment=result.vertices)
    pag = CausalPAG.from_mag(MaximalAncestralGraph.from_text("X --> M --> Y"))
    assert isinstance(find_adjustment_set(pag, interventions="X", outcomes="Y"),
                      AdjustmentNotFound)


def test_every_three_vertex_causal_pag_against_its_entire_mag_class():
    from causal_graph_algorithms import Mark
    nodes = ("A", "B", "C")
    for _, (members, _) in no_selection_census(nodes).classes.items():
        mags = tuple(MaximalAncestralGraph.from_graph(EndpointMarkedGraph(
            nodes=nodes, edges=((a, Mark.ARROW if x else Mark.TAIL,
                                 b, Mark.ARROW if y else Mark.TAIL) for a, x, b, y in member)))
                     for member in members)
        pag = CausalPAG.from_mag(mags[0])
        for x, y in combinations(nodes, 2):
            for x, y in ((x, y), (y, x)):
                candidates = tuple(v for v in nodes if v not in (x, y))
                valid = []
                for z in subsets(candidates):
                    expected = all(adjustment_set_is_valid(mag, interventions=x,
                        outcomes=y, adjustment=z) for mag in mags)
                    assert literal_gac(nodes, endpoint_atoms(pag), {x}, {y}, z, "pag") == expected
                    for mag in mags:
                        assert adjustment_set_is_valid(mag, interventions=x, outcomes=y,
                            adjustment=z) == literal_gac(nodes, endpoint_atoms(mag), {x}, {y}, z, "mag")
                    assert adjustment_set_is_valid(pag, interventions=x, outcomes=y,
                                                   adjustment=z) == expected
                    if expected:
                        valid.append(z)
                result = find_minimal_adjustment_set(pag, interventions=x, outcomes=y)
                assert isinstance(result, AdjustmentNotFound) == (not valid)
                if valid:
                    assert frozenset(result.vertices) in valid


def test_visible_pag_edge_and_mag_witness_are_bound_to_the_input():
    from causal_graph_algorithms.visibility import find_mag_directed_edge_visibility_witness
    mag = MaximalAncestralGraph.from_text("W --> X <-- Z; X --> Y")
    witness = find_mag_directed_edge_visibility_witness(mag, "X", "Y")
    assert witness.graph is mag
    assert witness.path in (("W", "X"), ("Z", "X"))
    pag = CausalPAG.from_mag(mag)
    result = find_minimal_adjustment_set(pag, interventions="X", outcomes="Y")
    assert isinstance(result, AdjustmentSet) and result.vertices == ()


def test_known_admg_with_bows_against_latent_dag_path_definition():
    from itertools import product
    nodes = ("A", "B", "C")
    pairs = tuple(combinations(nodes, 2))
    for directions in product(range(3), repeat=3):
        directed = [(a, b) if s == 1 else (b, a)
                    for (a, b), s in zip(pairs, directions) if s]
        for flags in product((False, True), repeat=3):
            try:
                graph = AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
                    nodes=nodes, directed=directed, bidirected=[p for p, f in zip(pairs, flags) if f]))
            except ValueError:
                continue
            expanded, edges = latent_dag_atoms(graph)
            for x, y in pairs:
                for x, y in ((x, y), (y, x)):
                    for z in subsets(tuple(v for v in nodes if v not in (x, y))):
                        assert adjustment_set_is_valid(graph, interventions=x, outcomes=y,
                            adjustment=z) == literal_gac(expanded, edges, {x}, {y}, z, "dag")


def test_all_four_vertex_causal_pag_completions_stay_in_independent_mag_class():
    from causal_graph_algorithms import Mark
    from causal_graph_algorithms.adjustment import _complete_no_selection_pag
    from tests.python.oracles import m_separation_signature_by_all_conditioning_subsets
    nodes = ("A", "B", "C", "D")
    for signature, (members, _) in no_selection_census(nodes).classes.items():
        source = MaximalAncestralGraph.from_graph(EndpointMarkedGraph(nodes=nodes,
            edges=((a, Mark.ARROW if x else Mark.TAIL, b, Mark.ARROW if y else Mark.TAIL)
                   for a, x, b, y in members[0])))
        member = _complete_no_selection_pag(CausalPAG.from_mag(source))
        atoms = tuple((a, x is Mark.ARROW, b, y is Mark.ARROW)
                      for a, x, b, y in member.topology.edges)
        assert m_separation_signature_by_all_conditioning_subsets(nodes, atoms) == signature
