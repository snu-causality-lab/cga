from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    AncestralGraph,
    DAG,
    DirectedBidirectedGraph,
    EndpointMarkedGraph,
    GraphValidationError,
    LossyConversionError,
    ProjectionDomainError,
    directed_bidirected_to_endpoint_marked,
    graph_to_canonical_json,
    induced_subgraph,
    latent_projection,
    m_separated,
)

from .oracles import (
    MixedEdgeAtom,
    latent_projection_by_simple_edge_paths,
    m_separated_by_canonical_dag,
    m_separated_by_simple_edge_paths,
)


NODES3 = ("A", "B", "C")
PAIRS3 = tuple(itertools.combinations(NODES3, 2))
NODES4 = ("A", "B", "C", "D")
PAIRS4 = tuple(itertools.combinations(NODES4, 2))


def edge_atoms(graph: DirectedBidirectedGraph) -> tuple[MixedEdgeAtom, ...]:
    atoms: list[MixedEdgeAtom] = [
        (source, False, target, True) for source, target in graph.directed
    ]
    atoms.extend((left, True, right, True) for left, right in graph.bidirected)
    return tuple(atoms)


def choice_graph(
    nodes: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
    choices: tuple[int, ...],
) -> DirectedBidirectedGraph:
    directed: list[tuple[str, str]] = []
    bidirected: list[tuple[str, str]] = []
    for (left, right), choice in zip(pairs, choices, strict=True):
        if choice in (1, 4):
            directed.append((left, right))
        if choice in (2, 5):
            directed.append((right, left))
        if choice in (3, 4, 5):
            bidirected.append((left, right))
    return DirectedBidirectedGraph(
        nodes=nodes, directed=directed, bidirected=bidirected
    )


def retained_from_bits(nodes: tuple[str, ...], bits: int) -> tuple[str, ...]:
    return tuple(
        node for index, node in enumerate(nodes) if bits & (1 << index)
    )


def assert_matches_literal_oracle(
    view: AcyclicDirectedBidirectedGraph | DAG,
    retained: tuple[str, ...],
) -> AcyclicDirectedBidirectedGraph:
    topology = view.topology
    expected_nodes, expected_directed, expected_bidirected = (
        latent_projection_by_simple_edge_paths(
            topology.nodes, edge_atoms(topology), retained
        )
    )
    projected = latent_projection(view, retained)
    assert type(projected) is AcyclicDirectedBidirectedGraph
    assert projected.topology.nodes == expected_nodes
    assert projected.topology.directed == expected_directed
    assert projected.topology.bidirected == expected_bidirected
    return projected


def assert_separation_correspondence(
    source: AcyclicDirectedBidirectedGraph | DAG,
    projected: AcyclicDirectedBidirectedGraph,
) -> None:
    source_atoms = edge_atoms(source.topology)
    projected_atoms = edge_atoms(projected.topology)
    for left, right in itertools.combinations(projected.topology.nodes, 2):
        remaining = tuple(
            node
            for node in projected.topology.nodes
            if node not in {left, right}
        )
        for size in range(len(remaining) + 1):
            for given in itertools.combinations(remaining, size):
                expected_path = m_separated_by_simple_edge_paths(
                    source.topology.nodes,
                    source_atoms,
                    [left],
                    [right],
                    given,
                )
                expected_dag = m_separated_by_canonical_dag(
                    source.topology.nodes,
                    source_atoms,
                    [left],
                    [right],
                    given,
                )
                assert expected_path == expected_dag
                assert m_separated(source, left, right, given) == expected_path
                assert m_separated(projected, left, right, given) == expected_path
                assert m_separated_by_simple_edge_paths(
                    projected.topology.nodes,
                    projected_atoms,
                    [left],
                    [right],
                    given,
                ) == expected_path


def load_paper_fixture(name: str) -> dict[str, object]:
    path = Path(__file__).parents[1] / "fixtures" / "papers" / name
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "rers_2023_figures1_3_latent_projection.json",
        "evans_2021_figure1_latent_projection.json",
    ],
)
def test_published_latent_projection_fixtures(fixture_name: str) -> None:
    fixture = load_paper_fixture(fixture_name)
    topology = DirectedBidirectedGraph(
        nodes=fixture["nodes"],
        directed=fixture["directed"],
        bidirected=fixture["bidirected"],
    )
    view = DAG.from_graph(topology)
    expected = DirectedBidirectedGraph(
        nodes=fixture["retained"],
        directed=fixture["projected_directed"],
        bidirected=fixture["projected_bidirected"],
    )

    projected = assert_matches_literal_oracle(
        view, tuple(fixture["retained"])
    )

    assert projected.topology == expected
    assert_separation_correspondence(view, projected)


def test_evans_fixture_pins_bow_output_and_canonical_bytes() -> None:
    graph = DAG.from_graph(
        DirectedBidirectedGraph(
            nodes=["h", "a", "b", "c"],
            directed=[("a", "b"), ("b", "c"), ("h", "b"), ("h", "c")],
        )
    )

    projected = latent_projection(graph, retained=["c", "a", "b"])

    assert projected.topology.has_directed("b", "c")
    assert projected.topology.has_bidirected("b", "c")
    assert graph_to_canonical_json(projected.topology) == (
        '{"schema":"cga.graph/1","family":"directed-bidirected",'
        '"nodes":["a","b","c"],"directed":[["a","b"],["b","c"]],'
        '"bidirected":[["b","c"]]}'
    )
    with pytest.raises(LossyConversionError):
        directed_bidirected_to_endpoint_marked(projected.topology)
    with pytest.raises(GraphValidationError, match="bidirected"):
        DAG.from_graph(projected.topology)


def test_definition_level_chain_fork_collider_and_spouse_paths() -> None:
    cases = (
        (
            DirectedBidirectedGraph(
                nodes=["A", "L1", "L2", "B"],
                directed=[("A", "L1"), ("L1", "L2"), ("L2", "B")],
            ),
            DirectedBidirectedGraph(nodes=["A", "B"], directed=[("A", "B")]),
        ),
        (
            DirectedBidirectedGraph(
                nodes=["A", "L", "B"],
                directed=[("L", "A"), ("L", "B")],
            ),
            DirectedBidirectedGraph(nodes=["A", "B"], bidirected=[("A", "B")]),
        ),
        (
            DirectedBidirectedGraph(
                nodes=["A", "L", "B"],
                directed=[("A", "L"), ("B", "L")],
            ),
            DirectedBidirectedGraph(nodes=["A", "B"]),
        ),
        (
            DirectedBidirectedGraph(
                nodes=["A", "L", "B"],
                directed=[("L", "B")],
                bidirected=[("A", "L")],
            ),
            DirectedBidirectedGraph(nodes=["A", "B"], bidirected=[("A", "B")]),
        ),
    )
    for source, expected in cases:
        view = AcyclicDirectedBidirectedGraph.from_graph(source)
        projected = assert_matches_literal_oracle(view, ("A", "B"))
        assert projected.topology == expected


def test_retained_mediator_and_existing_bow_are_not_rewritten() -> None:
    source = DirectedBidirectedGraph(
        nodes=["A", "M", "B"],
        directed=[("A", "M"), ("M", "B")],
        bidirected=[("A", "M")],
    )
    view = AcyclicDirectedBidirectedGraph.from_graph(source)
    source_bytes = graph_to_canonical_json(source)

    projected = latent_projection(view, retained=["A", "M", "B"])

    assert graph_to_canonical_json(source) == source_bytes
    assert projected.topology == source
    assert ("A", "B") not in projected.topology.directed
    assert projected.topology.has_directed("A", "M")
    assert projected.topology.has_bidirected("A", "M")


def test_empty_singleton_all_and_declared_order_contract() -> None:
    source = AcyclicDirectedBidirectedGraph.from_graph(
        DirectedBidirectedGraph(
            nodes=["Z", "A", "M"],
            directed=[("Z", "A"), ("A", "M")],
            bidirected=[("Z", "M")],
        )
    )

    assert latent_projection(source, []).topology.nodes == ()
    assert latent_projection(source, "A").topology.nodes == ("A",)
    assert latent_projection(
        source, ["M", "Z", "A", "M"]
    ).topology == source.topology


def test_projection_requires_a_validated_directed_bidirected_view() -> None:
    raw = DirectedBidirectedGraph(nodes=["A", "B"], directed=[("A", "B")])
    ancestral = AncestralGraph.from_graph(EndpointMarkedGraph(nodes=["A", "B"]))

    with pytest.raises(ProjectionDomainError):
        latent_projection(raw, ["A"])  # type: ignore[arg-type]
    with pytest.raises(ProjectionDomainError):
        latent_projection(ancestral, ["A"])  # type: ignore[arg-type]
    with pytest.raises(GraphValidationError, match="unknown vertex"):
        latent_projection(DAG.from_graph(raw), ["A", "missing"])

    cyclic = DirectedBidirectedGraph(
        nodes=["A", "B"], directed=[("A", "B"), ("B", "A")]
    )
    with pytest.raises(GraphValidationError, match="cycle"):
        AcyclicDirectedBidirectedGraph.from_graph(cyclic)
    with pytest.raises(ProjectionDomainError):
        latent_projection(cyclic, ["A"])  # type: ignore[arg-type]


def test_projection_is_not_an_induced_subgraph() -> None:
    source = DAG.from_graph(
        DirectedBidirectedGraph(
            nodes=["A", "L", "B"],
            directed=[("A", "L"), ("L", "B")],
        )
    )

    induced = induced_subgraph(source, ["A", "B"])
    projected = latent_projection(source, ["A", "B"])

    assert induced.topology.directed == frozenset()
    assert projected.topology.directed == frozenset({("A", "B")})


def test_exhaustive_three_vertex_bow_universe_and_composition() -> None:
    accepted = 0
    for choices in itertools.product(range(6), repeat=len(PAIRS3)):
        topology = choice_graph(NODES3, PAIRS3, choices)
        try:
            view = AcyclicDirectedBidirectedGraph.from_graph(topology)
        except GraphValidationError:
            continue
        accepted += 1
        for retained_bits in range(1 << len(NODES3)):
            retained = retained_from_bits(NODES3, retained_bits)
            projected = assert_matches_literal_oracle(view, retained)
            assert_separation_correspondence(view, projected)

            intermediate = tuple(
                node
                for index, node in enumerate(NODES3)
                if retained_bits & (1 << index) or index % 2 == 0
            )
            first = latent_projection(view, intermediate)
            composed = latent_projection(first, retained)
            assert composed == projected
    assert accepted == 200


def test_all_four_vertex_dags_match_the_literal_oracle() -> None:
    directed_pairs = tuple(
        (left, right) for left in NODES4 for right in NODES4 if left != right
    )
    accepted = 0
    compared = 0
    for bits in range(1 << len(directed_pairs)):
        topology = DirectedBidirectedGraph(
            nodes=NODES4,
            directed=[
                edge
                for index, edge in enumerate(directed_pairs)
                if bits & (1 << index)
            ],
        )
        try:
            view = DAG.from_graph(topology)
        except GraphValidationError:
            continue
        accepted += 1
        for retained_bits in range(1 << len(NODES4)):
            assert_matches_literal_oracle(
                view, retained_from_bits(NODES4, retained_bits)
            )
            compared += 1
    assert accepted == 543
    assert compared == 8_688


def test_fixed_seed_four_vertex_bow_sample_matches_literal_oracle() -> None:
    rng = random.Random(0xC0A53)
    accepted = 0
    attempts = 0
    while accepted < 4_000:
        attempts += 1
        choices = tuple(rng.randrange(6) for _ in PAIRS4)
        topology = choice_graph(NODES4, PAIRS4, choices)
        try:
            view = AcyclicDirectedBidirectedGraph.from_graph(topology)
        except GraphValidationError:
            continue
        accepted += 1
        for retained_bits in range(1 << len(NODES4)):
            assert_matches_literal_oracle(
                view, retained_from_bits(NODES4, retained_bits)
            )
    assert attempts == 5_365
