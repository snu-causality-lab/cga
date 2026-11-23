from __future__ import annotations

from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import MarkovEquivalenceDomainError
from .maximality import MaximalAncestralGraph
from .directed_bidirected import DAG


def dag_markov_equivalent(left: DAG, right: DAG) -> bool:
    """Compare labelled DAG skeletons and unshielded colliders (Verma--Pearl)."""
    if not isinstance(left, DAG) or not isinstance(right, DAG):
        raise MarkovEquivalenceDomainError("DAG equivalence requires two DAG views")
    if set(left.topology.nodes) != set(right.topology.nodes):
        return False
    from .pdag import _colliders
    def endpoint(graph: DAG) -> EndpointMarkedGraph:
        return EndpointMarkedGraph(nodes=graph.topology.nodes,
            edges=((a, Mark.TAIL, b, Mark.ARROW) for a, b in graph.topology.directed))
    skeleton_left = frozenset(frozenset(e) for e in left.topology.directed)
    skeleton_right = frozenset(frozenset(e) for e in right.topology.directed)
    return skeleton_left == skeleton_right and _colliders(endpoint(left)) == _colliders(endpoint(right))


_Triple = tuple[str, str, str]
_PairVertex = tuple[str, str]


class _TopologyIndex:
    """Local adjacency index for Tables A.1--A.3 of Ali et al. (2009)."""

    def __init__(self, graph: EndpointMarkedGraph) -> None:
        self.nodes = graph.nodes
        self.neighbors: dict[str, set[str]] = {
            node: set() for node in graph.nodes
        }
        self.parents: dict[str, set[str]] = {
            node: set() for node in graph.nodes
        }
        self.spouses: dict[str, set[str]] = {
            node: set() for node in graph.nodes
        }
        arrow_at: dict[tuple[str, str], bool] = {}

        for left, left_mark, right, right_mark in graph.edges:
            self.neighbors[left].add(right)
            self.neighbors[right].add(left)
            arrow_at[(left, right)] = left_mark is Mark.ARROW
            arrow_at[(right, left)] = right_mark is Mark.ARROW
            if left_mark is Mark.TAIL and right_mark is Mark.ARROW:
                self.parents[right].add(left)
            elif left_mark is Mark.ARROW and right_mark is Mark.TAIL:
                self.parents[left].add(right)
            elif left_mark is Mark.ARROW and right_mark is Mark.ARROW:
                self.spouses[left].add(right)
                self.spouses[right].add(left)

        colliders: set[_Triple] = set()
        for middle in graph.nodes:
            arrow_neighbors = tuple(
                neighbor
                for neighbor in graph.nodes
                if arrow_at.get((middle, neighbor), False)
            )
            for first in arrow_neighbors:
                for third in arrow_neighbors:
                    if first != third:
                        colliders.add((first, middle, third))
        self.colliders = frozenset(colliders)

    def adjacent(self, left: str, right: str) -> bool:
        return right in self.neighbors[left]


def _reachable_pair_vertices(
    vertices: set[_PairVertex],
    triples: frozenset[_Triple],
    source: _PairVertex,
) -> set[_PairVertex]:
    """Run Table A.1 on the edge-pair graph constructed in Table A.2."""
    reached = {source}
    pending = [source]
    while pending:
        first, middle = pending.pop()
        for candidate in vertices:
            if candidate in reached or candidate[0] != middle:
                continue
            if (first, middle, candidate[1]) in triples:
                reached.add(candidate)
                pending.append(candidate)
    return reached


def _triples(index: _TopologyIndex) -> frozenset[_Triple]:
    """Return the Table A.2 set between OCol(G) and ICol(G)."""
    triples = frozenset(
        triple
        for triple in index.colliders
        if not index.adjacent(triple[0], triple[2])
    )

    while True:
        updated = set(triples)
        for first, middle, third in index.colliders - triples:
            if first not in index.spouses[middle] or (
                first not in index.parents[third]
            ):
                continue

            pair_vertices: set[_PairVertex] = {(middle, first)}
            for left_parent in index.parents[third]:
                for right_parent in index.spouses[left_parent]:
                    if right_parent in index.parents[third]:
                        pair_vertices.add((left_parent, right_parent))

            reachable = _reachable_pair_vertices(
                pair_vertices, triples, (middle, first)
            )
            has_nonadjacent_endpoint = any(
                not index.adjacent(endpoint, third)
                for before, at, endpoint in triples
                if (before, at) in reachable
            )
            if has_nonadjacent_endpoint:
                updated.add((first, middle, third))
                updated.add((third, middle, first))

        next_triples = frozenset(updated)
        if next_triples == triples:
            return triples
        triples = next_triples


def mag_markov_equivalent(
    left: MaximalAncestralGraph,
    right: MaximalAncestralGraph,
) -> bool:
    """Test MAG Markov equivalence by Ali et al. (2009), Tables A.1--A.3.

    Vertex identity is label based, so declaration order may differ.  Both
    arguments must be validated maximal-ancestral proof views over the same
    labels.
    """
    if not isinstance(left, MaximalAncestralGraph) or not isinstance(
        right, MaximalAncestralGraph
    ):
        raise MarkovEquivalenceDomainError(
            "MAG Markov equivalence requires two MAG proof views"
        )
    if frozenset(left.topology.nodes) != frozenset(right.topology.nodes):
        raise MarkovEquivalenceDomainError(
            "MAG Markov equivalence requires matching vertex labels"
        )

    left_index = _TopologyIndex(left.topology)
    right_index = _TopologyIndex(right.topology)
    left_adjacencies = frozenset(
        frozenset((node, neighbor))
        for node, neighbors in left_index.neighbors.items()
        for neighbor in neighbors
    )
    right_adjacencies = frozenset(
        frozenset((node, neighbor))
        for node, neighbors in right_index.neighbors.items()
        for neighbor in neighbors
    )
    if left_adjacencies != right_adjacencies:
        return False

    left_triples = _triples(left_index)
    right_triples = _triples(right_index)
    return left_triples <= right_index.colliders and (
        right_triples <= left_index.colliders
    )
