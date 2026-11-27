from collections import defaultdict
from itertools import combinations, permutations

import pytest

from causal_graph_algorithms import (
    CPDAG,
    DAG,
    DirectedBidirectedGraph,
    Mark,
    PDAGDomainError,
    ges_delete_is_valid,
    ges_insert_is_valid,
)


def test_insert_requires_clique_and_blocks_semidirected_paths():
    chain = CPDAG.from_dag(DAG.from_text("X --> P --> Q --> Y"))
    assert not ges_insert_is_valid(chain, "X", "Y")
    assert ges_insert_is_valid(chain, "X", "Y", tails="Q")

    fork = CPDAG.from_dag(DAG.from_text("X; Y --> P; Y --> Q"))
    assert ges_insert_is_valid(fork, "X", "Y", tails=("P",))
    assert not ges_insert_is_valid(fork, "X", "Y", tails=("P", "Q"))


def test_insert_respects_arrow_direction_along_the_path():
    graph = CPDAG.from_dag(DAG.from_text("X --> P <-- Q; P --> Y"))
    assert ges_insert_is_valid(graph, "X", "Y")
    assert not ges_insert_is_valid(graph, "Y", "X")


def test_delete_requires_remaining_common_neighbors_to_be_a_clique():
    graph = CPDAG.from_dag(
        DAG.from_text("X --> Y; X --> P; X --> Q; Y --> P; Y --> Q")
    )
    assert not ges_delete_is_valid(graph, "X", "Y")
    assert ges_delete_is_valid(graph, "X", "Y", heads="P")
    assert ges_delete_is_valid(graph, "Y", "X", heads=("P", "Q"))


def test_rejects_inapplicable_operators_on_known_vertices():
    chain = CPDAG.from_dag(DAG.from_text("X --> P --> Y"))
    assert not ges_insert_is_valid(chain, "X", "P")
    assert not ges_insert_is_valid(chain, "X", "Y", tails="P")
    assert not ges_delete_is_valid(chain, "X", "Y")
    assert not ges_delete_is_valid(chain, "X", "P", heads="Y")
    assert not ges_insert_is_valid(chain, "X", "X")
    assert not ges_delete_is_valid(chain, "X", "X")

    collider = CPDAG.from_dag(DAG.from_text("X --> Y <-- P"))
    assert ges_delete_is_valid(collider, "X", "Y")
    assert not ges_delete_is_valid(collider, "Y", "X")


@pytest.mark.parametrize(
    ("check", "parameter"),
    [(ges_insert_is_valid, "tails"), (ges_delete_is_valid, "heads")],
)
def test_requires_cpdag_and_known_vertices(check, parameter):
    graph = DAG.from_text("X --> Y")
    with pytest.raises(PDAGDomainError, match="CPDAG"):
        check(graph, "X", "Y")
    cpdag = CPDAG.from_dag(graph)
    with pytest.raises(PDAGDomainError, match="unknown vertex"):
        check(cpdag, "X", "missing")
    with pytest.raises(PDAGDomainError, match="unknown vertex"):
        check(cpdag, "X", "Y", **{parameter: "missing"})


def _subsets(items):
    ordered = tuple(items)
    for size in range(len(ordered) + 1):
        yield from (frozenset(chosen) for chosen in combinations(ordered, size))


def _skeleton(arrows):
    return frozenset(frozenset(edge) for edge in arrows)


def _colliders(nodes, skeleton, arrows):
    return frozenset(
        (frozenset((left, right)), middle)
        for middle in nodes
        for left, right in combinations(nodes, 2)
        if frozenset((left, right)) not in skeleton
        and {(left, middle), (right, middle)} <= arrows
    )


def test_admissible_operators_match_single_edits_of_enumerated_four_vertex_dags():
    nodes = ("X", "Y", "P", "Q")
    # Every DAG arises by keeping a subset of the forward arrows in a vertex order.
    all_dags = {arrows for order in permutations(nodes) for arrows in _subsets(combinations(order, 2))}
    by_signature = defaultdict(set)
    for arrows in all_dags:
        skeleton = _skeleton(arrows)
        by_signature[skeleton, _colliders(nodes, skeleton, arrows)].add(arrows)

    def has_edited_extension(original, source, target, skeleton, compelled, *, insert):
        # Compare against the edited PDAG's exact skeleton, arrows and colliders.
        # This check uses neither the clique test nor the semidirected-path test.
        new_edge = {(source, target)}
        allowed = by_signature.get((skeleton, _colliders(nodes, skeleton, compelled)), ())
        for arrows in original:
            if not insert and not new_edge <= arrows:
                continue
            edited = arrows | new_edge if insert else arrows - new_edge
            if edited in allowed and compelled <= edited:
                return True
        return False

    for (skeleton, _), representatives in by_signature.items():
        compelled = frozenset.intersection(*representatives)
        reversible = skeleton - _skeleton(compelled)
        cpdag = CPDAG.from_dag(
            DAG(DirectedBidirectedGraph(nodes=nodes, directed=next(iter(representatives))))
        )
        for left, right in combinations(nodes, 2):
            if (left, right) in compelled:
                expected_edge = (Mark.TAIL, Mark.ARROW)
            elif (right, left) in compelled:
                expected_edge = (Mark.ARROW, Mark.TAIL)
            elif frozenset((left, right)) in reversible:
                expected_edge = (Mark.TAIL, Mark.TAIL)
            else:
                expected_edge = None
            assert cpdag.topology.edge(left, right) == expected_edge
        for source, target in permutations(nodes, 2):
            pair = frozenset((source, target))
            neighbors = {
                v for v in nodes if frozenset((target, v)) in reversible
            }
            common = {v for v in neighbors if frozenset((source, v)) in skeleton}
            if pair not in skeleton:
                for tails in _subsets(neighbors - common):
                    arrows = compelled | {(source, target)} | {(tail, target) for tail in tails}
                    expected = has_edited_extension(
                        representatives, source, target, skeleton | {pair}, arrows, insert=True
                    )
                    assert ges_insert_is_valid(cpdag, source, target, tails=tails) == expected
            elif pair in reversible or (source, target) in compelled:
                for heads in _subsets(common):
                    arrows = (compelled - {(source, target)}) | {(target, head) for head in heads}
                    arrows |= {(source, head) for head in heads if frozenset((source, head)) in reversible}
                    expected = has_edited_extension(
                        representatives, source, target, skeleton - {pair}, arrows, insert=False
                    )
                    assert ges_delete_is_valid(cpdag, source, target, heads=heads) == expected
