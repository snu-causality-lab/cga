"""Definition-level test oracles that do not import production traversal code."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache


MixedEdgeAtom = tuple[str, bool, str, bool]
EndpointMarkAtom = tuple[str, str, str, str]
ConditionalIndependenceSignature = tuple[
    tuple[str, str, tuple[str, ...], bool], ...
]
MAGEndpointClass = tuple[tuple[MixedEdgeAtom, ...], ...]
FullPAGOracleClass = tuple[MAGEndpointClass, tuple[EndpointMarkAtom, ...]]
NO_SELECTION_STATES: tuple[None | tuple[bool, bool], ...] = (
    None,
    (False, True),
    (True, False),
    (True, True),
)


@dataclass(frozen=True, slots=True)
class NoSelectionCensus:
    """Literal no-selection MAG/PAG census over one declared node order."""

    raw_count: int
    ancestral_count: int
    mags: tuple[tuple[MixedEdgeAtom, ...], ...]
    classes: dict[ConditionalIndependenceSignature, FullPAGOracleClass]


def has_topological_order(
    nodes: tuple[str, ...], directed: frozenset[tuple[str, str]]
) -> bool:
    """Literal acyclicity oracle: enumerate every vertex order."""

    for order in itertools.permutations(nodes):
        position = {node: index for index, node in enumerate(order)}
        if all(position[left] < position[right] for left, right in directed):
            return True
    return False


def strict_ancestors_by_simple_paths(
    nodes: tuple[str, ...],
    directed: frozenset[tuple[str, str]],
    seeds: Iterable[str],
) -> frozenset[str]:
    """Literal positive-path oracle, with every seed removed from the result."""

    seed_set = frozenset(seeds)
    parents = {
        node: frozenset(left for left, right in directed if right == node)
        for node in nodes
    }
    found: set[str] = set()

    def visit(current: str, path: frozenset[str]) -> None:
        for parent in parents[current]:
            if parent in path:
                continue
            found.add(parent)
            visit(parent, path | {parent})

    for seed in seed_set:
        visit(seed, frozenset({seed}))
    return frozenset(found - seed_set)


def strict_descendants_by_simple_paths(
    nodes: tuple[str, ...],
    directed: frozenset[tuple[str, str]],
    seeds: Iterable[str],
) -> frozenset[str]:
    reversed_edges = frozenset((right, left) for left, right in directed)
    return strict_ancestors_by_simple_paths(nodes, reversed_edges, seeds)


def possible_descendants_by_simple_paths(
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str, str, str], ...],
    seeds: Iterable[str],
) -> frozenset[str]:
    """Enumerate simple paths whose departing endpoint is not an arrowhead."""

    seed_set = frozenset(seeds)
    adjacency: dict[str, list[tuple[str, str]]] = {node: [] for node in nodes}
    for left, left_mark, right, right_mark in edges:
        adjacency[left].append((right, left_mark))
        adjacency[right].append((left, right_mark))
    found: set[str] = set()

    def visit(current: str, path: frozenset[str]) -> None:
        for neighbor, departing_mark in adjacency[current]:
            if departing_mark == "arrow" or neighbor in path:
                continue
            found.add(neighbor)
            visit(neighbor, path | {neighbor})

    for seed in seed_set:
        visit(seed, frozenset({seed}))
    return frozenset(found - seed_set)


def possible_ancestors_by_simple_paths(
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str, str, str], ...],
    seeds: Iterable[str],
) -> frozenset[str]:
    """Literal candidate-by-candidate reversal of possible reachability."""

    seed_set = frozenset(seeds)
    found = {
        candidate
        for candidate in nodes
        if possible_descendants_by_simple_paths(nodes, edges, [candidate]) & seed_set
    }
    return frozenset(found - seed_set)


def m_separated_by_simple_edge_paths(
    nodes: tuple[str, ...],
    edges: tuple[MixedEdgeAtom, ...],
    left: Iterable[str],
    right: Iterable[str],
    given: Iterable[str],
) -> bool:
    """Literal m-separation oracle over vertex-simple, edge-atom paths."""

    left_set = frozenset(left)
    right_set = frozenset(right)
    given_set = frozenset(given)
    directed = {
        (first, second)
        for first, arrow_first, second, arrow_second in edges
        if not arrow_first and arrow_second
    } | {
        (second, first)
        for first, arrow_first, second, arrow_second in edges
        if arrow_first and not arrow_second
    }
    ancestors = set(given_set)
    changed = True
    while changed:
        changed = False
        for parent, child in directed:
            if child in ancestors and parent not in ancestors:
                ancestors.add(parent)
                changed = True

    adjacency: dict[str, list[tuple[str, bool, bool]]] = {
        node: [] for node in nodes
    }
    for first, arrow_first, second, arrow_second in edges:
        adjacency[first].append((second, arrow_first, arrow_second))
        adjacency[second].append((first, arrow_second, arrow_first))

    def reaches_right(
        current: str,
        incoming_arrow: bool | None,
        visited: frozenset[str],
    ) -> bool:
        for neighbor, arrow_current, arrow_neighbor in adjacency[current]:
            if neighbor in visited:
                continue
            if incoming_arrow is not None:
                collider = incoming_arrow and arrow_current
                if collider and current not in ancestors:
                    continue
                if not collider and current in given_set:
                    continue
            if neighbor in right_set:
                return True
            if reaches_right(
                neighbor, arrow_neighbor, visited | frozenset({neighbor})
            ):
                return True
        return False

    return not any(
        reaches_right(source, None, frozenset({source})) for source in left_set
    )


def m_separated_by_canonical_dag(
    nodes: tuple[str, ...],
    edges: tuple[MixedEdgeAtom, ...],
    left: Iterable[str],
    right: Iterable[str],
    given: Iterable[str],
) -> bool:
    """Independent latent/selection expansion followed by ancestral moralization."""

    observed = {node: ("observed", node) for node in nodes}
    dag_nodes: set[tuple[str, object]] = set(observed.values())
    directed: set[tuple[tuple[str, object], tuple[str, object]]] = set()
    conditioned = {observed[node] for node in given}

    for index, (first, arrow_first, second, arrow_second) in enumerate(edges):
        first_node = observed[first]
        second_node = observed[second]
        if not arrow_first and arrow_second:
            directed.add((first_node, second_node))
        elif arrow_first and not arrow_second:
            directed.add((second_node, first_node))
        elif arrow_first and arrow_second:
            latent = ("latent", index)
            dag_nodes.add(latent)
            directed.add((latent, first_node))
            directed.add((latent, second_node))
        else:
            selection = ("selection", index)
            dag_nodes.add(selection)
            directed.add((first_node, selection))
            directed.add((second_node, selection))
            conditioned.add(selection)

    parents: dict[tuple[str, object], set[tuple[str, object]]] = {
        node: set() for node in dag_nodes
    }
    for parent, child in directed:
        parents[child].add(parent)

    relevant = (
        {observed[node] for node in left}
        | {observed[node] for node in right}
        | conditioned
    )
    pending = list(relevant)
    while pending:
        current = pending.pop()
        for parent in parents[current]:
            if parent not in relevant:
                relevant.add(parent)
                pending.append(parent)

    moral: dict[tuple[str, object], set[tuple[str, object]]] = {
        node: set() for node in relevant
    }
    for parent, child in directed:
        if parent in relevant and child in relevant:
            moral[parent].add(child)
            moral[child].add(parent)
    for child in relevant:
        child_parents = tuple(parent for parent in parents[child] if parent in relevant)
        for first, second in itertools.combinations(child_parents, 2):
            moral[first].add(second)
            moral[second].add(first)

    blocked = conditioned
    targets = {observed[node] for node in right}
    reached = {observed[node] for node in left}
    pending = list(reached)
    while pending:
        current = pending.pop()
        for neighbor in moral[current]:
            if neighbor in blocked or neighbor in reached:
                continue
            if neighbor in targets:
                return False
            reached.add(neighbor)
            pending.append(neighbor)
    return targets.isdisjoint(reached)


def endpoint_graph_is_ancestral_by_definition(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> bool:
    """Richardson--Spirtes conditions, checked without production validators."""

    directed: set[tuple[str, str]] = set()
    parents = {node: set() for node in nodes}
    spouses = {node: set() for node in nodes}
    undirected_neighbors = {node: set() for node in nodes}
    adjacency: dict[str, list[tuple[str, bool]]] = {node: [] for node in nodes}

    for first, arrow_first, second, arrow_second in edges:
        adjacency[first].append((second, arrow_first))
        adjacency[second].append((first, arrow_second))
        if not arrow_first and arrow_second:
            directed.add((first, second))
            parents[second].add(first)
        elif arrow_first and not arrow_second:
            directed.add((second, first))
            parents[first].add(second)
        elif arrow_first and arrow_second:
            spouses[first].add(second)
            spouses[second].add(first)
        else:
            undirected_neighbors[first].add(second)
            undirected_neighbors[second].add(first)

    if not has_topological_order(nodes, frozenset(directed)):
        return False
    for node in nodes:
        if undirected_neighbors[node] and (parents[node] or spouses[node]):
            return False

    def anterior_reachable(source: str, target: str) -> bool:
        def visit(current: str, seen: frozenset[str]) -> bool:
            for neighbor, arrow_at_current in adjacency[current]:
                if arrow_at_current or neighbor in seen:
                    continue
                if neighbor == target:
                    return True
                if visit(neighbor, seen | frozenset({neighbor})):
                    return True
            return False

        return source == target or visit(source, frozenset({source}))

    return all(
        not anterior_reachable(node, arrow_neighbor)
        for node in nodes
        for arrow_neighbor in parents[node] | spouses[node]
    )


def strict_anterior_vertices_by_simple_paths(
    nodes: tuple[str, ...],
    edges: tuple[EndpointMarkAtom, ...],
    seeds: Iterable[str],
) -> frozenset[str]:
    """Literal R&S anterior paths over tail-tail and tail-arrow edges."""

    seed_set = frozenset(seeds)
    adjacency: dict[str, list[tuple[str, str, str]]] = {
        node: [] for node in nodes
    }
    for first, first_mark, second, second_mark in edges:
        adjacency[first].append((second, first_mark, second_mark))
        adjacency[second].append((first, second_mark, first_mark))

    def has_anterior_path(source: str, target: str) -> bool:
        def visit(
            current: str,
            visited: frozenset[str],
        ) -> bool:
            for neighbor, current_mark, neighbor_mark in adjacency[current]:
                if neighbor in visited or current_mark != "tail":
                    continue
                if neighbor_mark not in ("tail", "arrow"):
                    continue
                if neighbor == target:
                    return True
                if visit(
                    neighbor,
                    visited | frozenset({neighbor}),
                ):
                    return True
            return False

        return source == target or visit(source, frozenset({source}))

    return frozenset(
        candidate
        for candidate in nodes
        if candidate not in seed_set
        and any(has_anterior_path(candidate, seed) for seed in seed_set)
    )


def primitive_inducing_paths_by_simple_paths(
    nodes: tuple[str, ...],
    edges: tuple[MixedEdgeAtom, ...],
    left: str,
    right: str,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate primitive inducing paths from R&S Section 4.2.3."""

    positions = {node: index for index, node in enumerate(nodes)}
    adjacency: dict[str, list[tuple[str, bool, bool]]] = {
        node: [] for node in nodes
    }
    directed: set[tuple[str, str]] = set()
    for first, arrow_first, second, arrow_second in edges:
        adjacency[first].append((second, arrow_first, arrow_second))
        adjacency[second].append((first, arrow_second, arrow_first))
        if not arrow_first and arrow_second:
            directed.add((first, second))
        elif arrow_first and not arrow_second:
            directed.add((second, first))

    ancestors = strict_ancestors_by_simple_paths(
        nodes, frozenset(directed), (left, right)
    )
    found: list[tuple[str, ...]] = []

    def visit(
        current: str,
        path: tuple[str, ...],
        incoming_arrow: bool | None,
    ) -> None:
        for neighbor, arrow_current, arrow_neighbor in adjacency[current]:
            if neighbor in path:
                continue
            if incoming_arrow is not None and (
                not incoming_arrow
                or not arrow_current
                or current not in ancestors
            ):
                continue
            next_path = path + (neighbor,)
            if neighbor == right:
                found.append(next_path)
            else:
                visit(neighbor, next_path, arrow_neighbor)

    visit(left, (left,), None)
    found.sort(
        key=lambda path: (
            len(path),
            tuple(positions[vertex] for vertex in path),
        )
    )
    return tuple(found)


def nonmaximal_pair_by_all_conditioning_sets(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> tuple[str, str] | None:
    """Return the first missing pair unseparated by every admissible set."""

    adjacent = {
        frozenset((first, second))
        for first, _, second, _ in edges
    }
    for left_index, right_index in itertools.combinations(range(len(nodes)), 2):
        left = nodes[left_index]
        right = nodes[right_index]
        if frozenset((left, right)) in adjacent:
            continue
        remaining = tuple(
            node
            for index, node in enumerate(nodes)
            if index not in (left_index, right_index)
        )
        has_separator = any(
            m_separated_by_simple_edge_paths(
                nodes, edges, (left,), (right,), given
            )
            for size in range(len(remaining) + 1)
            for given in itertools.combinations(remaining, size)
        )
        if not has_separator:
            return left, right
    return None


def nonmaximality_path_by_simple_paths(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> tuple[str, ...] | None:
    """Return the first canonical primitive path between a missing pair."""

    adjacent = {
        frozenset((first, second))
        for first, _, second, _ in edges
    }
    for left, right in itertools.combinations(nodes, 2):
        if frozenset((left, right)) in adjacent:
            continue
        paths = primitive_inducing_paths_by_simple_paths(
            nodes, edges, left, right
        )
        if paths:
            return paths[0]
    return None


def maximal_extension_by_all_conditioning_sets(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> tuple[MixedEdgeAtom, ...]:
    """Construct G[empty, empty] from the all-conditioning-sets definition."""

    positions = {node: index for index, node in enumerate(nodes)}
    adjacent = {
        frozenset((first, second))
        for first, _, second, _ in edges
    }
    result = list(edges)
    for left_index, right_index in itertools.combinations(range(len(nodes)), 2):
        left = nodes[left_index]
        right = nodes[right_index]
        if frozenset((left, right)) in adjacent:
            continue
        remaining = tuple(
            node
            for index, node in enumerate(nodes)
            if index not in (left_index, right_index)
        )
        if all(
            not m_separated_by_simple_edge_paths(
                nodes, edges, (left,), (right,), given
            )
            for size in range(len(remaining) + 1)
            for given in itertools.combinations(remaining, size)
        ):
            result.append((left, True, right, True))
    result.sort(key=lambda edge: (positions[edge[0]], positions[edge[2]]))
    return tuple(result)


def ancestral_transform_by_all_conditioning_sets(
    nodes: tuple[str, ...],
    edges: tuple[MixedEdgeAtom, ...],
    conditioned: Iterable[str],
    marginalized: Iterable[str],
) -> tuple[tuple[str, ...], tuple[MixedEdgeAtom, ...]]:
    """Evaluate Richardson--Spirtes Definition 4.2.1 literally."""

    conditioned_set = frozenset(conditioned)
    marginalized_set = frozenset(marginalized)
    removed = conditioned_set | marginalized_set
    retained = tuple(node for node in nodes if node not in removed)
    endpoint_edges: tuple[EndpointMarkAtom, ...] = tuple(
        (
            first,
            "arrow" if arrow_first else "tail",
            second,
            "arrow" if arrow_second else "tail",
        )
        for first, arrow_first, second, arrow_second in edges
    )

    def is_anterior_to_any(source: str, targets: frozenset[str]) -> bool:
        return source in targets or source in (
            strict_anterior_vertices_by_simple_paths(
                nodes,
                endpoint_edges,
                targets,
            )
        )

    result: list[MixedEdgeAtom] = []
    for left_index, right_index in itertools.combinations(range(len(retained)), 2):
        left = retained[left_index]
        right = retained[right_index]
        remaining = tuple(
            node
            for index, node in enumerate(retained)
            if index not in (left_index, right_index)
        )
        adjacent_in_transform = all(
            not m_separated_by_simple_edge_paths(
                nodes,
                edges,
                (left,),
                (right,),
                tuple(given) + tuple(conditioned_set),
            )
            for size in range(len(remaining) + 1)
            for given in itertools.combinations(remaining, size)
        )
        if not adjacent_in_transform:
            continue
        arrow_left = not is_anterior_to_any(
            left, conditioned_set | frozenset({right})
        )
        arrow_right = not is_anterior_to_any(
            right, conditioned_set | frozenset({left})
        )
        result.append((left, arrow_left, right, arrow_right))
    return retained, tuple(result)


def ancestral_transform_by_inducing_paths(
    nodes: tuple[str, ...],
    edges: tuple[MixedEdgeAtom, ...],
    conditioned: Iterable[str],
    marginalized: Iterable[str],
) -> tuple[tuple[str, ...], tuple[MixedEdgeAtom, ...]]:
    """Literal simple-path oracle for R&S Section 4.2.3."""

    conditioned_set = frozenset(conditioned)
    marginalized_set = frozenset(marginalized)
    removed = conditioned_set | marginalized_set
    retained = tuple(node for node in nodes if node not in removed)
    adjacency: dict[str, list[tuple[str, bool, bool]]] = {
        node: [] for node in nodes
    }
    directed_children: dict[str, set[str]] = {node: set() for node in nodes}
    for first, arrow_first, second, arrow_second in edges:
        adjacency[first].append((second, arrow_first, arrow_second))
        adjacency[second].append((first, arrow_second, arrow_first))
        if not arrow_first and arrow_second:
            directed_children[first].add(second)
        elif arrow_first and not arrow_second:
            directed_children[second].add(first)

    def has_directed_path(source: str, targets: frozenset[str]) -> bool:
        def visit(current: str, visited: frozenset[str]) -> bool:
            for child in directed_children[current]:
                if child in visited:
                    continue
                if child in targets:
                    return True
                if visit(child, visited | frozenset({child})):
                    return True
            return False

        return source in targets or visit(source, frozenset({source}))

    def is_anterior_to_any(source: str, targets: frozenset[str]) -> bool:
        def visit(current: str, visited: frozenset[str]) -> bool:
            for neighbor, arrow_current, _ in adjacency[current]:
                if arrow_current or neighbor in visited:
                    continue
                if neighbor in targets:
                    return True
                if visit(neighbor, visited | frozenset({neighbor})):
                    return True
            return False

        return source in targets or visit(source, frozenset({source}))

    def has_inducing_path(left: str, right: str) -> bool:
        collider_targets = conditioned_set | frozenset({left, right})

        def visit(
            current: str,
            incoming_arrow: bool | None,
            visited: frozenset[str],
        ) -> bool:
            for neighbor, arrow_current, arrow_neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                if incoming_arrow is not None:
                    collider = incoming_arrow and arrow_current
                    if collider:
                        if not has_directed_path(current, collider_targets):
                            continue
                    elif current not in marginalized_set:
                        continue
                if neighbor == right:
                    return True
                if visit(
                    neighbor,
                    arrow_neighbor,
                    visited | frozenset({neighbor}),
                ):
                    return True
            return False

        return visit(left, None, frozenset({left}))

    result: list[MixedEdgeAtom] = []
    for left, right in itertools.combinations(retained, 2):
        if not has_inducing_path(left, right):
            continue
        arrow_left = not is_anterior_to_any(
            left, conditioned_set | frozenset({right})
        )
        arrow_right = not is_anterior_to_any(
            right, conditioned_set | frozenset({left})
        )
        result.append((left, arrow_left, right, arrow_right))
    return retained, tuple(result)


def inducing_path_is_valid_by_definition(
    nodes: tuple[str, ...],
    edges: tuple[MixedEdgeAtom, ...],
    path: Iterable[str],
    conditioned: Iterable[str],
    marginalized: Iterable[str],
) -> bool:
    """Check one stated R&S inducing path without searching for alternatives."""

    path_tuple = tuple(path)
    if len(path_tuple) < 2 or len(set(path_tuple)) != len(path_tuple):
        return False
    if any(vertex not in nodes for vertex in path_tuple):
        return False
    conditioned_set = frozenset(conditioned)
    marginalized_set = frozenset(marginalized)
    if path_tuple[0] in conditioned_set | marginalized_set or (
        path_tuple[-1] in conditioned_set | marginalized_set
    ):
        return False

    marks: dict[tuple[str, str], tuple[bool, bool]] = {}
    directed_children: dict[str, set[str]] = {node: set() for node in nodes}
    for first, arrow_first, second, arrow_second in edges:
        marks[(first, second)] = (arrow_first, arrow_second)
        marks[(second, first)] = (arrow_second, arrow_first)
        if not arrow_first and arrow_second:
            directed_children[first].add(second)
        elif arrow_first and not arrow_second:
            directed_children[second].add(first)

    collider_targets = conditioned_set | frozenset(
        (path_tuple[0], path_tuple[-1])
    )

    def is_directed_ancestor(source: str) -> bool:
        def visit(current: str, visited: frozenset[str]) -> bool:
            for child in directed_children[current]:
                if child in visited:
                    continue
                if child in collider_targets:
                    return True
                if visit(child, visited | frozenset({child})):
                    return True
            return False

        return source in collider_targets or visit(
            source, frozenset({source})
        )

    for previous, current, following in zip(
        path_tuple, path_tuple[1:], path_tuple[2:]
    ):
        previous_edge = marks.get((previous, current))
        following_edge = marks.get((current, following))
        if previous_edge is None or following_edge is None:
            return False
        collider = previous_edge[1] and following_edge[0]
        if collider:
            if not is_directed_ancestor(current):
                return False
        elif current not in marginalized_set:
            return False
    return marks.get((path_tuple[0], path_tuple[1])) is not None and (
        marks.get((path_tuple[-2], path_tuple[-1])) is not None
    )


def m_separation_signature_by_all_conditioning_subsets(
    nodes: tuple[str, ...], edges: tuple[MixedEdgeAtom, ...]
) -> ConditionalIndependenceSignature:
    """Return the label-based singleton CI model by literal path search."""

    labels = tuple(sorted(nodes))
    signature: list[tuple[str, str, tuple[str, ...], bool]] = []
    for left_index, right_index in itertools.combinations(range(len(labels)), 2):
        left = labels[left_index]
        right = labels[right_index]
        remaining = tuple(
            node
            for index, node in enumerate(labels)
            if index not in (left_index, right_index)
        )
        for size in range(len(remaining) + 1):
            for given in itertools.combinations(remaining, size):
                signature.append(
                    (
                        left,
                        right,
                        given,
                        m_separated_by_simple_edge_paths(
                            nodes,
                            edges,
                            (left,),
                            (right,),
                            given,
                        ),
                    )
                )
    return tuple(signature)


def full_pag_mark_intersection_by_definition(
    nodes: tuple[str, ...],
    members: Iterable[tuple[MixedEdgeAtom, ...]],
) -> tuple[EndpointMarkAtom, ...]:
    """Intersect endpoint marks over one nonempty FULL MAG class.

    An invariant tail or arrowhead is retained.  An endpoint is a circle
    exactly when both a tail witness and an arrowhead witness occur among
    the class members.  This oracle enumerates class members and does not
    use orientation rules or production PAG code.
    """

    positions = {node: index for index, node in enumerate(nodes)}
    member_tuple = tuple(tuple(edges) for edges in members)
    if not member_tuple:
        raise ValueError("a FULL MAG class must be nonempty")

    def mark_map(
        edges: tuple[MixedEdgeAtom, ...],
    ) -> dict[tuple[str, str], tuple[bool, bool]]:
        result: dict[tuple[str, str], tuple[bool, bool]] = {}
        for first, arrow_first, second, arrow_second in edges:
            if positions[first] < positions[second]:
                pair = (first, second)
                marks = (arrow_first, arrow_second)
            else:
                pair = (second, first)
                marks = (arrow_second, arrow_first)
            if pair in result:
                raise ValueError("a MAG class member repeats an adjacency")
            result[pair] = marks
        return result

    maps = tuple(mark_map(edges) for edges in member_tuple)
    skeleton = frozenset(maps[0])
    if any(frozenset(marks) != skeleton for marks in maps[1:]):
        raise ValueError("a FULL MAG class must have one common skeleton")

    def intersected_mark(values: set[bool]) -> str:
        if values == {False}:
            return "tail"
        if values == {True}:
            return "arrow"
        if values == {False, True}:
            return "circle"
        raise ValueError("an endpoint mark must be a tail or arrowhead")

    return tuple(
        (
            left,
            intersected_mark({marks[(left, right)][0] for marks in maps}),
            right,
            intersected_mark({marks[(left, right)][1] for marks in maps}),
        )
        for left, right in itertools.combinations(nodes, 2)
        if (left, right) in skeleton
    )


def full_pag_classes_by_all_conditioning_subsets(
    nodes: tuple[str, ...],
    mags: Iterable[tuple[MixedEdgeAtom, ...]],
) -> dict[ConditionalIndependenceSignature, FullPAGOracleClass]:
    """Group MAGs by literal all-Z CI models and intersect their marks."""

    grouped: dict[
        ConditionalIndependenceSignature,
        list[tuple[MixedEdgeAtom, ...]],
    ] = {}
    for edges in mags:
        normalized = tuple(edges)
        signature = m_separation_signature_by_all_conditioning_subsets(
            nodes, normalized
        )
        grouped.setdefault(signature, []).append(normalized)
    return {
        signature: (
            tuple(members),
            full_pag_mark_intersection_by_definition(nodes, members),
        )
        for signature, members in grouped.items()
    }


@cache
def no_selection_census(nodes: tuple[str, ...]) -> NoSelectionCensus:
    """Enumerate all tiny no-selection endpoint graphs and FULL MAG classes."""

    pairs = tuple(itertools.combinations(nodes, 2))
    raw_count = 0
    ancestral_count = 0
    mags: list[tuple[MixedEdgeAtom, ...]] = []
    for choices in itertools.product(
        range(len(NO_SELECTION_STATES)), repeat=len(pairs)
    ):
        raw_count += 1
        edges = tuple(
            (left, state[0], right, state[1])
            for (left, right), choice in zip(pairs, choices, strict=True)
            if (state := NO_SELECTION_STATES[choice]) is not None
        )
        if not endpoint_graph_is_ancestral_by_definition(nodes, edges):
            continue
        ancestral_count += 1
        if nonmaximal_pair_by_all_conditioning_sets(nodes, edges) is None:
            mags.append(edges)
    normalized = tuple(mags)
    return NoSelectionCensus(
        raw_count,
        ancestral_count,
        normalized,
        full_pag_classes_by_all_conditioning_subsets(nodes, normalized),
    )


def directed_edge_visibility_paths_by_simple_paths(
    nodes: tuple[str, ...],
    edges: tuple[EndpointMarkAtom, ...],
    source: str,
    target: str,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate every path satisfying Zhang's visibility definition.

    This is deliberately a literal, factorial-time test oracle. It does not
    use orientation rules, production adjacency helpers, or a production path
    search. A candidate path starts at a vertex nonadjacent to ``target`` and
    ends at ``source`` with an arrowhead at ``source``. Every nonendpoint
    vertex must be a collider on that path and an exact directed parent of
    ``target``. Chords do not invalidate a path.
    """

    positions = {node: index for index, node in enumerate(nodes)}
    if len(positions) != len(nodes):
        raise ValueError("visibility-oracle nodes must be unique")
    if source not in positions or target not in positions:
        raise ValueError("visibility-oracle endpoints must be vertices")
    if source == target:
        raise ValueError("visibility-oracle endpoints must be distinct")

    marks: dict[tuple[str, str], tuple[str, str]] = {}
    adjacency: set[frozenset[str]] = set()
    for first, first_mark, second, second_mark in edges:
        if first not in positions or second not in positions or first == second:
            raise ValueError("visibility-oracle edge endpoints are invalid")
        if first_mark not in {"tail", "arrow", "circle"} or second_mark not in {
            "tail",
            "arrow",
            "circle",
        }:
            raise ValueError("visibility-oracle endpoint mark is invalid")
        pair = frozenset((first, second))
        if pair in adjacency:
            raise ValueError("visibility-oracle repeats an adjacency")
        adjacency.add(pair)
        marks[(first, second)] = (first_mark, second_mark)
        marks[(second, first)] = (second_mark, first_mark)

    if marks.get((source, target)) != ("tail", "arrow"):
        raise ValueError("visibility oracle requires a directed query edge")

    def adjacent(left: str, right: str) -> bool:
        return frozenset((left, right)) in adjacency

    def arrow_at(vertex: str, neighbor: str) -> bool:
        edge = marks.get((vertex, neighbor))
        return edge is not None and edge[0] == "arrow"

    def exact_parent(parent: str, child: str) -> bool:
        return marks.get((parent, child)) == ("tail", "arrow")

    candidates: list[tuple[str, ...]] = []
    for external in nodes:
        if external in (source, target) or adjacent(external, target):
            continue
        available_internal = tuple(
            node for node in nodes if node not in (external, source, target)
        )
        for internal_count in range(len(available_internal) + 1):
            for internal in itertools.permutations(available_internal, internal_count):
                path = (external, *internal, source)
                if any(
                    not adjacent(left, right)
                    for left, right in itertools.pairwise(path)
                ):
                    continue
                if not arrow_at(source, path[-2]):
                    continue
                if any(
                    not arrow_at(current, previous)
                    or not arrow_at(current, following)
                    or not exact_parent(current, target)
                    for previous, current, following in zip(path, path[1:], path[2:])
                ):
                    continue
                candidates.append(path)

    return tuple(
        sorted(
            candidates,
            key=lambda path: (
                len(path),
                tuple(positions[vertex] for vertex in path),
            ),
        )
    )


def directed_edge_visibility_witness_by_simple_paths(
    nodes: tuple[str, ...],
    edges: tuple[EndpointMarkAtom, ...],
    source: str,
    target: str,
) -> tuple[str, ...] | None:
    """Return the shortest, then declared-order visibility path, if any."""

    paths = directed_edge_visibility_paths_by_simple_paths(nodes, edges, source, target)
    return paths[0] if paths else None
