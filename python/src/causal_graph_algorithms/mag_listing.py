from __future__ import annotations

import itertools
from collections.abc import Callable, Collection

from .endpoint_marked import EndpointMarkedGraph, Mark
from .equivalence import mag_markov_equivalent
from .errors import (
    GraphValidationError,
    MAGListingDomainError,
    MAGListingResourceLimitError,
)
from .maximality import MaximalAncestralGraph
from .pag import (
    ClosedPAG,
    _Conclusion,
    _PAGState,
    _RULES,
    _uncovered_pd_reachable,
)


class _ListingState(_PAGState):
    """Exact mutable PMG state used only during MAG listing."""


def _circle_neighbors(state: _PAGState, node: str) -> tuple[str, ...]:
    return tuple(
        neighbor
        for neighbor in state.ordered_neighbors(node)
        if state.mark(node, neighbor) is Mark.CIRCLE
    )


def _directed(state: _PAGState, source: str, target: str) -> bool:
    return state.mark(source, target) is Mark.TAIL and (
        state.mark(target, source) is Mark.ARROW
    )


def _directed_ancestors_reflexive(
    state: _PAGState, seeds: Collection[str]
) -> frozenset[str]:
    parents: dict[str, set[str]] = {node: set() for node in state.nodes}
    for source in state.nodes:
        for target in state.ordered_neighbors(source):
            if _directed(state, source, target):
                parents[target].add(source)

    reached = set(seeds)
    pending = list(seeds)
    while pending:
        current = pending.pop()
        for parent in parents[current]:
            if parent not in reached:
                reached.add(parent)
                pending.append(parent)
    return frozenset(reached)


def _arrow_into(state: _PAGState, node: str) -> frozenset[str]:
    return frozenset(
        neighbor
        for neighbor in state.ordered_neighbors(node)
        if state.mark(node, neighbor) is Mark.ARROW
    )


def _circle_circle(state: _PAGState, left: str, right: str) -> bool:
    return state.mark(left, right) is Mark.CIRCLE and (
        state.mark(right, left) is Mark.CIRCLE
    )


def _circle_into_anchor(
    state: _PAGState, body: str, anchor: str
) -> bool:
    return state.mark(body, anchor) is Mark.CIRCLE and (
        state.mark(anchor, body) is Mark.ARROW
    )


def _bidirected(state: _PAGState, left: str, right: str) -> bool:
    return state.mark(left, right) is Mark.ARROW and (
        state.mark(right, left) is Mark.ARROW
    )


def _has_unbridged_path(
    state: _PAGState,
    *,
    anchors: Collection[str],
    eligible_body: Collection[str],
) -> bool:
    """Decide Wang--Du--Zhou Definition 1 on a transient listing state.

    Attachment checks reject cheap negative cases before path search.  The
    remaining search enumerates circle paths and checks minimality against
    every nonconsecutive adjacency in the full partial mixed graph.  That
    exact check is needed because individual rule and branching states need
    not yet satisfy the compatibility properties of a completed local
    transform.  The worst-case search is intentionally not given a
    polynomial-time contract.
    """
    anchor_nodes = tuple(
        sorted(set(anchors), key=state.positions.__getitem__)
    )
    eligible = set(eligible_body) - set(anchor_nodes)
    if len(anchor_nodes) < 2 or not eligible:
        return False

    for first_anchor in anchor_nodes:
        for second_anchor in anchor_nodes:
            if first_anchor == second_anchor:
                continue
            if state.adjacent(first_anchor, second_anchor) and not (
                _bidirected(state, first_anchor, second_anchor)
            ):
                continue

            first_attached = {
                node
                for node in eligible
                if _circle_into_anchor(state, node, first_anchor)
            }
            second_attached = {
                node
                for node in eligible
                if _circle_into_anchor(state, node, second_anchor)
            }
            if first_attached & second_attached:
                return True

            first_body = {
                node
                for node in first_attached
                if not state.adjacent(node, second_anchor)
            }
            last_body = {
                node
                for node in second_attached
                if not state.adjacent(node, first_anchor)
            }
            if not first_body or not last_body:
                continue
            pending: list[tuple[str, ...]] = [
                (node,)
                for node in reversed(state.nodes)
                if node in first_body
            ]
            while pending:
                path = pending.pop()
                current = path[-1]
                for following in state.ordered_neighbors(current):
                    if (
                        following not in eligible
                        or following in path
                        or not _circle_circle(state, current, following)
                    ):
                        continue
                    if state.adjacent(following, first_anchor):
                        continue
                    if any(
                        state.adjacent(following, earlier)
                        for earlier in path[:-1]
                    ):
                        continue
                    if following in last_body:
                        return True
                    if state.adjacent(following, second_anchor):
                        continue
                    pending.append((*path, following))
    return False


def _r11(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for first in state.nodes:
        source_set = _arrow_into(state, first) | {first}
        ancestors = _directed_ancestors_reflexive(state, source_set)
        for second in _circle_neighbors(state, first):
            if _uncovered_pd_reachable(state, first, second) & ancestors:
                conclusions.append((first, second, Mark.ARROW))
    return conclusions


def _r12(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for first in state.nodes:
        source_set = _arrow_into(state, first) | {first}
        for second in _circle_neighbors(state, first):
            reachable = _uncovered_pd_reachable(state, first, second)
            eligible = reachable - source_set - {second}
            if _has_unbridged_path(
                state,
                anchors=source_set,
                eligible_body=eligible,
            ):
                conclusions.append((first, second, Mark.ARROW))
    return conclusions


def _r13(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for first in state.nodes:
        for endpoint in _circle_neighbors(state, first):
            for middle in state.ordered_neighbors(first):
                if middle == endpoint or not state.adjacent(middle, endpoint):
                    continue
                if state.mark(middle, endpoint) is not Mark.TAIL:
                    continue
                for outside in state.ordered_neighbors(first):
                    if outside in (endpoint, middle):
                        continue
                    if not state.adjacent(outside, middle):
                        continue
                    if state.mark(middle, outside) is not Mark.ARROW:
                        continue
                    if state.adjacent(outside, endpoint):
                        continue
                    conclusions.append((first, endpoint, Mark.TAIL))
    return conclusions


def _prior_base_relation(
    state: _PAGState, relative_to: str
) -> dict[str, set[str]]:
    candidates = set(_circle_neighbors(state, relative_to))
    relation: dict[str, set[str]] = {
        candidate: set() for candidate in candidates
    }
    incoming_to_relative = _arrow_into(state, relative_to)

    for source in candidates:
        possible_reach = _uncovered_pd_reachable(
            state, relative_to, source
        )
        for target in candidates - {source}:
            # Definition 3(1) is direct and does not depend on the
            # possible-directed first step from relative_to to source.
            if _directed(state, source, target):
                relation[source].add(target)
                continue

            if any(
                _directed(state, last, target)
                for last in possible_reach
            ):
                relation[source].add(target)
                continue

            target_set = incoming_to_relative | {target}
            if _has_unbridged_path(
                state,
                anchors=target_set,
                eligible_body=(
                    possible_reach - target_set - {source}
                ),
            ):
                relation[source].add(target)
    return relation


def _prior_vertices(
    state: _PAGState, relative_to: str, target: str
) -> frozenset[str]:
    relation = _prior_base_relation(state, relative_to)
    reverse: dict[str, set[str]] = {
        node: set() for node in relation
    }
    for source, targets in relation.items():
        for endpoint in targets:
            reverse[endpoint].add(source)

    reached: set[str] = set()
    pending = [target]
    while pending:
        current = pending.pop()
        for previous in reverse.get(current, set()):
            if previous not in reached:
                reached.add(previous)
                pending.append(previous)
    reached.discard(target)
    return frozenset(reached)


def _r14(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for relative_to in state.nodes:
        circle_neighbors = set(_circle_neighbors(state, relative_to))
        for endpoint in state.ordered_neighbors(relative_to):
            if not (
                state.mark(relative_to, endpoint) is Mark.CIRCLE
                and state.mark(endpoint, relative_to) is Mark.ARROW
            ):
                continue
            prior = tuple(
                sorted(
                    _prior_vertices(state, relative_to, endpoint),
                    key=state.positions.__getitem__,
                )
            )
            for first, second in itertools.combinations(prior, 2):
                if not state.adjacent(first, second):
                    conclusions.append(
                        (relative_to, endpoint, Mark.TAIL)
                    )
                    break
                if _has_unbridged_path(
                    state,
                    anchors=(first, second),
                    eligible_body=circle_neighbors - {first, second},
                ):
                    conclusions.append(
                        (relative_to, endpoint, Mark.TAIL)
                    )
                    break
    return conclusions


def _r15(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for first in state.nodes:
        for endpoint in state.ordered_neighbors(first):
            if not (
                state.mark(first, endpoint) is Mark.CIRCLE
                and state.mark(endpoint, first) is Mark.ARROW
            ):
                continue
            for middle in state.ordered_neighbors(first):
                if middle == endpoint:
                    continue
                if state.mark(first, middle) is not Mark.TAIL:
                    continue
                if _directed(state, middle, endpoint):
                    conclusions.append((first, endpoint, Mark.TAIL))
                    break
    return conclusions


def _r16(state: _PAGState) -> list[_Conclusion]:
    conclusions: list[_Conclusion] = []
    for middle in state.nodes:
        arrow_neighbors = frozenset(
            neighbor
            for neighbor in state.ordered_neighbors(middle)
            if state.mark(middle, neighbor) is Mark.ARROW
        )
        if not arrow_neighbors:
            continue
        for endpoint in state.ordered_neighbors(middle):
            if not (arrow_neighbors - {endpoint}):
                continue
            if (
                state.mark(middle, endpoint) is Mark.TAIL
                and state.mark(endpoint, middle) is Mark.CIRCLE
            ):
                conclusions.append((endpoint, middle, Mark.ARROW))
            elif (
                state.mark(middle, endpoint) is Mark.CIRCLE
                and state.mark(endpoint, middle) is Mark.TAIL
            ):
                conclusions.append((middle, endpoint, Mark.ARROW))
    return conclusions


_Rule = Callable[[_PAGState], list[_Conclusion]]
_LISTING_RULES: tuple[_Rule, ...] = (
    *_RULES,
    _r11,
    _r12,
    _r13,
    _r14,
    _r15,
    _r16,
)


def _apply_listing_rules(state: _ListingState) -> None:
    maximum_changing_passes = len(state.marks) + 1
    for _ in range(maximum_changing_passes):
        changed = False
        for rule_number, rule in enumerate(_LISTING_RULES, start=1):
            changed = state.apply(rule_number, rule(state)) or changed
        if not changed:
            return
    raise RuntimeError("MAG listing orientation failed to reach a fixed point")


def _perfect_elimination_order(
    topology: EndpointMarkedGraph,
) -> tuple[str, ...]:
    circle_neighbors: dict[str, set[str]] = {
        node: set() for node in topology.nodes
    }
    for left, left_mark, right, right_mark in topology.edges:
        if left_mark is Mark.CIRCLE and right_mark is Mark.CIRCLE:
            circle_neighbors[left].add(right)
            circle_neighbors[right].add(left)

    remaining = set(topology.nodes)
    order: list[str] = []
    while remaining:
        selected: str | None = None
        for candidate in topology.nodes:
            if candidate not in remaining:
                continue
            neighbors = circle_neighbors[candidate] & remaining
            if all(
                right in circle_neighbors[left]
                for left, right in itertools.combinations(neighbors, 2)
            ):
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(
                "closed PAG circle component is not chordal"
            )
        order.append(selected)
        remaining.remove(selected)
    return tuple(order)


def _constructive_mag(pag: ClosedPAG) -> MaximalAncestralGraph:
    """Construct one Definition 4 MAG using only the PAG topology."""
    topology = pag.topology
    elimination_order = _perfect_elimination_order(topology)
    eliminated_at = {
        node: position for position, node in enumerate(elimination_order)
    }
    completed_edges: list[tuple[str, Mark, str, Mark]] = []

    for left, left_mark, right, right_mark in topology.edges:
        if left_mark is Mark.CIRCLE and right_mark is Mark.CIRCLE:
            if eliminated_at[left] < eliminated_at[right]:
                completed_edges.append(
                    (left, Mark.ARROW, right, Mark.TAIL)
                )
            else:
                completed_edges.append(
                    (left, Mark.TAIL, right, Mark.ARROW)
                )
        elif left_mark is Mark.CIRCLE:
            replacement = (
                Mark.ARROW if right_mark is Mark.TAIL else Mark.TAIL
            )
            completed_edges.append((left, replacement, right, right_mark))
        elif right_mark is Mark.CIRCLE:
            replacement = (
                Mark.ARROW if left_mark is Mark.TAIL else Mark.TAIL
            )
            completed_edges.append((left, left_mark, right, replacement))
        else:
            completed_edges.append((left, left_mark, right, right_mark))

    try:
        member = MaximalAncestralGraph.from_graph(
            EndpointMarkedGraph(
                nodes=topology.nodes,
                edges=completed_edges,
            )
        )
    except GraphValidationError as error:
        raise RuntimeError(
            "closed PAG constructive completion did not produce a MAG"
        ) from error
    if ClosedPAG.from_mag(member).topology != topology:
        raise RuntimeError(
            "closed PAG constructive completion changed the equivalence class"
        )
    return member


def _separators_from_member(
    pag: ClosedPAG, member: MaximalAncestralGraph
) -> dict[frozenset[str], frozenset[str]]:
    separators: dict[frozenset[str], frozenset[str]] = {}
    topology = pag.topology
    for left, right in itertools.combinations(topology.nodes, 2):
        if topology.edge(left, right) is None:
            separators[frozenset((left, right))] = (
                member.anterior_strict((left, right))
            )
    return separators


def _first_circle_vertex(state: _ListingState) -> str | None:
    for node in state.nodes:
        if _circle_neighbors(state, node):
            return node
    return None


def _leaf_key(topology: EndpointMarkedGraph) -> tuple[int, ...]:
    bits: list[int] = []
    for _, left_mark, _, right_mark in topology.edges:
        if left_mark is Mark.CIRCLE or right_mark is Mark.CIRCLE:
            raise RuntimeError("MAG listing reached a leaf with circles")
        bits.extend(
            (
                int(left_mark is Mark.ARROW),
                int(right_mark is Mark.ARROW),
            )
        )
    return tuple(bits)


def _preserves_fixed_endpoints(
    root: EndpointMarkedGraph, leaf: EndpointMarkedGraph
) -> bool:
    for left, left_mark, right, right_mark in root.edges:
        leaf_marks = leaf.edge(left, right)
        if leaf_marks is None:
            return False
        leaf_left, leaf_right = leaf_marks
        if left_mark is not Mark.CIRCLE and leaf_left is not left_mark:
            return False
        if right_mark is not Mark.CIRCLE and leaf_right is not right_mark:
            return False
    return True


def _validate_leaf(
    topology: EndpointMarkedGraph,
    *,
    pag: ClosedPAG,
    representative: MaximalAncestralGraph,
) -> MaximalAncestralGraph:
    if not _preserves_fixed_endpoints(pag.topology, topology):
        raise RuntimeError("MAG listing changed a fixed PAG endpoint")
    try:
        member = MaximalAncestralGraph.from_graph(topology)
    except GraphValidationError as error:
        raise RuntimeError("MAG listing produced a non-MAG leaf") from error
    if not mag_markov_equivalent(member, representative):
        raise RuntimeError("MAG listing produced an inequivalent MAG")
    if ClosedPAG.from_mag(member).topology != pag.topology:
        raise RuntimeError("MAG listing produced a MAG outside the closed PAG")
    return member


def list_mag_equivalence_class(
    pag: ClosedPAG,
    *,
    max_members: int,
) -> tuple[MaximalAncestralGraph, ...]:
    """List the FULL MAG equivalence class represented by ``pag`` atomically."""
    if not isinstance(pag, ClosedPAG):
        raise MAGListingDomainError(
            "MAG listing requires a closed PAG proof view"
        )
    if (
        not isinstance(max_members, int)
        or isinstance(max_members, bool)
        or max_members < 0
    ):
        raise MAGListingDomainError(
            "max_members must be a nonnegative integer"
        )

    representative = _constructive_mag(pag)
    root = _ListingState.from_topology(
        pag.topology,
        separators=_separators_from_member(pag, representative),
    )
    outer_pending = [root]
    members: dict[tuple[int, ...], MaximalAncestralGraph] = {}

    while outer_pending:
        outer_state = outer_pending.pop()
        selected = _first_circle_vertex(outer_state)
        if selected is None:
            topology = outer_state.topology()
            key = _leaf_key(topology)
            member = _validate_leaf(
                topology,
                pag=pag,
                representative=representative,
            )
            if key in members:
                raise RuntimeError("MAG listing produced a duplicate leaf")
            if len(members) == max_members:
                raise MAGListingResourceLimitError(
                    "MAG listing exceeded max_members before class completion"
                )
            members[key] = member
            continue

        inner_pending = [outer_state]
        locally_completed: list[_ListingState] = []
        while inner_pending:
            state = inner_pending.pop()
            circle_neighbors = _circle_neighbors(state, selected)
            if not circle_neighbors:
                locally_completed.append(state)
                continue
            endpoint = circle_neighbors[0]
            for wanted in (Mark.ARROW, Mark.TAIL):
                child = state.clone()
                child.marks[(selected, endpoint)] = wanted
                _apply_listing_rules(child)
                inner_pending.append(child)
        outer_pending.extend(locally_completed)

    return tuple(members[key] for key in sorted(members))
