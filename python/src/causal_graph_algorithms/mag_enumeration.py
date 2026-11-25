"""Exhaustive small-graph MAG enumeration from endpoint-marked topology."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeAlias

from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import MAGListingDomainError, MAGListingResourceLimitError
from .maximality import MaximalAncestralGraph


# A pair (left, right), left < right, has these states in this order:
# absent, left -> right, left <- right, left <-> right, left --- right.
_REVERSED = (0, 2, 1, 3, 4)
_MARKS = (
    (Mark.TAIL, Mark.TAIL),  # State zero is omitted from the topology.
    (Mark.TAIL, Mark.ARROW),
    (Mark.ARROW, Mark.TAIL),
    (Mark.ARROW, Mark.ARROW),
    (Mark.TAIL, Mark.TAIL),
)
_AncestralState: TypeAlias = tuple[
    tuple[int, ...], tuple[int, ...], int, int
]


def _extend_ancestral(
    state: _AncestralState, left: int, right: int, code: int
) -> _AncestralState | None:
    """Extend strict descendant/spouse masks and reject irreversible violations."""
    descendants, spouses, arrows, undirected = state
    if code == 0:
        return state
    left_bit, right_bit = 1 << left, 1 << right
    endpoints = left_bit | right_bit
    if code == 4:
        if arrows & endpoints:
            return None
        return descendants, spouses, arrows, undirected | endpoints
    if code == 3:
        if undirected & endpoints:
            return None
        if descendants[left] & right_bit or descendants[right] & left_bit:
            return None
        updated_spouses = list(spouses)
        updated_spouses[left] |= right_bit
        updated_spouses[right] |= left_bit
        return descendants, tuple(updated_spouses), arrows | endpoints, undirected

    source, target = (left, right) if code == 1 else (right, left)
    source_bit, target_bit = 1 << source, 1 << target
    if undirected & target_bit or descendants[target] & source_bit:
        return None
    reachable = descendants[target] | target_bit
    updated_descendants = list(descendants)
    for ancestor, previous in enumerate(descendants):
        if ancestor == source or previous & source_bit:
            following = previous | reachable
            if following & spouses[ancestor]:
                return None
            updated_descendants[ancestor] = following
    return tuple(updated_descendants), spouses, arrows | target_bit, undirected


def _canonical_prefix(matrix: list[list[int]], size: int, pinned: int) -> bool:
    """Test the completed induced prefix under all its allowed permutations.

    A permutation is kept live only while its completed vertex groups equal
    the identity's groups. A smaller group rejects the graph immediately; a
    larger group cannot improve its code. Interchangeable vertices have the
    same marks to every other vertex and a symmetric edge to each other, so
    swapping them is an automorphism and only one choice is necessary.

    This backtracking stores partial permutations, never a factorial table.
    Its time can still be factorial on difficult asymmetric prefixes.
    """
    if size - pinned < 2:
        return True
    classes: dict[int, int] = {}
    representatives: list[int] = []
    for vertex in range(pinned, size):
        for representative in representatives:
            if matrix[vertex][representative] != matrix[representative][vertex]:
                continue
            if all(
                matrix[vertex][other] == matrix[representative][other]
                for other in range(size)
                if other not in (vertex, representative)
            ):
                classes[vertex] = representative
                break
        else:
            classes[vertex] = vertex
            representatives.append(vertex)
    if len(representatives) == 1:
        return True

    # An explicit stack also avoids a recursion-depth bound on the node count.
    order = list(range(pinned))
    used = (1 << pinned) - 1
    candidates = [iter(range(pinned, size))]
    tried_classes: list[set[int]] = [set()]
    while candidates:
        candidate = next(candidates[-1], None)
        if candidate is None:
            candidates.pop()
            tried_classes.pop()
            if order and len(order) > pinned:
                used ^= 1 << order.pop()
            continue
        vertex = candidate
        if used & (1 << vertex) or classes[vertex] in tried_classes[-1]:
            continue
        tried_classes[-1].add(classes[vertex])
        position = len(order)
        comparison = 0
        for earlier, image in enumerate(order):
            comparison = matrix[image][vertex] - matrix[earlier][position]
            if comparison:
                break
        if comparison < 0:
            return False
        if comparison > 0 or position + 1 == size:
            continue
        order.append(vertex)
        used |= 1 << vertex
        candidates.append(iter(range(pinned, size)))
        tried_classes.append(set())
    return True


def _is_maximal(
    descendants: tuple[int, ...],
    spouses: tuple[int, ...],
    pairs: list[tuple[int, int]],
    codes: list[int],
) -> bool:
    """Exclude primitive inducing paths between nonadjacent endpoints.

    In an ancestral graph such a path is bidirected, and all its vertices
    are directed ancestors of either endpoint. The final public view still
    validates maximality independently of these search-state bitmasks.
    """
    for (left, right), code in zip(pairs, codes):
        if code:
            continue
        endpoints = (1 << left) | (1 << right)
        allowed = endpoints
        for vertex, reachable in enumerate(descendants):
            if reachable & endpoints:
                allowed |= 1 << vertex
        reached = 1 << left
        pending = reached
        while pending:
            bit = pending & -pending
            pending ^= bit
            following = spouses[bit.bit_length() - 1] & allowed & ~reached
            if following & (1 << right):
                return False
            reached |= following
            pending |= following
    return True


def _enumerate(
    nodes: tuple[str, ...],
    pinned: int,
    up_to_isomorphism: bool,
    max_output: int | None,
    include_undirected: bool,
) -> Iterator[MaximalAncestralGraph]:
    size = len(nodes)
    pairs = [(left, right) for right in range(size) for left in range(right)]
    codes = [0] * len(pairs)
    next_codes = [0] * len(pairs)
    matrix = [[0] * size for _ in range(size)]
    empty_masks = (0,) * size
    states: list[_AncestralState] = [(empty_masks, empty_masks, 0, 0)]
    state_count = 5 if include_undirected else 4
    emitted = 0
    depth = 0
    while depth >= 0:
        if depth == len(pairs):
            descendants, spouses, _, _ = states[-1]
            if _is_maximal(descendants, spouses, pairs, codes):
                topology = EndpointMarkedGraph(
                    nodes=nodes,
                    edges=(
                        (nodes[left], _MARKS[code][0], nodes[right], _MARKS[code][1])
                        for (left, right), code in zip(pairs, codes)
                        if code
                    ),
                )
                member = MaximalAncestralGraph.from_graph(topology)
                if max_output is not None and emitted >= max_output:
                    raise MAGListingResourceLimitError(
                        "MAG enumeration exceeded max_output; the yielded prefix "
                        "is incomplete"
                    )
                emitted += 1
                yield member
            states.pop()
            depth -= 1
            continue

        left, right = pairs[depth]
        if next_codes[depth] == state_count:
            next_codes[depth] = 0
            matrix[left][right] = matrix[right][left] = 0
            states.pop()
            depth -= 1
            continue
        code = next_codes[depth]
        next_codes[depth] += 1
        extended = _extend_ancestral(states[-1], left, right, code)
        if extended is None:
            continue
        matrix[left][right] = code
        matrix[right][left] = _REVERSED[code]
        if (
            up_to_isomorphism
            and left + 1 == right
            and not _canonical_prefix(matrix, right + 1, min(pinned, right + 1))
        ):
            continue
        codes[depth] = code
        states.append(extended)
        depth += 1


def enumerate_mags(
    n: int,
    *,
    pinned: Sequence[str] = (),
    up_to_isomorphism: bool = True,
    max_output: int | None = None,
    include_undirected: bool = True,
) -> Iterator[MaximalAncestralGraph]:
    """Stream every MAG on ``n`` vertices, optionally modulo isomorphism.

    Nodes are the pinned labels in their supplied order, followed by ``V0``,
    ``V1``, ... with collisions skipped. Each pinned label is fixed
    individually; only unpinned nodes can be permuted. Pinned input is
    snapshotted and all arguments are validated when this function is called.

    The domain is simple endpoint-marked graphs: at most one edge per pair,
    no loops, and no circle marks. By default directed, bidirected, and
    undirected edges are all included. ``include_undirected=False`` restricts
    the domain to directed and bidirected edges. Graphs are never deduplicated
    by Markov equivalence.

    The deterministic pair order is (0, 1), (0, 2), (1, 2), (0, 3), ... .
    Pair states are absent, forward directed, reverse directed, bidirected,
    undirected, in that order. With ``up_to_isomorphism=True``, return exactly
    the least code in each orbit under the allowed permutations; otherwise
    return every labelled graph. Search uses incremental ancestral pruning
    and exact canonicity on completed vertex groups. Memory does not grow
    with the number of outputs and no factorial permutation table is built;
    exhaustive enumeration and isomorphism search can still take exponential
    or factorial time. This is a small-graph operation with no delay guarantee.

    ``max_output`` is a nonnegative output budget, not truncation. Already
    yielded graphs remain valid, but discovering the first additional valid
    output raises ``MAGListingResourceLimitError`` and means the prefix is
    incomplete. Exhausting a universe of exactly that size ends normally.
    Invalid arguments raise ``MAGListingDomainError``.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise MAGListingDomainError("n must be a nonnegative integer")
    if isinstance(pinned, (str, bytes, bytearray)) or not isinstance(pinned, Sequence):
        raise MAGListingDomainError("pinned must be a sequence of unique nonempty strings")
    labels = tuple(pinned)
    if (
        len(labels) > n
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise MAGListingDomainError(
            "pinned must contain at most n unique nonempty strings"
        )
    if not isinstance(up_to_isomorphism, bool):
        raise MAGListingDomainError("up_to_isomorphism must be a bool")
    if not isinstance(include_undirected, bool):
        raise MAGListingDomainError("include_undirected must be a bool")
    if max_output is not None and (
        isinstance(max_output, bool)
        or not isinstance(max_output, int)
        or max_output < 0
    ):
        raise MAGListingDomainError("max_output must be a nonnegative integer or None")
    nodes = list(labels)
    occupied = set(labels)
    suffix = 0
    while len(nodes) < n:
        label = f"V{suffix}"
        suffix += 1
        if label not in occupied:
            nodes.append(label)
    return _enumerate(
        tuple(nodes), len(labels), up_to_isomorphism, max_output, include_undirected
    )
