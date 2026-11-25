"""Small-universe MAG enumeration with no production graph/traversal imports.

Every pair assignment is completed before testing ancestrality. Maximality is
the definition: every nonadjacent pair must have some m-separating subset of
the other vertices. It is not an inducing-path test. Relabeling enumerates all
permutations fixing each pinned position, not a refinement/canonical-labeling
algorithm. These choices deliberately differ from an optimized enumerator.

These test-only helpers assume valid inputs: nonnegative integer n, exactly
n * (n - 1) // 2 states from 0..4 in each word, and 0 <= pinned_count <= n.
Pins are the first pinned_count vertex positions, each fixed individually.
They are not input validators. Complete corpora and permutation maps stay in
process-local caches; exhaustive callers must keep the universe small.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from functools import cache


Word = tuple[int, ...]
# In the upper triangle: absent, tail-tail, tail-arrow, arrow-tail, arrow-arrow.
MARKS = ((False, False), (False, False), (False, True), (True, False), (True, True))
REVERSED = (0, 1, 3, 2, 4)


@cache
def pairs(n: int) -> tuple[tuple[int, int], ...]:
    return tuple(itertools.combinations(range(n), 2))


def is_mag_by_definition(n: int, word: Word) -> bool:
    """Check a completed word by ancestry and all-subsets m-separation."""

    descendants = [0] * n
    undirected = 0
    arrowheads = 0
    for (left, right), state in zip(pairs(n), word, strict=True):
        if state == 1:
            undirected |= (1 << left) | (1 << right)
        elif state == 2:
            descendants[left] |= 1 << right
            arrowheads |= 1 << right
        elif state == 3:
            descendants[right] |= 1 << left
            arrowheads |= 1 << left
        elif state == 4:
            arrowheads |= (1 << left) | (1 << right)
    if undirected & arrowheads:
        return False
    # Boolean Floyd--Warshall on the completed directed subgraph.
    for middle in range(n):
        for source in range(n):
            if descendants[source] & (1 << middle):
                descendants[source] |= descendants[middle]
    if any(descendants[node] & (1 << node) for node in range(n)):
        return False
    for (left, right), state in zip(pairs(n), word, strict=True):
        if state == 4 and (
            descendants[left] & (1 << right) or descendants[right] & (1 << left)
        ):
            return False

    adjacency: list[list[tuple[int, bool, bool]]] = [[] for _ in range(n)]
    for (left, right), state in zip(pairs(n), word, strict=True):
        if state:
            left_arrow, right_arrow = MARKS[state]
            adjacency[left].append((right, left_arrow, right_arrow))
            adjacency[right].append((left, right_arrow, left_arrow))

    for (left, right), state in zip(pairs(n), word, strict=True):
        if state:
            continue
        # A path is open iff its noncolliders avoid Z and all colliders have
        # a directed descendant in Z (including themselves).
        paths: list[tuple[int, int]] = []

        def visit(
            current: int,
            seen: int,
            incoming_arrow: bool,
            colliders: int,
            noncolliders: int,
        ) -> None:
            for neighbor, arrow_here, arrow_there in adjacency[current]:
                if seen & (1 << neighbor):
                    continue
                next_colliders, next_noncolliders = colliders, noncolliders
                if current != left:
                    if incoming_arrow and arrow_here:
                        next_colliders |= 1 << current
                    else:
                        next_noncolliders |= 1 << current
                if neighbor == right:
                    paths.append((next_colliders, next_noncolliders))
                else:
                    visit(
                        neighbor,
                        seen | (1 << neighbor),
                        arrow_there,
                        next_colliders,
                        next_noncolliders,
                    )

        visit(left, 1 << left, False, 0, 0)
        available = tuple(node for node in range(n) if node not in (left, right))
        for chosen in itertools.product((False, True), repeat=len(available)):
            conditioned = sum(
                1 << node for node, take in zip(available, chosen) if take
            )
            ancestors = conditioned
            for node, reachable in enumerate(descendants):
                if reachable & conditioned:
                    ancestors |= 1 << node
            if not any(
                not (noncolliders & conditioned) and not (colliders & ~ancestors)
                for colliders, noncolliders in paths
            ):
                break
        else:
            return False
    return True


@cache
def naive_mag_words(n: int, include_undirected: bool = True) -> frozenset[Word]:
    """Literal Cartesian universe, with validation only at completed leaves."""

    states = (0, 1, 2, 3, 4) if include_undirected else (0, 2, 3, 4)
    return frozenset(
        word
        for word in itertools.product(states, repeat=len(pairs(n)))
        if is_mag_by_definition(n, word)
    )


@cache
def permutation_maps(
    n: int, pinned_count: int
) -> tuple[tuple[tuple[int, bool], ...], ...]:
    """For every relabeling, locate each new pair in the old upper triangle."""

    pair_index = {pair: index for index, pair in enumerate(pairs(n))}
    result = []
    for free in itertools.permutations(range(pinned_count, n)):
        permutation = tuple(range(pinned_count)) + free
        remap = []
        for left, right in pairs(n):
            old_left, old_right = permutation[left], permutation[right]
            remap.append(
                (pair_index[tuple(sorted((old_left, old_right)))], old_left > old_right)
            )
        result.append(tuple(remap))
    return tuple(result)


def relabelings(word: Word, n: int, pinned_count: int = 0) -> frozenset[Word]:
    return frozenset(
        tuple(
            REVERSED[word[index]] if reverse else word[index]
            for index, reverse in remap
        )
        for remap in permutation_maps(n, pinned_count)
    )


def canonical_word(word: Word, n: int, pinned_count: int = 0) -> Word:
    return min(relabelings(word, n, pinned_count))


def orbit_keys(words: Iterable[Word], n: int, pinned_count: int = 0) -> frozenset[Word]:
    """Return represented orbit keys, which need not belong to the input subset."""

    remaining = set(words)
    representatives = set()
    while remaining:
        orbit = relabelings(remaining.pop(), n, pinned_count)
        representatives.add(min(orbit))
        remaining.difference_update(orbit)
    return frozenset(representatives)
