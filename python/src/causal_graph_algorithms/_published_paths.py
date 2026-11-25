"""Literal paths with the nonlocal arrowhead condition of the cited papers."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from .endpoint_marked import EndpointMarkedGraph, Mark


@dataclass(slots=True)
class _PathSearch:
    raw: EndpointMarkedGraph
    limit: int
    error: type[RuntimeError]
    visited: int = 0

    def paths(self, starts: frozenset[str], *, excluded: frozenset[str] = frozenset(),
              possible: bool = True, first_undirected: bool = False) -> Iterator[tuple[str, ...]]:
        pending: deque[tuple[str, ...]] = deque()
        for start in self.raw.nodes:
            if start in starts:
                self._append(pending, (start,))
        while pending:
            path = pending.popleft()
            yield path
            for following in self.raw.nodes:
                if following in path or following in excluded:
                    continue
                if self.raw.edge(path[-1], following) is None:
                    continue
                if len(path) == 1 and first_undirected and self.raw.edge(path[0], following) != (Mark.TAIL, Mark.TAIL):
                    continue
                if possible and any(
                    (edge := self.raw.edge(earlier, following)) is not None
                    and edge[0] is Mark.ARROW for earlier in path
                ):
                    continue
                self._append(pending, (*path, following))

    def _append(self, pending: deque[tuple[str, ...]], path: tuple[str, ...]) -> None:
        self.visited += 1
        if self.visited > self.limit:
            raise self.error("literal path-state budget exhausted; no mathematical decision returned")
        pending.append(path)

    def descendants(self, seeds: frozenset[str]) -> frozenset[str]:
        return frozenset(path[-1] for path in self.paths(seeds))

    def ancestors(self, targets: frozenset[str]) -> frozenset[str]:
        if not targets:
            return frozenset()
        found = set(targets)
        for start in self.raw.nodes:
            if start in found:
                continue
            for path in self.paths(frozenset((start,))):
                if path[-1] in targets:
                    found.add(start)
                    break
        return frozenset(found)


def _positive_limit(value: int, role: str, error: type[ValueError]) -> int:
    if type(value) is not int or value < 1:
        raise error(f"{role} must be a positive integer")
    return value
