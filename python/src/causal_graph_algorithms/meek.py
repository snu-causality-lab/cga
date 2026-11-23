"""Meek R1--R4 closure with explicit direct-edge background knowledge."""
from __future__ import annotations

from itertools import combinations

from .endpoint_marked import EndpointMarkedGraph, Mark
from .pdag import BackgroundKnowledge as BackgroundKnowledge
from .pdag import (CPDAG, MPDAG, PDAG, PDAGDomainError, PDAGView,
                   NoConsistentExtension, _TOKEN, _colliders, _consistent_extension)


def _close_meek_topology(raw: EndpointMarkedGraph) -> EndpointMarkedGraph | None:
    nodes = raw.nodes
    skeleton = {frozenset((a, b)) for a, _, b, _ in raw.edges}
    directed = {(a, b) if x is Mark.TAIL else (b, a)
                for a, x, b, y in raw.edges if x is not y}
    undirected = {frozenset((a, b)) for a, x, b, y in raw.edges if x is y}

    def adjacent(a: str, b: str) -> bool:
        return frozenset((a, b)) in skeleton

    def unoriented(a: str, b: str) -> bool:
        return frozenset((a, b)) in undirected

    while True:
        proposals: set[tuple[str, str]] = set()
        for a in nodes:
            for b in nodes:
                if not unoriented(a, b):
                    continue
                # R1: c -> a - b, with c and b nonadjacent.
                r1 = any((c, a) in directed and not adjacent(c, b) for c in nodes if c != b)
                # R2: a -> c -> b, a - b.
                r2 = any((a, c) in directed and (c, b) in directed for c in nodes)
                # R3: a-c, a-d, c->b<-d, with c and d nonadjacent.
                candidates = [c for c in nodes if unoriented(a, c) and (c, b) in directed]
                r3 = any(not adjacent(c, d) for c, d in combinations(candidates, 2))
                # R4: a-c, c->d->b, with c and b nonadjacent.
                r4 = any(unoriented(a, c) and not adjacent(c, b)
                         and (c, d) in directed and (d, b) in directed
                         for c in nodes for d in nodes if len({a, b, c, d}) == 4)
                if r1 or r2 or r3 or r4:
                    proposals.add((a, b))
        if not proposals:
            break
        if any((b, a) in proposals for a, b in proposals):
            return None
        directed.update(proposals)
        undirected.difference_update(frozenset(e) for e in proposals)
    return EndpointMarkedGraph(nodes=nodes, edges=(
        (a, Mark.ARROW if (b, a) in directed else Mark.TAIL,
         b, Mark.ARROW if (a, b) in directed else Mark.TAIL)
        for a, b in combinations(nodes, 2) if adjacent(a, b)))


def meek_closure(graph: PDAGView, *, knowledge: BackgroundKnowledge | None = None
                 ) -> MPDAG | NoConsistentExtension:
    """Compute all invariant orientations of the consistent, knowledge-restricted DAGs."""
    if not isinstance(graph, (PDAG, CPDAG, MPDAG)):
        raise PDAGDomainError("Meek closure requires a PDAG, CPDAG, or MPDAG view")
    if knowledge is not None and not isinstance(knowledge, BackgroundKnowledge):
        raise PDAGDomainError("knowledge must be BackgroundKnowledge")
    raw = graph.topology
    original = _colliders(raw)
    background = knowledge if knowledge is not None else BackgroundKnowledge()
    if any(a not in raw.nodes or b not in raw.nodes
           for a, b in (*background.required, *background.forbidden)):
        raise PDAGDomainError("background knowledge contains an unknown vertex")
    if _consistent_extension(raw) is None:
        return NoConsistentExtension(graph, "input_unextendable", knowledge=knowledge, _token=_TOKEN)
    required, forbidden = set(background.required), set(background.forbidden)
    imposed = set(required)
    for a, b in forbidden:
        if raw.edge(a, b) is not None:
            imposed.add((b, a))
    marks = {(a, b): (x, y) for a, x, b, y in raw.edges}
    for a, b in sorted(imposed):
        edge = raw.edge(a, b)
        if edge is None or (b, a) in imposed or edge == (Mark.ARROW, Mark.TAIL):
            return NoConsistentExtension(graph, "background_inconsistent", knowledge=knowledge, _token=_TOKEN)
        first, second = (a, b) if raw.nodes.index(a) < raw.nodes.index(b) else (b, a)
        marks[(first, second)] = ((Mark.TAIL, Mark.ARROW) if first == a
                                 else (Mark.ARROW, Mark.TAIL))
    restricted = EndpointMarkedGraph(nodes=raw.nodes,
        edges=((a, x, b, y) for (a, b), (x, y) in marks.items()))
    if _consistent_extension(restricted, original_colliders=original) is None:
        return NoConsistentExtension(graph, "background_inconsistent", knowledge=knowledge, _token=_TOKEN)
    closed = _close_meek_topology(restricted)
    if closed is None or _consistent_extension(closed, original_colliders=original) is None:
        raise RuntimeError("Meek closure changed a nonempty consistent extension set")
    return MPDAG(closed, _token=_TOKEN)
