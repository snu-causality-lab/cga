"""DAG classes from literal orders, skeletons, and unshielded colliders."""
from functools import cache
from itertools import combinations, permutations, product


def signature(nodes, directed):
    skeleton = frozenset(frozenset(e) for e in directed)
    colliders = frozenset((frozenset((a, c)), b)
                         for b in nodes for a, c in combinations(nodes, 2)
                         if (a, b) in directed and (c, b) in directed
                         and frozenset((a, c)) not in skeleton)
    return skeleton, colliders


@cache
def dag_classes(nodes):
    pairs = tuple(combinations(nodes, 2))
    orders = tuple({n: i for i, n in enumerate(p)} for p in permutations(nodes))
    groups = {}
    for states in product((0, 1, 2), repeat=len(pairs)):
        directed = frozenset((a, b) if s == 1 else (b, a)
                             for (a, b), s in zip(pairs, states) if s)
        if any(all(p[a] < p[b] for a, b in directed) for p in orders):
            groups.setdefault(signature(nodes, directed), []).append(directed)
    return groups


def consistent_members(raw, knowledge=None):
    directed = set()
    skeleton = set()
    for a, x, b, y in raw.edges:
        skeleton.add(frozenset((a, b)))
        if (x.value, y.value) == ("tail", "arrow"):
            directed.add((a, b))
        elif (x.value, y.value) == ("arrow", "tail"):
            directed.add((b, a))
    # Shields belong to the full PDAG skeleton, including undirected edges.
    expected = frozenset((frozenset((a, c)), b)
                         for b in raw.nodes for a, c in combinations(raw.nodes, 2)
                         if (a, b) in directed and (c, b) in directed
                         and frozenset((a, c)) not in skeleton)
    candidates = dag_classes(raw.nodes).get((frozenset(skeleton), expected), ())
    return tuple(d for d in candidates if directed <= d
                 and (knowledge is None or (set(knowledge.required) <= d
                      and not set(knowledge.forbidden) & d)))


def invariant_edges(members):
    return frozenset.intersection(*members) if members else frozenset()
