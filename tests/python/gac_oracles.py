"""Literal GAC definitions on original graphs; no production reductions."""
from itertools import combinations


def paths(nodes, edges, start, ends, prohibited=()):
    adjacency = {n: [] for n in nodes}
    for a, x, b, y in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    def extend(path):
        if len(path) > 1 and path[-1] in ends:
            yield path
        for neighbor in adjacency[path[-1]]:
            if neighbor not in path and neighbor not in prohibited:
                yield from extend(path + (neighbor,))
    yield from extend((start,))


def literal_gac(nodes, edges, x, y, z, kind):
    marks = {}
    for a, ma, b, mb in edges:
        marks[a, b] = ma
        marks[b, a] = mb
    directed = {(a, b) for (a, b), ma in marks.items()
                if ma == "tail" and marks[b, a] == "arrow"}
    def possible(path):
        return all(marks[a, b] != "arrow" for a, b in zip(path, path[1:]))
    def visibility(a, b):
        for v in nodes:
            if v in (a, b) or (v, b) in marks:
                continue
            for path in paths(nodes, edges, v, {a}):
                if marks[a, path[-2]] != "arrow":
                    continue
                if all(marks[m, p] == marks[m, q] == "arrow" and (m, b) in directed
                       for p, m, q in zip(path, path[1:], path[2:])):
                    return True
        return False
    proper_paths = tuple(path for start in x
                         for path in paths(nodes, edges, start, y, x))
    causal = tuple(path for path in proper_paths if possible(path))
    if any((marks[path[0], path[1]], marks[path[1], path[0]]) != ("tail", "arrow")
           or (kind in ("mag", "pag") and not visibility(path[0], path[1]))
           for path in causal):
        return False
    forbidden = set()
    for path in causal:
        for v in path[1:]:
            forbidden.add(v)
            forbidden.update(p[-1] for p in paths(nodes, edges, v, set(nodes)) if possible(p))
    if z & (forbidden | x | y):
        return False
    ancestors = set(z)
    for v in nodes:
        if any(all((a, b) in directed for a, b in zip(p, p[1:]))
               for p in paths(nodes, edges, v, z)):
            ancestors.add(v)
    for path in proper_paths:
        if possible(path):
            continue
        active = True
        for prev, middle, following in zip(path, path[1:], path[2:]):
            left, right = marks[middle, prev], marks[middle, following]
            if left == right == "arrow":
                if middle not in ancestors:
                    active = False
                    break
            elif "tail" in (left, right) or (
                left == right == "circle" and (prev, following) not in marks):
                if middle in z:
                    active = False
                    break
            else:
                active = False  # indefinite status: outside the blocking condition
                break
        if active:
            return False
    return True


def endpoint_atoms(graph, cpdag=False):
    return tuple((a, "circle" if cpdag and x == y else x.value,
                  b, "circle" if cpdag and x == y else y.value)
                 for a, x, b, y in graph.topology.edges)


def latent_dag_atoms(graph):
    raw = graph.topology
    nodes = list(raw.nodes)
    edges = [(a, "tail", b, "arrow") for a, b in raw.directed]
    for k, (a, b) in enumerate(sorted(raw.bidirected)):
        latent = f"oracle_latent_{k}"
        assert latent not in nodes
        nodes.append(latent)
        edges.extend(((latent, "tail", a, "arrow"), (latent, "tail", b, "arrow")))
    return tuple(nodes), tuple(edges)
