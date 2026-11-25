"""Small-graph definition-level Runge oracles, with no production traversal calls."""

from __future__ import annotations

from itertools import combinations, permutations, product

from causal_graph_algorithms.directed_bidirected import DirectedBidirectedGraph


def closure(graph: DirectedBidirectedGraph, seeds: set[str], *, reverse: bool = False) -> set[str]:
    result = set(seeds)
    while True:
        expanded = result | {
            (a if reverse else b)
            for a, b in graph.directed
            if (b if reverse else a) in result
        }
        if result == expanded:
            return result
        result = expanded


def edge_choices(graph: DirectedBidirectedGraph, a: str, b: str) -> list[tuple[bool, bool]]:
    choices = []
    if (a, b) in graph.directed:
        choices.append((False, True))
    if (b, a) in graph.directed:
        choices.append((True, False))
    if (a, b) in graph.bidirected or (b, a) in graph.bidirected:
        choices.append((True, True))
    return choices


def open_path_exists(
    graph: DirectedBidirectedGraph, x: str, y: str, given: set[str], *, noncausal: bool = False
) -> bool:
    ancestors = closure(graph, given, reverse=True)
    middle = [v for v in graph.nodes if v not in (x, y)]
    for length in range(len(middle) + 1):
        for interior in permutations(middle, length):
            path = (x, *interior, y)
            for marks in product(*(edge_choices(graph, a, b) for a, b in zip(path, path[1:]))):
                if noncausal and all(edge == (False, True) for edge in marks):
                    continue
                if all(
                    (node in ancestors if marks[k][1] and marks[k + 1][0] else node not in given)
                    for k, node in enumerate(interior)
                ):
                    return True
    return False


def forbidden_nodes(graph: DirectedBidirectedGraph, x: str, y: str) -> set[str]:
    mediators = closure(graph, {x}) & closure(graph, {y}, reverse=True) - {x, y}
    return {x} | closure(graph, {y} | mediators)


def valid_sets(graph: DirectedBidirectedGraph, x: str, y: str, fixed: set[str]) -> list[set[str]]:
    forbidden = forbidden_nodes(graph, x, y)
    eligible = [node for node in graph.nodes if node not in forbidden | fixed]
    return [
        set(subset)
        for size in range(len(eligible) + 1)
        for subset in combinations(eligible, size)
        if not open_path_exists(graph, x, y, fixed | set(subset), noncausal=True)
    ]


def latent_dag_adjustment_valid(
    graph: DirectedBidirectedGraph, x: str, y: str, total: set[str]
) -> bool:
    """Independent latent-DAG moralization oracle for larger paper figures.

    This uses integer-indexed matrices and does not call production separation,
    graph closure, or the edge-state traversal. Latent vertices stay unavailable.
    """
    n = len(graph.nodes)
    index = {node: i for i, node in enumerate(graph.nodes)}
    causal = [[i == j or (a, b) in graph.directed for j, b in enumerate(graph.nodes)]
              for i, a in enumerate(graph.nodes)]
    for k in range(n):
        for i in range(n):
            if causal[i][k]:
                for j in range(n):
                    causal[i][j] = causal[i][j] or causal[k][j]
    xi, yi = index[x], index[y]
    targets = {v for v in range(n) if v != xi and causal[xi][v] and causal[v][yi]}
    forbidden = {xi} | {v for v in range(n) if any(causal[w][v] for w in targets)}
    conditions = {index[v] for v in total}
    if conditions & forbidden:
        return False
    size = n + len(graph.bidirected)
    parents = [set() for _ in range(size)]
    for a, b in graph.directed:
        if not (a == x and index[b] in targets):
            parents[index[b]].add(index[a])
    for latent, (a, b) in enumerate(sorted(graph.bidirected), start=n):
        parents[index[a]].add(latent)
        parents[index[b]].add(latent)
    relevant = {xi, yi} | conditions
    while True:
        extended = relevant | set().union(*(parents[v] for v in relevant))
        if relevant == extended:
            break
        relevant = extended
    available = relevant - conditions
    moral = [[i == j for j in range(size)] for i in range(size)]
    for child in relevant:
        for a in parents[child]:
            if child in available and a in available:
                moral[a][child] = moral[child][a] = True
            for b in parents[child]:
                if a in available and b in available:
                    moral[a][b] = moral[b][a] = True
    for k in available:
        for i in available:
            if moral[i][k]:
                for j in available:
                    moral[i][j] = moral[i][j] or moral[k][j]
    return not moral[xi][yi]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(len(vector)):
        pivot = max(range(column, len(vector)), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(len(vector)):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return [row[-1] for row in augmented]


def sem_covariance(
    graph: DirectedBidirectedGraph,
    coefficients: dict[tuple[str, str], float],
    latent_loadings: dict[tuple[str, str], tuple[float, float]],
    variances: dict[str, float],
) -> list[list[float]]:
    """Population covariance of a fully specified linear latent-variable SEM."""
    nodes = graph.nodes
    size = len(nodes)
    index = {v: i for i, v in enumerate(nodes)}
    equations = [[float(i == j) for j in range(size)] for i in range(size)]
    noise = [[variances[nodes[i]] if i == j else 0. for j in range(size)] for i in range(size)]
    for (a, b), coefficient in coefficients.items():
        equations[index[b]][index[a]] -= coefficient
    for (a, b), (loading_a, loading_b) in latent_loadings.items():
        ai, bi = index[a], index[b]
        noise[ai][ai] += loading_a * loading_a
        noise[bi][bi] += loading_b * loading_b
        noise[ai][bi] += loading_a * loading_b
        noise[bi][ai] += loading_a * loading_b
    columns = [_solve(equations, [float(i == j) for i in range(size)]) for j in range(size)]
    transform = [[columns[j][i] for j in range(size)] for i in range(size)]
    return [[
        sum(transform[i][k] * noise[k][column] * transform[j][column]
            for k in range(size) for column in range(size))
        for j in range(size)
    ] for i in range(size)]


def ols_variance(
    graph: DirectedBidirectedGraph, sigma: list[list[float]], x: str, y: str, total: set[str]
) -> float:
    """Population asymptotic OLS variance, omitting the common factor 1/n."""
    positions = {v: i for i, v in enumerate(graph.nodes)}

    def residual(vertex: str, given: set[str]) -> float:
        index = positions[vertex]
        indices = [positions[v] for v in sorted(given)]
        if not indices:
            return sigma[index][index]
        weights = _solve([[sigma[i][j] for j in indices] for i in indices], [sigma[i][index] for i in indices])
        return sigma[index][index] - sum(sigma[index][i] * weight for i, weight in zip(indices, weights))

    return residual(y, total | {x}) / residual(x, total)


def bidirected_paths(
    graph: DirectedBidirectedGraph, start: str, targets: set[str], inside: set[str]
) -> list[tuple[str, ...]]:
    result = []
    candidates = [v for v in graph.nodes if v in inside and v != start]
    for count in range(len(candidates) + 1):
        for interior in permutations(candidates, count):
            for target in targets:
                if target == start or target in interior:
                    continue
                path = (start, *interior, target)
                if all((True, True) in edge_choices(graph, a, b) for a, b in zip(path, path[1:])):
                    result.append(path)
    return result


def components(
    graph: DirectedBidirectedGraph, x: str, y: str, fixed: set[str]
) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    targets = (closure(graph, {x}) & closure(graph, {y}, reverse=True)) - {x}
    forbidden = forbidden_nodes(graph, x, y)
    ancestors = closure(graph, {x, y} | fixed, reverse=True) - forbidden
    parents = {a for a, b in graph.directed if b in targets} - forbidden
    allowed = {
        v for v in graph.nodes
        if v not in forbidden and (v in ancestors or not open_path_exists(graph, x, v, ancestors))
    }
    colliders = set()
    for target in targets:
        for node in allowed:
            if bidirected_paths(graph, target, {node}, allowed):
                colliders.add(node)
    collider_parents = {a for a, b in graph.directed if b in colliders}
    return parents, colliders, collider_parents, ancestors, targets


def criteria(
    graph: DirectedBidirectedGraph, x: str, y: str, fixed: set[str]
) -> tuple[bool, bool]:
    parents, colliders, collider_parents, ancestors, targets = components(graph, x, y, fixed)
    oset = parents | colliders | collider_parents
    n_nodes = {
        v for v in graph.nodes
        if any((True, True) in edge_choices(graph, v, w) for w in targets | colliders)
    } - forbidden_nodes(graph, x, y) - oset - fixed
    first = all(
        not valid_sets(graph, x, y, fixed | set(path[:-1]))
        for node in n_nodes
        for path in bidirected_paths(graph, node, targets, colliders)
    )
    second = True
    for node in oset - parents - fixed:
        if not open_path_exists(graph, x, node, (oset | fixed) - {node}):
            continue
        if any((True, True) in edge_choices(graph, node, w) for w in targets):
            continue
        has_extension = False
        for child in colliders & ancestors - {node}:
            if (node, child) not in graph.directed and (True, True) not in edge_choices(graph, node, child):
                continue
            if bidirected_paths(graph, child, targets, (colliders & ancestors) - {node}):
                has_extension = True
        if not has_extension:
            second = False
    return first, second
