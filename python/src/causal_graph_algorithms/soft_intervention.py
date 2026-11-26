"""Augmented MAGs for the two soft-intervention equivalence criteria."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations, count

from ._validation import node_positions, normalize_seeds
from .ancestral import AncestralGraph
from .ancestral_transform import ancestral_marginal_condition_transform
from .directed_bidirected import DAG
from .endpoint_marked import EndpointMarkedGraph, Mark
from .equivalence import mag_markov_equivalent
from .errors import GraphValidationError
from .maximality import MaximalAncestralGraph


def _observed(graph: DAG, observed: str | Iterable[str]) -> frozenset[str]:
    if not isinstance(graph, DAG):
        raise GraphValidationError("soft-intervention equivalence requires a DAG with explicit latent vertices")
    return normalize_seeds(node_positions(graph.topology.nodes), observed)


def _controlled_differences(
    observed: frozenset[str], targets: Iterable[str | Iterable[str]],
) -> tuple[frozenset[str], ...]:
    if isinstance(targets, str):
        raise GraphValidationError("use a collection of target sets, such as ((), 'X')")
    positions = node_positions(tuple(sorted(observed)))
    regimes = tuple(normalize_seeds(positions, target) for target in targets)
    if not regimes:
        raise GraphValidationError("provide at least one target set")
    differences = {left ^ right for left, right in combinations(regimes, 2) if left != right}
    return tuple(sorted(differences, key=lambda vertices: tuple(sorted(vertices))))


def _mechanism_maps(
    observed: frozenset[str], regimes: Iterable[Mapping[str, str | int]],
) -> tuple[dict[str, str | int], ...]:
    maps: list[dict[str, str | int]] = []
    for regime in regimes:
        if not isinstance(regime, Mapping):
            raise GraphValidationError("each regime must map intervention targets to mechanism labels")
        if regime.keys() - observed:
            raise GraphValidationError("mechanism labels must be for observed variables")
        if any(type(label) not in (str, int) for label in regime.values()):
            raise GraphValidationError("mechanism labels must be strings or integers")
        maps.append(dict(regime))
    if not maps:
        raise GraphValidationError("provide at least one regime")
    return tuple(maps)


def _mechanism_differences(
    regimes: tuple[dict[str, str | int], ...],
) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(node for node in left.keys() | right.keys() if left.get(node) != right.get(node))
        for left, right in combinations(regimes, 2)
    )


def _comparison_names(reserved: set[str], size: int) -> tuple[str, ...]:
    names: list[str] = []
    for index in count():
        if len(names) == size:
            break
        name = f"__F{index}"
        if name not in reserved:
            names.append(name)
    return tuple(names)


def _augmented_mag(
    graph: DAG,
    observed: frozenset[str],
    differences: tuple[frozenset[str], ...],
    names: tuple[str, ...],
) -> MaximalAncestralGraph:
    directed = [*graph.topology.directed]
    for f_node, children in zip(names, differences, strict=True):
        directed.extend((f_node, child) for child in children)
    topology = EndpointMarkedGraph(
        nodes=(*graph.topology.nodes, *names),
        edges=((parent, Mark.TAIL, child, Mark.ARROW) for parent, child in directed),
    )
    return ancestral_marginal_condition_transform(
        AncestralGraph.from_graph(topology), marginalized=set(graph.topology.nodes) - observed,
    )


def controlled_soft_intervention_mag(
    graph: DAG,
    *,
    observed: str | Iterable[str],
    targets: Iterable[str | Iterable[str]],
) -> MaximalAncestralGraph:
    """Project Kocaoglu et al. (2019)'s controlled-experiment augmentation."""
    vertices = _observed(graph, observed)
    differences = _controlled_differences(vertices, targets)
    names = _comparison_names(set(graph.topology.nodes), len(differences))
    return _augmented_mag(graph, vertices, differences, names)


def controlled_soft_intervention_markov_equivalent(
    left: DAG,
    right: DAG,
    *,
    observed: str | Iterable[str],
    targets: Iterable[str | Iterable[str]],
) -> bool:
    """Compare the augmented MAGs with the same known controlled targets."""
    vertices = _observed(left, observed)
    _observed(right, vertices)
    differences = _controlled_differences(vertices, targets)
    names = _comparison_names(set(left.topology.nodes) | set(right.topology.nodes), len(differences))
    return mag_markov_equivalent(
        _augmented_mag(left, vertices, differences, names),
        _augmented_mag(right, vertices, differences, names),
    )


def psi_soft_intervention_mag(
    graph: DAG,
    *,
    observed: str | Iterable[str],
    regimes: Iterable[Mapping[str, str | int]],
) -> MaximalAncestralGraph:
    """Project Jaber et al. (2020)'s mechanism-sensitive augmentation."""
    vertices = _observed(graph, observed)
    differences = _mechanism_differences(_mechanism_maps(vertices, regimes))
    names = _comparison_names(set(graph.topology.nodes), len(differences))
    return _augmented_mag(graph, vertices, differences, names)


def psi_markov_equivalent(
    left: DAG,
    left_regimes: Iterable[Mapping[str, str | int]],
    right: DAG,
    right_regimes: Iterable[Mapping[str, str | int]],
    *,
    observed: str | Iterable[str],
) -> bool:
    """Compare causal graphs and proposed regime mechanisms in tuple order."""
    vertices = _observed(left, observed)
    _observed(right, vertices)
    left_maps = _mechanism_maps(vertices, left_regimes)
    right_maps = _mechanism_maps(vertices, right_regimes)
    if len(left_maps) != len(right_maps):
        return False
    left_differences = _mechanism_differences(left_maps)
    right_differences = _mechanism_differences(right_maps)
    names = _comparison_names(
        set(left.topology.nodes) | set(right.topology.nodes), len(left_differences),
    )
    return mag_markov_equivalent(
        _augmented_mag(left, vertices, left_differences, names),
        _augmented_mag(right, vertices, right_differences, names),
    )
