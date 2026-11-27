"""Single-world intervention templates for directed acyclic graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import final

from ._immutable import _ImmutableValue
from ._validation import node_positions, normalize_nodes, normalize_seeds
from .directed_bidirected import DAG
from .errors import GraphValidationError
from .identification import _topological_order


_FACTORY_TOKEN = object()


@final
@dataclass(frozen=True, slots=True)
class SWIGVertex(_ImmutableValue):
    """A random or fixed vertex identified by its original variable and kind."""

    variable: str
    is_fixed: bool = False

    def __post_init__(self) -> None:
        normalize_nodes((self.variable,))
        if not isinstance(self.is_fixed, bool):
            raise GraphValidationError("is_fixed must be a Boolean")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("SWIGVertex cannot be subclassed")


@final
@dataclass(frozen=True, slots=True, init=False)
class SWIGTemplate(_ImmutableValue):
    """The immutable vertex lists, directed edges, and labels of a SWIG template."""

    random_vertices: tuple[SWIGVertex, ...]
    fixed_vertices: tuple[SWIGVertex, ...]
    directed: tuple[tuple[SWIGVertex, SWIGVertex], ...]
    _labels: Mapping[str, tuple[str, ...]] = field(hash=False, repr=False)

    def __init__(
        self,
        factory_token: object,
        *,
        random_vertices: tuple[SWIGVertex, ...],
        fixed_vertices: tuple[SWIGVertex, ...],
        directed: tuple[tuple[SWIGVertex, SWIGVertex], ...],
        labels: dict[str, tuple[str, ...]],
    ) -> None:
        if factory_token is not _FACTORY_TOKEN:
            raise TypeError("SWIGTemplate values are constructed by swig_template")
        object.__setattr__(self, "random_vertices", random_vertices)
        object.__setattr__(self, "fixed_vertices", fixed_vertices)
        object.__setattr__(self, "directed", directed)
        object.__setattr__(self, "_labels", MappingProxyType(dict(labels)))

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("SWIGTemplate cannot be subclassed")

    def fixed_ancestors(self, vertex: str) -> tuple[str, ...]:
        """Return the original names of fixed ancestors of the random vertex."""
        try:
            return self._labels[vertex]
        except (KeyError, TypeError) as error:
            raise GraphValidationError(f"unknown vertex: {vertex!r}") from error


def swig_template(
    graph: DAG, *, interventions: str | Iterable[str]
) -> SWIGTemplate:
    """Split the selected vertices and label the random vertices by fixed ancestry.

    Follows Richardson and Robins (2013), Section 3.3.1, before assigning
    intervention values. Vertices, edges, and labels follow input vertex order.
    """
    if not isinstance(graph, DAG):
        raise GraphValidationError("SWIG construction requires a validated DAG")

    topology = graph.topology
    nodes = topology.nodes
    positions = node_positions(nodes)
    targets = normalize_seeds(positions, interventions)
    random = {node: SWIGVertex(node) for node in nodes}
    fixed = {node: SWIGVertex(node, is_fixed=True) for node in nodes if node in targets}
    children: dict[str, list[str]] = {node: [] for node in nodes}
    for parent, child in topology.directed:
        children[parent].append(child)

    ancestors: dict[str, set[str]] = {node: set() for node in nodes}
    for parent in _topological_order(topology):
        # The fixed half takes the outgoing edges and is itself a root.
        inherited = {parent} if parent in targets else ancestors[parent]
        for child in children[parent]:
            ancestors[child].update(inherited)

    return SWIGTemplate(
        _FACTORY_TOKEN,
        random_vertices=tuple(random.values()),
        fixed_vertices=tuple(fixed.values()),
        directed=tuple(
            (fixed.get(parent, random[parent]), random[child])
            for parent in nodes
            for child in sorted(children[parent], key=positions.__getitem__)
        ),
        labels={
            node: tuple(sorted(ancestors[node], key=positions.__getitem__)) for node in nodes
        },
    )
