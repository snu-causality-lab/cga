"""The recanting-district definition of Shpitser (2013)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise

from ._validation import node_positions, normalize_seeds, require_vertex
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG
from .errors import IdentificationDomainError
from .subgraph import induced_subgraph


@dataclass(frozen=True)
class RecantingDistrictWitness:
    """A treatment, a district in graph order, and two proper directed paths."""

    treatment: str
    district: tuple[str, ...]
    selected_path: tuple[str, ...]
    excluded_path: tuple[str, ...]


def _paths_to_outcomes(
    start: tuple[str, str],
    children: dict[str, list[str]],
    outcomes: frozenset[str],
) -> Iterator[tuple[str, ...]]:
    path = list(start)
    pending = [iter(children[start[-1]])]
    if start[-1] in outcomes:
        yield start
    while pending:
        child = next(pending[-1], None)
        if child is None:
            pending.pop()
            path.pop()
            continue
        path.append(child)
        pending.append(iter(children[child]))
        # A directed continuation may end at another outcome.
        if child in outcomes:
            yield tuple(path)


def find_recanting_district(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    *,
    interventions: str | Iterable[str],
    outcomes: str | Iterable[str],
    selected_paths: Iterable[Sequence[str]],
) -> RecantingDistrictWitness | None:
    """Find a witness for Definition 2 of Shpitser (2013), if one exists.

    Each selected path starts at a treatment, ends at an outcome, and
    follows directed edges without visiting another treatment. Duplicate
    paths are ignored. The two witness paths may share their first edge.
    """
    if not isinstance(graph, (AcyclicDirectedBidirectedGraph, DAG)):
        raise IdentificationDomainError("recanting districts require an ADMG or DAG view")
    topology = graph.topology
    positions = node_positions(topology.nodes)
    treatment_set = normalize_seeds(positions, interventions)
    outcome_set = normalize_seeds(positions, outcomes)
    if not treatment_set or not outcome_set or treatment_set & outcome_set:
        raise IdentificationDomainError("interventions and outcomes must be nonempty and disjoint")

    try:
        path_entries = iter(selected_paths)
    except TypeError as error:
        raise IdentificationDomainError("selected_paths must be iterable") from error
    selected: set[tuple[str, ...]] = set()
    for entry in path_entries:
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence):
            raise IdentificationDomainError("each selected path must be an ordered sequence of vertices")
        path = tuple(require_vertex(positions, vertex) for vertex in entry)
        if (
            len(path) < 2
            or path[0] not in treatment_set
            or path[-1] not in outcome_set
            or treatment_set.intersection(path[1:])
            or any(edge not in topology.directed for edge in pairwise(path))
        ):
            raise IdentificationDomainError(f"invalid proper directed path: {path!r}")
        selected.add(path)
    if not selected:
        return None

    untreated = induced_subgraph(topology, positions.keys() - treatment_set)
    region = untreated.directed_ancestors_reflexive(outcome_set)
    region_graph = induced_subgraph(untreated, region)

    district_of: dict[str, tuple[str, ...]] = {}
    selected_in_district: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = {}
    for path in sorted(selected, key=lambda path: tuple(positions[node] for node in path)):
        first_child = path[1]
        if first_child not in district_of:
            district = tuple(sorted(region_graph.district(first_child), key=positions.__getitem__))
            district_of.update((node, district) for node in district)
        selected_in_district.setdefault((path[0], district_of[first_child]), path)

    children: dict[str, list[str]] = {node: [] for node in region_graph.nodes}
    for parent, child in region_graph.directed:
        children[parent].append(child)
    for successors in children.values():
        successors.sort(key=positions.__getitem__)

    first_edges = (
        (parent, child)
        for parent, child in topology.directed
        if parent in treatment_set and child in district_of
    )
    for treatment, child in sorted(first_edges, key=lambda edge: (positions[edge[0]], positions[edge[1]])):
        district = district_of[child]
        matching_path = selected_in_district.get((treatment, district))
        if matching_path is None:
            continue
        # Every prefix in the region can reach an outcome. Before finding
        # an excluded path we only emit paths supplied in selected_paths.
        for path in _paths_to_outcomes((treatment, child), children, outcome_set):
            if path not in selected:
                return RecantingDistrictWitness(treatment, district, matching_path, path)
    return None
