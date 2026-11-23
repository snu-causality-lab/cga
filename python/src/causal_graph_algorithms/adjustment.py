"""General adjustment across known graphs and equivalence classes.

Perković et al. (JMLR 2018), Definitions 2--4 and Lemma 10: amenability and
forbidden sets are checked in the original graph before a representative reduction.
MAG/PAG guarantees quantify over their latent causal models, not merely over a
canonical latent expansion. A known ADMG instead specifies its own latent SCM.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from functools import lru_cache
from typing import Literal, TypeAlias, final

from ._immutable import _ImmutableValue
from ._validation import node_positions, normalize_seeds
from .ancestral import AncestralGraph
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import GraphValidationError
from .maximality import MaximalAncestralGraph
from .pag import CausalPAG
from .pdag import CPDAG, NoConsistentExtension, find_consistent_dag_extension
from .separation import SeparationView, m_separated_sets
from .separators import SeparatorNotFound, find_m_separator, find_minimal_m_separator
from .visibility import _directed_edge_visibility_path


class AdjustmentDomainError(ValueError):
    """Malformed adjustment query or unsupported causal graph semantics."""


AdjustmentGraph: TypeAlias = DAG | AcyclicDirectedBidirectedGraph | MaximalAncestralGraph | CausalPAG | CPDAG
_TOKEN = object()


@final
@dataclass(frozen=True, slots=True)
class AdjustmentSet(_ImmutableValue):
    """A total valid adjustment set; minimality holds relative to mandatory I."""
    graph: AdjustmentGraph
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    required: tuple[str, ...]
    allowed: tuple[str, ...]
    vertices: tuple[str, ...]
    minimal: bool
    forbidden_vertices: tuple[str, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("adjustment results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AdjustmentSet cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class AdjustmentNotFound(_ImmutableValue):
    """No ordinary adjustment set under the constraints, not nonidentification."""
    graph: AdjustmentGraph
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    required: tuple[str, ...]
    allowed: tuple[str, ...]
    reason: Literal["not_amenable", "required_forbidden", "no_separator"]
    forbidden_vertices: tuple[str, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("adjustment results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AdjustmentNotFound cannot be subclassed")


AdjustmentResult: TypeAlias = AdjustmentSet | AdjustmentNotFound


def _ordered(nodes: tuple[str, ...], values: Iterable[str]) -> tuple[str, ...]:
    selected = frozenset(values)
    return tuple(v for v in nodes if v in selected)


def _query(graph: AdjustmentGraph, interventions: str | Iterable[str],
           outcomes: str | Iterable[str]) -> tuple[tuple[str, ...], frozenset[str], frozenset[str]]:
    if not isinstance(graph, (DAG, AcyclicDirectedBidirectedGraph,
                              MaximalAncestralGraph, CausalPAG, CPDAG)):
        raise AdjustmentDomainError("adjustment requires a DAG, known ADMG, no-selection MAG, causal PAG, or CPDAG")
    raw = graph.topology
    if isinstance(graph, MaximalAncestralGraph) and any(
        a is Mark.TAIL and b is Mark.TAIL for _, a, _, b in graph.topology.edges
    ):
        raise AdjustmentDomainError("selection-bias MAGs are outside the generalized adjustment domain")
    positions = node_positions(raw.nodes)
    try:
        x, y = normalize_seeds(positions, interventions), normalize_seeds(positions, outcomes)
    except GraphValidationError as error:
        raise AdjustmentDomainError(str(error)) from error
    if not x or not y or x & y:
        raise AdjustmentDomainError("interventions and outcomes must be nonempty and disjoint")
    return raw.nodes, x, y


def _endpoint(raw: DirectedBidirectedGraph | EndpointMarkedGraph) -> EndpointMarkedGraph:
    if isinstance(raw, EndpointMarkedGraph):
        return raw
    # A bow cannot be collapsed to one edge. This index is only for causal paths,
    # which use the directed part; all bidirected paths remain in the separator view.
    return EndpointMarkedGraph(nodes=raw.nodes,
        edges=((a, Mark.TAIL, b, Mark.ARROW) for a, b in raw.directed))


def _possible_reached(raw: EndpointMarkedGraph, seeds: frozenset[str],
                      excluded: frozenset[str] = frozenset()) -> frozenset[str]:
    reached = set(seeds - excluded)
    pending = list(reached)
    adjacency: dict[str, list[str]] = {v: [] for v in raw.nodes}
    for a, at_a, b, at_b in raw.edges:
        if at_a is not Mark.ARROW:
            adjacency[a].append(b)
        if at_b is not Mark.ARROW:
            adjacency[b].append(a)
    while pending:
        v = pending.pop()
        for w in adjacency[v]:
            if w not in reached and w not in excluded:
                reached.add(w)
                pending.append(w)
    return frozenset(reached)


def _first_edges(raw: EndpointMarkedGraph, x: frozenset[str], y: frozenset[str]
                 ) -> frozenset[tuple[str, str]]:
    first = set()
    for a in x:
        for b in raw.nodes:
            edge = raw.edge(a, b)
            if b not in x and edge is not None and edge[0] is not Mark.ARROW:
                if _possible_reached(raw, frozenset((b,)), x) & y:
                    first.add((a, b))
    return frozenset(first)


def _original_conditions(graph: AdjustmentGraph, x: frozenset[str], y: frozenset[str]
                         ) -> tuple[bool, frozenset[str]]:
    raw = _endpoint(graph.topology)
    first = _first_edges(raw, x, y)
    forbidden = _possible_reached(raw, frozenset(b for _, b in first))
    amenable = all(raw.edge(a, b) == (Mark.TAIL, Mark.ARROW) and
                   (not isinstance(graph, (MaximalAncestralGraph, CausalPAG)) or
                    _directed_edge_visibility_path(raw, a, b) is not None)
                   for a, b in first)
    return amenable, forbidden


@lru_cache(maxsize=128)
def _complete_no_selection_pag(pag: CausalPAG) -> MaximalAncestralGraph:
    """Zhang completion, with MAG validity and causal-PAG roundtrip validation."""
    from .mag_listing import _perfect_elimination_order
    raw = pag.topology
    order = _perfect_elimination_order(raw)
    positions = {v: i for i, v in enumerate(order)}
    edges = []
    for a, at_a, b, at_b in raw.edges:
        if at_a is Mark.CIRCLE and at_b is Mark.CIRCLE:
            at_a, at_b = ((Mark.ARROW, Mark.TAIL) if positions[a] < positions[b]
                          else (Mark.TAIL, Mark.ARROW))
        elif at_a is Mark.CIRCLE:
            at_a = Mark.ARROW if at_b is Mark.TAIL else Mark.TAIL
        elif at_b is Mark.CIRCLE:
            at_b = Mark.ARROW if at_a is Mark.TAIL else Mark.TAIL
        edges.append((a, at_a, b, at_b))
    try:
        member = MaximalAncestralGraph.from_graph(EndpointMarkedGraph(nodes=raw.nodes, edges=edges))
    except GraphValidationError as error:
        raise RuntimeError("causal PAG completion failed MAG validation") from error
    if CausalPAG.from_mag(member).topology != raw:
        raise RuntimeError("causal PAG completion changed its no-selection equivalence class")
    return member


def _proper_backdoor(graph: AdjustmentGraph, x: frozenset[str], y: frozenset[str]
                     ) -> SeparationView:
    member: DAG | AcyclicDirectedBidirectedGraph | MaximalAncestralGraph
    if isinstance(graph, CPDAG):
        extension = find_consistent_dag_extension(graph)
        if isinstance(extension, NoConsistentExtension):
            raise RuntimeError("CPDAG proof view unexpectedly has no extension")
        member = extension
    elif isinstance(graph, CausalPAG):
        member = _complete_no_selection_pag(graph)
    else:
        member = graph
    first = _first_edges(_endpoint(member.topology), x, y)
    raw = member.topology
    if isinstance(raw, DirectedBidirectedGraph):
        topology = DirectedBidirectedGraph(nodes=raw.nodes,
            directed=raw.directed - first, bidirected=raw.bidirected)
        return (DAG.from_graph(topology) if isinstance(member, DAG)
                else AcyclicDirectedBidirectedGraph.from_graph(topology))
    # Deleting edges can destroy maximality, but preserves ancestral validity.
    return AncestralGraph.from_graph(EndpointMarkedGraph(nodes=raw.nodes, edges=(
        (a, at_a, b, at_b) for a, at_a, b, at_b in raw.edges
        if not ((at_a is Mark.TAIL and at_b is Mark.ARROW and (a, b) in first) or
                (at_a is Mark.ARROW and at_b is Mark.TAIL and (b, a) in first)))))


def adjustment_set_is_valid(graph: AdjustmentGraph, *, interventions: str | Iterable[str],
                            outcomes: str | Iterable[str], adjustment: str | Iterable[str]) -> bool:
    """Check ordinary total-effect adjustment with the graph's declared causal semantics."""
    nodes, x, y = _query(graph, interventions, outcomes)
    try:
        z = normalize_seeds(node_positions(nodes), adjustment)
    except GraphValidationError as error:
        raise AdjustmentDomainError(str(error)) from error
    if z & (x | y):
        return False
    amenable, forbidden = _original_conditions(graph, x, y)
    if not amenable or z & forbidden:
        return False
    return m_separated_sets(_proper_backdoor(graph, x, y), x, y, z)


def _find(graph: AdjustmentGraph, interventions: str | Iterable[str], outcomes: str | Iterable[str],
          required: str | Iterable[str], allowed: str | Iterable[str] | None, *, minimal: bool
          ) -> AdjustmentResult:
    nodes, x, y = _query(graph, interventions, outcomes)
    try:
        i = normalize_seeds(node_positions(nodes), required)
        r = (frozenset(nodes) - x - y if allowed is None
             else normalize_seeds(node_positions(nodes), allowed))
    except GraphValidationError as error:
        raise AdjustmentDomainError(str(error)) from error
    if (i | r) & (x | y) or not i <= r:
        raise AdjustmentDomainError("require I ⊆ R, with both excluding query endpoints")
    amenable, forbidden = _original_conditions(graph, x, y)
    metadata = (graph, _ordered(nodes, x), _ordered(nodes, y), _ordered(nodes, i), _ordered(nodes, r))
    forbidden_vertices = _ordered(nodes, forbidden)
    if not amenable:
        return AdjustmentNotFound(*metadata, reason="not_amenable", forbidden_vertices=forbidden_vertices, _token=_TOKEN)
    if i & forbidden:
        return AdjustmentNotFound(*metadata, reason="required_forbidden", forbidden_vertices=forbidden_vertices, _token=_TOKEN)
    finder = find_minimal_m_separator if minimal else find_m_separator
    result = finder(_proper_backdoor(graph, x, y), x, y, required=i, allowed=r - forbidden)
    if isinstance(result, SeparatorNotFound):
        return AdjustmentNotFound(*metadata, reason="no_separator", forbidden_vertices=forbidden_vertices, _token=_TOKEN)
    return AdjustmentSet(*metadata, vertices=result.vertices, minimal=minimal,
                         forbidden_vertices=forbidden_vertices, _token=_TOKEN)


def find_adjustment_set(graph: AdjustmentGraph, *, interventions: str | Iterable[str],
                        outcomes: str | Iterable[str], required: str | Iterable[str] = (),
                        allowed: str | Iterable[str] | None = None) -> AdjustmentResult:
    """Find a valid total adjustment set under inclusion and observability constraints."""
    return _find(graph, interventions, outcomes, required, allowed, minimal=False)


def find_minimal_adjustment_set(graph: AdjustmentGraph, *, interventions: str | Iterable[str],
                                outcomes: str | Iterable[str], required: str | Iterable[str] = (),
                                allowed: str | Iterable[str] | None = None) -> AdjustmentResult:
    """Find an I-minimal adjustment set; neither minimum size nor variance optimality."""
    return _find(graph, interventions, outcomes, required, allowed, minimal=True)
