"""Graphical recovery and testability criteria for missing-data graphs."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from ._validation import node_positions, normalize_seeds, require_vertex
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .errors import GraphValidationError


@dataclass(frozen=True, init=False, slots=True)
class MissingnessGraph:
    """A missing-data graph with the deterministic proxies suppressed.

    Map each partially observed substantive variable to its response indicator.
    The remaining vertices are fully observed substantive variables. Latent
    causes are represented by bidirected edges in the ADMG.
    """

    topology: DirectedBidirectedGraph
    response_pairs: tuple[tuple[str, str], ...]

    def __init__(
        self, graph: DAG | AcyclicDirectedBidirectedGraph, responses: Mapping[str, str]
    ) -> None:
        if not isinstance(graph, (DAG, AcyclicDirectedBidirectedGraph)):
            raise GraphValidationError("missingness requires a validated DAG or ADMG")
        if not isinstance(responses, Mapping):
            raise GraphValidationError("responses must map true variables to indicators")
        topology = graph.topology
        positions = node_positions(topology.nodes)
        pairs = {
            require_vertex(positions, variable): require_vertex(positions, indicator)
            for variable, indicator in responses.items()
        }
        indicators = frozenset(pairs.values())
        if len(indicators) != len(pairs) or indicators & pairs.keys():
            raise GraphValidationError("responses must be distinct and separate from true variables")
        if any(source in indicators and target not in indicators
               for source, target in topology.directed):
            raise GraphValidationError("response indicators cannot point to substantive variables")
        object.__setattr__(self, "topology", topology)
        object.__setattr__(
            self, "response_pairs", tuple((node, pairs[node]) for node in topology.nodes if node in pairs)
        )


def find_colluding_path(graph: MissingnessGraph, variable: str) -> tuple[str, ...] | None:
    """Find a path from a true variable to its indicator with only interior colliders."""
    topology = graph.topology
    responses = dict(graph.response_pairs)
    indicator = responses[require_vertex(responses, variable)]
    if (variable, indicator) in topology.directed:
        return (variable, indicator)

    spouses: dict[str, set[str]] = {node: set() for node in topology.nodes}
    for left, right in topology.bidirected:
        if {left, right} == {variable, indicator}:
            return (variable, indicator)
        spouses[left].add(right)
        spouses[right].add(left)

    positions = node_positions(topology.nodes)
    endpoints = {variable, indicator}
    outgoing = (topology.directed_children(variable) | spouses[variable]) - endpoints
    incoming = (topology.directed_children(indicator) | spouses[indicator]) - endpoints
    prior = {vertex: variable for vertex in outgoing}
    queue = deque(sorted(outgoing, key=positions.__getitem__))
    while queue:
        current = queue.popleft()
        if current in incoming:
            path = [indicator, current]
            while current != variable:
                current = prior[current]
                path.append(current)
            return tuple(reversed(path))
        # Both ends of a step between interior vertices must be arrowheads.
        unseen = spouses[current] - endpoints - prior.keys()
        for neighbor in sorted(unseen, key=positions.__getitem__):
            prior[neighbor] = current
            queue.append(neighbor)
    return None


def missing_data_full_law_obstructions(graph: MissingnessGraph) -> dict[str, tuple[str, ...]]:
    """Find one colluding path per obstructed variable (Nabi et al., Theorem 4)."""
    return {
        variable: path
        for variable, _ in graph.response_pairs
        if (path := find_colluding_path(graph, variable)) is not None
    }


@dataclass(frozen=True, slots=True)
class MohanResponseFactor:
    indicator: str
    parents: tuple[str, ...]
    observed_parent_indicators: tuple[str, ...]


def mohan_joint_recovery_factors(
    graph: MissingnessGraph,
) -> tuple[MohanResponseFactor, ...] | None:
    """Return Eq. (6)'s denominator factors, or None when self-censoring occurs.

    Mohan, Pearl and Tian (2013), Theorem 2. The response indicators may
    have substantive parents but may not share directed edges or have
    latent parents. The recovering functional requires complete-case support.
    """
    topology = graph.topology
    responses = dict(graph.response_pairs)
    indicators = frozenset(responses.values())
    if any(source in indicators for source, _ in topology.directed):
        raise GraphValidationError("Mohan et al. Theorem 2 excludes edges between response indicators")
    if any(left in indicators or right in indicators for left, right in topology.bidirected):
        raise GraphValidationError("Mohan et al. Theorem 2 excludes latent parents of response indicators")
    if any((variable, indicator) in topology.directed for variable, indicator in graph.response_pairs):
        return None

    factors = []
    for _, indicator in graph.response_pairs:
        parents = topology.directed_parents(indicator)
        response_parents = {responses[var] for var in parents if var in responses}
        factors.append(MohanResponseFactor(
            indicator,
            tuple(node for node in topology.nodes if node in parents),
            tuple(node for node in topology.nodes if node in response_parents),
        ))
    return tuple(factors)


def mohan_direct_test_witnesses(
    graph: MissingnessGraph,
    left: str | Iterable[str],
    right: str | Iterable[str],
    given: str | Iterable[str] = (),
) -> tuple[str, ...]:
    """List comparison variables satisfying the direct-testability criterion.

    Applies the three syntactic conditions of Mohan and Pearl (2014),
    Theorem 1. Indicators required to reveal the compared variables stay
    fixed at their observed value. The proposed independence has its
    own graphical implication and statistical support.
    """
    positions = node_positions(graph.topology.nodes)
    x, y, z = (normalize_seeds(positions, part) for part in (left, right, given))
    if not x or not y or x & y or x & z or y & z:
        raise GraphValidationError("conditional independence needs disjoint sets and nonempty endpoints")

    responses = dict(graph.response_pairs)
    rx, ry, rz = ({responses[node] for node in part if node in responses} for part in (x, y, z))
    if not rx <= x | y | z or not ry | rz <= y | z:
        return ()
    candidates = y - rx - ry - rz
    return tuple(node for node in graph.topology.nodes if node in candidates)


def mohan_binary_independence_interval(
    first: tuple[int | Fraction, int | Fraction, int | Fraction],
    second: tuple[int | Fraction, int | Fraction, int | Fraction],
) -> tuple[Fraction, Fraction] | None:
    """Return the common P(Y=0) values for an independent binary completion.

    Each row for the fully observed X contains mass for Y=0, Y=1 and Y
    missing, in that order. The inputs may use counts or probabilities.
    Mohan and Pearl (2014), Theorem 3.
    """
    bounds = []
    for row in (first, second):
        if len(row) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, Fraction)) or value < 0
            for value in row
        ):
            raise GraphValidationError("a binary response row needs three nonnegative rational cells")
        total = sum(row, Fraction())
        if not total:
            raise GraphValidationError("each fully observed binary stratum needs positive mass")
        observed_zero, _, missing = row
        bounds.append((observed_zero / total, (observed_zero + missing) / total))
    lower = max(left for left, _ in bounds)
    upper = min(right for _, right in bounds)
    return (lower, upper) if lower <= upper else None
