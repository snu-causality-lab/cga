"""LaPlante--Perković (AISTATS 2024) conditional adjustment, not general CIDP.

The conditioning set must exclude every possible treatment descendant. The source
Theorems 3/9 characterize exactly this adjustment functional; Theorems 5(a)/10
construct a set whenever one exists. These calls assume a causal interpretation,
no latent selection, and the usual positive-density/query-support interpretation.
They accept no data and cannot verify causal correctness or positivity. Literal
path traversal is bounded explicitly and has no polynomial-time claim.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from typing import Literal, TypeAlias, final

from ._immutable import _ImmutableValue
from ._published_paths import _PathSearch, _positive_limit
from ._validation import node_positions, normalize_seeds
from .directed_bidirected import DAG
from .endpoint_marked import EndpointMarkedGraph, Mark
from .errors import GraphValidationError
from .pag import CausalPAG
from .pdag import CPDAG, MPDAG
from .probability import Probability, ProbabilityExpression, Product, Summation
from .visibility import _directed_edge_visibility_path


class ConditionalAdjustmentDomainError(ValueError):
    """Unsupported proof view, query roles, or possible-descendant conditioning."""


class ConditionalAdjustmentResourceLimitError(RuntimeError):
    """The explicit search budget was exhausted; this is not an adjustment negative."""


ConditionalAdjustmentGraph: TypeAlias = DAG | CPDAG | MPDAG | CausalPAG
_TOKEN = object()


@final
@dataclass(frozen=True, slots=True)
class ConditionalAdjustmentSet(_ImmutableValue):
    """An adjustment set S for P(Y | do(X), Z), with its observational formula."""
    graph: ConditionalAdjustmentGraph
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    vertices: tuple[str, ...]
    formula: ProbabilityExpression
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("conditional-adjustment results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ConditionalAdjustmentSet cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class ConditionalAdjustmentNotFound(_ImmutableValue):
    """No set for this conditional adjustment formula; general CID may succeed."""
    graph: ConditionalAdjustmentGraph
    interventions: tuple[str, ...]
    outcomes: tuple[str, ...]
    conditioned_on: tuple[str, ...]
    reason: Literal["not_amenable", "no_adjustment_set"]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("conditional-adjustment results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ConditionalAdjustmentNotFound cannot be subclassed")


ConditionalAdjustmentResult: TypeAlias = ConditionalAdjustmentSet | ConditionalAdjustmentNotFound


def _ordered(raw: EndpointMarkedGraph, values: frozenset[str]) -> tuple[str, ...]:
    return tuple(v for v in raw.nodes if v in values)


def _query(graph: ConditionalAdjustmentGraph, interventions: str | Iterable[str],
           outcomes: str | Iterable[str], conditioned_on: str | Iterable[str],
           max_path_states: int,
           ) -> tuple[_PathSearch, frozenset[str], frozenset[str], frozenset[str]]:
    if not isinstance(graph, (DAG, CPDAG, MPDAG, CausalPAG)):
        raise ConditionalAdjustmentDomainError("requires a DAG, CPDAG, MPDAG, or no-selection CausalPAG proof view")
    raw = (EndpointMarkedGraph(nodes=graph.topology.nodes, edges=(
        (a, Mark.TAIL, b, Mark.ARROW) for a, b in graph.topology.directed))
        if isinstance(graph, DAG) else graph.topology)
    positions = node_positions(raw.nodes)
    try:
        x = normalize_seeds(positions, interventions)
        y = normalize_seeds(positions, outcomes)
        z = normalize_seeds(positions, conditioned_on)
    except GraphValidationError as error:
        raise ConditionalAdjustmentDomainError(str(error)) from error
    if not x or not y or x & y or (x | y) & z:
        raise ConditionalAdjustmentDomainError("X and Y must be nonempty and X,Y,Z pairwise disjoint")
    search = _PathSearch(raw, _positive_limit(max_path_states, "max_path_states",
                         ConditionalAdjustmentDomainError), ConditionalAdjustmentResourceLimitError)
    if z & search.descendants(x):
        raise ConditionalAdjustmentDomainError("conditioning contains a possible descendant of treatment")
    return search, x, y, z


def _conditions(search: _PathSearch, x: frozenset[str], y: frozenset[str], *, pag: bool
                ) -> tuple[bool, frozenset[str]]:
    mediators: set[str] = set()
    for path in search.paths(x, excluded=x):
        if len(path) < 2 or path[-1] not in y:
            continue
        if search.raw.edge(path[0], path[1]) != (Mark.TAIL, Mark.ARROW):
            return False, frozenset()
        if pag and _directed_edge_visibility_path(search.raw, path[0], path[1]) is None:
            return False, frozenset()
        mediators.update(path[1:])
    return True, search.descendants(frozenset(mediators))


def _valid(search: _PathSearch, x: frozenset[str], y: frozenset[str], z: frozenset[str],
           s: frozenset[str], forbidden: frozenset[str]) -> bool:
    if s & (x | y | z | forbidden):
        return False
    given = s | z
    ancestors = search.raw.directed_ancestors_reflexive(given)
    for path in search.paths(x, excluded=x, possible=False):
        if len(path) < 2 or path[-1] not in y:
            continue
        causal = all((edge := search.raw.edge(a, b)) is None or edge[0] is not Mark.ARROW
                     for i, a in enumerate(path) for b in path[i + 1:])
        if causal:
            continue
        active = True
        for a, b, c in zip(path, path[1:], path[2:]):
            left, right = search.raw.edge(b, a), search.raw.edge(b, c)
            assert left is not None and right is not None
            if left[0] is right[0] is Mark.ARROW:
                if b not in ancestors:
                    active = False
                    break
            elif left == (Mark.TAIL, Mark.ARROW) or right == (Mark.TAIL, Mark.ARROW) or (
                left[0] is left[1] and right[0] is right[1]
                and search.raw.edge(a, c) is None
            ):
                if b in given:
                    active = False
                    break
            else:
                active = False  # Indefinite status is outside the criterion.
                break
        if active:
            return False
    return True


def conditional_adjustment_set_is_valid(
    graph: ConditionalAdjustmentGraph, *, interventions: str | Iterable[str],
    outcomes: str | Iterable[str], conditioned_on: str | Iterable[str],
    adjustment: str | Iterable[str], max_path_states: int = 100_000,
) -> bool:
    """Check the source's complete conditional adjustment criterion in its stated domain.

    False rejects S, not identification by arbitrary formulas. Conditioning on a
    possible treatment descendant raises a domain error, rather than False.
    """
    search, x, y, z = _query(graph, interventions, outcomes, conditioned_on, max_path_states)
    try:
        s = normalize_seeds(node_positions(search.raw.nodes), adjustment)
    except GraphValidationError as error:
        raise ConditionalAdjustmentDomainError(str(error)) from error
    amenable, forbidden = _conditions(search, x, y, pag=isinstance(graph, CausalPAG))
    return amenable and _valid(search, x, y, z, s, forbidden)


def find_conditional_adjustment_set(
    graph: ConditionalAdjustmentGraph, *, interventions: str | Iterable[str],
    outcomes: str | Iterable[str], conditioned_on: str | Iterable[str],
    max_path_states: int = 100_000,
) -> ConditionalAdjustmentResult:
    """Construct Theorem 5(a)/10's explicit S, or prove no S in this formula class.

    The returned set need not be minimal, cheapest, or variance-optimal. No
    observability restriction is inferred. Unsupported queries and path exhaustion
    raise distinct exceptions. Positivity and causal interpretation are caller premises.
    """
    search, x, y, z = _query(graph, interventions, outcomes, conditioned_on, max_path_states)
    metadata = (graph, _ordered(search.raw, x), _ordered(search.raw, y), _ordered(search.raw, z))
    amenable, forbidden = _conditions(search, x, y, pag=isinstance(graph, CausalPAG))
    if not amenable:
        return ConditionalAdjustmentNotFound(*metadata, "not_amenable", _token=_TOKEN)
    z_ancestors = (search.ancestors(z) if isinstance(graph, CausalPAG)
                   else search.raw.directed_ancestors_reflexive(z))
    s = (search.ancestors(x | y) | z_ancestors) - forbidden - x - y - z
    if not _valid(search, x, y, z, s, forbidden):
        return ConditionalAdjustmentNotFound(*metadata, "no_adjustment_set", _token=_TOKEN)
    selected = _ordered(search.raw, s)
    first = Probability(_ordered(search.raw, y), conditioned_on=_ordered(search.raw, x | z | s))
    formula: ProbabilityExpression = first
    if selected:
        formula = Summation(selected, Product((first, Probability(selected, conditioned_on=_ordered(search.raw, z)))))
    return ConditionalAdjustmentSet(*metadata, selected, formula, _token=_TOKEN)
