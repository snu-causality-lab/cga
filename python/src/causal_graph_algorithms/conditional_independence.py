"""CI statements, exact oracle adapters, and budgeted graphoid deductions.

Sadeghi--Soo (JMLR 2022), Section 2.2, records the selected axioms. Derivability
under them is not a decision procedure for general probabilistic CI implication.
An underivable statement is not a dependence fact. Intersection and composition
are caller-supplied assumptions, never inferred from an oracle's return type.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from itertools import combinations
from typing import Literal, Protocol, final

from ._immutable import _ImmutableValue
from ._validation import normalize_nodes
from .errors import GraphValidationError, SeparationDomainError
from .separation import SeparationView, _view_topology, m_separated_sets


class CIDomainError(ValueError):
    """Malformed CI statement, universe, assumption, or exact oracle answer."""


class CIResourceLimitError(RuntimeError):
    """A deduction budget was exhausted; no completed closure is returned."""


def _labels(values: str | Iterable[str]) -> frozenset[str]:
    try:
        raw = (values,) if isinstance(values, str) else tuple(values)
        # CI statements have set semantics, but invalid labels are never discarded.
        return frozenset(normalize_nodes(tuple(dict.fromkeys(raw))))
    except (TypeError, GraphValidationError) as error:
        raise CIDomainError("CI labels must be nonempty strings") from error


@final
@dataclass(frozen=True, slots=True, init=False)
class CIStatement(_ImmutableValue):
    """A disjoint set-valued CI statement, canonical modulo symmetry."""
    left: frozenset[str]
    right: frozenset[str]
    given: frozenset[str]

    def __init__(self, left: str | Iterable[str], right: str | Iterable[str],
                 given: str | Iterable[str] = ()) -> None:
        a, b, c = (_labels(v) for v in (left, right, given))
        if a & b or a & c or b & c:
            raise CIDomainError("CI roles must be pairwise disjoint")
        if tuple(sorted(b)) < tuple(sorted(a)):
            a, b = b, a
        object.__setattr__(self, "left", a)
        object.__setattr__(self, "right", b)
        object.__setattr__(self, "given", c)

    @property
    def trivial(self) -> bool:
        return not self.left or not self.right

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CIStatement cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class CIScope(_ImmutableValue):
    """Declared vertex universe; statements themselves do not invent a universe."""
    nodes: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            if isinstance(self.nodes, str):
                raise CIDomainError("scope nodes must be an ordered collection")
            object.__setattr__(self, "nodes", normalize_nodes(self.nodes))
        except GraphValidationError as error:
            raise CIDomainError(str(error)) from error

    def validate(self, statement: CIStatement) -> None:
        if not isinstance(statement, CIStatement):
            raise CIDomainError("expected a CIStatement")
        if not (statement.left | statement.right | statement.given) <= set(self.nodes):
            raise CIDomainError("CI statement contains vertices outside its scope")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CIScope cannot be subclassed")


class CIOracle(Protocol):
    """An exact yes/no oracle; statistical uncertainty needs a separate interface."""
    def __call__(self, statement: CIStatement, /) -> bool: ...


@final
@dataclass(frozen=True, slots=True)
class SeparationCIOracle(_ImmutableValue):
    """Exact separation: True implies CI under Markov; False needs faithfulness for dependence."""
    graph: SeparationView

    def __post_init__(self) -> None:
        try:
            _view_topology(self.graph)
        except SeparationDomainError as error:
            raise CIDomainError(str(error)) from error

    @property
    def scope(self) -> CIScope:
        return CIScope(self.graph.topology.nodes)

    def __call__(self, statement: CIStatement, /) -> bool:
        self.scope.validate(statement)
        return statement.trivial or m_separated_sets(
            self.graph, statement.left, statement.right, statement.given)

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("SeparationCIOracle cannot be subclassed")


@final
class CachedCIOracle:
    """Symmetry-aware bounded cache; preserves the exact underlying oracle answers."""
    def __init__(self, oracle: CIOracle, *, max_entries: int = 10_000) -> None:
        if not callable(oracle) or type(max_entries) is not int or max_entries < 1:
            raise CIDomainError("a callable oracle and positive cache capacity are required")
        self._oracle = oracle
        self._cache: dict[CIStatement, bool] = {}
        self._max_entries = max_entries
        self._calls = 0
        self._cache_hits = 0

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    def __call__(self, statement: CIStatement, /) -> bool:
        if not isinstance(statement, CIStatement):
            raise CIDomainError("expected a CIStatement")
        if statement in self._cache:
            self._cache_hits += 1
            return self._cache[statement]
        answer = self._oracle(statement)
        self._calls += 1
        if type(answer) is not bool:
            raise CIDomainError("an exact CI oracle must return bool, not a score or unknown")
        if len(self._cache) == self._max_entries:
            del self._cache[next(iter(self._cache))]
        self._cache[statement] = answer
        return answer


@final
@dataclass(frozen=True, slots=True)
class GraphoidAxioms(_ImmutableValue):
    """Semi-graphoid axioms plus explicitly assumed intersection/composition."""
    intersection: bool = False
    composition: bool = False

    def __post_init__(self) -> None:
        if type(self.intersection) is not bool or type(self.composition) is not bool:
            raise CIDomainError("axiom flags must be bool")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("GraphoidAxioms cannot be subclassed")


_Rule = Literal["premise", "decomposition", "weak_union", "contraction", "intersection", "composition"]


@final
@dataclass(frozen=True, slots=True)
class CIDerivation(_ImmutableValue):
    statement: CIStatement
    rule: _Rule
    premises: tuple[CIStatement, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CIDerivation cannot be subclassed")


_TOKEN = object()


@final
@dataclass(frozen=True, slots=True)
class CIClosure(_ImmutableValue):
    """Completed selected-axiom closure; absence means underivable, not dependent."""
    scope: CIScope
    axioms: GraphoidAxioms
    statements: frozenset[CIStatement]
    derivations: tuple[CIDerivation, ...]
    rule_checks: int
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("CI closures are factory-created")

    def contains(self, statement: CIStatement) -> bool:
        """Whether the statement is trivial or derived under the stored assumptions."""
        self.scope.validate(statement)
        return statement.trivial or statement in self.statements

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("CIClosure cannot be subclassed")


def _orientations(s: CIStatement) -> tuple[tuple[frozenset[str], frozenset[str], frozenset[str]], ...]:
    return ((s.left, s.right, s.given), (s.right, s.left, s.given))


def derive_ci_closure(
    scope: CIScope, premises: Iterable[CIStatement], *, max_statements: int,
    axioms: GraphoidAxioms = GraphoidAxioms(), max_rule_checks: int = 1_000_000,
) -> CIClosure:
    """Close finite premises under selected axioms, atomically or with a resource error.

    Trivial statements are implicit and do not consume the statement budget.
    The closure is complete for these rules, not for probabilistic CI implication.
    Neither positivity nor composition is established by this operation.
    """
    if not isinstance(scope, CIScope) or not isinstance(axioms, GraphoidAxioms):
        raise CIDomainError("scope and axioms must be their declared immutable types")
    if any(type(v) is not int or v < 1 for v in (max_statements, max_rule_checks)):
        raise CIDomainError("deduction budgets must be positive integers")
    known: dict[CIStatement, CIDerivation] = {}
    checks = 0

    def count() -> None:
        nonlocal checks
        checks += 1
        if checks > max_rule_checks:
            raise CIResourceLimitError("CI rule-check budget exhausted")

    def add(s: CIStatement, rule: _Rule, parents: tuple[CIStatement, ...]) -> None:
        if s.trivial or s in known:
            return
        if len(known) == max_statements:
            raise CIResourceLimitError("CI statement budget exhausted")
        known[s] = CIDerivation(s, rule, parents)

    try:
        initial = tuple(premises)
    except TypeError as error:
        raise CIDomainError("premises must be an iterable of CIStatement values") from error
    for statement in initial:
        scope.validate(statement)
        add(statement, "premise", ())
    changed = True
    while changed:
        before = len(known)
        current = tuple(known)
        for s in current:
            for a, b, c in _orientations(s):
                labels = tuple(sorted(b))
                for k in range(1, len(labels)):
                    for raw_subset in combinations(labels, k):
                        count()
                        subset = frozenset(raw_subset)
                        add(CIStatement(a, subset, c), "decomposition", (s,))
                        add(CIStatement(a, subset, c | (b - subset)), "weak_union", (s,))
                for t in current:
                    for other_a, d, other_c in _orientations(t):
                        count()
                        if a != other_a or b & d:
                            continue
                        if other_c == c | b:
                            add(CIStatement(a, b | d, c), "contraction", (s, t))
                        if axioms.composition and c == other_c:
                            add(CIStatement(a, b | d, c), "composition", (s, t))
                        if axioms.intersection and d <= c and b <= other_c and c - d == other_c - b:
                            add(CIStatement(a, b | d, c - d), "intersection", (s, t))
        changed = len(known) != before
    return CIClosure(scope, axioms, frozenset(known), tuple(known.values()), checks, _token=_TOKEN)
