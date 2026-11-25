"""Corrected LSAS sufficient rules with an exact independence oracle.

The causal meaning requires Markov/faithfulness and pretreatment assumptions.
No statistical CI tests, effect estimation or general completeness claim are
provided. See docs/LSAS_SELECTION.md for the proposed source corrections.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import InitVar, dataclass, field
from itertools import combinations
from typing import Literal, NoReturn, Self, SupportsIndex, TypeAlias, final


IndependenceOracle: TypeAlias = Callable[[str, str, frozenset[str]], bool]
_TOKEN = object()


class LSASDomainError(ValueError):
    """The observed labels, query, or oracle violate the LSAS input contract."""


class LSASOracleError(TypeError):
    """The supplied independence callable returned something other than bool."""


class _ImmutableResult:
    __slots__ = ()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("pickle is unsupported for LSAS results")


@final
@dataclass(frozen=True, slots=True)
class LSASAdjustmentSet(_ImmutableResult):
    """An R1 adjustment set, possibly empty; not a nonzero-effect claim."""

    observed: tuple[str, ...]
    treatment: str
    outcome: str
    adjustment: tuple[str, ...]
    witness: str
    status: Literal["adjustment"] = field(default="adjustment", init=False)
    rule: Literal["R1"] = field(default="R1", init=False)
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("LSAS results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("LSASAdjustmentSet cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class LSASZeroEffect(_ImmutableResult):
    """An R2 conclusion of no causal influence, conditional on the premises."""

    observed: tuple[str, ...]
    treatment: str
    outcome: str
    conditioning: tuple[str, ...]
    witness: str | None
    rule: Literal["R2(i)", "R2(ii)"]
    status: Literal["zero"] = field(default="zero", init=False)
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("LSAS results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("LSASZeroEffect cannot be subclassed")


@final
@dataclass(frozen=True, slots=True)
class LSASUnresolved(_ImmutableResult):
    """No local rule applies; this does not assert nonidentification."""

    observed: tuple[str, ...]
    treatment: str
    outcome: str
    reason: Literal["no_rule_applies"] = field(default="no_rule_applies", init=False)
    status: Literal["unknown"] = field(default="unknown", init=False)
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("LSAS results are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("LSASUnresolved cannot be subclassed")


LSASResult: TypeAlias = LSASAdjustmentSet | LSASZeroEffect | LSASUnresolved


def _subsets(nodes: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    for size in range(len(nodes) + 1):
        yield from combinations(nodes, size)


def find_lsas_adjustment(
    observed: Sequence[str], *, treatment: str, outcome: str,
    independent: IndependenceOracle,
) -> LSASResult:
    """Search corrected LSAS sufficient rules using exact CI.

    The oracle receives disjoint query labels and frozen conditioning sets;
    True means independence. It must describe a Markov-faithful acyclic causal
    model without selection, with outcome not an ancestor of treatment and
    every other observed vertex a nondescendant of both. These assumptions
    are not verified from the callable. Oracle exceptions propagate.

    Under these assumptions, R1 returns a valid adjustment set and R2 implies
    no causal influence. R1 does not imply a nonzero ATE, minimum set size or
    minimum estimator variance. Numerical use of an adjustment functional
    requires suitable support/positivity, which this selector does not check.

    R2(i) is searched without a witness requirement. Then witnesses follow
    observed order; each conditioning search starts with the empty set and
    increases in size. Blankets use total-conditioning CI. Symmetric queries
    are memoized. No matches means unresolved, not nonidentified. There is no
    subset cap or general completeness guarantee; search is exponential.
    """
    if isinstance(observed, (str, bytes, bytearray)) or not isinstance(observed, Sequence):
        raise LSASDomainError("observed must be an ordered sequence of labels")
    nodes = tuple(observed)
    if any(not isinstance(node, str) or not node for node in nodes):
        raise LSASDomainError("observed labels must be nonempty strings")
    if len(set(nodes)) != len(nodes):
        raise LSASDomainError("observed labels must be unique")
    if (
        not isinstance(treatment, str) or not isinstance(outcome, str)
        or treatment not in nodes or outcome not in nodes or treatment == outcome
    ):
        raise LSASDomainError("treatment and outcome must be distinct observed labels")
    if not callable(independent):
        raise LSASDomainError("independent must be callable")

    positions = {node: index for index, node in enumerate(nodes)}
    universe = frozenset(nodes)
    answers: dict[tuple[str, str, frozenset[str]], bool] = {}

    def ci(left: str, right: str, given: frozenset[str]) -> bool:
        if positions[left] > positions[right]:
            left, right = right, left
        key = (left, right, given)
        if key not in answers:
            answer = independent(left, right, given)
            if type(answer) is not bool:
                raise LSASOracleError("independent must return a built-in bool")
            answers[key] = answer
        return answers[key]

    mb_y = tuple(node for node in nodes if node != outcome
                 and not ci(outcome, node, universe - {outcome, node}))
    candidates = tuple(node for node in mb_y if node != treatment)

    # R2(i) has no witness variable and must work when MB(X) is empty.
    for conditioning in _subsets(candidates):
        if ci(treatment, outcome, frozenset(conditioning)):
            return LSASZeroEffect(
                nodes, treatment, outcome, conditioning, None, "R2(i)", _token=_TOKEN,
            )

    mb_x = tuple(node for node in nodes if node != treatment
                 and not ci(treatment, node, universe - {treatment, node}))
    for witness in mb_x:
        if witness == outcome:
            continue
        local = tuple(node for node in candidates if node != witness)
        for conditioning in _subsets(local):
            given = frozenset(conditioning)
            if ci(witness, outcome, given):
                if not ci(witness, treatment, given):
                    return LSASZeroEffect(
                        nodes, treatment, outcome, conditioning, witness, "R2(ii)",
                        _token=_TOKEN,
                    )
            elif ci(witness, outcome, given | {treatment}):
                return LSASAdjustmentSet(
                    nodes, treatment, outcome, conditioning, witness, _token=_TOKEN,
                )

    return LSASUnresolved(nodes, treatment, outcome, _token=_TOKEN)
