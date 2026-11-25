from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path

import pytest

from causal_graph_algorithms import (
    LSASAdjustmentSet,
    LSASDomainError,
    LSASOracleError,
    LSASUnresolved,
    LSASZeroEffect,
    find_lsas_adjustment,
)
from . import lsas_oracles as oracle


def select(nodes, directed=(), bidirected=()):
    return find_lsas_adjustment(
        nodes, treatment="X", outcome="Y",
        independent=oracle.ci_oracle(nodes, directed, bidirected),
    )


def test_isolated_treatment_does_not_require_a_witness():
    nodes = ("X", "W", "Y")
    edges = (("W", "Y"),)
    assert not oracle.printed_rules_apply(nodes, edges, ())
    result = select(nodes, edges)
    assert isinstance(result, LSASZeroEffect)
    assert result.rule == "R2(i)"
    assert result.witness is None
    assert result.conditioning == ()


def test_empty_adjustment_set_is_success_not_unresolved():
    result = select(("S", "X", "Y"), (("S", "X"), ("X", "Y")))
    assert isinstance(result, LSASAdjustmentSet)
    assert result.adjustment == ()
    assert result.witness == "S"
    assert result.rule == "R1"


def test_proof_counterexample_has_a_valid_zero_rule():
    result = select(("S", "B", "X", "Y"),
                    (("S", "B"), ("B", "X")), (("B", "Y"), ("X", "Y")))
    assert isinstance(result, LSASZeroEffect)
    assert result.rule == "R2(ii)"
    assert result.witness == "S"
    assert result.conditioning == ()


def test_nonempty_adjustment_set_blocks_observed_confounding():
    nodes = ("S", "Z", "X", "Y")
    edges = (("S", "X"), ("Z", "X"), ("Z", "Y"), ("X", "Y"))
    result = select(nodes, edges)
    assert isinstance(result, LSASAdjustmentSet)
    assert result.adjustment == ("Z",)
    assert result.witness == "S"
    assert oracle.adjustment_valid(nodes, edges, (), "X", "Y", result.adjustment)


def test_exact_gaussian_ci_and_adjusted_effect_use_an_independent_numeric_source():
    # Independent unit Gaussian noises, including latent U:
    # S=eS, Z=U+eZ, X=S+Z+eX, Y=X+U+eY. Direct do(X=x) slope is 1.
    # The covariance below is derived from those equations, not from the
    # production selector or the graph-oracle traversal.
    nodes = ("S", "Z", "X", "Y")
    covariance = tuple(tuple(Fraction(value) for value in row) for row in (
        (1, 0, 1, 1), (0, 2, 2, 3), (1, 2, 4, 5), (1, 3, 5, 8),
    ))
    positions = {node: index for index, node in enumerate(nodes)}

    def conditional_covariance(left, right, given):
        i, j = positions[left], positions[right]
        indices = sorted(positions[node] for node in given)
        value = covariance[i][j]
        if len(indices) == 1:
            k = indices[0]
            value -= covariance[i][k] * covariance[k][j] / covariance[k][k]
        elif len(indices) == 2:
            k, ell = indices
            a, b, d = covariance[k][k], covariance[k][ell], covariance[ell][ell]
            value -= (
                covariance[i][k] * (d * covariance[k][j] - b * covariance[ell][j])
                + covariance[i][ell] * (a * covariance[ell][j] - b * covariance[k][j])
            ) / (a * d - b * b)
        else:
            assert not indices
        return value

    def exact_ci(left, right, given):
        assert left != right and not ({left, right} & given)
        return conditional_covariance(left, right, given) == 0

    result = find_lsas_adjustment(nodes, treatment="X", outcome="Y", independent=exact_ci)
    assert isinstance(result, LSASAdjustmentSet)
    assert result.adjustment == ("Z",) and result.witness == "S"
    adjusted_slope = conditional_covariance("X", "Y", result.adjustment) / (
        conditional_covariance("X", "X", result.adjustment))
    assert adjusted_slope == 1  # Direct intervention equation gives Y=x+U+eY.
    assert covariance[2][3] / covariance[2][2] == Fraction(5, 4)

    # Separately verify observed faithfulness on all 24 pairwise CI queries.
    graph_ci = oracle.ci_oracle(nodes, (("S", "X"), ("Z", "X"), ("X", "Y")),
                                (("Z", "Y"),))
    count = 0
    for left, right in combinations(nodes, 2):
        remaining = tuple(node for node in nodes if node not in (left, right))
        for size in range(len(remaining) + 1):
            for subset in combinations(remaining, size):
                given = frozenset(subset)
                assert exact_ci(left, right, given) == graph_ci(left, right, given)
                count += 1
    assert count == 24


def test_unresolved_is_not_a_nonidentification_claim():
    # Even a true unconfounded X -> Y is observationally indistinguishable
    # from confounding when there is no additional witness variable.
    result = select(("X", "Y"), (("X", "Y"),))
    assert isinstance(result, LSASUnresolved)
    assert result.reason == "no_rule_applies"


def test_conditional_zero_case_precedes_witness_search():
    result = select(("Z", "X", "Y"), (("Z", "X"), ("Z", "Y")))
    assert isinstance(result, LSASZeroEffect)
    assert result.rule == "R2(i)"
    assert result.conditioning == ("Z",)


def test_paper_fixtures_keep_printed_code_and_repaired_outcomes_distinct():
    path = Path(__file__).parents[1] / "fixtures/papers/li_2025_lsas.json"
    data = json.loads(path.read_text())
    for case in data["cases"]:
        result = select(case["nodes"], case["directed"], case["bidirected"])
        assert result.status == case["corrected_status"]
        assert oracle.printed_rules_apply(
            case["nodes"], case["directed"], case["bidirected"]
        ) == case["printed_rules_apply"]
    # The upstream implementation's behavior is pinned external evidence,
    # not inferred from this implementation or asserted to be rerun here.
    assert data["upstream_commit"] == "213c04d69136f171c8fb6c8323bf8e205e4c1d3a"
    assert data["cases"][0]["upstream_outcome"] == "zero"
    assert data["cases"][1]["upstream_outcome"] == "unresolved"


@pytest.mark.parametrize("nodes,treatment,outcome", [
    ("XY", "X", "Y"),
    ({"X", "Y"}, "X", "Y"),
    ((), "X", "Y"),
    (("X", "X", "Y"), "X", "Y"),
    (("X", "", "Y"), "X", "Y"),
    (("X", 1, "Y"), "X", "Y"),
    (("X", "Y"), "missing", "Y"),
    (("X", "Y"), "X", "missing"),
    (("X", "Y"), "X", "X"),
    (("X", "Y"), [], "Y"),
])
def test_invalid_inputs_fail_before_querying_the_oracle(nodes, treatment, outcome):
    def never(*args):
        pytest.fail("invalid input reached the oracle")
    with pytest.raises(LSASDomainError):
        find_lsas_adjustment(nodes, treatment=treatment, outcome=outcome, independent=never)


def test_noncallable_oracle_is_a_domain_error():
    with pytest.raises(LSASDomainError):
        find_lsas_adjustment(("X", "Y"), treatment="X", outcome="Y", independent=None)


@pytest.mark.parametrize("answer", [None, 0, 1, [], "false", object()])
def test_oracle_must_return_a_builtin_bool(answer):
    with pytest.raises(LSASOracleError, match="bool"):
        find_lsas_adjustment(("X", "Y"), treatment="X", outcome="Y",
                             independent=lambda *_: answer)


def test_arbitrary_oracle_errors_are_not_relabelled():
    class OracleFailure(Exception):
        pass
    def fails(*args):
        raise OracleFailure("upstream")
    with pytest.raises(OracleFailure, match="upstream"):
        find_lsas_adjustment(("X", "Y"), treatment="X", outcome="Y", independent=fails)


def test_queries_are_disjoint_immutable_and_memoized():
    nodes = ("S", "Z", "X", "Y")
    edges = (("S", "X"), ("Z", "X"), ("Z", "Y"), ("X", "Y"))
    ci = oracle.ci_oracle(nodes, edges)
    seen = set()
    def checked(left, right, given):
        key = (frozenset((left, right)), given)
        assert key not in seen
        seen.add(key)
        return ci(left, right, given)
    result = find_lsas_adjustment(nodes, treatment="X", outcome="Y", independent=checked)
    assert isinstance(result, LSASAdjustmentSet)


def test_order_snapshot_and_results_are_immutable():
    nodes = ["T", "S", "X", "Y"]
    edges = (("T", "X"), ("S", "X"), ("X", "Y"))
    result = select(nodes, edges)
    assert isinstance(result, LSASAdjustmentSet)
    assert result.witness == "T"
    nodes.append("later")
    assert result.observed == ("T", "S", "X", "Y")
    assert result.treatment == "X" and result.outcome == "Y"
    assert copy.copy(result) is result
    assert copy.deepcopy(result) is result
    with pytest.raises(FrozenInstanceError):
        result.adjustment = ("T",)
    with pytest.raises(TypeError, match="factory"):
        LSASAdjustmentSet(("X", "Y"), "X", "Y", (), "S")


@pytest.mark.parametrize("nodes", [("S", "X", "Y"), ("S", "Z", "X", "Y")])
def test_small_pretreatment_universes_against_independent_causal_checks(nodes):
    statuses = set()
    count = 0
    for directed, bidirected in oracle.pretreatment_graphs(nodes):
        result = select(nodes, directed, bidirected)
        statuses.add(result.status)
        count += 1
        if isinstance(result, LSASAdjustmentSet):
            assert oracle.adjustment_valid(nodes, directed, bidirected, "X", "Y", result.adjustment), (
                directed, bidirected, result)
        elif isinstance(result, LSASZeroEffect):
            assert not oracle.has_directed_path(directed, "X", "Y"), (directed, bidirected, result)
        else:
            assert isinstance(result, LSASUnresolved)
    assert statuses == {"adjustment", "zero", "unknown"}
    assert count == (64 if len(nodes) == 3 else 4096)
