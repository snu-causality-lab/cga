from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
import itertools
import json
import random
from pathlib import Path

import pytest

from causal_graph_algorithms.directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from causal_graph_algorithms.runge_adjustment import (
    RungeAdjustmentDomainError,
    RungeAdjustmentResourceLimitError,
    RungeAdjustmentResult,
    runge_o_set,
    runge_optimal_adjustment,
)
from tests.python import runge_oracles as oracle


def graph(nodes, directed=(), bidirected=()):
    return AcyclicDirectedBidirectedGraph(DirectedBidirectedGraph(
        nodes=nodes, directed=directed, bidirected=bidirected
    ))


PAPER_GRAPH_LABELS = ("1A", "2A", "2B", "2C", "2D", "2E", "2F", "2G", "S1")
# Visual transcription of arXiv:2102.10324v4 pp. 3, 6, 12.
# PDF SHA-256: 4302151468edfbed2863b30246212e0bd13c2ff1b1571458afbedf1ecba521ac.
PAPER_FIXTURES = json.loads(
    (Path(__file__).parents[1] / "fixtures/papers/runge_2023_figures.json").read_text(encoding="utf-8")
)


def paper_graph(label):
    row = next(row for row in PAPER_FIXTURES["cases"] if row["figure"] == label)
    return (
        graph(row["nodes"], row["directed"], row["bidirected"]),
        tuple(row["conditioned_on"]), tuple(row["o_set"]), row["graphical_optimality"],
    )


@pytest.mark.parametrize("label", PAPER_GRAPH_LABELS)
def test_paper_graph_figures(label):
    """All nine graph panels in the main paper and its theoretical supplement."""
    view, fixed, expected, optimal = paper_graph(label)
    result = runge_optimal_adjustment(view, "X", "Y", conditioned_on=fixed)
    assert result.adjustment_set == expected, label
    assert result.graphical_optimality is optimal
    assert oracle.latent_dag_adjustment_valid(view.topology, "X", "Y", set(expected) | set(fixed))
    if label != "2B":
        alternatives = oracle.valid_sets(view.topology, "X", "Y", set(fixed))
        assert set(expected) in alternatives
        assert (result.criterion_i, result.criterion_ii) == oracle.criteria(view.topology, "X", "Y", set(fixed))
        if label == "1A":
            assert {frozenset(z) for z in alternatives} == {
                frozenset(z) for z in (("Z1",), ("Z2",), ("Z1", "Z2"),
                                      ("Z2", "Z3"), ("Z1", "Z3"), ("Z1", "Z2", "Z3"))
            }
    if label == "2B":
        assert result.parents == ("Z1", "Z2", "Z3", "Z4")
        assert result.colliders == ("Z5", "Z6", "Z8")
        assert result.collider_parents == ("Z1", "Z7", "Z8")
        assert result.forbidden == ("X", "Y", "M", "Z9", "Z10")
        assert result.valid_ancestors == ("S", "Z1", "Z2", "Z3", "Z4", "Z11")
        assert "Z12" not in result.adjustment_set


def test_latent_moral_oracle_matches_original_path_oracle():
    rng = random.Random(24680)
    nodes = ("A", "X", "B", "C", "Y")
    for _ in range(65):
        directed = {pair for pair in itertools.combinations(nodes, 2) if rng.random() < .3}
        directed.add(("X", "Y"))
        view = graph(nodes, directed, [pair for pair in itertools.combinations(nodes, 2) if rng.random() < .24])
        optional = set(nodes) - oracle.forbidden_nodes(view.topology, "X", "Y")
        for size in range(len(optional) + 1):
            for candidate in itertools.combinations(optional, size):
                assert oracle.latent_dag_adjustment_valid(view.topology, "X", "Y", set(candidate)) == (
                    not oracle.open_path_exists(view.topology, "X", "Y", set(candidate), noncausal=True)
                )


def test_s1_comparison_term_i_is_zero_and_term_ii_is_positive():
    """B.2 p13 overstates positivity of term (i); its first argument is empty."""
    view, fixed, _, _ = paper_graph("S1")
    result = runge_optimal_adjustment(view, "X", "Y", conditioned_on=fixed)
    oset = set(result.o_set)
    ancestors = set(result.valid_ancestors)
    assert oset == {"Z2", "Z3"}
    assert ancestors == {"Z1", "Z2", "Z3", "S"}
    # Eq(7)(i) is I(O\vancs ; Y | vancs,X,S), necessarily zero.
    assert not oset - ancestors
    # The nonempty positive term (ii) is I(X ; Z1 | O,S), under faithfulness.
    assert ancestors - oset - set(fixed) == {"Z1"}
    assert oracle.open_path_exists(view.topology, "X", "Z1", oset | set(fixed))
    assert not oracle.open_path_exists(view.topology, "Y", "Z1", oset | set(fixed) | {"X"})


def test_population_covariance_oracle_has_an_analytic_reference():
    view = graph(("X", "Y", "Z"), [("X", "Y")], [("Y", "Z")])
    sigma = oracle.sem_covariance(view.topology, {("X", "Y"): 2.}, {("Y", "Z"): (3., 4.)},
                                  {"X": 1., "Y": 1., "Z": 1.})
    assert sigma == [[1., 2., 0.], [2., 14., 12.], [0., 12., 17.]]
    assert oracle.ols_variance(view.topology, sigma, "X", "Y", set()) == pytest.approx(10.)
    assert oracle.ols_variance(view.topology, sigma, "X", "Y", {"Z"}) == pytest.approx(10. - 144./17.)


@pytest.mark.parametrize("label", PAPER_GRAPH_LABELS)
def test_paper_graph_population_variances_and_nonoptimal_reversals(label):
    """All paper graphs, all valid covariate sets, fixed S and mixed scales/signs."""
    view, fixed, expected, optimal = paper_graph(label)
    eligible = [v for v in view.topology.nodes
                if v not in oracle.forbidden_nodes(view.topology, "X", "Y") | set(fixed)]
    alternatives = [set(subset) for size in range(len(eligible) + 1)
                    for subset in itertools.combinations(eligible, size)
                    if oracle.latent_dag_adjustment_valid(view.topology, "X", "Y", set(subset) | set(fixed))]
    assert alternatives
    if label == "2B":
        assert len(alternatives) == 384
        assert all("Z1" in z and bool(z & {"Z2", "Z11"}) for z in alternatives)
    rng = random.Random(123456 + PAPER_GRAPH_LABELS.index(label))
    # The two nonoptimal graphs use a predeclared coefficient/loading grid;
    # neither its ranges nor inclusion of cases depends on observed rankings.
    scales = tuple(itertools.product((.1, 1., 10.), repeat=2)) if not optimal else ((.2, .2), (1., 1.), (3., 3.), (.2, 3.), (3., .2))
    relative_rankings = []
    for directed_scale, latent_scale in scales:
        coefficients = {edge: rng.choice((-1, 1)) * rng.uniform(.2, 1.8) * directed_scale
                        for edge in sorted(view.topology.directed)}
        loadings = {edge: (rng.uniform(.2, 1.8) * latent_scale,
                           rng.choice((-1, 1)) * rng.uniform(.2, 1.8) * latent_scale)
                    for edge in sorted(view.topology.bidirected)}
        variances = {v: rng.uniform(.5, 2.) for v in view.topology.nodes}
        sigma = oracle.sem_covariance(view.topology, coefficients, loadings, variances)
        actual = oracle.ols_variance(view.topology, sigma, "X", "Y", set(expected) | set(fixed))
        if optimal:
            for alternative in alternatives:
                variance = oracle.ols_variance(view.topology, sigma, "X", "Y", alternative | set(fixed))
                assert actual <= variance + 1e-7 * max(1., variance), (label, alternative)
        else:
            other = set() if label == "2E" else {"Z1", "Z2"}
            difference = actual - oracle.ols_variance(view.topology, sigma, "X", "Y", other)
            relative_rankings.append(difference)
    if not optimal:
        assert min(relative_rankings) < -1e-6 and max(relative_rankings) > 1e-6


def test_dag_mediator_parents_and_declared_order():
    topology = DirectedBidirectedGraph(nodes=("Y", "P", "M", "X", "C", "S"), directed=(
        ("X", "M"), ("M", "Y"), ("P", "M"), ("C", "X"), ("C", "Y"), ("S", "Y")
    ))
    result = runge_optimal_adjustment(DAG(topology), "X", "Y", conditioned_on="S")
    assert result.adjustment_set == ("P", "C")
    assert result.o_set == ("P", "C", "S")
    assert result.parents == result.o_set
    assert result.mediators == ("M",)
    assert result.forbidden == ("Y", "M", "X")
    assert result.graphical_optimality is True
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.adjustment_set = ()


def test_no_adjustment_is_distinct_from_nonoptimal_and_empty_adjustment():
    view = graph(("X", "Y"), [("X", "Y")], [("X", "Y")])
    result = runge_optimal_adjustment(view, "X", "Y")
    assert result.adjustment_set is None
    assert result.graphical_optimality is None
    empty = runge_optimal_adjustment(graph(("X", "Y"), [("X", "Y")]), "X", "Y")
    assert empty.adjustment_set == ()
    assert empty.graphical_optimality is True
    assert empty.unique_adjustment_set is True


def test_conditioning_opens_an_unadjustable_collider_path():
    view = graph(("X", "Y", "C", "S"), [("X", "Y"), ("C", "S")], [("X", "C"), ("C", "Y")])
    assert runge_o_set(view, "X", "Y").adjustment_set == ()
    assert runge_o_set(view, "X", "Y", conditioned_on="S").adjustment_set is None


def test_construction_does_not_claim_optimality():
    result = runge_o_set(graph(("X", "Y"), [("X", "Y")]), "X", "Y")
    assert result.adjustment_set == ()
    assert result.graphical_optimality is None
    assert result.criterion_i is None


def test_public_result_constructor_cannot_forge_optimality():
    view, fixed, _, _ = paper_graph("2E")
    result = runge_optimal_adjustment(view, "X", "Y", conditioned_on=fixed)
    arguments = {field.name: getattr(result, field.name) for field in fields(result)}
    arguments["graphical_optimality"] = True
    with pytest.raises(TypeError, match="factory-created"):
        RungeAdjustmentResult(**arguments)


def test_dataclass_replace_cannot_forge_optimality():
    view, fixed, _, _ = paper_graph("2E")
    result = runge_optimal_adjustment(view, "X", "Y", conditioned_on=fixed)
    assert result.graphical_optimality is False
    with pytest.raises(TypeError, match="factory-created"):
        replace(result, graphical_optimality=True)
    assert result.graphical_optimality is False


def test_posttreatment_conditioning_estimand_mismatch_exactly():
    """Eq(3) differs from adjusted OLS despite Assumptions1's S∩forb=empty."""
    view = graph(("X", "U", "S", "Y"),
                 [("X", "Y"), ("X", "S"), ("U", "S"), ("U", "Y")])
    assert "S" not in oracle.forbidden_nodes(view.topology, "X", "Y")
    assert oracle.valid_sets(view.topology, "X", "Y", {"S"}) == [{"U"}]
    parents, colliders, collider_parents, _, _ = oracle.components(view.topology, "X", "Y", {"S"})
    assert parents | colliders | collider_parents == {"U"}

    # X,U,eps_S,eps_Y are independent N(0,1); S=X+U+eps_S, Y=X+U+eps_Y.
    # Under do(X=x), E[U|S=s]=(s-x)/2, hence E[Y|do(x),s]=(x+s)/2.
    loadings = [[Fraction(v) for v in row] for row in (
        (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 1, 0), (1, 1, 0, 1)
    )]
    covariance = [[sum((a * b for a, b in zip(row_i, row_j)), Fraction(0))
                   for row_j in loadings] for row_i in loadings]
    # Intervention removes the exogenous X column from the stochastic part.
    covariance_ys_under_do = sum((a * b for a, b in zip(loadings[3][1:], loadings[2][1:])), Fraction(0))
    variance_s_under_do = sum((v * v for v in loadings[2][1:]), Fraction(0))
    causal_contrast_at_fixed_s = Fraction(1) - covariance_ys_under_do / variance_s_under_do
    assert causal_contrast_at_fixed_s == Fraction(1, 2)

    def solve(matrix, vector):
        rows = [list(row) + [value] for row, value in zip(matrix, vector)]
        for column in range(len(vector)):
            divisor = rows[column][column]
            rows[column] = [value / divisor for value in rows[column]]
            for row in range(len(vector)):
                if row != column:
                    factor = rows[row][column]
                    rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
        return [row[-1] for row in rows]

    # Population normal equations for regressors X,U,S: beta=(1,1,0).
    coefficients = solve([row[:3] for row in covariance[:3]], [row[3] for row in covariance[:3]])
    assert coefficients == [Fraction(1), Fraction(1), Fraction(0)]
    assert coefficients[0] != causal_contrast_at_fixed_s

    # Check faithfulness exactly: all pairwise Gaussian conditional covariances
    # vanish iff the graph separates the pair, for every conditioning subset.
    for a, b in itertools.combinations(range(4), 2):
        remaining = [v for v in range(4) if v not in (a, b)]
        for size in range(3):
            for given in itertools.combinations(remaining, size):
                weights = solve([[covariance[i][j] for j in given] for i in given],
                                [covariance[i][b] for i in given])
                conditional_covariance = covariance[a][b] - sum(
                    (covariance[a][i] * weight for i, weight in zip(given, weights)), Fraction(0)
                )
                connected = oracle.open_path_exists(view.topology, view.topology.nodes[a], view.topology.nodes[b],
                                                    {view.topology.nodes[i] for i in given})
                assert (conditional_covariance != 0) == connected


@pytest.mark.parametrize("algorithm", (runge_o_set, runge_optimal_adjustment))
def test_posttreatment_conditioning_rejected_by_both_public_apis(algorithm):
    view = graph(("X", "U", "S", "Y"),
                 [("X", "Y"), ("X", "S"), ("U", "S"), ("U", "Y")])
    with pytest.raises(RungeAdjustmentDomainError, match="descendants of treatment"):
        algorithm(view, "X", "Y", conditioned_on="S")


@pytest.mark.parametrize("fixed", [("Y",), ("X",), ("M",), ("D",), ("missing",)])
def test_bad_fixed_conditions_fail_closed(fixed):
    view = graph(("X", "M", "Y", "D"), [("X", "M"), ("M", "Y"), ("Y", "D")])
    with pytest.raises(RungeAdjustmentDomainError):
        runge_o_set(view, "X", "Y", conditioned_on=fixed)


def test_domain_validation_and_explicit_resource_limit():
    view = graph(("X", "Y", "Z1", "Z2"), [("X", "Y"), ("X", "Z1"), ("Z1", "Z2")], [("Z2", "Y")])
    with pytest.raises(RungeAdjustmentDomainError):
        runge_o_set(view.topology, "X", "Y")
    with pytest.raises(RungeAdjustmentDomainError):
        runge_o_set(view, "Y", "X")
    with pytest.raises(RungeAdjustmentDomainError):
        runge_o_set(view, "X", "X")
    with pytest.raises(RungeAdjustmentResourceLimitError):
        runge_optimal_adjustment(view, "X", "Y", max_collider_paths=0)
    with pytest.raises(RungeAdjustmentDomainError):
        runge_optimal_adjustment(view, "X", "Y", max_collider_paths=True)


def test_random_graphs_against_original_path_definitions():
    rng = random.Random(210210324)
    nodes = ("A", "X", "B", "C", "Y")
    checked = 0
    rejected_posttreatment = 0
    rejected_with_adjustment = 0
    for _ in range(110):
        directed = {pair for pair in itertools.combinations(nodes, 2) if rng.random() < .3}
        directed.add(("X", "Y"))
        bidirected = [pair for pair in itertools.combinations(nodes, 2) if rng.random() < .2]
        view = graph(nodes, directed, bidirected)
        forbidden = oracle.forbidden_nodes(view.topology, "X", "Y")
        fixed = {v for v in nodes if v not in forbidden and rng.random() < .22}
        expected = oracle.valid_sets(view.topology, "X", "Y", fixed)
        if fixed & oracle.closure(view.topology, {"X"}):
            rejected_posttreatment += 1
            rejected_with_adjustment += bool(expected)
            for algorithm in (runge_o_set, runge_optimal_adjustment):
                with pytest.raises(RungeAdjustmentDomainError, match="descendants of treatment"):
                    algorithm(view, "X", "Y", conditioned_on=fixed)
            continue
        result = runge_optimal_adjustment(view, "X", "Y", conditioned_on=fixed)
        assert (result.adjustment_set is not None) == bool(expected)
        if not expected:
            continue
        checked += 1
        assert set(result.adjustment_set) in expected
        p, c, pc, ancestors, _ = oracle.components(view.topology, "X", "Y", fixed)
        assert set(result.o_set) == p | c | pc
        assert set(result.colliders) == c
        assert set(result.valid_ancestors) == ancestors
        criteria = oracle.criteria(view.topology, "X", "Y", fixed)
        assert (result.criterion_i, result.criterion_ii) == criteria
        assert result.unique_adjustment_set == (len(expected) == 1)
        assert result.graphical_optimality == (len(expected) == 1 or all(criteria))
    assert checked > 40
    assert rejected_posttreatment > 0
    assert rejected_with_adjustment > 0


def test_long_bidirected_path_and_collider_parent():
    view = graph(("X", "Y", "C1", "C2", "P", "D"),
                 [("X", "Y"), ("P", "C2"), ("Y", "D")],
                 [("Y", "C1"), ("C1", "C2"), ("C2", "D")])
    result = runge_optimal_adjustment(view, "X", "Y")
    assert result.o_set == ("C1", "C2", "P")
    assert result.colliders == ("C1", "C2")
    assert result.collider_parents == ("P",)
    assert result.graphical_optimality is True
    assert set(result.adjustment_set) in oracle.valid_sets(view.topology, "X", "Y", set())


def test_six_vertex_admgs_with_mediators_and_fixed_conditions():
    rng = random.Random(271828)
    nodes = ("A", "X", "B", "C", "D", "Y")
    checked = 0
    rejected_posttreatment = 0
    rejected_with_adjustment = 0
    for _ in range(80):
        directed = {pair for pair in itertools.combinations(nodes, 2) if rng.random() < .24}
        directed.update({("X", "C"), ("C", "Y")})
        bidirected = [pair for pair in itertools.combinations(nodes, 2) if rng.random() < .16]
        view = graph(nodes, directed, bidirected)
        forbidden = oracle.forbidden_nodes(view.topology, "X", "Y")
        fixed = {v for v in nodes if v not in forbidden and rng.random() < .3}
        expected = oracle.valid_sets(view.topology, "X", "Y", fixed)
        if fixed & oracle.closure(view.topology, {"X"}):
            rejected_posttreatment += 1
            rejected_with_adjustment += bool(expected)
            for algorithm in (runge_o_set, runge_optimal_adjustment):
                with pytest.raises(RungeAdjustmentDomainError, match="descendants of treatment"):
                    algorithm(view, "X", "Y", conditioned_on=fixed)
            continue
        result = runge_optimal_adjustment(view, "X", "Y", conditioned_on=fixed)
        assert (result.adjustment_set is not None) == bool(expected)
        if not expected:
            continue
        checked += 1
        assert set(result.adjustment_set) in expected
        p, c, pc, _, _ = oracle.components(view.topology, "X", "Y", fixed)
        assert set(result.o_set) == p | c | pc
        criteria = oracle.criteria(view.topology, "X", "Y", fixed)
        assert (result.criterion_i, result.criterion_ii) == criteria
        assert result.unique_adjustment_set == (len(expected) == 1)
    assert checked > 25
    assert rejected_posttreatment > 0
    assert rejected_with_adjustment > 0


def test_graphically_optimal_sets_have_smallest_population_ols_variance():
    """Independent SEM covariance calculations check implications beyond syntax."""
    rng = random.Random(314159)
    nodes = ("A", "X", "B", "C", "Y")
    positions = {v: i for i, v in enumerate(nodes)}

    def solve(matrix, vector):
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

    def covariance(view):
        size = len(nodes)
        coefficients = [[0.] * size for _ in nodes]
        noise = [[float(i == j) for j in range(size)] for i in range(size)]
        for a, b in sorted(view.topology.directed):
            coefficients[positions[b]][positions[a]] = rng.choice((-1, 1)) * rng.uniform(.2, 1.8)
        for a, b in sorted(view.topology.bidirected):
            latent_loading = [0.] * size
            latent_loading[positions[a]] = rng.uniform(.2, 1.8)
            latent_loading[positions[b]] = rng.choice((-1, 1)) * rng.uniform(.2, 1.8)
            for i in range(size):
                for j in range(size):
                    noise[i][j] += latent_loading[i] * latent_loading[j]
        # Nodes are generated in a topological order; propagate exogenous terms.
        transform = [[float(i == j) for j in range(size)] for i in range(size)]
        for i in range(size):
            for j in range(size):
                transform[i][j] += sum(coefficients[i][k] * transform[k][j] for k in range(i))
        return [[
            sum(transform[i][k] * noise[k][column] * transform[j][column]
                for k in range(size) for column in range(size))
            for j in range(size)
        ] for i in range(size)]

    def ols_variance(sigma, adjustment):
        def residual(vertex, given):
            index = positions[vertex]
            indices = [positions[v] for v in given]
            if not indices:
                return sigma[index][index]
            weights = solve([[sigma[i][j] for j in indices] for i in indices], [sigma[i][index] for i in indices])
            return sigma[index][index] - sum(sigma[index][i] * weight for i, weight in zip(indices, weights))
        return residual("Y", adjustment | {"X"}) / residual("X", adjustment)

    checked = 0
    for _ in range(65):
        directed = {pair for pair in itertools.combinations(nodes, 2) if rng.random() < .3}
        directed.add(("X", "Y"))
        view = graph(nodes, directed, [pair for pair in itertools.combinations(nodes, 2) if rng.random() < .2])
        result = runge_optimal_adjustment(view, "X", "Y")
        if result.graphical_optimality is not True:
            continue
        checked += 1
        alternatives = oracle.valid_sets(view.topology, "X", "Y", set())
        for _ in range(6):
            sigma = covariance(view)
            actual = ols_variance(sigma, set(result.adjustment_set))
            for alternative in alternatives:
                assert actual <= ols_variance(sigma, alternative) + 1e-9
    assert checked > 20
