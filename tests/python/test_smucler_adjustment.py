from __future__ import annotations

import copy
import itertools
import json
import pickle
from pathlib import Path

import pytest

from causal_graph_algorithms import DAG, DirectedBidirectedGraph
from causal_graph_algorithms.smucler_adjustment import (
    SmuclerAdjustmentDomainError,
    SmuclerAdjustmentNotApplicable,
    SmuclerAdjustmentSet,
    _maximum_vertex_flow,
    _optimal_minimum,
    smucler_adjustment_set,
)

from .smucler_oracles import (
    binary_joint, d_separated, descendants, efficiency_bound, simple_paths, subsets, valid_sets,
)


OBJECTIVES = ("optimal", "optimal_minimal", "optimal_minimum")


def _dag(nodes: tuple[str, ...], edges: list[tuple[str, str]]) -> DAG:
    return DAG.from_graph(DirectedBidirectedGraph(nodes=nodes, directed=edges))


def _branch_graph() -> DAG:
    # Smucler et al. arXiv v3 Figure 5 / Example 2, K=3.
    return _dag(("Y", "W3", "A", "W1", "T", "W4", "W2"),
                [("T", "A"), ("A", "Y"), ("W4", "Y")]
                + [(w, v) for w in ("W1", "W2", "W3") for v in ("T", "Y")])


@pytest.mark.parametrize(("objective", "expected"), [
    ("optimal", ("W3", "W1", "W4", "W2")),
    ("optimal_minimal", ("W3", "W1", "W2")),
    ("optimal_minimum", ("T",)),
])
def test_paper_figure_5_has_three_different_objectives(objective: str, expected: tuple[str, ...]) -> None:
    graph = _branch_graph()
    result = smucler_adjustment_set(graph, "A", "Y", objective=objective)  # type: ignore[arg-type]
    assert isinstance(result, SmuclerAdjustmentSet)
    assert result.vertices == expected
    assert result.objective == objective
    candidates = valid_sets(graph.topology.nodes, graph.topology.directed,
                            "A", "Y", frozenset(graph.topology.nodes))
    assert frozenset(expected) in candidates


def test_hidden_component_is_marginalized_after_moralization() -> None:
    graph = _dag(("A", "H1", "Z", "Y", "H2", "M"),
                 [("Z", "A"), ("Z", "H1"), ("H1", "H2"),
                  ("H2", "Y"), ("A", "M"), ("M", "Y")])
    for objective in OBJECTIVES:
        result = smucler_adjustment_set(graph, "A", "Y", observed=("A", "Y", "Z", "M"),
                                       objective=objective)  # type: ignore[arg-type]
        assert isinstance(result, SmuclerAdjustmentSet)
        assert result.vertices == ("Z",)
        assert result.reduction.ignored == ("H1", "H2", "M")
        assert result.reduction.vertices == ("A", "Z", "Y")
        assert result.reduction.edges == (("A", "Z"), ("Z", "Y"))


def test_policy_covariates_are_mandatory_even_when_isolated() -> None:
    graph = _dag(("L", "A", "Y"), [("A", "Y")])
    for objective in OBJECTIVES:
        result = smucler_adjustment_set(graph, "A", "Y", policy_covariates=("L",),
                                       objective=objective)  # type: ignore[arg-type]
        assert isinstance(result, SmuclerAdjustmentSet)
        assert result.vertices == ("L",)


def test_no_observed_adjustment_is_a_distinct_algorithm_outcome() -> None:
    graph = _dag(("A", "Y", "U"), [("U", "A"), ("U", "Y"), ("A", "Y")])
    for objective in OBJECTIVES:
        result = smucler_adjustment_set(graph, "A", "Y", observed=("A", "Y"),
                                       objective=objective)  # type: ignore[arg-type]
        assert isinstance(result, SmuclerAdjustmentNotApplicable)
        assert result.reason == "no_observed_adjustment_set"


def test_global_condition_uses_observed_subset_of_ancestors_not_reverse() -> None:
    graph = _dag(("A", "Y", "U", "F"), [("A", "Y"), ("U", "Y")])
    result = smucler_adjustment_set(graph, "A", "Y", observed=("A", "Y", "F"), objective="optimal")
    assert isinstance(result, SmuclerAdjustmentNotApplicable)
    assert result.reason == "global_optimality_not_established"
    assert result.valid_candidate == ()
    # Hidden ancestor U does not violate N <= ancestors when F is unobserved.
    result = smucler_adjustment_set(graph, "A", "Y", observed=("A", "Y"), objective="optimal")
    assert isinstance(result, SmuclerAdjustmentSet)
    assert result.vertices == ()
    # Missing sufficient condition does not mean a global optimum cannot exist.
    for objective in ("optimal_minimal", "optimal_minimum"):
        result = smucler_adjustment_set(graph, "A", "Y", observed=("A", "Y", "F"),
                                       objective=objective)
        assert isinstance(result, SmuclerAdjustmentSet)
        assert result.vertices == ()


def test_minimum_selects_outcome_nearest_cut_on_each_disjoint_path() -> None:
    graph = _dag(("A", "B1", "B2", "C1", "C2", "Y"),
                 [("B1", "A"), ("B1", "B2"), ("B2", "Y"),
                  ("C1", "A"), ("C1", "C2"), ("C2", "Y"), ("A", "Y")])
    result = smucler_adjustment_set(graph, "A", "Y", objective="optimal_minimum")
    assert isinstance(result, SmuclerAdjustmentSet)
    assert result.vertices == ("B2", "C2")


def test_node_split_flow_reroutes_previously_selected_path() -> None:
    nodes = ("A", "u", "v", "Y", "w", "z")
    edges = (("A", "u"), ("u", "v"), ("v", "Y"),
             ("A", "w"), ("w", "v"), ("u", "z"), ("z", "Y"))
    adjacency = {v: set[str]() for v in nodes}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    size, paths = _maximum_vertex_flow(nodes, adjacency, "A", "Y", extract_paths=True)
    assert size == 2
    assert len(paths) == 2
    assert set(paths[0][1:-1]).isdisjoint(paths[1][1:-1])
    for path in paths:
        assert path[0] == "A" and path[-1] == "Y"
        assert all(b in adjacency[a] for a, b in zip(path, path[1:]))


def test_all_five_vertex_undirected_flow_cuts_against_lattice_definition() -> None:
    nodes = ("A", "v1", "v2", "v3", "Y")
    possible = tuple(edge for edge in itertools.combinations(nodes, 2) if edge != ("A", "Y"))
    for present in itertools.product((False, True), repeat=len(possible)):
        edges = frozenset(edge for edge, include in zip(possible, present) if include)
        paths = simple_paths(nodes, edges, "A", "Y")
        cuts = [z for z in subsets(nodes[1:-1]) if all(set(path) & z for path in paths)]
        smallest = min(map(len, cuts))
        minimum_cuts = [z for z in cuts if len(z) == smallest]
        # Halin's infimum: boundary of Y's component after deleting union of cuts.
        union = frozenset().union(*minimum_cuts)
        remaining = tuple(v for v in nodes if v not in union)
        component = {v for v in remaining if simple_paths(remaining, edges, "Y", v)}
        expected = {v for v in nodes if v not in component
                    and any((v, w) in edges or (w, v) in edges for w in component)}
        adjacency = {v: set[str]() for v in nodes}
        for a, b in edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        assert _optimal_minimum(nodes, adjacency, "A", "Y") == expected


def test_polynomial_path_algorithm_handles_exponentially_many_minimum_cuts() -> None:
    # 2**32 minimum cuts; the production routine should only run vertex flows.
    roots = tuple(f"R{i}" for i in range(32))
    near_y = tuple(f"Q{i}" for i in range(32))
    nodes = ("A", *roots, *near_y, "Y")
    edges = [("A", "Y")]
    for root, child in zip(roots, near_y):
        edges.extend(((root, "A"), (root, child), (child, "Y")))
    result = smucler_adjustment_set(_dag(nodes, edges), "A", "Y", objective="optimal_minimum")
    assert isinstance(result, SmuclerAdjustmentSet)
    assert result.vertices == near_y


@pytest.mark.parametrize("kwargs", [
    {"treatment": True}, {"outcome": 1}, {"treatment": "missing"},
    {"outcome": "A"}, {"observed": ("A",)}, {"observed": ("A", "Y", "bad")},
    {"observed": ("A", "A", "Y")}, {"observed": "A"},
    {"policy_covariates": ("A",)},
    {"policy_covariates": ("Y",)}, {"policy_covariates": (True,)},
    {"objective": "minimum"},
])
def test_domain_validation_fails_closed(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {"treatment": "A", "outcome": "Y"}
    arguments.update(kwargs)
    with pytest.raises(SmuclerAdjustmentDomainError):
        smucler_adjustment_set(_dag(("A", "Y", "L"), [("A", "Y")]), **arguments)  # type: ignore[arg-type]


def test_requires_validated_dag_and_paper_inclusion_premises() -> None:
    with pytest.raises(SmuclerAdjustmentDomainError, match="DAG"):
        smucler_adjustment_set(DirectedBidirectedGraph(nodes=("A", "Y")), "A", "Y")  # type: ignore[arg-type]
    with pytest.raises(SmuclerAdjustmentDomainError, match="descendant"):
        smucler_adjustment_set(_dag(("A", "Y"), []), "A", "Y")
    graph = _dag(("A", "Y", "L"), [("A", "Y"), ("A", "L")])
    with pytest.raises(SmuclerAdjustmentDomainError, match="non-descendant"):
        smucler_adjustment_set(graph, "A", "Y", policy_covariates=("L",))
    with pytest.raises(SmuclerAdjustmentDomainError, match="observed"):
        smucler_adjustment_set(graph, "A", "Y", observed=("A", "Y"), policy_covariates=("L",))


def test_string_selector_means_one_declared_vertex_including_multichar_labels() -> None:
    graph = _dag(("A", "Y", "Long policy"), [("A", "Y")])
    result = smucler_adjustment_set(graph, "A", "Y", policy_covariates="Long policy")
    assert isinstance(result, SmuclerAdjustmentSet)
    assert result.vertices == ("Long policy",)
    # This recognized singleton fails inclusion, not character-wise lookup.
    with pytest.raises(SmuclerAdjustmentDomainError, match="must be observed"):
        smucler_adjustment_set(graph, "A", "Y", observed="Long policy")


def test_all_paper_figures_reproduce_drawn_dag_h0_h1_and_reported_sets() -> None:
    fixture = json.loads((Path(__file__).parents[1] / "fixtures" / "papers"
                          / "smucler_2020_figures.json").read_text())
    assert fixture["source"]["arxiv_version"] == "2004.10521v3"
    assert {case["figure"] for case in fixture["cases"]} == {1, 2, 3, 4, 5}
    for case in fixture["cases"]:
        graph = _dag(tuple(case["nodes"]), [tuple(edge) for edge in case["directed"]])
        observed, policy = frozenset(case["observed"]), frozenset(case["policy"])
        _assert_optima(graph, "A", "Y", observed, policy)
        for objective in OBJECTIVES:
            result = smucler_adjustment_set(graph, "A", "Y", observed=observed,
                                           policy_covariates=policy, objective=objective)  # type: ignore[arg-type]
            assert set(result.reduction.h0_vertices) == set(case["h0_nodes"])
            assert {frozenset(edge) for edge in result.reduction.h0_edges} == {
                frozenset(edge) for edge in case["h0_edges"]}
            assert set(result.reduction.vertices) == set(case["h1_nodes"])
            assert {frozenset(edge) for edge in result.reduction.edges} == {
                frozenset(edge) for edge in case["h1_edges"]}
            assert set(result.reduction.forbidden) == set(case["forbidden"])
            assert set(result.reduction.ignored) == set(case["ignored"])
            expected = case["expected"][objective]
            if expected is None:
                assert isinstance(result, SmuclerAdjustmentNotApplicable)
                assert result.reason == "global_optimality_not_established"
                assert set(result.valid_candidate or ()) == set(case["valid_global_candidate"])
            else:
                assert isinstance(result, SmuclerAdjustmentSet)
                assert set(result.vertices) == set(expected)


def test_figure_3_efficiency_reversal_uses_two_independent_binary_models() -> None:
    nodes = ("A", "Y", "Z1", "Z2", "U")
    edges = frozenset((("Z1", "A"), ("Z1", "Z2"), ("U", "Z2"),
                       ("U", "Y"), ("A", "Y")))
    candidates = valid_sets(nodes, edges, "A", "Y", frozenset(nodes) - {"U"})
    assert set(candidates) == {frozenset(), frozenset(("Z1",)), frozenset(("Z1", "Z2"))}
    # These are newly constructed witnesses, not numerical parameters reported
    # by the source. Exact binary summation independently confirms the reversal.
    for seed, empty_is_better in ((0, True), (6, False)):
        joint = binary_joint(nodes, edges, seed)
        empty = efficiency_bound(nodes, joint, "A", "Y", frozenset())
        proxy = efficiency_bound(nodes, joint, "A", "Y", frozenset(("Z1", "Z2")))
        if empty_is_better:
            assert empty + 1e-5 < proxy
        else:
            assert proxy + 1e-5 < empty
        assert empty <= efficiency_bound(nodes, joint, "A", "Y", frozenset(("Z1",))) + 1e-10


def test_figure_4_global_optimum_beyond_theorem2_sufficient_condition() -> None:
    nodes = ("A", "Y", "L", "F", "U")
    edges = frozenset((("L", "A"), ("A", "Y"), ("U", "Y"), ("U", "F")))
    candidates = valid_sets(nodes, edges, "A", "Y", frozenset(nodes) - {"U"}, frozenset(("L",)))
    assert set(candidates) == {frozenset(("L",)), frozenset(("L", "F"))}
    # Proposition 2 proves the source's claim for {L,F}; the public algorithm
    # deliberately reports that Theorem 2 alone does not establish globality.
    assert d_separated(nodes, edges, ("A",), ("F",), frozenset(("L",)))
    for seed in range(8):
        joint = binary_joint(nodes, edges, seed)
        policy = frozenset(("L",))
        assert efficiency_bound(nodes, joint, "A", "Y", frozenset(("L", "F")), policy) <= (
            efficiency_bound(nodes, joint, "A", "Y", frozenset(("L",)), policy) + 1e-10)


def test_result_values_are_immutable_sealed_and_copy_safe() -> None:
    result = smucler_adjustment_set(_branch_graph(), "A", "Y")
    assert isinstance(result, SmuclerAdjustmentSet)
    for value in (result, result.reduction):
        assert not hasattr(value, "__dict__")
        assert copy.copy(value) is value
        assert copy.deepcopy(value) is value
        with pytest.raises(TypeError):
            pickle.dumps(value)
        with pytest.raises(TypeError):
            type("Mutable", (type(value),), {})
    with pytest.raises(AttributeError):
        result.vertices = ()  # type: ignore[misc]


def _assert_optima(graph: DAG, treatment: str, outcome: str,
                   observed: frozenset[str], policy: frozenset[str]) -> int:
    nodes, edges = graph.topology.nodes, graph.topology.directed
    candidates = valid_sets(nodes, edges, treatment, outcome, observed, policy)
    minimum = [z for z in candidates if len(z) == min(map(len, candidates))] if candidates else []
    minimal = [z for z in candidates if not any(w < z for w in candidates)]
    checked = 0
    for objective, comparison in zip(OBJECTIVES, (candidates, minimal, minimum)):
        result = smucler_adjustment_set(graph, treatment, outcome, observed=observed,
                                       policy_covariates=policy, objective=objective)  # type: ignore[arg-type]
        if not candidates:
            assert isinstance(result, SmuclerAdjustmentNotApplicable)
            assert result.reason == "no_observed_adjustment_set"
            continue
        if isinstance(result, SmuclerAdjustmentNotApplicable):
            assert objective == "optimal"
            assert result.reason == "global_optimality_not_established"
            assert frozenset(result.valid_candidate or ()) in candidates
            continue
        choice = frozenset(result.vertices)
        assert choice in comparison, (edges, treatment, outcome, observed, policy, objective, choice)
        for other in comparison:
            assert d_separated(nodes, edges, (treatment,), choice - other, other)
            assert d_separated(nodes, edges, (outcome,), other - choice, choice | {treatment})
        checked += 1
    return checked


def test_exhaustive_four_vertex_dags_against_original_path_adjustment_definition() -> None:
    nodes = ("V0", "V1", "V2", "V3")
    possible = tuple(itertools.combinations(nodes, 2))
    checked = 0
    for present in itertools.product((False, True), repeat=len(possible)):
        edges = [edge for edge, include in zip(possible, present) if include]
        graph = _dag(nodes, edges)
        for treatment, outcome in possible:
            if outcome not in descendants(nodes, frozenset(edges), treatment):
                continue
            others = set(nodes) - {treatment, outcome}
            for observable in subsets(others):
                observed = observable | {treatment, outcome}
                eligible_policy = observed - descendants(nodes, frozenset(edges), treatment)
                for policy in subsets(eligible_policy):
                    checked += _assert_optima(graph, treatment, outcome, observed, policy)
    assert checked > 2000


def test_seeded_six_vertex_dags_and_exact_binary_efficiency_bounds() -> None:
    import random

    rng = random.Random(4521)
    nodes = ("L", "U", "A", "B", "C", "Y")
    for index in range(48):
        edges = [pair for pair in itertools.combinations(nodes, 2) if rng.random() < 0.3]
        if ("A", "Y") not in edges:
            edges.append(("A", "Y"))
        graph = _dag(nodes, edges)
        observed = frozenset(v for v in nodes if v in ("A", "Y", "L") or rng.random() < 0.7)
        policy = frozenset(("L",)) if index % 2 else frozenset()
        _assert_optima(graph, "A", "Y", observed, policy)
        candidates = valid_sets(nodes, frozenset(edges), "A", "Y", observed, policy)
        if not candidates:
            continue
        joint = binary_joint(nodes, frozenset(edges), index)
        bounds = {z: efficiency_bound(nodes, joint, "A", "Y", z, policy) for z in candidates}
        for objective in OBJECTIVES:
            result = smucler_adjustment_set(graph, "A", "Y", observed=observed,
                                           policy_covariates=policy, objective=objective)  # type: ignore[arg-type]
            if not isinstance(result, SmuclerAdjustmentSet):
                continue
            comparison = candidates
            if objective == "optimal_minimal":
                comparison = [z for z in candidates if not any(w < z for w in candidates)]
            elif objective == "optimal_minimum":
                comparison = [z for z in candidates if len(z) == min(map(len, candidates))]
            assert all(bounds[frozenset(result.vertices)] <= bounds[z] + 1e-10 for z in comparison)
