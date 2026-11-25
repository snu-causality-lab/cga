from __future__ import annotations

import itertools
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import pytest

from causal_graph_algorithms import Probability, Product, Ratio, Summation

from .gid_oracles import (
    all_gid_local_problems_n3,
    all_intervention_targets,
    evaluate_generalized_probability_expression,
    generalized_identification_oracle,
    minimal_finite_formula_cases_n3,
    n3_gid_census,
    n3_gid_query_analyses,
)
from .id_oracles import (
    TinyADMG,
    all_bow_admgs_n3,
    all_nonempty_outcome_queries_n3,
    exact_binary_interventional_probability,
    literal_exact_query_hedges,
    literal_id_fails,
    obsolete_subset_literal_hedge_criterion,
)


def _load_fixture(name: str) -> dict[str, Any]:
    path = Path(__file__).parents[1] / "fixtures" / "papers" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _tiny_graph(
    *,
    nodes: list[str],
    directed: list[list[str]],
    bidirected: list[list[str]],
) -> tuple[TinyADMG, dict[str, int]]:
    positions = {node: index for index, node in enumerate(nodes)}
    return (
        TinyADMG(
            tuple(range(len(nodes))),
            frozenset((positions[left], positions[right]) for left, right in directed),
            frozenset(
                (
                    min(positions[left], positions[right]),
                    max(positions[left], positions[right]),
                )
                for left, right in bidirected
            ),
        ),
        positions,
    )


def _sets(
    labels: list[list[str]], positions: dict[str, int]
) -> tuple[frozenset[int], ...]:
    return tuple(frozenset(positions[label] for label in target) for target in labels)


def test_n3_generalized_id_oracle_exhausts_every_collection_and_pins_census() -> None:
    graphs = all_bow_admgs_n3()
    queries = all_nonempty_outcome_queries_n3()
    sources = all_intervention_targets((0, 1, 2))
    analyses = n3_gid_query_analyses()
    census = n3_gid_census()

    assert len(graphs) == 200
    assert len(queries) == 19
    assert len(sources) == 8
    assert len(analyses) == 3_800
    assert census.total_rows == 200 * 19 * 256 == 972_800
    assert (census.identified, census.nonidentified) == (783_466, 189_334)
    assert census.by_collection_size == (
        (3_800, 0),
        (20_972, 9_428),
        (46_308, 60_092),
        (54_908, 157_892),
        (39_686, 226_314),
        (18_264, 194_536),
        (4_873, 101_527),
        (523, 29_877),
        (0, 3_800),
    )
    assert census.no_cover_failures == 154_810
    assert census.all_local_failures == 34_524
    assert census.source_hedges == 42_732

    assert sum(analysis.identified_for_mask(0) for analysis in analyses) == 0
    assert sum(analysis.identified_for_mask(1) for analysis in analyses) == 3_164
    assert sum(analysis.identified_for_mask(255) for analysis in analyses) == 3_800
    assert (
        sum(
            district.covering_mask.bit_count()
            for analysis in analyses
            for district in analysis.districts
        )
        == 16_822
    )


def test_all_3100_local_problems_match_id_single_and_literal_exact_hedges() -> None:
    local_problems = all_gid_local_problems_n3()
    exact_failures = 0
    obsolete_false_hedges = 0

    assert len(local_problems) == 3_100
    for problem in local_problems:
        id_single_fails = literal_id_fails(
            problem.source_graph,
            interventions=problem.local_interventions,
            outcomes=problem.district,
        )
        literal_hedges = literal_exact_query_hedges(
            problem.source_graph,
            interventions=problem.local_interventions,
            outcomes=problem.district,
        )
        exact_fails = bool(problem.hedges)

        assert exact_fails == bool(literal_hedges) == id_single_fails
        exact_failures += exact_fails
        obsolete_fails = obsolete_subset_literal_hedge_criterion(
            problem.source_graph,
            interventions=problem.local_interventions,
            outcomes=problem.district,
        )
        obsolete_false_hedges += obsolete_fails and not exact_fails

    assert exact_failures == 489
    assert obsolete_false_hedges == 12


def test_obsolete_arbitrary_subquery_criterion_has_1536_overall_false_failures() -> (
    None
):
    census = n3_gid_census()
    assert census.obsolete_false_nonidentifications == 1_536

    false_identifications = sum(
        not analysis.identified_for_mask(mask)
        and analysis.identified_for_mask(mask, obsolete=True)
        for analysis in n3_gid_query_analyses()
        for mask in range(256)
    )
    assert false_identifications == 0


def test_minimal_formula_campaign_source_sets_are_independently_selected() -> None:
    cases = minimal_finite_formula_cases_n3()
    nonobservational_sizes = Counter(
        len(case.experiment_interventions)
        for case in cases
        if frozenset() not in case.experiment_interventions
    )

    assert len(cases) == 3_800
    assert nonobservational_sizes == {1: 3_168, 2: 316}
    assert sum(frozenset() in case.experiment_interventions for case in cases) == 316


def test_finite_scm_evaluator_rejects_unsupplied_and_inexact_source_atoms() -> None:
    graph = TinyADMG((0, 1, 2), frozenset(), frozenset())
    labels = ("X", "Z", "Y")
    environment = {"X": 0, "Z": 1, "Y": 0}

    wrong_denominator = Ratio(
        Probability("Y", interventions="X"),
        Probability("Y", interventions="Z"),
    )
    with pytest.raises(AssertionError, match="exact supplied source"):
        evaluate_generalized_probability_expression(
            wrong_denominator,
            graph=graph,
            node_labels=labels,
            experiment_interventions=(frozenset({0}),),
            environment=environment,
            variant=0,
        )

    downward_closure_atom = Probability("Y", interventions="X")
    with pytest.raises(AssertionError, match="exact supplied source"):
        evaluate_generalized_probability_expression(
            downward_closure_atom,
            graph=graph,
            node_labels=labels,
            experiment_interventions=(frozenset({0, 1}),),
            environment=environment,
            variant=0,
        )


def test_lee_figure1_statuses_formula_and_failed_sources_are_adjudicated() -> None:
    fixture = _load_fixture("lee_correa_bareinboim_2019_gid_figures1_2.json")
    figure = fixture["figure_1"]
    positions: dict[str, int]
    for panel_name, panel in figure["panels"].items():
        graph, positions = _tiny_graph(
            nodes=figure["nodes"],
            directed=figure["common_directed"],
            bidirected=figure["common_bidirected"] + panel["added_bidirected"],
        )
        query = figure["query"]
        decision = generalized_identification_oracle(
            graph,
            outcomes=frozenset(positions[label] for label in query["outcomes"]),
            interventions=frozenset(
                positions[label] for label in query["interventions"]
            ),
            experiment_interventions=_sets(
                figure["experiment_interventions"], positions
            ),
        )
        expected_identified = panel["adjudicated_status"] == "identified"
        assert decision.identified is expected_identified

        if expected_identified:
            formula = Summation(
                "W",
                Product(
                    (
                        Probability(
                            "Y",
                            conditioned_on="W",
                            interventions="X2",
                        ),
                        Probability("W", interventions="X1"),
                    )
                ),
            )
            labels = tuple(figure["nodes"])
            sources = _sets(figure["experiment_interventions"], positions)
            for variant in (0, 1):
                for bits in itertools.product((0, 1), repeat=3):
                    environment = dict(zip(("X1", "X2", "Y"), bits, strict=True))
                    expected = exact_binary_interventional_probability(
                        graph,
                        outcomes=(positions["Y"],),
                        interventions=(positions["X1"], positions["X2"]),
                        values={
                            positions[label]: value
                            for label, value in environment.items()
                        },
                        variant=variant,
                    )
                    actual = evaluate_generalized_probability_expression(
                        formula,
                        graph=graph,
                        node_labels=labels,
                        experiment_interventions=sources,
                        environment=environment,
                        variant=variant,
                    )
                    assert actual == expected
            continue

        assert decision.unresolved_district == frozenset(
            positions[label] for label in panel["expected_unresolved_district"]
        )
        assert decision.target_vertices == frozenset(
            positions[label] for label in panel["expected_target_vertices"]
        )
        assert tuple(
            source.experiment_interventions for source in decision.source_hedges
        ) == _sets(panel["expected_failed_sources"], positions)

    assert figure["panels"]["a"]["reported_status"] == "identified"
    assert figure["panels"]["b"]["reported_status"] == "identified"
    assert figure["panels"]["c"]["reported_status"] == "not_identified"
    assert figure["panels"]["d"]["reported_status"] == "not_identified"


def test_lee_figure2_caption_not_conflicting_prose_matches_exact_local_oracle() -> None:
    fixture = _load_fixture("lee_correa_bareinboim_2019_gid_figures1_2.json")
    figure = fixture["figure_2"]
    observed_statuses: dict[str, dict[str, str]] = {}

    for panel_name, panel in figure["panels"].items():
        graph, positions = _tiny_graph(
            nodes=figure["nodes"],
            directed=figure["common_directed"] + panel["added_directed"],
            bidirected=figure["common_bidirected"],
        )
        observed_statuses[panel_name] = {}
        for query_name, query in figure["queries"].items():
            decision = generalized_identification_oracle(
                graph,
                outcomes=frozenset(positions[label] for label in query["outcomes"]),
                interventions=frozenset(
                    positions[label] for label in query["interventions"]
                ),
                experiment_interventions=_sets(
                    figure["experiment_interventions"], positions
                ),
            )
            status = "identified" if decision.identified else "not_identified"
            observed_statuses[panel_name][query_name] = status
            assert status == query["adjudicated_status"][panel_name]
            assert status == figure["reported_caption"][panel_name][query_name]

    assert observed_statuses["b"]["P_x1(Y1)"] == "identified"
    assert observed_statuses["c"]["P_x1(Y1)"] == "identified"
    assert figure["reported_prose"]["b"]["P_x1(Y1)"] == "not_identified"
    assert figure["reported_prose"]["c"]["P_x1(Y1)"] == "not_identified"
    assert fixture["official_erratum"]["figure_2_correction_found"] is False
    assert "does not mention" in fixture["official_erratum"]["scope"]


def _conditional_probability(
    *,
    model: int,
    outcome: str,
    value: int,
    u2: int,
    treatment: int,
) -> Fraction:
    predicates: dict[tuple[int, str], Callable[[], bool]] = {
        (1, "Y1"): lambda: value == u2,
        (1, "Y2"): lambda: value == (u2 ^ treatment),
        (2, "Y1"): lambda: value == (u2 ^ treatment),
        (2, "Y2"): lambda: value == u2,
    }
    preferred = predicates[(model, outcome)]()
    if model == 1:
        return Fraction(1, 3) if preferred else Fraction(2, 3)
    return Fraction(2, 3) if preferred else Fraction(1, 3)


def _kivva_observational_table(model: int) -> dict[tuple[int, ...], Fraction]:
    table: dict[tuple[int, ...], Fraction] = {}
    for observed in itertools.product((0, 1), repeat=4):
        x1, x2, y1, y2 = observed
        probability = Fraction(0)
        for u1, u2 in itertools.product((0, 1), repeat=2):
            if x1 != u1 or x2 != u1:
                continue
            probability += (
                Fraction(1, 4)
                * _conditional_probability(
                    model=model,
                    outcome="Y1",
                    value=y1,
                    u2=u2,
                    treatment=x1,
                )
                * _conditional_probability(
                    model=model,
                    outcome="Y2",
                    value=y2,
                    u2=u2,
                    treatment=x2,
                )
            )
        table[observed] = probability
    return table


def _kivva_target_probability(model: int) -> Fraction:
    return sum(
        (
            Fraction(1, 2)
            * _conditional_probability(
                model=model,
                outcome="Y1",
                value=0,
                u2=u2,
                treatment=0,
            )
            * _conditional_probability(
                model=model,
                outcome="Y2",
                value=0,
                u2=u2,
                treatment=1,
            )
            for u2 in (0, 1)
        ),
        Fraction(0),
    )


def test_kivva_example2_numeric_error_is_adjudicated_and_outside_phase12_domain() -> (
    None
):
    fixture = _load_fixture("kivva_et_al_2022_example2_nonpositive.json")
    graph, positions = _tiny_graph(
        nodes=fixture["nodes"],
        directed=fixture["directed"],
        bidirected=fixture["bidirected"],
    )
    query = fixture["query"]
    positive_domain_decision = generalized_identification_oracle(
        graph,
        outcomes=frozenset(positions[label] for label in query["outcomes"]),
        interventions=frozenset(positions[label] for label in query["interventions"]),
        experiment_interventions=(frozenset(),),
    )
    first = _kivva_observational_table(1)
    second = _kivva_observational_table(2)

    assert positive_domain_decision.identified
    assert first == second
    assert sum(first.values(), Fraction(0)) == 1
    assert all(
        first[(0, 1, y1, y2)] == 0 for y1, y2 in itertools.product((0, 1), repeat=2)
    )
    assert _kivva_target_probability(1) == Fraction(2, 9)
    assert _kivva_target_probability(2) == Fraction(5, 18)
    assert fixture["models"]["M1"]["reported_target_probability"] == "4/9"
    assert fixture["models"]["M2"]["reported_target_probability"] == "5/9"
    assert fixture["models"]["M1"]["adjudicated_target_probability"] == "2/9"
    assert fixture["models"]["M2"]["adjudicated_target_probability"] == "5/18"
    assert fixture["official_numeric_erratum_found"] is False
    assert fixture["phase12_adjudication"] == "outside_domain"
    assert "strict-positivity" in fixture["phase12_reason"]
