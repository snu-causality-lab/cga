"""Definition-level RED oracle tests for no-selection marginal IDP."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from .id_oracles import (
    all_bow_admgs_n3,
    bow_admg_ci_signature_by_canonical_dag,
    literal_id_fails,
)
from .oracles import (
    ConditionalIndependenceSignature,
    no_selection_census,
)
from .pag_idp_oracles import (
    _sum_out,
    OracleIDPResult,
    OracleProbability,
    OracleProduct,
    OracleRatio,
    OracleSummation,
    OracleUnit,
    TinyPAG,
    all_nonempty_outcome_queries,
    circle_buckets_by_paths,
    expression_free_variables,
    identify_pag_by_definition,
    pc_related_by_simple_paths,
    pc_vertices_one_shot,
    possible_ancestors_reflexive_by_simple_paths,
    possible_descendants_reflexive_by_simple_paths,
    region_by_definition,
    visible_directed_edges_by_definition,
)


def _load_fixture(name: str) -> dict[str, Any]:
    path = Path(__file__).parents[1] / "fixtures" / "papers" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _endpoint_atoms(raw: object) -> tuple[tuple[str, str, str, str], ...]:
    assert isinstance(raw, list)
    atoms = tuple(tuple(edge) for edge in raw)
    assert all(
        len(edge) == 4 and all(isinstance(value, str) for value in edge)
        for edge in atoms
    )
    return atoms  # type: ignore[return-value]


def _tiny_pag(raw: dict[str, Any]) -> TinyPAG:
    return TinyPAG(tuple(raw["nodes"]), _endpoint_atoms(raw["edges"]))


def _result_for_query(
    pag: TinyPAG, outcomes: tuple[str, ...], interventions: tuple[str, ...]
) -> OracleIDPResult:
    return identify_pag_by_definition(
        pag,
        outcomes=frozenset(outcomes),
        interventions=frozenset(interventions),
    )


def test_r86_fixture_is_an_adjudicated_rendered_source_record() -> None:
    fixture = _load_fixture("jaber_2022_r86_idp_figures.json")
    assert fixture["primary_source"] == {
        "sha256": "768cfc03e8a01abd3c99406f96ab896a7d6b2ea14620cebecebdc3727e777fca",
        "page_count": 35,
        "independently_read_pages": [4, 5, 6, 7, 8, 29, 30],
    }
    assert fixture["related_primary_source"]["sha256"] == (
        "a8b1e9121b148ec27c406f13ba996ac6272379e7608a18db70b94fa311bbcb6e"
    )

    expected_statuses = {
        "2": [False],
        "3a": [True],
        "4a": [True, False],
        "9": [True],
    }
    for figure_name, expected in expected_statuses.items():
        figure = fixture["figures"][figure_name]
        assert figure["rendered_image_edge_check"]
        pag = _tiny_pag(figure)
        assert visible_directed_edges_by_definition(pag) == frozenset(
            tuple(edge) for edge in figure["visible_directed_edges"]
        )
        actual = [
            _result_for_query(
                pag,
                tuple(query["outcomes"]),
                tuple(query["interventions"]),
            )
            for query in figure["queries"]
        ]
        assert [result.identified for result in actual] == expected
        assert [
            query["independent_status"] == "identified" for query in figure["queries"]
        ] == expected

    figure4 = fixture["figures"]["4a"]
    assert ["C", "arrow", "D", "tail"] in figure4["edges"]
    assert ["B", "tail", "C", "arrow"] in figure4["edges"]


def test_line_six_uses_reflexive_possde_and_a_nonstrict_bucket_subset() -> None:
    figure = _load_fixture("jaber_2022_r86_idp_figures.json")["figures"]["4a"]
    pag = _tiny_pag(figure)
    result = _result_for_query(pag, ("A", "C", "D"), ("B",))

    first = result.trace[0]
    assert first.action == "eliminate"
    assert first.bucket == ("B",)
    assert first.possible_descendants == ("B", "C")
    assert set(first.bucket) == set(first.kernel_vertices) - set(first.target_vertices)

    q = OracleProbability(("A", "B", "C", "D"))
    expected = OracleRatio(
        OracleProduct((q, OracleSummation(("B", "C"), q))),
        OracleSummation(("C",), q),
    )
    assert result.formula == expected
    assert isinstance(result.formula, OracleRatio)
    numerator = result.formula.numerator
    denominator = result.formula.denominator
    assert isinstance(numerator, OracleProduct)
    assert isinstance(numerator.factors[1], OracleSummation)
    assert isinstance(denominator, OracleSummation)
    assert numerator.factors[0] is numerator.factors[1].body
    assert numerator.factors[0] is denominator.body


def test_line_six_keeps_q_denominator_when_possde_equals_bucket() -> None:
    pag = TinyPAG(("A", "B"), ())
    result = identify_pag_by_definition(
        pag,
        outcomes=frozenset({"B"}),
        interventions=frozenset({"A"}),
    )
    q = OracleProbability(("A", "B"))
    assert result.formula == OracleRatio(
        OracleProduct((q, OracleSummation(("A",), q))),
        q,
    )
    assert isinstance(result.formula, OracleRatio)
    assert isinstance(result.formula.numerator, OracleProduct)
    assert result.formula.denominator is result.formula.numerator.factors[0]


def test_definition_five_is_pairwise_nontransitive_and_region_is_one_shot() -> None:
    figure = _load_fixture("jaber_2022_r86_idp_figures.json")["figures"]["4a"]
    pag = _tiny_pag(figure)
    ambient = frozenset(pag.nodes)

    assert pc_related_by_simple_paths(pag, "B", "D", ambient)
    assert pc_related_by_simple_paths(pag, "D", "C", ambient)
    assert not pc_related_by_simple_paths(pag, "B", "C", ambient)
    assert region_by_definition(pag, frozenset({"B"}), ambient) == frozenset(
        {"A", "B", "D"}
    )
    assert "C" not in region_by_definition(pag, frozenset({"B"}), ambient)

    failure = _result_for_query(pag, ("B", "C", "D"), ("A",))
    first = failure.trace[0]
    assert (
        first.action,
        first.bucket,
        first.left_region,
        first.right_region,
        first.intersection,
    ) == (
        "region",
        ("B",),
        ("B", "D"),
        ("C", "D"),
        ("D",),
    )
    assert failure.terminal_target_vertices == ("B", "D")
    assert failure.terminal_kernel_vertices == ("A", "B", "D")


def test_visibility_is_inherited_from_the_original_pag_for_induced_paths() -> None:
    pag = TinyPAG(
        ("Z", "X", "Y"),
        (
            ("Z", "tail", "X", "arrow"),
            ("X", "tail", "Y", "arrow"),
        ),
    )
    assert visible_directed_edges_by_definition(pag) == frozenset({("X", "Y")})
    assert not pc_related_by_simple_paths(
        pag,
        "X",
        "Y",
        frozenset({"X", "Y"}),
    )


def test_circle_departures_are_possible_directions_and_closures_are_reflexive() -> None:
    pag = TinyPAG(
        ("A", "B"),
        (("A", "circle", "B", "arrow"),),
    )
    ambient = frozenset(pag.nodes)
    assert (
        possible_descendants_reflexive_by_simple_paths(pag, frozenset({"A"}), ambient)
        == ambient
    )
    assert possible_descendants_reflexive_by_simple_paths(
        pag, frozenset({"B"}), ambient
    ) == frozenset({"B"})
    assert (
        possible_ancestors_reflexive_by_simple_paths(pag, frozenset({"B"}), ambient)
        == ambient
    )


def test_figure_nine_exercises_line_nine_and_is_identified() -> None:
    figure = _load_fixture("jaber_2022_r86_idp_figures.json")["figures"]["9"]
    pag = _tiny_pag(figure)
    result = _result_for_query(pag, ("Y",), ("X",))

    assert result.identified
    ambient = frozenset(pag.nodes)
    bucket = frozenset({"X"})
    pc_vertices = pc_vertices_one_shot(pag, bucket, ambient)
    possible_descendants = possible_descendants_reflexive_by_simple_paths(
        pag, bucket, ambient
    )
    reflexive_possible_children = frozenset({"X", "A"})
    assert pc_vertices == frozenset({"W", "X", "B", "C"})
    assert pc_vertices & reflexive_possible_children <= bucket
    assert pc_vertices & possible_descendants == frozenset({"X", "C"})
    assert not pc_vertices & possible_descendants <= bucket

    region = next(step for step in result.trace if step.action == "region")
    required = figure["queries"][0]["required_first_region"]
    assert (
        region.bucket,
        region.left_region,
        region.right_region,
        region.intersection,
    ) == (
        tuple(required["bucket"]),
        tuple(required["left"]),
        tuple(required["right"]),
        tuple(required["intersection"]),
    )
    assert isinstance(result.unbound_formula, OracleSummation)
    line_nine_formula = result.unbound_formula.body
    assert isinstance(line_nine_formula, OracleRatio)
    assert line_nine_formula.denominator == OracleUnit()


def test_line_nine_chooses_buckets_from_p_c_not_p_t() -> None:
    # The source bucket in P_C is a singleton.  Taking buckets from P_T instead
    # merges A-o-o-X-o-o-B and skips the published line-9 decomposition.
    pag = TinyPAG(
        ("A", "X", "B"),
        (
            ("A", "circle", "X", "circle"),
            ("X", "circle", "B", "circle"),
        ),
    )
    target = frozenset({"A", "B"})
    assert circle_buckets_by_paths(pag, target) == (("A",), ("B",))
    assert circle_buckets_by_paths(pag, frozenset(pag.nodes)) == (("A", "X", "B"),)

    result = identify_pag_by_definition(
        pag,
        outcomes=target,
        interventions=frozenset({"X"}),
    )
    assert not result.identified
    assert result.trace[0].action == "region"
    assert result.trace[0].bucket == ("A",)
    assert result.terminal_target_vertices == ("A",)
    assert result.terminal_kernel_vertices == ("A", "X", "B")


def test_icml_figure_two_pins_nonempty_line_nine_intersection_denominator() -> None:
    fixture = _load_fixture("jaber_2019_icml_idp_figure2.json")
    assert fixture["primary_source"]["sha256"] == (
        "a8b1e9121b148ec27c406f13ba996ac6272379e7608a18db70b94fa311bbcb6e"
    )
    result = _result_for_query(
        _tiny_pag(fixture),
        tuple(fixture["query"]["outcomes"]),
        tuple(fixture["query"]["interventions"]),
    )
    assert result.identified
    first = result.trace[0]
    required = fixture["query"]["required_first_region"]
    assert (
        first.action,
        first.bucket,
        first.left_region,
        first.right_region,
        first.intersection,
    ) == (
        "region",
        tuple(required["bucket"]),
        tuple(required["left"]),
        tuple(required["right"]),
        tuple(required["intersection"]),
    )
    assert isinstance(result.unbound_formula, OracleRatio)
    assert not isinstance(result.unbound_formula.denominator, OracleUnit)


def test_empty_query_tiers_are_explicit_and_observational_shortcut_is_exact() -> None:
    pag = TinyPAG(
        ("A", "B", "C"),
        (("A", "circle", "B", "circle"),),
    )
    empty = identify_pag_by_definition(
        pag,
        outcomes=frozenset(),
        interventions=frozenset({"A"}),
    )
    observational = identify_pag_by_definition(
        pag,
        outcomes=frozenset({"B"}),
    )
    all_outcomes = identify_pag_by_definition(
        pag,
        outcomes=frozenset(pag.nodes),
    )

    assert empty.formula == OracleUnit()
    assert empty.trace == ()
    q = OracleProbability(pag.nodes)
    assert observational.formula == OracleSummation(("A", "C"), q)
    assert observational.possible_ancestors == ("A", "B")
    assert observational.trace == ()
    assert all_outcomes.formula == q
    with pytest.raises(
        AssertionError,
        match="oracle summation requested a non-free binder",
    ):
        _sum_out(pag, OracleProbability(("A",)), frozenset({"B"}))


def test_weighted_auxiliary_binding_uses_a_nonempty_intervention_query() -> None:
    # This is the smallest connected n=3 class found by exhaustive enumeration
    # with a raw structural extra.  C is tied to the intervention A; it is not
    # an isolated observational P(C) factor.
    pag = TinyPAG(
        ("A", "B", "C"),
        (
            ("A", "arrow", "B", "circle"),
            ("A", "arrow", "C", "circle"),
        ),
    )
    result = identify_pag_by_definition(
        pag,
        outcomes=frozenset({"B"}),
        interventions=frozenset({"A"}),
    )
    assert result.identified
    assert [step.bucket for step in result.trace if step.action == "eliminate"] == [
        ("A",),
        ("C",),
    ]
    assert result.unbound_formula is not None
    assert expression_free_variables(result.unbound_formula) == frozenset(
        {"A", "B", "C"}
    )
    assert result.formula == OracleSummation(
        ("C",),
        OracleProduct(
            (
                result.unbound_formula,
                OracleProbability(("C",), conditioned_on=("A",)),
            )
        ),
    )
    assert expression_free_variables(result.formula) == frozenset({"A", "B"})


def test_n3_census_matches_all_compatible_bow_admg_id_failures() -> None:
    nodes = ("A", "B", "C")
    census = no_selection_census(nodes)
    assert census.raw_count == 64
    assert census.ancestral_count == 56
    assert len(census.mags) == 56
    assert len(census.classes) == 11
    assert len(all_bow_admgs_n3()) == 200

    bow_groups: dict[ConditionalIndependenceSignature, list[int]] = {}
    for index in range(len(all_bow_admgs_n3())):
        signature = bow_admg_ci_signature_by_canonical_dag(
            all_bow_admgs_n3()[index], nodes
        )
        bow_groups.setdefault(signature, []).append(index)
    assert len(bow_groups) == 11
    assert sorted(map(len, bow_groups.values())) == [
        1,
        5,
        5,
        5,
        9,
        9,
        9,
        9,
        9,
        9,
        130,
    ]

    identified = 0
    nonidentified = 0
    all_role_rows = 0
    empty_outcome_rows = 0
    observational_rows = 0
    discriminative_rows = 0
    empty_intervention_identified = 0
    discriminative_identified = 0
    discriminative_nonidentified = 0
    labels = {"A": 0, "B": 1, "C": 2}
    queries = all_nonempty_outcome_queries(nodes)
    assert len(queries) == 19
    for signature, (_members, pag_edges) in census.classes.items():
        pag = TinyPAG(nodes, pag_edges)
        compatible = bow_groups[signature]
        for roles in itertools.product(
            ("other", "outcome", "intervention"), repeat=len(nodes)
        ):
            role_outcomes = frozenset(
                node
                for node, role in zip(nodes, roles, strict=True)
                if role == "outcome"
            )
            role_interventions = frozenset(
                node
                for node, role in zip(nodes, roles, strict=True)
                if role == "intervention"
            )
            all_role_rows += 1
            if not role_outcomes:
                empty_outcome_rows += 1
                unit = identify_pag_by_definition(
                    pag,
                    outcomes=role_outcomes,
                    interventions=role_interventions,
                )
                assert unit.formula == OracleUnit()
            elif not role_interventions:
                observational_rows += 1
            else:
                discriminative_rows += 1
        for outcomes, interventions in queries:
            result = identify_pag_by_definition(
                pag,
                outcomes=outcomes,
                interventions=interventions,
            )
            bow_failure = any(
                literal_id_fails(
                    all_bow_admgs_n3()[index],
                    outcomes=frozenset(labels[node] for node in outcomes),
                    interventions=frozenset(labels[node] for node in interventions),
                )
                for index in compatible
            )
            assert result.identified is not bow_failure
            if result.identified:
                identified += 1
            else:
                nonidentified += 1
            if not interventions:
                assert result.identified
                empty_intervention_identified += 1
            elif result.identified:
                discriminative_identified += 1
            else:
                discriminative_nonidentified += 1

    assert all_role_rows == 297
    assert empty_outcome_rows == 88
    assert observational_rows == 77
    assert discriminative_rows == 132
    assert empty_intervention_identified == 77
    assert (discriminative_identified, discriminative_nonidentified) == (51, 81)
    assert (identified, nonidentified) == (128, 81)
    assert identified + nonidentified == 209


def test_n4_full_public_query_census_is_shard_transparent() -> None:
    nodes = ("A", "B", "C", "D")
    census = no_selection_census(nodes)
    assert census.raw_count == 4_096
    assert census.ancestral_count == 2_504
    assert len(census.mags) == 2_492
    assert len(census.classes) == 248

    queries = all_nonempty_outcome_queries(nodes)
    assert len(queries) == 65
    shards = [[0, 0] for _ in range(8)]
    region_steps = 0
    for class_index, (_members, pag_edges) in enumerate(census.classes.values()):
        pag = TinyPAG(nodes, pag_edges)
        for outcomes, interventions in queries:
            result = identify_pag_by_definition(
                pag,
                outcomes=outcomes,
                interventions=interventions,
            )
            shards[class_index % 8][0 if result.identified else 1] += 1
            region_steps += sum(step.action == "region" for step in result.trace)

    assert [tuple(shard) for shard in shards] == [
        (1_126, 889),
        (1_077, 938),
        (1_145, 870),
        (986, 1_029),
        (1_171, 844),
        (1_112, 903),
        (1_182, 833),
        (1_107, 908),
    ]
    assert tuple(map(sum, zip(*shards, strict=True))) == (8_906, 7_214)
    assert sum(map(sum, shards)) == 16_120
    # This is a repository-independent enumeration result, not a paper-reported
    # count. Selecting line-9 candidates from P_C (rather than the legacy P_T
    # misreading) exposes these frames without changing the status totals;
    # that is why the crossing-bucket terminal-C/T regression above is needed.
    assert region_steps == 1_528
