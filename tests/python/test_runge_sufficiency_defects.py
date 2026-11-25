"""One in-hypothesis proof-step counterexample and two weaker Runge observations.

These pin observations about the paper's ARGUMENT, not about its statement and not about this
implementation. On every case below the verdict matches a separate criterion evaluation, so the
tests are regressions in the documentary sense the repository already uses for paper
fixtures: they record what the source claims, what an independent adjudication found, and
which of the two this code agrees with.

The three cases are NOT of equal force, and the fixture records which is which in
`hypotheses_of_the_step_in_force`. Only the first is a counterexample to a step with its
hypotheses in force. The second refutes a hypothesis-free reading of a step: Condition (II)
is false on that graph, so the step's own hypotheses are not met, and whether that step is
complete on its own terms is left open. Do not cite them at the same level.

The third refutes a proposed stronger repair, not a claim of the paper. These tests do
not adjudicate earlier externally reported necessity-proof problems. The fixed-S
estimand and Figure S1 errors recorded elsewhere are on neither half of that proof.

Each case was adjudicated against a second, definition-level implementation of m-separation
and adjustment validity written from the published definitions. The witness adjustment
sets are also checked here by the separate literal-path and latent-DAG moralization
oracles; the third E-set is recomputed by literal paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_graph_algorithms.directed_bidirected import (
    AcyclicDirectedBidirectedGraph,
    DirectedBidirectedGraph,
)
from causal_graph_algorithms.runge_adjustment import runge_o_set, runge_optimal_adjustment
from causal_graph_algorithms.separation import m_separated_sets
from tests.python.runge_oracles import latent_dag_adjustment_valid, open_path_exists, valid_sets

FIXTURES = json.loads(
    (Path(__file__).parents[1] / "fixtures/papers/runge_2023_sufficiency_defects.json")
    .read_text(encoding="utf-8")
)
CASES = {row["case"]: row for row in FIXTURES["cases"]}
CASE_IDS = tuple(CASES)


def _graph(row):
    return AcyclicDirectedBidirectedGraph(DirectedBidirectedGraph(
        nodes=tuple(row["nodes"]),
        directed=tuple((a, b) for a, b in row["directed"]),
        bidirected=tuple((a, b) for a, b in row["bidirected"]),
    ))


@pytest.mark.parametrize("case", CASE_IDS)
def test_o_set_and_verdict_match_the_adjudicated_values(case):
    """The criterion itself is unaffected: these graphs get the right verdict."""
    row = CASES[case]
    graph = _graph(row)
    result = runge_optimal_adjustment(graph, row["interventions"][0], row["outcomes"][0])
    assert sorted(result.o_set) == sorted(row["o_set"])
    assert result.criterion_i is row["criterion_i"]
    assert result.criterion_ii is row["criterion_ii"]
    assert result.graphical_optimality is row["graphical_optimality"]


@pytest.mark.parametrize("case", CASE_IDS)
def test_o_set_construction_matches(case):
    row = CASES[case]
    construction = runge_o_set(_graph(row), row["interventions"][0], row["outcomes"][0])
    for field in ("parents", "colliders", "collider_parents", "valid_ancestors"):
        if field in row:
            assert sorted(getattr(construction, field)) == sorted(row[field]), field


def test_only_the_first_case_has_the_step_hypotheses_in_force():
    """Guards the distinction itself, so a later edit cannot flatten the three cases."""
    in_force = {c: CASES[c]["hypotheses_of_the_step_in_force"] for c in CASE_IDS}
    assert in_force == {
        "condition-I-step": True,
        "condition-II-step-hypothesis-free-reading": False,
        "e-set-not-forced-into-every-valid-set": False,
    }


@pytest.mark.parametrize("case", CASE_IDS)
def test_each_recorded_separation(case):
    """Each separation is evaluated at the fixture's stated level of claim.

    Case 1 refutes an in-hypothesis assertion; case 2 refutes only the hypothesis-free
    reading. Case 3 checks the weaker separation, not a statement made by the paper.
    """
    row = CASES[case]
    failing = row["failing_separation"]
    assert m_separated_sets(
        _graph(row), set(failing["left"]), set(failing["right"]), set(failing["given"])
    ) is failing["m_separated"]


def test_condition_i_step_the_open_path_has_nothing_to_do_with_n_nodes():
    """The blocking vertex is in the O-set but on the A-side, so it is not conditioned on.

    This is the substance of the first defect: the step attributes a non-zero term to an
    N-node mechanism, and here the term is non-zero for an unrelated reason.
    """
    row = CASES["condition-I-step"]
    graph = _graph(row)
    o_set = set(runge_optimal_adjustment(graph, "X", "Y").o_set)
    witness = set(row["witness_adjustment_set"])
    shared = o_set & witness              # B, the part of O the witness set retains
    assert shared == set(), "the witness set was chosen to share no vertex with the O-set"
    assert not m_separated_sets(graph, {"Y"}, witness - o_set, shared | {"X"})
    # Conditioning additionally on the single A-side vertex V restores the separation,
    # which is the subset-witness step the paper does not take.
    assert m_separated_sets(graph, {"Y"}, witness - o_set, shared | {"X", "V"})


def test_condition_ii_hypothesis_free_reading_is_refuted():
    """A node outside the valid ancestors can still be m-connected to X given O minus it.

    This refutes the implication read WITHOUT Condition (II). It is not a counterexample to
    the published Case (2), whose hypotheses are not met here: P has no bidirected edge, so
    no type-2 witness exists, and Condition (II) is false on this graph.
    """
    row = CASES["condition-II-step-hypothesis-free-reading"]
    graph = _graph(row)
    construction = runge_o_set(graph, "X", "Y")
    assert "P" not in set(construction.valid_ancestors)
    o_set = set(runge_optimal_adjustment(graph, "X", "Y").o_set)
    assert not m_separated_sets(graph, {"X"}, {"P"}, o_set - {"P"})
    # The limit of this witness, pinned so it cannot be cited as more than it is: the step's
    # own hypotheses do not hold here, because Condition (II) is false.
    assert row["hypotheses_of_the_step_in_force"] is False
    assert runge_optimal_adjustment(graph, "X", "Y").criterion_ii is False


def test_each_case_records_its_distinct_claim_status():
    assert {case: row["claim_status"] for case, row in CASES.items()} == {
        "condition-I-step": "in_hypothesis_counterexample",
        "condition-II-step-hypothesis-free-reading": "hypothesis_free_counterexample_only",
        "e-set-not-forced-into-every-valid-set": "attempted_repair_counterexample",
    }
    assert "Each is a counterexample to a step of the published proof" not in FIXTURES["source"]["transcription"]
    context = CASES["condition-II-step-hypothesis-free-reading"]["source_context"]
    assert context["condition_ii_invocation_line"] == 1195
    assert context["case_2_lines"] == "1199-1210"
    assert context["published_case_2_completeness"] == "not_determined"


@pytest.mark.parametrize("case", [case for case, row in CASES.items() if "witness_adjustment_set" in row])
def test_witness_adjustment_validity_has_two_in_repo_definition_checks(case):
    row = CASES[case]
    graph = _graph(row).topology
    x, y = row["interventions"][0], row["outcomes"][0]
    witness = set(row["witness_adjustment_set"])
    fixed = set(row["conditioned_on"])
    literal_valid = witness in valid_sets(graph, x, y, fixed)
    moral_valid = latent_dag_adjustment_valid(graph, x, y, witness | fixed)
    assert literal_valid is moral_valid is row["witness_adjustment_set_is_valid"] is True


def test_third_witness_e_set_is_exact_and_is_not_forced_into_a_valid_set():
    row = CASES["e-set-not-forced-into-every-valid-set"]
    graph = _graph(row).topology
    o_set = set(row["o_set"])
    e_set = {
        e for e in o_set - set(row["parents"])
        if open_path_exists(graph, "X", e, o_set - {e})
    }
    assert e_set == set(row["e_set"]) == {"C"}
    assert e_set - set(row["witness_adjustment_set"]) == {"C"}
    assert not open_path_exists(graph, "X", "C", {"A"})



def test_api_negative_interpretation_names_its_unverified_necessity_dependency():
    """Documentation scope check, not a validation of the necessity theorem."""
    doc = (Path(__file__).parents[2] / "docs/reference/RUNGE_ADJUSTMENT.md").read_text()
    api = " ".join(doc.split("## API and the meaning of optimality\n")[1].split("\n## ")[0].split())
    assert "computes the graphical criterion" in api
    assert "Runge's published Theorem 3" in api
    assert "necessity direction" in api
    assert "do not independently validate necessity" in api


def test_open_path_information_claim_retains_its_faithfulness_premise():
    """An open path alone does not force positive MI for every Markov distribution."""
    doc = (Path(__file__).parents[2] / "docs/reference/RUNGE_ADJUSTMENT.md").read_text()
    case = " ".join(doc.split("### Condition (I) step,")[1].split("\n### ")[0].split())
    assert "For a faithful distribution, term (iii') is positive" in case
    assert "not imply positive mutual information for every Markov distribution" in case
