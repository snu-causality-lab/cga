"""R-115 ancestral multi-world networks and ctf-calculus side conditions.

Production claims are checked against calculus_oracles.py, which evaluates
explicit finite SCMs unit by unit, builds Definition 2.4 and Algorithm 1
literally with its own path search, and imports no production ancestry,
exclusion, AMWN or separation helper. Paper statements live in the fixture
correa_bareinboim_2025_calculus.json with reported and adjudicated results.
A claim must hold in every oracle model; a non-claim asserts nothing, so named
non-claims are additionally shown to be dependent in some positive model.
"""
from __future__ import annotations

import copy
from itertools import combinations, product
import json
import os
from pathlib import Path
import pickle
import random

import pytest

import causal_graph_algorithms as cga
from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph, AncestralMultiWorldNetwork, CounterfactualAtom,
    CounterfactualCalculusDomainError, CounterfactualDomainError, CounterfactualEvent,
    CounterfactualTerm, DAG, DirectedBidirectedGraph, InterventionAssignment, ValueLiteral,
    ancestral_multi_world_network, consistency_rule_applies, counterfactual_ancestors_reflexive,
    counterfactual_d_separated, counterfactual_exclusion, exclusion_rule_applies,
)

from .calculus_oracles import (
    algorithm_1, clamps, definition_2_4, generic_scm, independent, positive_kernels,
    term_independent,
)

ROOT = Path(__file__).parents[2]
FIXTURE = json.loads((ROOT / "tests/fixtures/papers/correa_bareinboim_2025_calculus.json")
                     .read_text(encoding="utf-8"))
BINARY = ("0", "1")
PUBLIC = ("AncestralMultiWorldNetwork", "CounterfactualCalculusDomainError",
          "ancestral_multi_world_network", "consistency_rule_applies", "counterfactual_ancestors_reflexive",
          "counterfactual_d_separated", "counterfactual_exclusion", "exclusion_rule_applies")


def term(variable, clamp=None):
    return CounterfactualTerm(variable, tuple(
        InterventionAssignment(v, ValueLiteral(v, s)) for v, s in (clamp or {}).items()))


def spec(raw):
    variable, clamp = raw
    return term(variable, clamp)


def event(pairs):
    return CounterfactualEvent(tuple(CounterfactualAtom(t, ValueLiteral(t.variable, s)) for t, s in pairs))


def fixture_event(raw):
    return event([(spec(t), s) for t, s in raw])


def graph(name):
    data = FIXTURE["graphs"][name]
    return AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
        nodes=data["nodes"], directed=data["directed"], bidirected=data["bidirected"]))


def generic(name, seed, domains=None, **options):
    data = FIXTURE["graphs"][name]
    spaces = domains or {v: BINARY for v in data["nodes"]}
    return generic_scm(data["nodes"], data["directed"], data["bidirected"], spaces, seed=seed, **options)


def model_key(case, entry):
    """Identity of the model a fixture entry names; its free-text note is not part of it."""
    return json.dumps([case["graph"], {k: v for k, v in entry.items() if k != "note"}], sort_keys=True)


MODELS = {}


def model_for(case, entry):
    """Build the named model once; the regressions never mutate a model."""
    key = model_key(case, entry)
    if key in MODELS:
        return MODELS[key]
    kind = entry["kind"]
    if kind == "generic":
        options = {key: entry[key] for key in ("max_tables", "latent_states") if key in entry}
        domains = ({v: tuple(states) for v, states in entry["domains"].items()}
                   if "domains" in entry else None)
        model = generic(case["graph"], entry["seed"], domains, **options)
    else:
        raise AssertionError(f"unknown model kind {kind}")
    return MODELS.setdefault(key, model)


def roles(case):
    return tuple([spec(t) for t in case[role]] for role in ("left", "right", "given"))


def by_id(collection, identifier):
    return next(case for case in FIXTURE[collection] if case["id"] == identifier)


def oracle_term(value):
    """A term as the literal oracle writes it: (variable, frozenset of clamp items)."""
    return value.variable, frozenset(clamps(value).items())


def network_parts(network):
    """A production network as (term nodes, arrows, {served variables: children})."""
    meaning = {label: oracle_term(t) for label, t in network.terms}
    served = {label: frozenset(variables) for label, variables in network.exogenous}
    assert not set(meaning) & set(served)
    assert set(meaning) | set(served) == set(network.dag.topology.nodes)
    arrows, roots = set(), {variables: set() for variables in served.values()}
    for parent, child in network.dag.topology.directed:
        if parent in served:
            roots[served[parent]].add(meaning[child])
        else:
            arrows.add((meaning[parent], meaning[child]))
    return frozenset(meaning.values()), frozenset(arrows), {k: frozenset(v) for k, v in roots.items()}


def literal_ancestors(name, seeds):
    data = FIXTURE["graphs"][name]
    return set().union(*(definition_2_4(data["nodes"], data["directed"], t) for t in seeds))


# ---------------------------------------------------------------- operators


def test_exclusion_operator_matches_lemma_2_3_examples():
    chain = graph("r115_figure4a")
    assert counterfactual_exclusion(chain, term("Y", {"X": "1", "Z": "0"})) == term("Y", {"Z": "0"})
    assert counterfactual_exclusion(chain, term("Y", {"X": "1"})) == term("Y", {"X": "1"})
    four = graph("r115_figure4b")
    assert counterfactual_exclusion(four, term("Y", {"X": "1", "W": "1"})) == term("Y", {"W": "1"})
    assert counterfactual_exclusion(four, term("Z", {"X": "0"})) == term("Z")
    assert counterfactual_exclusion(graph("x_to_y"), term("X", {"Y": "0"})) == term("X")


@pytest.mark.parametrize("case", FIXTURE["ancestor_cases"], ids=lambda case: case["id"])
def test_counterfactual_ancestors_match_definition_2_4(case):
    seeds = [spec(t) for t in case["terms"]]
    found = counterfactual_ancestors_reflexive(graph(case["graph"]), seeds)
    adjudicated = {spec(t) for t in case["adjudicated"]}
    assert len(found) == len(set(found)) and set(found) == adjudicated
    # The adjudicated set is Definition 2.4 applied literally, not just the printed one.
    assert literal_ancestors(case["graph"], seeds) == {oracle_term(t) for t in adjudicated}
    # The relation is reflexive: each seed contributes its own reduction.
    assert {counterfactual_exclusion(graph(case["graph"]), t) for t in seeds} <= set(found)


@pytest.mark.parametrize("case", FIXTURE["network_cases"], ids=lambda case: case["id"])
def test_network_matches_algorithm_1_figures(case):
    data, seeds, expected = FIXTURE["graphs"][case["graph"]], [spec(t) for t in case["terms"]], case["adjudicated"]
    adjudicated = (
        frozenset(oracle_term(spec(t)) for t in expected["nodes"]),
        frozenset((oracle_term(spec(a)), oracle_term(spec(b))) for a, b in expected["edges"]),
        {frozenset(root["serves"]): frozenset(oracle_term(spec(c)) for c in root["children"])
         for root in expected["exogenous"]},
    )
    assert network_parts(ancestral_multi_world_network(graph(case["graph"]), seeds)) == adjudicated
    assert algorithm_1(data["nodes"], data["directed"], data["bidirected"], seeds) == adjudicated


def small_terms(vertices, states):
    for y in vertices:
        others = [v for v in vertices if v != y]
        for values in product((None, *states), repeat=len(others)):
            yield term(y, {v: s for v, s in zip(others, values) if s is not None})


def test_ancestors_and_network_match_literal_constructions_on_small_graphs():
    # Every ADMG on A, B, C with directed edges along that order: each term with
    # clamps in {0, 1} alone, and each pair of terms with clamps in {1}. Then a
    # seeded sample of four-vertex ADMGs, declared in shuffled order, with up to
    # three seeds of up to three clamps.
    three = ("A", "B", "C")
    pairs = tuple(combinations(three, 2))
    subsets = [chosen for size in range(len(pairs) + 1) for chosen in combinations(pairs, size)]
    seed_sets = [[t] for t in small_terms(three, BINARY)] + [
        list(pair) for pair in combinations(small_terms(three, ("1",)), 2)]
    checked = 0
    for directed, bidirected in product(subsets, repeat=2):
        g = AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
            nodes=three, directed=directed, bidirected=bidirected))
        for seeds in seed_sets:
            literal = set().union(*(definition_2_4(three, directed, t) for t in seeds))
            assert {oracle_term(t) for t in counterfactual_ancestors_reflexive(g, seeds)} == literal
            assert network_parts(ancestral_multi_world_network(g, seeds)) == algorithm_1(
                three, directed, bidirected, seeds)
            checked += 1
    four = ("A", "B", "C", "D")
    rng, three_clamps = random.Random(2025), 0
    for _ in range(80):
        directed = [pair for pair in combinations(four, 2) if rng.random() < 0.5]
        bidirected = [pair for pair in combinations(four, 2) if rng.random() < 0.35]
        g = AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
            nodes=rng.sample(four, 4), directed=directed, bidirected=bidirected))
        for _ in range(6):
            seeds = []
            for _ in range(rng.randint(1, 3)):
                y = rng.choice(four)
                seeds.append(term(y, {v: rng.choice(BINARY) for v in four if v != y and rng.random() < 0.6}))
                three_clamps += len(seeds[-1].interventions) == 3
            literal = set().union(*(definition_2_4(four, directed, t) for t in seeds))
            assert {oracle_term(t) for t in counterfactual_ancestors_reflexive(g, seeds)} == literal
            assert network_parts(ancestral_multi_world_network(g, seeds)) == algorithm_1(
                four, directed, bidirected, seeds)
            checked += 1
    assert checked == 64 * (27 + 66) + 80 * 6 and three_clamps > 50


def test_documented_output_order():
    # Declared order W, Z, X, Y differs from label order. Ties between copies of
    # one variable break on their clamps as (declared position, state) pairs,
    # so Y_{W=1,X=1} precedes Y_{Z=0}, which precedes Y_{X=0}.
    g = graph("r115_figure6a")
    seeds = [term("Y", {"X": "0"}), term("X"), term("Y", {"X": "1", "W": "1"}), term("W"), term("Z"),
             term("Y", {"Z": "0"})]
    expected = (term("W"), term("Z"), term("Z", {"W": "1"}), term("X"), term("X", {"Z": "0"}),
                term("Y", {"X": "1", "W": "1"}), term("Y", {"Z": "0"}), term("Y", {"X": "0"}))
    assert counterfactual_ancestors_reflexive(g, seeds) == expected
    assert counterfactual_ancestors_reflexive(g, seeds[::-1]) == expected
    network = ancestral_multi_world_network(g, seeds)
    labels = ('["W",{}]', '["Z",{}]', '["Z",{"W":"1"}]', '["X",{}]', '["X",{"Z":"0"}]',
              '["Y",{"W":"1","X":"1"}]', '["Y",{"Z":"0"}]', '["Y",{"X":"0"}]')
    roots = ('{"U":["Z"]}', '{"U":["X"]}', '{"U":["Y"]}', '{"U":["Z","X"]}')
    assert network.dag.topology.nodes == labels + roots
    assert network.terms == tuple(zip(labels, expected))
    assert network.exogenous == tuple(zip(roots, (("Z",), ("X",), ("Y",), ("Z", "X"))))
    figure = by_id("ancestor_cases", "r115_fig6")
    assert counterfactual_ancestors_reflexive(g, [spec(t) for t in figure["terms"]]) == (
        term("W"), term("Z"), term("Z", {"W": "1"}), term("X"), term("Y", {"X": "1", "W": "1"}))


def test_exclusion_operator_and_rule_3_agree_with_unit_level_semantics():
    # Full response tables make every allowed edge active somewhere, so an
    # intervention that the operator keeps changes the response on some unit.
    # The four-vertex Figure 4(b) graph gives terms with three clamps.
    checked = three_clamps = 0
    for name in ("x_to_y", "bow", "chain_xyz", "front_door", "z_to_x_to_y", "r115_figure1", "r115_figure4b"):
        g, model = graph(name), generic(name, 7)
        nodes = FIXTURE["graphs"][name]["nodes"]
        for y in nodes:
            others = [v for v in nodes if v != y]
            for size in range(1, len(others) + 1):
                for clamped in combinations(others, size):
                    three_clamps += size == 3
                    full = term(y, {v: "1" for v in clamped})
                    reduced = counterfactual_exclusion(g, full)
                    assert set(clamps(reduced).items()) <= set(clamps(full).items())
                    pairs = model.joint([full, reduced])
                    assert all(a == b for a, b in pairs), (name, full)
                    for removed_size in range(1, size + 1):
                        for removed in combinations(clamped, removed_size):
                            kept = term(y, {v: "1" for v in clamped if v not in removed})
                            applies = exclusion_rule_applies(g, full, removed)
                            assert applies is set(removed).isdisjoint(clamps(reduced)), (name, full, removed)
                            same = all(a == b for a, b in model.joint([full, kept]))
                            assert same is applies, (name, full, removed)
                            checked += 1
    assert checked > 100 and three_clamps == 4


@pytest.mark.parametrize("case", FIXTURE["exclusion_cases"], ids=lambda case: case["id"])
def test_rule_3_cases(case):
    g, full = graph(case["graph"]), spec(case["term"])
    assert exclusion_rule_applies(g, full, case["removed"]) is case["adjudicated"]
    kept = term(full.variable, {v: s for v, s in clamps(full).items() if v not in case["removed"]})
    same = all(a == b for a, b in generic(case["graph"], 13).joint([full, kept]))
    assert same is case["adjudicated"]


@pytest.mark.parametrize("case", FIXTURE["consistency_cases"], ids=lambda case: case["id"])
def test_rule_1_cases(case):
    first, second = fixture_event(case["first"]), fixture_event(case["second"])
    assert consistency_rule_applies(first, second) is case["adjudicated"]
    assert consistency_rule_applies(second, first) is case["adjudicated"]
    # One Rule 1 step makes the two events the same set of units. A refused pair
    # can still have the same units, for example when two steps relate them.
    same_units = case.get("same_units", case["adjudicated"])
    atoms = list(dict.fromkeys(first.atoms + second.atoms))
    data = FIXTURE["graphs"][case["graph"]]
    for seed in (17, 18):
        model = generic_scm(data["nodes"], data["directed"], data["bidirected"],
                            {v: BINARY for v in data["nodes"]}, seed=seed)
        law = model.joint([atom.term for atom in atoms])
        def holds(states, chosen):
            return all(states[atoms.index(atom)] == atom.value.state for atom in chosen.atoms)
        agree = all(holds(states, first) == holds(states, second) for states in law)
        assert agree is same_units, (case["id"], seed)


# ------------------------------------------------------------ independence


@pytest.mark.parametrize("case", FIXTURE["independence_cases"], ids=lambda case: case["id"])
def test_named_independence_cases(case):
    left, right, given = roles(case)
    claimed = counterfactual_d_separated(graph(case["graph"]), left, right, given)
    assert claimed is (case["adjudicated"]["amwn"] == "claimed")
    for entry in case["models"]:
        expected = entry.get("expect", case["adjudicated"]["oracle"])
        assert expected in ("independent", "dependent")
        holds = term_independent(model_for(case, entry), left, right, given)
        assert holds is (expected == "independent"), (case["id"], entry)


def dependence_witnesses():
    """Each distinct (graph, model entry) that a named case uses to show a dependence."""
    found = {}
    for case in FIXTURE["independence_cases"]:
        for index, entry in enumerate(case["models"]):
            if entry.get("expect", case["adjudicated"]["oracle"]) == "dependent":
                found.setdefault(model_key(case, entry), pytest.param(case, entry, id=f'{case["id"]}-{index}'))
    return list(found.values())


@pytest.mark.parametrize("case, entry", dependence_witnesses())
def test_dependence_witnesses_are_positive(case, entry):
    # R-115 assumes positive laws (TR p.3). A witness is checked in its
    # observational law and every hard-intervention kernel.
    assert positive_kernels(model_for(case, entry), entry.get("positive_kernels")), entry


def test_fixed_binary_domains_are_beyond_the_criterion():
    # Proof for binary X = {0, 1}: on the cell X = i, Y = y consistency gives
    # Y_i = Y = y, so one term is constant on every conditioning cell and
    # P(a, b, c) P(c) = P(a, c) P(b, c) holds termwise in every model. A third
    # state X = 2 constrains neither term, and the shared root of Y couples them.
    left, right, given = [term("Y", {"X": "0"})], [term("Y", {"X": "1"})], [term("X"), term("Y")]
    assert not counterfactual_d_separated(graph("x_to_y"), left, right, given)
    for seed in range(20, 26):
        assert term_independent(generic("x_to_y", seed), left, right, given)
    for seed in range(20, 23):
        ternary = generic("x_to_y", seed, {"X": ("0", "1", "2"), "Y": BINARY})
        assert not term_independent(ternary, left, right, given)


# -------------------------------------------------------- semantic contract


def test_terms_equal_after_exclusion_are_one_random_variable():
    g, model = graph("x_to_y"), generic("x_to_y", 5)
    copy_of_x = term("X", {"Y": "0"})
    assert counterfactual_d_separated(g, term("X"), term("Y"), copy_of_x)
    assert term_independent(model, [term("X")], [term("Y")], [copy_of_x])
    assert not counterfactual_d_separated(g, term("X"), copy_of_x)
    assert not term_independent(model, [term("X")], [copy_of_x])
    assert counterfactual_d_separated(DAG.from_text("X --> Y"), term("Y", {"X": "0"}), term("X"))


def test_network_value_contract_and_labels():
    g = graph("bow")
    network = ancestral_multi_world_network(g, [term("Y", {"X": "0"}), term("Y", {"X": "1"}), term("X")])
    assert isinstance(network, AncestralMultiWorldNetwork) and isinstance(network.dag, DAG)
    assert network.graph is g
    assert dict(network.terms) == {'["X",{}]': term("X"), '["Y",{"X":"0"}]': term("Y", {"X": "0"}),
                                   '["Y",{"X":"1"}]': term("Y", {"X": "1"})}
    assert dict(network.exogenous) == {'{"U":["Y"]}': ("Y",), '{"U":["X","Y"]}': ("X", "Y")}
    assert network.node(term("Y", {"X": "0"})) == '["Y",{"X":"0"}]'
    assert network.node(term("X", {"Y": "1"})) == '["X",{}]'
    with pytest.raises(CounterfactualCalculusDomainError):
        network.node(term("Y"))
    with pytest.raises(TypeError):
        AncestralMultiWorldNetwork(g, network.dag, network.terms, network.exogenous)
    assert copy.copy(network) is network and copy.deepcopy(network) is network
    with pytest.raises(TypeError):
        pickle.dumps(network)
    again = ancestral_multi_world_network(g, [term("X"), term("Y", {"X": "1"}), term("Y", {"X": "0"})])
    assert again.dag.topology == network.dag.topology and again.terms == network.terms


def test_inputs_outside_the_hard_intervention_domain_fail_closed():
    g = graph("x_to_y")
    nested = CounterfactualTerm("Y", (InterventionAssignment("X", term("X", {"Y": "0"})),))
    calls = (
        lambda bad: counterfactual_exclusion(g, bad),
        lambda bad: counterfactual_ancestors_reflexive(g, [bad]),
        lambda bad: ancestral_multi_world_network(g, [bad]),
        lambda bad: counterfactual_d_separated(g, bad, term("X")),
        lambda bad: counterfactual_d_separated(g, term("X"), term("Y"), [bad]),
        lambda bad: exclusion_rule_applies(g, bad, "X"),
        lambda bad: consistency_rule_applies(event([(bad, "0")]), event([(term("Y"), "0")])),
    )
    for call in calls:
        with pytest.raises(CounterfactualCalculusDomainError, match=r"Corollary 2\.2.*Theorem 4"):
            call(nested)
        with pytest.raises(CounterfactualCalculusDomainError, match="own variable"):
            call(term("X", {"X": "0"}))
    g3 = graph("chain_xyz")
    for bad in (term("Q"), term("Y", {"Q": "0"})):
        with pytest.raises(CounterfactualCalculusDomainError, match="unknown"):
            counterfactual_d_separated(g3, bad, term("X"))
    raw = DirectedBidirectedGraph(nodes=("X", "Y"), directed=(("X", "Y"),))
    with pytest.raises(CounterfactualCalculusDomainError, match="DAG"):
        counterfactual_exclusion(raw, term("Y"))
    for left, right, given in ((term("Y"), (), ()), ((), term("Y"), ()), ("Y", term("X"), ()),
                               (term("Y"), term("Y"), ()), (term("Y"), term("X"), term("Y")),
                               (term("Y"), term("X"), 3)):
        with pytest.raises(CounterfactualCalculusDomainError):
            counterfactual_d_separated(g3, left, right, given)
    full = term("Y", {"X": "1"})
    for removed in ((), "Z", "Q", ("X", 1)):
        with pytest.raises(CounterfactualCalculusDomainError):
            exclusion_rule_applies(g3, full, removed)
    with pytest.raises(CounterfactualCalculusDomainError):
        consistency_rule_applies(event([(term("Y"), "0")]), [term("Y")])
    assert issubclass(CounterfactualCalculusDomainError, CounterfactualDomainError)


def test_public_names_and_reference_guide_are_discoverable():
    for name in PUBLIC:
        assert hasattr(cga, name) and name in cga.__all__, name
    guide = (ROOT / "docs/reference/COUNTERFACTUAL_CALCULUS.md").read_text(encoding="utf-8")
    for text in ("Theorem 2.5", "Algorithm 1", "independent", "CGA_CALCULUS_SWEEP", "Soundness basis",
                 "64dbee95584b97e4bd55f36bf528777d829ed5f710216bac3ac5adbf1699e7ba"):
        assert text in guide, text
    qualified = "not complete on fixed binary domains"
    for doc in (guide, cga.counterfactual_d_separated.__doc__):
        text = " ".join(doc.replace("*", "").split())
        assert text.count("not complete") == text.count(qualified) > 0 and "may be enlarged is open" in text


# ------------------------------------------------------------------ sweeps


def statements(size, max_side, max_given):
    """Unordered (left, right) pairs of disjoint index sets, then a disjoint given set."""
    indices = tuple(range(size))
    for left_size in range(1, max_side + 1):
        for left in combinations(indices, left_size):
            rest = tuple(i for i in indices if i not in left)
            for right_size in range(1, max_side + 1):
                for right in combinations(rest, right_size):
                    if right < left:
                        continue
                    remaining = tuple(i for i in rest if i not in right)
                    for given_size in range(max_given + 1):
                        for given in combinations(remaining, given_size):
                            yield left, right, given


def sweep(g, models, pool, max_side, max_given):
    laws = [model.joint(pool) for model in models]
    claims = refusals = refused_dependent = 0
    for left, right, given in statements(len(pool), max_side, max_given):
        claimed = counterfactual_d_separated(g, [pool[i] for i in left], [pool[i] for i in right],
                                             [pool[i] for i in given])
        verdicts = [independent(law, left, right, given) for law in laws]
        if claimed:
            claims += 1
            assert all(verdicts), [str(pool[i]) for i in left + right + given]
        else:
            refusals += 1
            refused_dependent += not all(verdicts)
    return claims, refusals, refused_dependent


TERNARY = ("0", "1", "2")
X_TO_Y_POOL = [["X", {}], ["Y", {}], ["Y", {"X": "0"}], ["Y", {"X": "1"}], ["X", {"Y": "0"}]]
BOW_POOL = [["X", {}], ["Y", {}], ["Y", {"X": "0"}], ["Y", {"X": "1"}], ["X", {"Y": "1"}]]
# (graph, ternary vertices, seeds, pool, (statements, claims, refusals dependent in
# some model)), with at most two terms per side and two given. Six refusals in
# each binary x_to_y and bow row hold in every binary model; the ternary rows
# extend those pools with a third state and show every refusal dependent.
DEFAULT_SWEEP = (
    ("x_to_y", (), (31, 32), X_TO_Y_POOL, (220, 45, 169)),
    ("bow", (), (33,), BOW_POOL, (220, 30, 184)),
    ("front_door", (), (34, 35), [["X", {}], ["M", {}], ["Y", {}], ["M", {"X": "0"}], ["Y", {"M": "0"}],
                                  ["Y", {"X": "0"}], ["Y", {"M": "0", "X": "1"}]], (2226, 164, 2062)),
    ("x_to_y", ("X",), (36,), X_TO_Y_POOL + [["Y", {"X": "2"}]], (765, 122, 643)),
    ("bow", ("X",), (37,), BOW_POOL + [["Y", {"X": "2"}]], (765, 68, 697)),
    ("front_door", ("M",), (38,), [["X", {}], ["M", {}], ["Y", {}], ["M", {"X": "0"}], ["Y", {"M": "0"}],
                                   ["Y", {"M": "2"}], ["Y", {"M": "0", "X": "1"}]], (2226, 230, 1996)),
)


@pytest.mark.parametrize("name, ternary, seeds, pool, counts", DEFAULT_SWEEP,
                         ids=[row[0] + "".join(f"-ternary_{v}" for v in row[1]) for row in DEFAULT_SWEEP])
def test_bounded_soundness_sweep(name, ternary, seeds, pool, counts):
    domains = {v: TERNARY if v in ternary else BINARY for v in FIXTURE["graphs"][name]["nodes"]}
    claims, refusals, refused_dependent = sweep(
        graph(name), [generic(name, seed, domains) for seed in seeds], [spec(t) for t in pool], 2, 2)
    assert (claims + refusals, claims, refused_dependent) == counts


def test_claims_hold_when_one_root_feeds_a_bidirected_triangle():
    # The model class lets one exogenous root feed a whole bidirected clique.
    # Here a single latent feeds A, B and C, and no other root is shared. The
    # triangle alone gives no claims (every pair of copies shares a root), so D
    # is added as an unconfounded parent of A.
    data = {"nodes": ["D", "A", "B", "C"], "directed": [["D", "A"], ["A", "B"]],
            "bidirected": [["A", "B"], ["B", "C"], ["A", "C"]]}
    g = AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(**data))
    models = [generic_scm(data["nodes"], data["directed"], data["bidirected"], {v: BINARY for v in data["nodes"]},
                          seed=seed, latent_groups=[("A", "B", "C")]) for seed in (61, 62)]
    pool = [term("D"), term("A"), term("B"), term("C"), term("B", {"A": "0"}), term("A", {"D": "0"})]
    claims, refusals, refused_dependent = sweep(g, models, pool, 2, 2)
    assert (claims + refusals, claims, refused_dependent) == (765, 18, 747)
    # The shared root alone couples C with A: refused, and dependent in each model.
    assert not counterfactual_d_separated(g, term("C"), term("A"))
    assert not any(term_independent(model, [term("C")], [term("A")]) for model in models)


SWEEP_GRAPHS = {
    **{name: FIXTURE["graphs"][name] for name in ("x_to_y", "bow", "y_to_x_confounded", "chain_xyz",
                                                   "front_door")},
    "backdoor": FIXTURE["graphs"]["r115_figure1"],
    "chain_xyz_confounded": {"nodes": ["X", "Y", "Z"], "directed": [["X", "Y"], ["Y", "Z"]],
                             "bidirected": [["Y", "Z"]]},
}
# (graph, ternary vertices, pool, (statements, claims)): the seven binary graphs,
# then one ternary vertex in three graphs. In the bow every pair of these copies
# shares a root, so its ternary row would claim nothing; the back-door graph has
# claims with a ternary treatment.
OPT_IN_SWEEP = (
    ("x_to_y", (), [["X", {}], ["Y", {}], ["Y", {"X": "0"}], ["Y", {"X": "1"}], ["X", {"Y": "0"}]], (220, 45)),
    ("bow", (), [["X", {}], ["Y", {}], ["Y", {"X": "0"}], ["Y", {"X": "1"}], ["X", {"Y": "0"}]], (220, 30)),
    ("y_to_x_confounded", (), [["Y", {}], ["X", {}], ["X", {"Y": "0"}], ["X", {"Y": "1"}], ["Y", {"X": "0"}]],
     (220, 30)),
    ("chain_xyz", (), [["X", {}], ["Y", {}], ["Z", {}], ["Y", {"X": "0"}], ["Y", {"X": "1"}],
                       ["Z", {"X": "0"}], ["Z", {"X": "1"}], ["Z", {"Y": "0"}]], (5614, 306)),
    ("front_door", (), [["X", {}], ["M", {}], ["Y", {}], ["M", {"X": "0"}], ["Y", {"M": "0"}],
                        ["Y", {"M": "1"}], ["Y", {"X": "0"}], ["Y", {"X": "1"}]], (5614, 40)),
    ("backdoor", (), [["Z", {}], ["X", {}], ["Y", {}], ["X", {"Z": "0"}], ["Y", {"X": "0"}],
                      ["Y", {"X": "1"}], ["Y", {"Z": "0"}], ["Y", {"X": "0", "Z": "0"}]], (5614, 138)),
    ("chain_xyz_confounded", (), [["X", {}], ["Y", {}], ["Z", {}], ["Y", {"X": "0"}], ["Y", {"X": "1"}],
                                  ["Z", {"X": "0"}], ["Z", {"Y": "0"}], ["Z", {"Y": "1"}]], (5614, 125)),
    ("x_to_y", ("X",), [["X", {}], ["Y", {}], ["Y", {"X": "0"}], ["Y", {"X": "1"}], ["Y", {"X": "2"}]], (220, 18)),
    ("front_door", ("M",), [["X", {}], ["M", {}], ["Y", {}], ["M", {"X": "0"}], ["Y", {"M": "0"}],
                            ["Y", {"M": "2"}], ["Y", {"X": "0"}]], (2226, 40)),
    ("backdoor", ("X",), [["Z", {}], ["X", {}], ["Y", {}], ["X", {"Z": "0"}], ["Y", {"X": "0"}],
                          ["Y", {"X": "2"}], ["Y", {"Z": "0"}], ["Y", {"X": "0", "Z": "0"}]], (5614, 138)),
)


def sweep_graph(name):
    data = SWEEP_GRAPHS[name]
    return AcyclicDirectedBidirectedGraph.from_graph(DirectedBidirectedGraph(
        nodes=data["nodes"], directed=data["directed"], bidirected=data["bidirected"]))


def test_opt_in_sweep_configurations_are_well_formed():
    assert {name for name, _, _, _ in OPT_IN_SWEEP} == set(SWEEP_GRAPHS)
    for name, ternary, pool, _ in OPT_IN_SWEEP:
        terms = [spec(t) for t in pool]
        assert len(set(terms)) == len(terms) <= 8
        assert counterfactual_ancestors_reflexive(sweep_graph(name), terms)
        states = {v: (TERNARY if v in ternary else BINARY) for v in SWEEP_GRAPHS[name]["nodes"]}
        assert all(s in states[v] for t in terms for v, s in clamps(t).items())
        # Each configuration must test at least one claim; the opt-in run pins the counts.
        assert any(counterfactual_d_separated(sweep_graph(name), [terms[i] for i in left],
                                              [terms[i] for i in right], [terms[i] for i in given])
                   for left, right, given in statements(len(terms), 2, 2)), (name, ternary)


@pytest.mark.skipif(os.environ.get("CGA_CALCULUS_SWEEP") != "1",
                    reason="larger exact Rule 2 soundness sweep: set CGA_CALCULUS_SWEEP=1")
def test_opt_in_soundness_sweep():
    # Every claim must hold exactly in three generic models; refusals are counted
    # and printed for the run record, since a refusal claims nothing.
    for name, ternary, pool, counts in OPT_IN_SWEEP:
        data = SWEEP_GRAPHS[name]
        states = {v: (TERNARY if v in ternary else BINARY) for v in data["nodes"]}
        models = [generic_scm(data["nodes"], data["directed"], data["bidirected"], states, seed=seed)
                  for seed in (41, 42, 43)]
        claims, refusals, refused_dependent = sweep(sweep_graph(name), models, [spec(t) for t in pool], 2, 2)
        print(f"{name} [{','.join(ternary) or 'binary'}]: {claims + refusals} statements, {claims} claims, "
              f"{refused_dependent} of {refusals} refusals dependent in some model")
        assert (claims + refusals, claims) == counts, (name, ternary)
