"""Independent finite checks; opt in on a GSDS compute node for full universes."""
import importlib
import json
import os
from itertools import combinations, product

import numpy as np
import pytest

from causal_graph_algorithms import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph, Probability
from tests.python.selection_recovery_oracles import BinarySCM, all_pairs, closure, evaluate, subsets


def universes(observed, mixed, limit=None, shard_index=0, shard_count=1, random_count=None):
    nodes = tuple(f"V{i}" for i in range(observed)) + ("context",)
    arrows = tuple(combinations(nodes, 2))
    bidirected = arrows if mixed else ()
    count = 0
    width = 2 ** len(bidirected)
    if random_count is None:
        indices = range(2 ** (len(arrows) + len(bidirected)))
    else:
        indices = np.random.default_rng(2026100204).choice(
            2 ** (len(arrows) + len(bidirected)), size=random_count, replace=False)
    for serial, index in enumerate(indices):
        if serial % shard_count != shard_index:
            continue
        dmask, bmask = divmod(int(index), width)
        directed = tuple(e for i, e in enumerate(arrows) if dmask & (1 << i))
        latent = tuple(e for i, e in enumerate(bidirected) if bmask & (1 << i))
        raw = DirectedBidirectedGraph(nodes=nodes, directed=directed, bidirected=latent)
        yield (AcyclicDirectedBidirectedGraph.from_graph(raw) if mixed else DAG.from_graph(raw))
        count += 1
        if limit is not None and count >= limit:
            return


def check_universe(observed, mixed, limit=None, shard_index=0, shard_count=1, random_count=None):
    recovery = importlib.import_module("causal_graph_algorithms.selection_recovery")
    report = dict(graphs=0, queries=0, pair_predicates=0, pair_searches=0,
                  pair_listings=0, adjustment_numeric=0, rc_numeric=0, max_error=0.)
    for g in universes(observed, mixed, limit, shard_index, shard_count, random_count):
        raw, context = g.topology, frozenset(("context",))
        observed_nodes = raw.nodes[:-1]
        data = Probability(observed_nodes, conditioned_on="context", conditioned_values={"context": 1})
        # Model draw is a function of this graph, independent of array partition.
        arrows = tuple(combinations(raw.nodes, 2))
        code = sum(1 << i for i, e in enumerate(arrows) if e in raw.directed)
        code += sum(1 << (i + len(arrows)) for i, e in enumerate(arrows) if e in raw.bidirected)
        scm = BinarySCM(raw, np.random.default_rng(20261002 + observed * 2 ** 21 + code))
        selected = scm.source_table(data)
        populations = {t: Probability(tuple(v for v in observed_nodes if v in t)) if t else None
                       for t in subsets(observed_nodes)}
        tables = {t: scm.source_table(p) if p is not None else None for t, p in populations.items()}
        for roles in product((0, 1, 2), repeat=observed):
            x = frozenset(v for v, role in zip(observed_nodes, roles) if role == 1)
            y = frozenset(v for v, role in zip(observed_nodes, roles) if role == 2)
            if not x or not y:
                continue
            report["queries"] += 1
            xo, yo = tuple(v for v in observed_nodes if v in x), tuple(v for v in observed_nodes if v in y)
            truth = scm.target(data, xo, yo)
            reference = all_pairs(raw, context, x, y, observed_nodes)
            for z in subsets(v for v in observed_nodes if v not in x | y):
                for zt in subsets(v for v in observed_nodes if v in z):
                    actual = recovery.selection_adjustment_pair_is_valid(
                        g, data, interventions=xo, outcomes=yo, adjustment=tuple(z),
                        population_adjustment=tuple(zt), population_data=populations[frozenset(observed_nodes)])
                    assert actual == ((z, zt) in reference), (raw.directed, raw.bidirected, x, y, z, zt)
                    report["pair_predicates"] += 1
            # Full pair/source enumeration is most expensive: exhaustive for the
            # smaller mixed universe, plus every source on the small smoke universe.
            source_sets = populations if observed <= 3 else (frozenset(), frozenset(observed_nodes))
            for t in source_sets:
                expected = {(z, zt) for z, zt in reference if zt <= t}
                population = populations[t]
                found = recovery.find_selection_adjustment_pair(
                    g, data, interventions=xo, outcomes=yo, population_data=population)
                assert isinstance(found, recovery.SelectionAdjustmentPair) == bool(expected)
                report["pair_searches"] += 1
                listed = list(recovery.iter_selection_adjustment_pairs(
                    g, data, interventions=xo, outcomes=yo, population_data=population))
                actual_pairs = [(frozenset(p.adjustment), frozenset(p.population_adjustment)) for p in listed]
                assert len(actual_pairs) == len(set(actual_pairs)) and set(actual_pairs) == expected
                report["pair_listings"] += 1
                if listed:
                    for pair in [found, *listed]:
                        actual = evaluate(pair.formula, data, selected, population, tables[t])
                        error = float(np.max(np.abs(actual - truth)))
                        assert error < 2e-10, (raw.directed, raw.bidirected, x, y, error)
                        report["max_error"] = max(report["max_error"], error)
                        report["adjustment_numeric"] += 1
            result = recovery.recover_causal_effect(g, data, interventions=xo, outcomes=yo)
            if not mixed:
                retained = frozenset(observed_nodes) - x
                d = closure(y, tuple((b, a) for a, b in raw.directed), retained)
                ancestors_context = closure(context, tuple((b, a) for a, b in raw.directed), set(raw.nodes))
                assert isinstance(result, recovery.RecoveredDistribution) == (not d & ancestors_context)
            if isinstance(result, recovery.RecoveredDistribution):
                assert result.formula.free_variables <= x | y
                actual = evaluate(result.formula, data, selected)
                error = float(np.max(np.abs(actual - truth)))
                assert error < 2e-10, (raw.directed, raw.bidirected, x, y, error)
                report["rc_numeric"] += 1
                report["max_error"] = max(report["max_error"], error)
            else:
                assert not any(not zt for _, zt in reference), "RC missed a selected-only valid pair"
        report["graphs"] += 1
    return report


def test_small_mixed_universe_against_independent_definitions_and_scms():
    assert check_universe(2, True)["graphs"] == 64


@pytest.mark.skipif(os.environ.get("CGA_SELECTION_EXHAUSTIVE") != "1",
                    reason="full scientific universes run separately on GSDS")
@pytest.mark.parametrize("observed,mixed", [(3, True), (4, False)])
def test_full_selection_recovery_universes(observed, mixed):
    index = int(os.environ.get("CGA_SELECTION_SHARD_INDEX", "0"))
    count = int(os.environ.get("CGA_SELECTION_SHARD_COUNT", "1"))
    assert 0 <= index < count
    report = check_universe(observed, mixed, shard_index=index, shard_count=count)
    print("SELECTION_RECOVERY_RESULT " + json.dumps(dict(observed=observed, mixed=mixed,
          universe="exhaustive", shard_index=index, shard_count=count, **report), sort_keys=True))
    size = 4096 if mixed else 1024
    assert report["graphs"] == len(range(index, size, count))


@pytest.mark.skipif(os.environ.get("CGA_SELECTION_EXHAUSTIVE") != "1",
                    reason="additional mixed-graph draws run separately on GSDS")
def test_random_four_observed_mixed_graphs():
    index = int(os.environ.get("CGA_SELECTION_SHARD_INDEX", "0"))
    count = int(os.environ.get("CGA_SELECTION_SHARD_COUNT", "1"))
    assert 0 <= index < count
    report = check_universe(4, True, shard_index=index, shard_count=count, random_count=2048)
    print("SELECTION_RECOVERY_RESULT " + json.dumps(dict(observed=4, mixed=True,
          universe="sample_2048", shard_index=index, shard_count=count, **report), sort_keys=True))
    assert report["graphs"] == len(range(index, 2048, count))
