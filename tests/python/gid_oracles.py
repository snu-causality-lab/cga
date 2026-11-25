"""Definition-level oracles for strict-positive generalized identification.

The structural oracle in this module uses only integer edge atoms and exhaustive
small-graph definitions.  It does not import production graph, traversal, or
identification code.  Public probability-expression nodes are imported only so the
finite-SCM evaluator can interpret a production formula after one has been returned.

The decision criterion is Kivva et al. (2022), Theorem 1 and Algorithm 2: reduce
each district of the post-intervention ancestral target to an exact classical-ID
problem in every available source graph that covers it.  Source failure is tested by
the exact-query hedge criterion, not by the obsolete arbitrary-subquery criterion.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterable

from causal_graph_algorithms.probability import (
    Probability,
    ProbabilityExpression,
    Product,
    Ratio,
    Summation,
)

from .id_oracles import (
    HedgeProof,
    NodeSet,
    TinyADMG,
    all_bow_admgs_n3,
    all_nonempty_outcome_queries_n3,
    exact_binary_interventional_probability,
    exact_query_hedges,
    obsolete_subset_literal_hedge_criterion,
)


ExperimentCollection = tuple[NodeSet, ...]


@dataclass(frozen=True, slots=True)
class GIDLocalProblem:
    """One exact ``Q[D]`` problem inside an available source graph."""

    graph: TinyADMG
    district: NodeSet
    experiment_interventions: NodeSet
    available_vertices: NodeSet
    source_graph: TinyADMG
    local_interventions: NodeSet
    hedges: tuple[HedgeProof, ...]

    @property
    def identified(self) -> bool:
        return not self.hedges


@dataclass(frozen=True, slots=True)
class GIDSourceHedge:
    """A source target paired with its exact local classical-ID hedge."""

    experiment_interventions: NodeSet
    source_graph: TinyADMG
    local_interventions: NodeSet
    hedge: HedgeProof


@dataclass(frozen=True, slots=True)
class GIDOracleDecision:
    """Definition-level generalized-ID status and canonical structural evidence."""

    identified: bool
    target_vertices: NodeSet
    districts: tuple[NodeSet, ...]
    selected_sources: ExperimentCollection
    unresolved_district: NodeSet | None
    source_hedges: tuple[GIDSourceHedge, ...]


@dataclass(frozen=True, slots=True)
class GIDDistrictMask:
    """All source-cover and source-success bits for one target district."""

    district: NodeSet
    covering_mask: int
    successful_mask: int
    obsolete_successful_mask: int


@dataclass(frozen=True, slots=True)
class GIDQueryMaskAnalysis:
    """A query analyzed once for all collections of the eight n=3 sources."""

    graph: TinyADMG
    interventions: NodeSet
    outcomes: NodeSet
    target_vertices: NodeSet
    districts: tuple[GIDDistrictMask, ...]

    def identified_for_mask(self, mask: int, *, obsolete: bool = False) -> bool:
        if mask < 0 or mask >= 256:
            raise ValueError("n=3 experiment mask must be in [0, 256)")
        return all(
            mask
            & (
                district.obsolete_successful_mask
                if obsolete
                else district.successful_mask
            )
            for district in self.districts
        )

    def first_unresolved_for_mask(self, mask: int) -> GIDDistrictMask | None:
        if mask < 0 or mask >= 256:
            raise ValueError("n=3 experiment mask must be in [0, 256)")
        return next(
            (
                district
                for district in self.districts
                if not mask & district.successful_mask
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class GIDCensus:
    """Pinned counts over all n=3 graphs, queries, and experiment collections."""

    total_rows: int
    identified: int
    nonidentified: int
    by_collection_size: tuple[tuple[int, int], ...]
    no_cover_failures: int
    all_local_failures: int
    source_hedges: int
    obsolete_false_nonidentifications: int


@dataclass(frozen=True, slots=True)
class FiniteFormulaCase:
    """One canonical n=3 formula-soundness case and its smallest source set."""

    graph: TinyADMG
    interventions: NodeSet
    outcomes: NodeSet
    experiment_interventions: ExperimentCollection


@lru_cache(maxsize=None)
def all_intervention_targets(nodes: tuple[int, ...]) -> ExperimentCollection:
    """Return every intervention target by cardinality then declared position."""

    return tuple(
        frozenset(target)
        for size in range(len(nodes) + 1)
        for target in itertools.combinations(nodes, size)
    )


def canonical_experiment_collection(
    nodes: tuple[int, ...],
    experiments: Iterable[NodeSet],
) -> ExperimentCollection:
    """Canonicalize an already set-valued experiment collection for the oracle."""

    node_set = frozenset(nodes)
    positions = {node: index for index, node in enumerate(nodes)}
    normalized = frozenset(experiments)
    if any(not target <= node_set for target in normalized):
        raise ValueError("experiment target must be a graph subset")
    return tuple(
        sorted(
            normalized,
            key=lambda target: (
                len(target),
                tuple(positions[node] for node in nodes if node in target),
            ),
        )
    )


def induced_tiny_admg(graph: TinyADMG, retained: NodeSet) -> TinyADMG:
    """Take a literal vertex-induced integer ADMG."""

    if not retained <= frozenset(graph.nodes):
        raise ValueError("retained vertices must be a graph subset")
    return TinyADMG(
        tuple(node for node in graph.nodes if node in retained),
        frozenset(
            (parent, child)
            for parent, child in graph.directed
            if parent in retained and child in retained
        ),
        frozenset(
            (left, right)
            for left, right in graph.bidirected
            if left in retained and right in retained
        ),
    )


def _literal_directed_path_exists(
    graph: TinyADMG,
    source: int,
    target: int,
    allowed: NodeSet,
) -> bool:
    if source not in allowed or target not in allowed:
        return False
    if source == target:
        return True
    internal_candidates = tuple(
        node for node in graph.nodes if node in allowed - {source, target}
    )
    for length in range(len(internal_candidates) + 1):
        for internal in itertools.permutations(internal_candidates, length):
            path = (source, *internal, target)
            if all(
                (path[index], path[index + 1]) in graph.directed
                for index in range(len(path) - 1)
            ):
                return True
    return False


def _literal_bidirected_path_exists(
    graph: TinyADMG,
    source: int,
    target: int,
    allowed: NodeSet,
) -> bool:
    if source not in allowed or target not in allowed:
        return False
    if source == target:
        return True
    internal_candidates = tuple(
        node for node in graph.nodes if node in allowed - {source, target}
    )
    for length in range(len(internal_candidates) + 1):
        for internal in itertools.permutations(internal_candidates, length):
            path = (source, *internal, target)
            if all(
                (min(path[index], path[index + 1]), max(path[index], path[index + 1]))
                in graph.bidirected
                for index in range(len(path) - 1)
            ):
                return True
    return False


def literal_target_vertices(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
) -> NodeSet:
    """Compute ``An(Y)`` in ``G[V \\ X]`` by enumerating simple paths."""

    node_set = frozenset(graph.nodes)
    if not outcomes or not outcomes <= node_set or not interventions <= node_set:
        raise ValueError("invalid generalized-ID query")
    if outcomes & interventions:
        raise ValueError("query roles must be disjoint")
    allowed = node_set - interventions
    return frozenset(
        candidate
        for candidate in allowed
        if any(
            _literal_directed_path_exists(graph, candidate, outcome, allowed)
            for outcome in outcomes
        )
    )


def literal_districts(graph: TinyADMG, vertices: NodeSet) -> tuple[NodeSet, ...]:
    """Partition a vertex set by exhaustive bidirected-path existence."""

    if not vertices <= frozenset(graph.nodes):
        raise ValueError("district vertices must be a graph subset")
    remaining = set(vertices)
    districts: list[NodeSet] = []
    for seed in graph.nodes:
        if seed not in remaining:
            continue
        district = frozenset(
            candidate
            for candidate in vertices
            if _literal_bidirected_path_exists(graph, seed, candidate, vertices)
        )
        remaining.difference_update(district)
        districts.append(district)
    return tuple(districts)


@lru_cache(maxsize=None)
def gid_local_problem(
    graph: TinyADMG,
    district: NodeSet,
    experiment_interventions: NodeSet,
) -> GIDLocalProblem:
    """Reduce one source/district pair to its exact classical-ID query."""

    node_set = frozenset(graph.nodes)
    available = node_set - experiment_interventions
    if not district or not district <= available:
        raise ValueError("source does not cover the target district")
    source_graph = induced_tiny_admg(graph, available)
    local_interventions = available - district
    hedges = exact_query_hedges(
        source_graph,
        interventions=local_interventions,
        outcomes=district,
    )
    if len(hedges) > 1:
        raise AssertionError("a single-c-component local query has multiple hedges")
    return GIDLocalProblem(
        graph=graph,
        district=district,
        experiment_interventions=experiment_interventions,
        available_vertices=available,
        source_graph=source_graph,
        local_interventions=local_interventions,
        hedges=hedges,
    )


def generalized_identification_oracle(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
    experiment_interventions: Iterable[NodeSet],
) -> GIDOracleDecision:
    """Apply the exact-local Kivva criterion with canonical source selection."""

    experiments = canonical_experiment_collection(graph.nodes, experiment_interventions)
    target = literal_target_vertices(
        graph,
        interventions=interventions,
        outcomes=outcomes,
    )
    districts = literal_districts(graph, target)
    node_set = frozenset(graph.nodes)
    selected: list[NodeSet] = []

    for district in districts:
        failed: list[GIDSourceHedge] = []
        successful_source: NodeSet | None = None
        for experiment in experiments:
            available = node_set - experiment
            if not district <= available:
                continue
            problem = gid_local_problem(graph, district, experiment)
            if problem.identified:
                successful_source = experiment
                break
            (hedge,) = problem.hedges
            failed.append(
                GIDSourceHedge(
                    experiment_interventions=experiment,
                    source_graph=problem.source_graph,
                    local_interventions=problem.local_interventions,
                    hedge=hedge,
                )
            )
        if successful_source is None:
            return GIDOracleDecision(
                identified=False,
                target_vertices=target,
                districts=districts,
                selected_sources=tuple(selected),
                unresolved_district=district,
                source_hedges=tuple(failed),
            )
        selected.append(successful_source)

    return GIDOracleDecision(
        identified=True,
        target_vertices=target,
        districts=districts,
        selected_sources=tuple(selected),
        unresolved_district=None,
        source_hedges=(),
    )


@lru_cache(maxsize=1)
def n3_gid_query_analyses() -> tuple[GIDQueryMaskAnalysis, ...]:
    """Precompute all 3,800 graph/query analyses for every source mask."""

    sources = all_intervention_targets((0, 1, 2))
    analyses: list[GIDQueryMaskAnalysis] = []
    for graph in all_bow_admgs_n3():
        node_set = frozenset(graph.nodes)
        for interventions, outcomes in all_nonempty_outcome_queries_n3():
            target = literal_target_vertices(
                graph,
                interventions=interventions,
                outcomes=outcomes,
            )
            district_masks: list[GIDDistrictMask] = []
            for district in literal_districts(graph, target):
                covering_mask = 0
                successful_mask = 0
                obsolete_successful_mask = 0
                for source_index, experiment in enumerate(sources):
                    if not district <= node_set - experiment:
                        continue
                    source_bit = 1 << source_index
                    covering_mask |= source_bit
                    problem = gid_local_problem(graph, district, experiment)
                    if problem.identified:
                        successful_mask |= source_bit
                    obsolete_fails = obsolete_subset_literal_hedge_criterion(
                        problem.source_graph,
                        interventions=problem.local_interventions,
                        outcomes=district,
                    )
                    if not obsolete_fails:
                        obsolete_successful_mask |= source_bit
                district_masks.append(
                    GIDDistrictMask(
                        district=district,
                        covering_mask=covering_mask,
                        successful_mask=successful_mask,
                        obsolete_successful_mask=obsolete_successful_mask,
                    )
                )
            analyses.append(
                GIDQueryMaskAnalysis(
                    graph=graph,
                    interventions=interventions,
                    outcomes=outcomes,
                    target_vertices=target,
                    districts=tuple(district_masks),
                )
            )
    return tuple(analyses)


@lru_cache(maxsize=1)
def all_gid_local_problems_n3() -> tuple[GIDLocalProblem, ...]:
    """Return the 3,100 unique local ``Q[D] @ G[A]`` problems in the census."""

    sources = all_intervention_targets((0, 1, 2))
    seen: dict[tuple[TinyADMG, NodeSet, NodeSet], None] = {}
    for analysis in n3_gid_query_analyses():
        for district_mask in analysis.districts:
            for source_index, experiment in enumerate(sources):
                if district_mask.covering_mask & (1 << source_index):
                    seen.setdefault(
                        (analysis.graph, district_mask.district, experiment), None
                    )
    return tuple(
        gid_local_problem(graph, district, experiment)
        for graph, district, experiment in seen
    )


@lru_cache(maxsize=1)
def n3_gid_census() -> GIDCensus:
    """Exhaust every one of the 200*19*256 generalized-ID rows."""

    identified = 0
    nonidentified = 0
    no_cover = 0
    all_local_fail = 0
    source_hedges = 0
    obsolete_false_nonidentified = 0
    by_size = [[0, 0] for _ in range(9)]

    for analysis in n3_gid_query_analyses():
        for mask in range(256):
            source_count = mask.bit_count()
            unresolved = analysis.first_unresolved_for_mask(mask)
            if unresolved is None:
                identified += 1
                by_size[source_count][1] += 1
                if not analysis.identified_for_mask(mask, obsolete=True):
                    obsolete_false_nonidentified += 1
                continue

            nonidentified += 1
            by_size[source_count][0] += 1
            failed_covering_mask = mask & unresolved.covering_mask
            if not failed_covering_mask:
                no_cover += 1
            else:
                all_local_fail += 1
                source_hedges += failed_covering_mask.bit_count()

    return GIDCensus(
        total_rows=identified + nonidentified,
        identified=identified,
        nonidentified=nonidentified,
        by_collection_size=tuple(tuple(row) for row in by_size),
        no_cover_failures=no_cover,
        all_local_failures=all_local_fail,
        source_hedges=source_hedges,
        obsolete_false_nonidentifications=obsolete_false_nonidentified,
    )


@lru_cache(maxsize=1)
def minimal_finite_formula_cases_n3() -> tuple[FiniteFormulaCase, ...]:
    """Choose a smallest nonobservational source set, or observational fallback."""

    sources = all_intervention_targets((0, 1, 2))
    nonobservational_indices = tuple(range(1, len(sources)))
    cases: list[FiniteFormulaCase] = []
    for analysis in n3_gid_query_analyses():
        selected_mask: int | None = None
        for size in range(1, len(nonobservational_indices) + 1):
            for indices in itertools.combinations(nonobservational_indices, size):
                mask = sum(1 << index for index in indices)
                if analysis.identified_for_mask(mask):
                    selected_mask = mask
                    break
            if selected_mask is not None:
                break
        if selected_mask is None:
            selected_mask = 1
            if not analysis.identified_for_mask(selected_mask):
                raise AssertionError(
                    "query has neither a nonobservational nor observational formula case"
                )
        experiments = tuple(
            source
            for index, source in enumerate(sources)
            if selected_mask & (1 << index)
        )
        cases.append(
            FiniteFormulaCase(
                graph=analysis.graph,
                interventions=analysis.interventions,
                outcomes=analysis.outcomes,
                experiment_interventions=experiments,
            )
        )
    return tuple(cases)


@lru_cache(maxsize=None)
def exact_binary_source_distribution(
    graph: TinyADMG,
    experiment_interventions: tuple[int, ...],
    intervention_values: tuple[int, ...],
    variant: int,
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    """Tabulate one source distribution directly from the truncated SCM product."""

    if len(experiment_interventions) != len(intervention_values):
        raise ValueError("source intervention values do not match their labels")
    intervention_set = frozenset(experiment_interventions)
    if not intervention_set <= frozenset(graph.nodes):
        raise ValueError("source intervention is outside the graph")
    available = tuple(node for node in graph.nodes if node not in intervention_set)
    fixed = dict(zip(experiment_interventions, intervention_values, strict=True))
    rows: list[tuple[tuple[int, ...], Fraction]] = []
    for assignment in itertools.product((0, 1), repeat=len(available)):
        values = dict(fixed)
        values.update(zip(available, assignment, strict=True))
        rows.append(
            (
                assignment,
                exact_binary_interventional_probability(
                    graph,
                    outcomes=available,
                    interventions=experiment_interventions,
                    values=values,
                    variant=variant,
                ),
            )
        )
    result = tuple(rows)
    if sum((probability for _, probability in result), Fraction(0)) != 1:
        raise AssertionError("finite-SCM source distribution is not normalized")
    if any(probability <= 0 for _, probability in result):
        raise AssertionError("finite-SCM source distribution is not positive")
    return result


def evaluate_generalized_probability_expression(
    expression: ProbabilityExpression,
    *,
    graph: TinyADMG,
    node_labels: tuple[str, ...],
    experiment_interventions: ExperimentCollection,
    environment: dict[str, int],
    variant: int,
) -> Fraction:
    """Interpret source-stamped probability IR against exact finite-SCM sources."""

    if len(node_labels) != len(graph.nodes) or len(set(node_labels)) != len(
        node_labels
    ):
        raise ValueError("node labels must correspond one-to-one with graph nodes")
    label_to_node = dict(zip(node_labels, graph.nodes, strict=True))
    available_sources = frozenset(experiment_interventions)

    def evaluate(
        current: ProbabilityExpression,
        current_environment: dict[str, int],
    ) -> Fraction:
        if isinstance(current, Probability):
            try:
                source = frozenset(
                    label_to_node[label] for label in current.interventions
                )
            except KeyError as error:
                raise AssertionError(
                    "formula atom contains an unknown label"
                ) from error
            if source not in available_sources:
                raise AssertionError(
                    "formula atom is not stamped with an exact supplied source"
                )
            source_order = tuple(node for node in graph.nodes if node in source)
            source_values = tuple(
                current_environment[node_labels[graph.nodes.index(node)]]
                for node in source_order
            )
            table = exact_binary_source_distribution(
                graph,
                source_order,
                source_values,
                variant,
            )
            available = tuple(node for node in graph.nodes if node not in source)
            positions = {node: index for index, node in enumerate(available)}

            def event_probability(labels: tuple[str, ...]) -> Fraction:
                try:
                    event = tuple(
                        (label_to_node[label], current_environment[label])
                        for label in labels
                    )
                except KeyError as error:
                    raise AssertionError(
                        "formula evaluation is missing a free variable"
                    ) from error
                if any(node in source for node, _ in event):
                    raise AssertionError(
                        "source intervention appears as a random formula variable"
                    )
                return sum(
                    (
                        probability
                        for assignment, probability in table
                        if all(
                            assignment[positions[node]] == value
                            for node, value in event
                        )
                    ),
                    Fraction(0),
                )

            numerator = event_probability(current.variables + current.conditioned_on)
            if not current.conditioned_on:
                return numerator
            denominator = event_probability(current.conditioned_on)
            if denominator <= 0:
                raise AssertionError("positive source has a zero conditioning event")
            return numerator / denominator

        if isinstance(current, Product):
            product = Fraction(1)
            for factor in current.factors:
                product *= evaluate(factor, current_environment)
            return product

        if isinstance(current, Summation):
            total = Fraction(0)
            for assignment in itertools.product((0, 1), repeat=len(current.variables)):
                bound_environment = dict(current_environment)
                bound_environment.update(
                    zip(current.variables, assignment, strict=True)
                )
                total += evaluate(current.body, bound_environment)
            return total

        if not isinstance(current, Ratio):
            raise AssertionError(f"unknown probability-expression node: {current!r}")
        denominator = evaluate(current.denominator, current_environment)
        if denominator <= 0:
            raise AssertionError("positive source has a zero formula denominator")
        return evaluate(current.numerator, current_environment) / denominator

    return evaluate(expression, environment)
