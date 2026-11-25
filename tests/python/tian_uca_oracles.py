"""Independent tiny-graph oracle for the corrected Tian-to-UCA compiler.

This test-only module operates on the literal integer atoms in ``id_oracles``.  It
does not import the production Tian-to-UCA implementation or any production graph
traversal.  The block oracle follows the independently adjudicated maximal-run
reading of Jung--Tian--Bareinboim (2024), Algorithm 1; the numerical oracle follows
Equation (4) one singleton chain factor at a time, rather than evaluating grouped
UCA factors with production helpers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction

from .id_oracles import TinyADMG


@dataclass(frozen=True, slots=True)
class OracleTianFactor:
    """One independently generated UCA block."""

    variables: tuple[int, ...]
    conditioned_on: tuple[int, ...]
    fixed_conditioning: tuple[int, ...]


def topological_orders_by_permutation(graph: TinyADMG) -> tuple[tuple[int, ...], ...]:
    """Enumerate topological orders directly from the edge inequalities."""

    return tuple(
        order
        for order in itertools.permutations(graph.nodes)
        if all(order.index(parent) < order.index(child) for parent, child in graph.directed)
    )


def _has_bidirected_path_by_permutation(
    graph: TinyADMG,
    source: int,
    target: int,
) -> bool:
    if source == target:
        return True
    canonical_edges = graph.bidirected
    internal = tuple(node for node in graph.nodes if node not in (source, target))
    for length in range(len(internal) + 1):
        for middle in itertools.permutations(internal, length):
            path = (source, *middle, target)
            if all(
                (min(left, right), max(left, right)) in canonical_edges
                for left, right in itertools.pairwise(path)
            ):
                return True
    return False


def treatment_district_by_all_simple_paths(
    graph: TinyADMG,
    treatment: int,
) -> frozenset[int]:
    """Use literal simple paths, independently of production district traversal."""

    return frozenset(
        node
        for node in graph.nodes
        if _has_bidirected_path_by_permutation(graph, treatment, node)
    )


def criterion_children_by_definition(
    graph: TinyADMG,
    *,
    treatment: int,
    variable_order: tuple[int, ...],
) -> tuple[int, ...]:
    """Return ``ch_G(X) intersect district_G(X)`` in the supplied order."""

    district = treatment_district_by_all_simple_paths(graph, treatment)
    children = frozenset(
        child
        for parent, child in graph.directed
        if parent == treatment and child in district
    )
    return tuple(node for node in variable_order if node in children)


def corrected_tian_factors_by_maximal_runs(
    graph: TinyADMG,
    *,
    treatment: int,
    outcome: int,
    variable_order: tuple[int, ...],
) -> tuple[OracleTianFactor, ...]:
    """Construct the corrected Algorithm-1 blocks without production helpers."""

    if variable_order[-1] != outcome:
        raise ValueError("the oracle requires the outcome last")
    district = treatment_district_by_all_simple_paths(graph, treatment)
    pre_outcome = variable_order[:-1]
    treatment_index = pre_outcome.index(treatment)

    first_stop = treatment_index + 1
    while first_stop < len(pre_outcome) and pre_outcome[first_stop] in district:
        first_stop += 1

    factors = [OracleTianFactor(pre_outcome[:first_stop], (), ())]
    cursor = first_stop
    while cursor < len(pre_outcome):
        run_inside = pre_outcome[cursor] in district
        stop = cursor + 1
        while stop < len(pre_outcome) and (
            (pre_outcome[stop] in district) == run_inside
        ):
            stop += 1
        preceding = pre_outcome[:cursor]
        if run_inside:
            conditioned_on = preceding
            fixed_conditioning: tuple[int, ...] = ()
        else:
            conditioned_on = tuple(node for node in preceding if node != treatment)
            fixed_conditioning = (treatment,)
        factors.append(
            OracleTianFactor(
                pre_outcome[cursor:stop],
                conditioned_on,
                fixed_conditioning,
            )
        )
        cursor = stop

    if outcome in district:
        final_conditioning = pre_outcome
        final_fixed: tuple[int, ...] = ()
    else:
        final_conditioning = tuple(
            node for node in pre_outcome if node != treatment
        )
        final_fixed = (treatment,)
    factors.append(
        OracleTianFactor((outcome,), final_conditioning, final_fixed)
    )
    return tuple(factors)


def _event_probability(
    table: tuple[tuple[tuple[int, ...], Fraction], ...],
    *,
    nodes: tuple[int, ...],
    event: dict[int, int],
) -> Fraction:
    positions = {node: index for index, node in enumerate(nodes)}
    return sum(
        (
            probability
            for assignment, probability in table
            if all(assignment[positions[node]] == value for node, value in event.items())
        ),
        Fraction(0),
    )


def _observational_conditional(
    table: tuple[tuple[tuple[int, ...], Fraction], ...],
    *,
    nodes: tuple[int, ...],
    values: dict[int, int],
    variables: tuple[int, ...],
    conditioning: tuple[int, ...],
) -> Fraction:
    conditioning_event = {node: values[node] for node in conditioning}
    denominator = _event_probability(
        table,
        nodes=nodes,
        event=conditioning_event,
    )
    if denominator <= 0:
        raise AssertionError("strictly positive oracle table has zero support")
    joint_event = dict(conditioning_event)
    joint_event.update((node, values[node]) for node in variables)
    numerator = _event_probability(table, nodes=nodes, event=joint_event)
    return numerator / denominator


def equation4_binary_outcome_mean(
    graph: TinyADMG,
    *,
    table: tuple[tuple[tuple[int, ...], Fraction], ...],
    treatment: int,
    outcome: int,
    variable_order: tuple[int, ...],
    treatment_value: int,
) -> Fraction:
    """Evaluate Tian's printed Equation (4) as literal singleton factors.

    ``X`` in the summation is the dummy value ``x'``.  A factor belonging to the
    treatment district sees that dummy value; an outside-district factor sees the
    queried intervention value ``x`` instead.  Since the oracle outcome is binary,
    its conditional mean is its conditional probability of value one.
    """

    district = treatment_district_by_all_simple_paths(graph, treatment)
    pre_outcome = variable_order[:-1]
    total = Fraction(0)
    for bits in itertools.product((0, 1), repeat=len(pre_outcome)):
        random_values = dict(zip(pre_outcome, bits, strict=True))
        weight = Fraction(1)
        for index, node in enumerate(pre_outcome):
            preceding = pre_outcome[:index]
            factor_values = dict(random_values)
            if node not in district and treatment in preceding:
                factor_values[treatment] = treatment_value
            weight *= _observational_conditional(
                table,
                nodes=graph.nodes,
                values=factor_values,
                variables=(node,),
                conditioning=preceding,
            )

        outcome_values = dict(random_values)
        outcome_values[outcome] = 1
        if outcome not in district:
            outcome_values[treatment] = treatment_value
        conditional_mean = _observational_conditional(
            table,
            nodes=graph.nodes,
            values=outcome_values,
            variables=(outcome,),
            conditioning=pre_outcome,
        )
        total += weight * conditional_mean
    return total


def grouped_uca_binary_outcome_mean(
    graph: TinyADMG,
    *,
    table: tuple[tuple[tuple[int, ...], Fraction], ...],
    outcome: int,
    variable_order: tuple[int, ...],
    treatment_value: int,
    factors: tuple[OracleTianFactor, ...],
) -> Fraction:
    """Evaluate grouped factors independently against an exact positive table."""

    if factors[-1].variables != (outcome,):
        raise AssertionError("the final UCA factor must contain exactly the outcome")
    pre_outcome = variable_order[:-1]
    total = Fraction(0)
    for bits in itertools.product((0, 1), repeat=len(pre_outcome)):
        random_values = dict(zip(pre_outcome, bits, strict=True))
        weight = Fraction(1)
        for factor in factors[:-1]:
            factor_values = dict(random_values)
            factor_values.update(
                (node, treatment_value) for node in factor.fixed_conditioning
            )
            weight *= _observational_conditional(
                table,
                nodes=graph.nodes,
                values=factor_values,
                variables=factor.variables,
                conditioning=(*factor.conditioned_on, *factor.fixed_conditioning),
            )

        final_factor = factors[-1]
        final_values = dict(random_values)
        final_values[outcome] = 1
        final_values.update(
            (node, treatment_value) for node in final_factor.fixed_conditioning
        )
        conditional_mean = _observational_conditional(
            table,
            nodes=graph.nodes,
            values=final_values,
            variables=(outcome,),
            conditioning=(
                *final_factor.conditioned_on,
                *final_factor.fixed_conditioning,
            ),
        )
        total += weight * conditional_mean
    return total
