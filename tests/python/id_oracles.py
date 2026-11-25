"""Independent tiny-graph oracles for scientific ID/hedge adjudication.

This test-only module intentionally uses plain integer graph atoms.  In particular, it
does not import production graph, traversal, fixing, or identification code.  The
finite-SCM oracle below imports only the public probability-expression node types so
that it can interpret, but never generate, a production formula.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from causal_graph_algorithms.probability import (
    Probability,
    ProbabilityExpression,
    Product,
    Ratio,
    Summation,
)

from .oracles import (
    ConditionalIndependenceSignature,
    MixedEdgeAtom,
    m_separated_by_canonical_dag,
)


NodeSet = frozenset[int]
DirectedEdge = tuple[int, int]
BidirectedEdge = tuple[int, int]
Query = tuple[NodeSet, NodeSet]
ConditionalQuery = tuple[NodeSet, NodeSet, NodeSet]


@dataclass(frozen=True, slots=True)
class TinyADMG:
    """A bow-capable ADMG represented only by literal integer edge atoms."""

    nodes: tuple[int, ...]
    directed: frozenset[DirectedEdge]
    bidirected: frozenset[BidirectedEdge]

    def __post_init__(self) -> None:
        node_set = frozenset(self.nodes)
        if tuple(sorted(node_set)) != self.nodes:
            raise ValueError("nodes must be unique and sorted")
        if any(
            left == right or left not in node_set or right not in node_set
            for left, right in self.directed
        ):
            raise ValueError("invalid directed edge")
        if any(
            left >= right or left not in node_set or right not in node_set
            for left, right in self.bidirected
        ):
            raise ValueError("bidirected edges must use canonical distinct endpoints")
        if not _has_topological_permutation(self.nodes, self.directed):
            raise ValueError("directed edges must be acyclic")


@dataclass(frozen=True, slots=True)
class CForestProof:
    """An actual selected-edge sink-rooted C-forest."""

    nodes: NodeSet
    roots: NodeSet
    directed: frozenset[DirectedEdge]
    bidirected_tree: frozenset[BidirectedEdge]


@dataclass(frozen=True, slots=True)
class HedgeProof:
    """Nested selected-edge C-forests plus the greedy fixing trace."""

    inner: CForestProof
    outer: CForestProof
    fixing_order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReachableClosure:
    """Result of greedily fixing vertices outside a target set."""

    target: NodeSet
    closure: NodeSet
    fixing_order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LiteralCForestProof:
    """A selected-edge C-forest found from Shpitser--Pearl Definition 5."""

    nodes: NodeSet
    roots: NodeSet
    directed: frozenset[DirectedEdge]
    bidirected_tree: frozenset[BidirectedEdge]


@dataclass(frozen=True, slots=True)
class LiteralHedgeProof:
    """A nested C-forest pair independent of fixing and the ID recursion."""

    inner: LiteralCForestProof
    outer: LiteralCForestProof


@dataclass(frozen=True, slots=True)
class LiteralIDCReduction:
    """Definition-level result of the iterative UAI-06 IDC reduction."""

    outcomes: NodeSet
    original_interventions: NodeSet
    original_conditions: NodeSet
    rule_two_interventions: tuple[int, ...]
    interventions: NodeSet
    residual_conditions: NodeSet
    joint_outcomes: NodeSet
    id_failed: bool


def _canonical_bidirected_edges(graph: TinyADMG) -> tuple[BidirectedEdge, ...]:
    return tuple(sorted(graph.bidirected))


def _node_kernel_probability(
    graph: TinyADMG,
    node: int,
    observed: dict[int, int],
    latent: tuple[int, ...],
    latent_edges: tuple[BidirectedEdge, ...],
    variant: int,
) -> Fraction:
    positions = {vertex: index for index, vertex in enumerate(graph.nodes)}
    parents = tuple(
        sorted(
            (parent for parent, child in graph.directed if child == node),
            key=positions.__getitem__,
        )
    )
    incident_latents = tuple(
        index for index, edge in enumerate(latent_edges) if node in edge
    )
    input_bits = tuple(observed[parent] for parent in parents) + tuple(
        latent[index] for index in incident_latents
    )
    code = sum(bit << index for index, bit in enumerate(input_bits))
    offset = (positions[node] + 1) * (variant + 1)
    probability_one = Fraction(
        offset + code + 1,
        offset + (1 << len(input_bits)) + 1,
    )
    return probability_one if observed[node] else 1 - probability_one


def _latent_assignment_probability(
    latent: tuple[int, ...], variant: int
) -> Fraction:
    probability = Fraction(1)
    for index, value in enumerate(latent):
        probability_one = Fraction(2 + ((index + variant) % 3), 7)
        probability *= probability_one if value else 1 - probability_one
    return probability


@lru_cache(maxsize=None)
def exact_binary_observational_distribution(
    graph: TinyADMG,
    variant: int,
) -> tuple[tuple[tuple[int, ...], Fraction], ...]:
    """Return a strictly positive observational table for a deterministic SCM."""

    latent_edges = _canonical_bidirected_edges(graph)
    table: list[tuple[tuple[int, ...], Fraction]] = []
    for observed_bits in itertools.product((0, 1), repeat=len(graph.nodes)):
        observed = dict(zip(graph.nodes, observed_bits, strict=True))
        probability = Fraction(0)
        for latent in itertools.product((0, 1), repeat=len(latent_edges)):
            term = _latent_assignment_probability(latent, variant)
            for node in graph.nodes:
                term *= _node_kernel_probability(
                    graph,
                    node,
                    observed,
                    latent,
                    latent_edges,
                    variant,
                )
            probability += term
        table.append((observed_bits, probability))
    result = tuple(table)
    if sum((probability for _, probability in result), Fraction(0)) != 1:
        raise AssertionError("finite-SCM observational table is not normalized")
    if any(probability <= 0 for _, probability in result):
        raise AssertionError("finite-SCM observational table is not strictly positive")
    return result


def exact_binary_interventional_probability(
    graph: TinyADMG,
    *,
    outcomes: tuple[int, ...],
    interventions: tuple[int, ...],
    values: dict[int, int],
    variant: int,
) -> Fraction:
    """Evaluate an intervention directly by the truncated SCM product."""

    latent_edges = _canonical_bidirected_edges(graph)
    intervention_set = frozenset(interventions)
    fixed = frozenset(outcomes) | intervention_set
    probability = Fraction(0)
    for observed_bits in itertools.product((0, 1), repeat=len(graph.nodes)):
        observed = dict(zip(graph.nodes, observed_bits, strict=True))
        if any(observed[node] != values[node] for node in fixed):
            continue
        for latent in itertools.product((0, 1), repeat=len(latent_edges)):
            term = _latent_assignment_probability(latent, variant)
            for node in graph.nodes:
                if node not in intervention_set:
                    term *= _node_kernel_probability(
                        graph,
                        node,
                        observed,
                        latent,
                        latent_edges,
                        variant,
                    )
            probability += term
    return probability


def exact_binary_conditional_interventional_probability(
    graph: TinyADMG,
    *,
    outcomes: tuple[int, ...],
    interventions: tuple[int, ...],
    conditioned_on: tuple[int, ...],
    values: dict[int, int],
    variant: int,
) -> Fraction:
    """Evaluate ``P_x(Y | Z)`` directly from the truncated finite SCM."""

    joint = exact_binary_interventional_probability(
        graph,
        outcomes=(*outcomes, *conditioned_on),
        interventions=interventions,
        values=values,
        variant=variant,
    )
    if not conditioned_on:
        return joint
    denominator = exact_binary_interventional_probability(
        graph,
        outcomes=conditioned_on,
        interventions=interventions,
        values=values,
        variant=variant,
    )
    if denominator <= 0:
        raise AssertionError("strictly positive SCM produced unsupported evidence")
    return joint / denominator


def evaluate_probability_expression(
    expression: ProbabilityExpression,
    *,
    table: tuple[tuple[tuple[int, ...], Fraction], ...],
    node_labels: tuple[str, ...],
    environment: dict[str, int],
) -> Fraction:
    """Structurally interpret the public IR against an exact observational table."""

    if isinstance(expression, Probability):
        if expression.interventions:
            raise AssertionError("ID formula contains an interventional atom")
        positions = {label: index for index, label in enumerate(node_labels)}

        def event_probability(labels: tuple[str, ...]) -> Fraction:
            return sum(
                (
                    probability
                    for assignment, probability in table
                    if all(
                        assignment[positions[label]] == environment[label]
                        for label in labels
                    )
                ),
                Fraction(0),
            )

        numerator = event_probability(
            expression.variables + expression.conditioned_on
        )
        if not expression.conditioned_on:
            return numerator
        denominator = event_probability(expression.conditioned_on)
        if denominator <= 0:
            raise AssertionError("positive SCM produced a zero conditioning event")
        return numerator / denominator

    if isinstance(expression, Product):
        value = Fraction(1)
        for factor in expression.factors:
            value *= evaluate_probability_expression(
                factor,
                table=table,
                node_labels=node_labels,
                environment=environment,
            )
        return value

    if isinstance(expression, Summation):
        value = Fraction(0)
        for assignment in itertools.product((0, 1), repeat=len(expression.variables)):
            bound_environment = dict(environment)
            bound_environment.update(
                zip(expression.variables, assignment, strict=True)
            )
            value += evaluate_probability_expression(
                expression.body,
                table=table,
                node_labels=node_labels,
                environment=bound_environment,
            )
        return value

    if not isinstance(expression, Ratio):
        raise AssertionError(f"unknown probability-expression node: {expression!r}")
    denominator = evaluate_probability_expression(
        expression.denominator,
        table=table,
        node_labels=node_labels,
        environment=environment,
    )
    if denominator <= 0:
        raise AssertionError("positive SCM produced a zero formula denominator")
    return evaluate_probability_expression(
        expression.numerator,
        table=table,
        node_labels=node_labels,
        environment=environment,
    ) / denominator


def _has_topological_permutation(
    nodes: tuple[int, ...], directed: frozenset[DirectedEdge]
) -> bool:
    for order in itertools.permutations(nodes):
        position = {node: index for index, node in enumerate(order)}
        if all(position[parent] < position[child] for parent, child in directed):
            return True
    return False


@lru_cache(maxsize=1)
def all_bow_admgs_n3() -> tuple[TinyADMG, ...]:
    """Enumerate all 25 labeled DAGs times all eight bidirected masks."""

    nodes = (0, 1, 2)
    unordered_pairs = tuple(itertools.combinations(nodes, 2))
    dags: list[frozenset[DirectedEdge]] = []
    for states in itertools.product((0, 1, 2), repeat=len(unordered_pairs)):
        directed: set[DirectedEdge] = set()
        for (left, right), state in zip(unordered_pairs, states, strict=True):
            if state == 1:
                directed.add((left, right))
            elif state == 2:
                directed.add((right, left))
        frozen_directed = frozenset(directed)
        if _has_topological_permutation(nodes, frozen_directed):
            dags.append(frozen_directed)

    graphs: list[TinyADMG] = []
    for directed in dags:
        for mask in range(1 << len(unordered_pairs)):
            bidirected = frozenset(
                edge
                for index, edge in enumerate(unordered_pairs)
                if mask & (1 << index)
            )
            graphs.append(TinyADMG(nodes, directed, bidirected))
    return tuple(graphs)


def labeled_bow_admg_atoms(
    graph: TinyADMG, labels: tuple[str, ...]
) -> tuple[MixedEdgeAtom, ...]:
    """Convert parallel directed/bidirected atoms without collapsing bows."""

    if len(labels) != len(graph.nodes) or len(set(labels)) != len(labels):
        raise ValueError("labels must correspond one-to-one with ADMG nodes")
    positions = {node: index for index, node in enumerate(graph.nodes)}
    node_labels = dict(zip(graph.nodes, labels, strict=True))
    atoms: list[MixedEdgeAtom] = []
    for parent, child in sorted(graph.directed):
        if positions[parent] < positions[child]:
            atoms.append((node_labels[parent], False, node_labels[child], True))
        else:
            atoms.append((node_labels[child], True, node_labels[parent], False))
    atoms.extend(
        (node_labels[left], True, node_labels[right], True)
        for left, right in sorted(graph.bidirected)
    )
    return tuple(atoms)


def bow_admg_ci_signature_by_canonical_dag(
    graph: TinyADMG, labels: tuple[str, ...]
) -> ConditionalIndependenceSignature:
    """Compute the observed CI model through an independent latent DAG."""

    atoms = labeled_bow_admg_atoms(graph, labels)
    signature: list[tuple[str, str, tuple[str, ...], bool]] = []
    for left_index, right_index in itertools.combinations(range(len(labels)), 2):
        left = labels[left_index]
        right = labels[right_index]
        remaining = tuple(
            label
            for index, label in enumerate(labels)
            if index not in (left_index, right_index)
        )
        for size in range(len(remaining) + 1):
            for given in itertools.combinations(remaining, size):
                signature.append(
                    (
                        left,
                        right,
                        given,
                        m_separated_by_canonical_dag(
                            labels,
                            atoms,
                            (left,),
                            (right,),
                            given,
                        ),
                    )
                )
    return tuple(signature)


@lru_cache(maxsize=1)
def all_nonempty_outcome_queries_n3() -> tuple[Query, ...]:
    """Enumerate all 19 disjoint (intervention, nonempty outcome) queries."""

    queries: list[Query] = []
    for roles in itertools.product((0, 1, 2), repeat=3):
        interventions = frozenset(node for node, role in enumerate(roles) if role == 1)
        outcomes = frozenset(node for node, role in enumerate(roles) if role == 2)
        if outcomes:
            queries.append((interventions, outcomes))
    return tuple(queries)


@lru_cache(maxsize=1)
def all_conditional_queries_n3() -> tuple[ConditionalQuery, ...]:
    """Enumerate all 37 disjoint ``(X, Y, Z)`` queries with nonempty ``Y``."""

    queries: list[ConditionalQuery] = []
    for roles in itertools.product((0, 1, 2, 3), repeat=3):
        interventions = frozenset(node for node, role in enumerate(roles) if role == 1)
        outcomes = frozenset(node for node, role in enumerate(roles) if role == 2)
        conditions = frozenset(node for node, role in enumerate(roles) if role == 3)
        if outcomes:
            queries.append((interventions, outcomes, conditions))
    return tuple(queries)


def _validate_query(graph: TinyADMG, interventions: NodeSet, outcomes: NodeSet) -> None:
    node_set = frozenset(graph.nodes)
    if not outcomes:
        raise ValueError("outcomes must be nonempty")
    if not interventions <= node_set or not outcomes <= node_set:
        raise ValueError("query vertices must belong to the graph")
    if interventions & outcomes:
        raise ValueError("interventions and outcomes must be disjoint")


# The classical ID oracle deliberately uses iterative scans/components, rather than the
# simple-path routines used by the modern fixing oracle below.
def _id_ancestors(
    nodes: NodeSet,
    seeds: NodeSet,
    directed: frozenset[DirectedEdge],
) -> NodeSet:
    ancestors = set(seeds)
    changed = True
    while changed:
        changed = False
        for parent, child in directed:
            if parent in nodes and child in ancestors and parent not in ancestors:
                ancestors.add(parent)
                changed = True
    return frozenset(ancestors)


def _id_districts(
    nodes: NodeSet, bidirected: frozenset[BidirectedEdge]
) -> tuple[NodeSet, ...]:
    remaining = set(nodes)
    districts: list[NodeSet] = []
    while remaining:
        component = {min(remaining)}
        changed = True
        while changed:
            changed = False
            for left, right in bidirected:
                if left not in nodes or right not in nodes:
                    continue
                if left in component and right not in component:
                    component.add(right)
                    changed = True
                if right in component and left not in component:
                    component.add(left)
                    changed = True
        remaining -= component
        districts.append(frozenset(component))
    return tuple(sorted(districts, key=lambda district: tuple(sorted(district))))


def literal_id_fails(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
) -> bool:
    """Run the published seven-line ID control flow with no probability algebra."""

    _validate_query(graph, interventions, outcomes)

    @lru_cache(maxsize=None)
    def recurse(current: NodeSet, y: NodeSet, x: NodeSet) -> bool:
        if not x:
            return False

        ancestors = _id_ancestors(current, y, graph.directed)
        if ancestors != current:
            return recurse(ancestors, y, x & ancestors)

        not_intervened = current - x
        post_intervention_ancestors = _id_ancestors(not_intervened, y, graph.directed)
        added_interventions = not_intervened - post_intervention_ancestors
        if added_interventions:
            return recurse(current, y, x | added_interventions)

        remaining_districts = _id_districts(not_intervened, graph.bidirected)
        if len(remaining_districts) > 1:
            return any(
                recurse(current, district, current - district)
                for district in remaining_districts
            )

        (remaining_district,) = remaining_districts
        current_districts = _id_districts(current, graph.bidirected)
        if len(current_districts) == 1:
            return True
        if remaining_district in current_districts:
            return False

        strict_supersets = tuple(
            district for district in current_districts if remaining_district < district
        )
        if len(strict_supersets) != 1:
            raise AssertionError("ID line 7 must have one containing district")
        (containing_district,) = strict_supersets
        return recurse(
            containing_district,
            y,
            x & containing_district,
        )

    return recurse(frozenset(graph.nodes), outcomes, interventions)


def _directed_path_exists(
    directed: frozenset[DirectedEdge],
    source: int,
    target: int,
    allowed: NodeSet,
) -> bool:
    if source not in allowed or target not in allowed:
        return False
    if source == target:
        return True

    def visit(current: int, visited: NodeSet) -> bool:
        for left, right in sorted(directed):
            if left != current or right not in allowed or right in visited:
                continue
            if right == target or visit(right, visited | {right}):
                return True
        return False

    return visit(source, frozenset({source}))


def _bidirected_path_exists(
    bidirected: frozenset[BidirectedEdge],
    source: int,
    target: int,
    allowed: NodeSet,
) -> bool:
    if source not in allowed or target not in allowed:
        return False
    if source == target:
        return True

    def visit(current: int, visited: NodeSet) -> bool:
        for left, right in sorted(bidirected):
            if left == current:
                neighbor = right
            elif right == current:
                neighbor = left
            else:
                continue
            if neighbor not in allowed or neighbor in visited:
                continue
            if neighbor == target or visit(neighbor, visited | {neighbor}):
                return True
        return False

    return visit(source, frozenset({source}))


def _path_districts(
    nodes: NodeSet, bidirected: frozenset[BidirectedEdge]
) -> tuple[NodeSet, ...]:
    remaining = set(nodes)
    districts: list[NodeSet] = []
    while remaining:
        seed = min(remaining)
        component = frozenset(
            candidate
            for candidate in nodes
            if _bidirected_path_exists(bidirected, seed, candidate, nodes)
        )
        remaining -= component
        districts.append(component)
    return tuple(sorted(districts, key=lambda district: tuple(sorted(district))))


def _is_fixable(graph: TinyADMG, vertex: int, random_nodes: NodeSet) -> bool:
    district = next(
        district
        for district in _path_districts(random_nodes, graph.bidirected)
        if vertex in district
    )
    descendants = frozenset(
        candidate
        for candidate in random_nodes
        if _directed_path_exists(graph.directed, vertex, candidate, random_nodes)
    )
    return district & descendants == frozenset({vertex})


def greedy_reachable_closure(graph: TinyADMG, target: NodeSet) -> ReachableClosure:
    """Greedily fix the first definition-level fixable vertex outside target."""

    if not target or not target <= frozenset(graph.nodes):
        raise ValueError("target must be a nonempty graph subset")
    random_nodes = frozenset(graph.nodes)
    fixing_order: list[int] = []
    while True:
        next_vertex = next(
            (
                vertex
                for vertex in sorted(random_nodes - target)
                if _is_fixable(graph, vertex, random_nodes)
            ),
            None,
        )
        if next_vertex is None:
            break
        fixing_order.append(next_vertex)
        random_nodes = random_nodes - {next_vertex}
    return ReachableClosure(target, random_nodes, tuple(fixing_order))


def validate_reachable_closure_trace(graph: TinyADMG, result: ReachableClosure) -> bool:
    """Replay the fixing trace and confirm that it stops exactly when stuck."""

    random_nodes = frozenset(graph.nodes)
    for vertex in result.fixing_order:
        if vertex in result.target or vertex not in random_nodes:
            return False
        if not _is_fixable(graph, vertex, random_nodes):
            return False
        random_nodes = random_nodes - {vertex}
    if random_nodes != result.closure:
        return False
    return not any(
        _is_fixable(graph, vertex, random_nodes)
        for vertex in random_nodes - result.target
    )


def _is_bidirected_tree(nodes: NodeSet, edges: frozenset[BidirectedEdge]) -> bool:
    if len(edges) != len(nodes) - 1:
        return False
    if any(left not in nodes or right not in nodes for left, right in edges):
        return False
    seed = min(nodes)
    return all(
        _bidirected_path_exists(edges, seed, candidate, nodes) for candidate in nodes
    )


def _first_spanning_tree(
    nodes: NodeSet,
    ambient_edges: frozenset[BidirectedEdge],
    required: frozenset[BidirectedEdge] = frozenset(),
) -> frozenset[BidirectedEdge]:
    if len(required) > len(nodes) - 1:
        raise AssertionError("required bidirected edges already contain a cycle")
    available = tuple(
        edge
        for edge in sorted(ambient_edges - required)
        if edge[0] in nodes and edge[1] in nodes
    )
    addition_count = len(nodes) - 1 - len(required)
    for additions in itertools.combinations(available, addition_count):
        candidate = required | frozenset(additions)
        if _is_bidirected_tree(nodes, candidate):
            return candidate
    raise AssertionError("C-forest node set has no bidirected spanning tree")


def _selected_directed_forest(
    graph: TinyADMG,
    nodes: NodeSet,
    roots: NodeSet,
    *,
    inner: CForestProof | None = None,
) -> frozenset[DirectedEdge]:
    selected = set(() if inner is None else inner.directed)
    protected_inner = frozenset() if inner is None else inner.nodes
    for source in sorted(nodes - roots - protected_inner):
        chosen = next(
            (
                (source, child)
                for parent, child in sorted(graph.directed)
                if parent == source
                and child in nodes
                and any(
                    _directed_path_exists(graph.directed, child, root, nodes)
                    for root in roots
                )
            ),
            None,
        )
        if chosen is None:
            raise AssertionError("C-forest vertex has no path toward a root")
        selected.add(chosen)
    return frozenset(selected)


def _construct_hedge(
    graph: TinyADMG,
    inner_nodes: NodeSet,
    outer_nodes: NodeSet,
    fixing_order: tuple[int, ...],
) -> HedgeProof:
    roots = frozenset(
        node
        for node in inner_nodes
        if not any(
            parent == node and child in inner_nodes for parent, child in graph.directed
        )
    )
    inner_tree = _first_spanning_tree(inner_nodes, graph.bidirected)
    inner = CForestProof(
        nodes=inner_nodes,
        roots=roots,
        directed=_selected_directed_forest(graph, inner_nodes, roots),
        bidirected_tree=inner_tree,
    )
    outer = CForestProof(
        nodes=outer_nodes,
        roots=roots,
        directed=_selected_directed_forest(graph, outer_nodes, roots, inner=inner),
        bidirected_tree=_first_spanning_tree(
            outer_nodes,
            graph.bidirected,
            required=inner_tree,
        ),
    )
    return HedgeProof(inner, outer, fixing_order)


def _is_selected_c_forest(graph: TinyADMG, forest: CForestProof) -> bool:
    if not forest.nodes or not forest.roots or not forest.roots <= forest.nodes:
        return False
    if any(edge not in graph.directed for edge in forest.directed):
        return False
    if any(edge not in graph.bidirected for edge in forest.bidirected_tree):
        return False
    if any(
        parent not in forest.nodes or child not in forest.nodes
        for parent, child in forest.directed
    ):
        return False

    outdegree = {
        node: sum(parent == node for parent, _ in forest.directed)
        for node in forest.nodes
    }
    if any(outdegree[root] != 0 for root in forest.roots):
        return False
    if any(outdegree[node] != 1 for node in forest.nodes - forest.roots):
        return False
    if frozenset(node for node, degree in outdegree.items() if degree == 0) != (
        forest.roots
    ):
        return False
    if not _has_topological_permutation(tuple(sorted(forest.nodes)), forest.directed):
        return False
    if any(
        not any(
            _directed_path_exists(forest.directed, node, root, forest.nodes)
            for root in forest.roots
        )
        for node in forest.nodes
    ):
        return False
    return _is_bidirected_tree(forest.nodes, forest.bidirected_tree)


def validate_hedge_by_definition(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
    witness: HedgeProof,
) -> bool:
    """Check selected edges, sink direction, nesting, and exact-query conditions."""

    _validate_query(graph, interventions, outcomes)
    inner = witness.inner
    outer = witness.outer
    if not inner.nodes < outer.nodes or inner.roots != outer.roots:
        return False
    expected_roots = frozenset(
        node
        for node in inner.nodes
        if not any(
            parent == node and child in inner.nodes for parent, child in graph.directed
        )
    )
    if inner.roots != expected_roots:
        return False
    if not _is_selected_c_forest(graph, inner):
        return False
    if not _is_selected_c_forest(graph, outer):
        return False
    if inner.directed != frozenset(
        edge
        for edge in outer.directed
        if edge[0] in inner.nodes and edge[1] in inner.nodes
    ):
        return False
    if inner.bidirected_tree != frozenset(
        edge
        for edge in outer.bidirected_tree
        if edge[0] in inner.nodes and edge[1] in inner.nodes
    ):
        return False
    if inner.nodes & interventions:
        return False
    if not (outer.nodes - inner.nodes) & interventions:
        return False

    allowed_ancestry_nodes = frozenset(graph.nodes) - interventions
    return all(
        any(
            _directed_path_exists(
                graph.directed,
                root,
                outcome,
                allowed_ancestry_nodes,
            )
            for outcome in outcomes
        )
        for root in inner.roots
    )


@lru_cache(maxsize=None)
def exact_query_hedges(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
) -> tuple[HedgeProof, ...]:
    """Construct every Proposition-1 district/closure hedge for the exact query."""

    _validate_query(graph, interventions, outcomes)
    allowed_ancestry_nodes = frozenset(graph.nodes) - interventions
    y_star = frozenset(
        candidate
        for candidate in allowed_ancestry_nodes
        if any(
            _directed_path_exists(
                graph.directed,
                candidate,
                outcome,
                allowed_ancestry_nodes,
            )
            for outcome in outcomes
        )
    )

    witnesses: list[HedgeProof] = []
    for district in _path_districts(y_star, graph.bidirected):
        closure = greedy_reachable_closure(graph, district)
        if not validate_reachable_closure_trace(graph, closure):
            raise AssertionError("invalid greedy fixing trace")
        if district == closure.closure:
            continue
        witness = _construct_hedge(
            graph,
            district,
            closure.closure,
            closure.fixing_order,
        )
        if not validate_hedge_by_definition(
            graph,
            interventions=interventions,
            outcomes=outcomes,
            witness=witness,
        ):
            raise AssertionError("district/closure pair is not an exact-query hedge")
        witnesses.append(witness)
    return tuple(witnesses)


def _definition_has_topological_permutation(
    nodes: NodeSet, directed: frozenset[DirectedEdge]
) -> bool:
    """A second literal acyclicity check used only for Definition 5."""

    for candidate_order in itertools.permutations(sorted(nodes)):
        ranks = {node: rank for rank, node in enumerate(candidate_order)}
        if all(ranks[parent] < ranks[child] for parent, child in directed):
            return True
    return False


def _definition_directed_path_exists(
    directed: frozenset[DirectedEdge],
    source: int,
    target: int,
    allowed: NodeSet,
) -> bool:
    """Enumerate every vertex-simple directed path as a vertex permutation."""

    if source not in allowed or target not in allowed:
        return False
    if source == target:
        return True
    possible_internal = tuple(sorted(allowed - {source, target}))
    for internal_count in range(len(possible_internal) + 1):
        for internal in itertools.permutations(possible_internal, internal_count):
            path = (source, *internal, target)
            if all(
                (path[index], path[index + 1]) in directed
                for index in range(len(path) - 1)
            ):
                return True
    return False


def _definition_bidirected_tree(
    nodes: NodeSet, edges: frozenset[BidirectedEdge]
) -> bool:
    """Check a tree by its edge count and every nontrivial vertex cut."""

    if not nodes or len(edges) != len(nodes) - 1:
        return False
    if any(left not in nodes or right not in nodes for left, right in edges):
        return False
    ordered_nodes = tuple(sorted(nodes))
    for side_size in range(1, len(ordered_nodes)):
        for side_tuple in itertools.combinations(ordered_nodes, side_size):
            side = frozenset(side_tuple)
            if not any((left in side) != (right in side) for left, right in edges):
                return False
    return True


def _definition_is_selected_c_forest(
    graph: TinyADMG, forest: LiteralCForestProof
) -> bool:
    """Apply Shpitser--Pearl (2006), Definition 5 to a selected subgraph."""

    if not forest.nodes or not forest.roots or not forest.roots <= forest.nodes:
        return False
    if any(edge not in graph.directed for edge in forest.directed):
        return False
    if any(edge not in graph.bidirected for edge in forest.bidirected_tree):
        return False
    if any(
        parent not in forest.nodes or child not in forest.nodes
        for parent, child in forest.directed
    ):
        return False
    if any(
        left not in forest.nodes or right not in forest.nodes
        for left, right in forest.bidirected_tree
    ):
        return False

    child_counts = {
        node: sum(parent == node for parent, _ in forest.directed)
        for node in forest.nodes
    }
    if any(child_counts[root] != 0 for root in forest.roots):
        return False
    if any(child_counts[node] != 1 for node in forest.nodes - forest.roots):
        return False
    if (
        frozenset(
            node for node, child_count in child_counts.items() if child_count == 0
        )
        != forest.roots
    ):
        return False
    if not _definition_has_topological_permutation(forest.nodes, forest.directed):
        return False
    if any(
        not any(
            _definition_directed_path_exists(
                forest.directed,
                node,
                root,
                forest.nodes,
            )
            for root in forest.roots
        )
        for node in forest.nodes
    ):
        return False
    return _definition_bidirected_tree(forest.nodes, forest.bidirected_tree)


@lru_cache(maxsize=None)
def _literal_c_forests(
    graph: TinyADMG, nodes: NodeSet, roots: NodeSet
) -> tuple[LiteralCForestProof, ...]:
    """Enumerate every selected directed forest and bidirected spanning tree."""

    internal_directed = tuple(
        edge for edge in sorted(graph.directed) if edge[0] in nodes and edge[1] in nodes
    )
    internal_bidirected = tuple(
        edge
        for edge in sorted(graph.bidirected)
        if edge[0] in nodes and edge[1] in nodes
    )
    forests: list[LiteralCForestProof] = []
    for directed_mask in range(1 << len(internal_directed)):
        directed = frozenset(
            edge
            for index, edge in enumerate(internal_directed)
            if directed_mask & (1 << index)
        )
        for tree_tuple in itertools.combinations(internal_bidirected, len(nodes) - 1):
            candidate = LiteralCForestProof(
                nodes=nodes,
                roots=roots,
                directed=directed,
                bidirected_tree=frozenset(tree_tuple),
            )
            if _definition_is_selected_c_forest(graph, candidate):
                forests.append(candidate)
    return tuple(forests)


@lru_cache(maxsize=None)
def all_literal_hedge_subgraphs(
    graph: TinyADMG,
) -> tuple[LiteralHedgeProof, ...]:
    """Enumerate all edge-minimal nested C-forest pairs in ``graph``.

    Definition 5 only requires bidirected connectivity.  Restricting selected
    bidirected edges to spanning trees is complete for existence because redundant
    bidirected edges can always be deleted without changing roots or connectivity.
    The returned tuple therefore enumerates every edge-minimal representative, not
    every nonminimal supergraph of such a representative.
    """

    ordered_nodes = tuple(graph.nodes)
    witnesses: list[LiteralHedgeProof] = []
    for inner_size in range(1, len(ordered_nodes)):
        for inner_tuple in itertools.combinations(ordered_nodes, inner_size):
            inner_nodes = frozenset(inner_tuple)
            for outer_size in range(inner_size + 1, len(ordered_nodes) + 1):
                for outer_tuple in itertools.combinations(ordered_nodes, outer_size):
                    outer_nodes = frozenset(outer_tuple)
                    if not inner_nodes < outer_nodes:
                        continue
                    for root_size in range(1, inner_size + 1):
                        for root_tuple in itertools.combinations(
                            inner_tuple, root_size
                        ):
                            roots = frozenset(root_tuple)
                            for inner in _literal_c_forests(graph, inner_nodes, roots):
                                for outer in _literal_c_forests(
                                    graph, outer_nodes, roots
                                ):
                                    if not inner.directed <= outer.directed:
                                        continue
                                    if not (
                                        inner.bidirected_tree <= outer.bidirected_tree
                                    ):
                                        continue
                                    witnesses.append(LiteralHedgeProof(inner, outer))
    return tuple(witnesses)


def validate_literal_hedge_by_definition(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
    witness: LiteralHedgeProof,
) -> bool:
    """Validate against Shpitser--Pearl (2006), Definitions 5--6."""

    _validate_query(graph, interventions, outcomes)
    inner = witness.inner
    outer = witness.outer
    if not inner.nodes < outer.nodes or inner.roots != outer.roots:
        return False
    if not _definition_is_selected_c_forest(graph, inner):
        return False
    if not _definition_is_selected_c_forest(graph, outer):
        return False
    if not inner.directed <= outer.directed:
        return False
    if not inner.bidirected_tree <= outer.bidirected_tree:
        return False
    if inner.nodes & interventions:
        return False
    if not (outer.nodes - inner.nodes) & interventions:
        return False

    allowed_ancestry_nodes = frozenset(graph.nodes) - interventions
    return all(
        any(
            _definition_directed_path_exists(
                graph.directed,
                root,
                outcome,
                allowed_ancestry_nodes,
            )
            for outcome in outcomes
        )
        for root in inner.roots
    )


@lru_cache(maxsize=None)
def literal_exact_query_hedges(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
) -> tuple[LiteralHedgeProof, ...]:
    """Filter the exhaustive Definition-6 forest pairs for the exact full query."""

    _validate_query(graph, interventions, outcomes)
    return tuple(
        witness
        for witness in all_literal_hedge_subgraphs(graph)
        if validate_literal_hedge_by_definition(
            graph,
            interventions=interventions,
            outcomes=outcomes,
            witness=witness,
        )
    )


def obsolete_subset_literal_hedge_criterion(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
) -> bool:
    """Quantify the obsolete criterion over literal exact-query hedges."""

    _validate_query(graph, interventions, outcomes)
    intervention_order = tuple(sorted(interventions))
    outcome_order = tuple(sorted(outcomes))
    for intervention_count in range(len(intervention_order) + 1):
        for intervention_subset_tuple in itertools.combinations(
            intervention_order, intervention_count
        ):
            intervention_subset = frozenset(intervention_subset_tuple)
            for outcome_count in range(1, len(outcome_order) + 1):
                for outcome_subset_tuple in itertools.combinations(
                    outcome_order, outcome_count
                ):
                    if literal_exact_query_hedges(
                        graph,
                        interventions=intervention_subset,
                        outcomes=frozenset(outcome_subset_tuple),
                    ):
                        return True
    return False


def obsolete_subset_hedge_criterion(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
) -> bool:
    """Backward-compatible name for the literal obsolete-criterion oracle."""

    return obsolete_subset_literal_hedge_criterion(
        graph,
        interventions=interventions,
        outcomes=outcomes,
    )


def _validate_conditional_query(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
    conditions: NodeSet,
) -> None:
    _validate_query(graph, interventions, outcomes)
    if not conditions <= frozenset(graph.nodes):
        raise ValueError("conditioning vertices must belong to the graph")
    if interventions & conditions or outcomes & conditions:
        raise ValueError("interventions, outcomes, and conditions must be disjoint")


def _backdoor_incident_atoms(
    graph: TinyADMG,
    vertex: int,
    actions: NodeSet,
) -> tuple[tuple[int, bool, bool], ...]:
    """Return original-graph edge atoms surviving incoming-action deletion.

    Each tuple is ``(neighbor, arrow_at_vertex, arrow_at_neighbor)``.  Parallel
    directed and bidirected atoms are intentionally retained when a bow is present.
    """

    atoms: list[tuple[int, bool, bool]] = []
    for parent, child in sorted(graph.directed):
        if child in actions:
            continue
        if parent == vertex:
            atoms.append((child, False, True))
        elif child == vertex:
            atoms.append((parent, True, False))
    for left, right in sorted(graph.bidirected):
        if left in actions or right in actions:
            continue
        if left == vertex:
            atoms.append((right, True, True))
        elif right == vertex:
            atoms.append((left, True, True))
    return tuple(atoms)


def _backdoor_collider_reaches_evidence_by_direct_paths(
    graph: TinyADMG,
    *,
    collider: int,
    evidence: NodeSet,
    actions: NodeSet,
    candidate: int,
) -> bool:
    """Check reflexive collider ancestry by literal simple directed paths."""

    if collider in evidence:
        return True
    available = tuple(
        node for node in graph.nodes if node not in {collider} and node not in evidence
    )
    surviving = frozenset(
        (parent, child)
        for parent, child in graph.directed
        if child not in actions and parent != candidate
    )
    for target in sorted(evidence):
        possible_internal = tuple(node for node in available if node != target)
        for internal_count in range(len(possible_internal) + 1):
            for internal in itertools.permutations(possible_internal, internal_count):
                path = (collider, *internal, target)
                if all(
                    (path[index], path[index + 1]) in surviving
                    for index in range(len(path) - 1)
                ):
                    return True
    return False


def literal_rule_two_eligible_by_backdoor_paths(
    graph: TinyADMG,
    *,
    outcomes: NodeSet,
    actions: NodeSet,
    conditions: NodeSet,
    candidate: int,
) -> bool:
    """Apply Rule 2 using the paper's literal back-door-path restatement.

    A path is explored in the original ADMG only when its first edge has an
    arrowhead at ``candidate``.  Incoming edges at current actions are filtered as
    the path is traversed.  Collider ancestry is checked by a separate exhaustive
    directed-path definition after removing incoming-action and outgoing-candidate
    arrows.
    """

    if candidate not in conditions:
        raise ValueError("Rule-2 candidate must be a current condition")
    if not outcomes or actions & conditions or outcomes & (actions | conditions):
        raise ValueError("Rule-2 sets must be disjoint with nonempty outcomes")
    evidence = actions | (conditions - {candidate})

    def active_extension_exists(
        current: int,
        incoming_arrow: bool,
        visited: frozenset[int],
    ) -> bool:
        for neighbor, outgoing_arrow, neighbor_arrow in _backdoor_incident_atoms(
            graph, current, actions
        ):
            if neighbor in visited:
                continue
            collider = incoming_arrow and outgoing_arrow
            if collider:
                if not _backdoor_collider_reaches_evidence_by_direct_paths(
                    graph,
                    collider=current,
                    evidence=evidence,
                    actions=actions,
                    candidate=candidate,
                ):
                    continue
            elif current in evidence:
                continue
            if neighbor in outcomes:
                return True
            if active_extension_exists(
                neighbor,
                neighbor_arrow,
                visited | {neighbor},
            ):
                return True
        return False

    for neighbor, candidate_arrow, neighbor_arrow in _backdoor_incident_atoms(
        graph, candidate, actions
    ):
        if not candidate_arrow:
            continue
        if neighbor in outcomes:
            return False
        if active_extension_exists(
            neighbor,
            neighbor_arrow,
            frozenset({candidate, neighbor}),
        ):
            return False
    return True


def _mutilated_pair_marks(
    graph: TinyADMG,
    left: int,
    right: int,
    *,
    actions: NodeSet,
    candidate: int,
) -> tuple[tuple[bool, bool], ...]:
    """Return edge-end marks in ``G[bar(actions), underline(candidate)]``."""

    marks: list[tuple[bool, bool]] = []
    if right not in actions and left != candidate and (left, right) in graph.directed:
        marks.append((False, True))
    if left not in actions and right != candidate and (right, left) in graph.directed:
        marks.append((True, False))
    canonical = (left, right) if left < right else (right, left)
    if left not in actions and right not in actions and canonical in graph.bidirected:
        marks.append((True, True))
    return tuple(marks)


def _mutilated_collider_ancestors(
    graph: TinyADMG,
    *,
    evidence: NodeSet,
    actions: NodeSet,
    candidate: int,
) -> NodeSet:
    """Compute reflexive evidence ancestry by an iterative reverse scan."""

    ancestors = set(evidence)
    changed = True
    while changed:
        changed = False
        for parent, child in graph.directed:
            if child in actions or parent == candidate:
                continue
            if child in ancestors and parent not in ancestors:
                ancestors.add(parent)
                changed = True
    return frozenset(ancestors)


def literal_rule_two_eligible_by_mutilated_paths(
    graph: TinyADMG,
    *,
    outcomes: NodeSet,
    actions: NodeSet,
    conditions: NodeSet,
    candidate: int,
) -> bool:
    """Independently apply Rule 2 by all paths in the explicitly mutilated ADMG."""

    if candidate not in conditions:
        raise ValueError("Rule-2 candidate must be a current condition")
    evidence = actions | (conditions - {candidate})
    collider_ancestors = _mutilated_collider_ancestors(
        graph,
        evidence=evidence,
        actions=actions,
        candidate=candidate,
    )
    possible_internal = tuple(
        node for node in graph.nodes if node != candidate and node not in outcomes
    )
    for outcome in sorted(outcomes):
        internal_pool = tuple(node for node in possible_internal if node != outcome)
        for internal_count in range(len(internal_pool) + 1):
            for internal in itertools.permutations(internal_pool, internal_count):
                path = (candidate, *internal, outcome)
                edge_choices: list[tuple[tuple[bool, bool], ...]] = []
                for left, right in zip(path, path[1:]):
                    marks = _mutilated_pair_marks(
                        graph,
                        left,
                        right,
                        actions=actions,
                        candidate=candidate,
                    )
                    if not marks:
                        break
                    edge_choices.append(marks)
                else:
                    for selected_marks in itertools.product(*edge_choices):
                        active = True
                        for index, vertex in enumerate(path[1:-1], start=1):
                            incoming_arrow = selected_marks[index - 1][1]
                            outgoing_arrow = selected_marks[index][0]
                            if incoming_arrow and outgoing_arrow:
                                if vertex not in collider_ancestors:
                                    active = False
                                    break
                            elif vertex in evidence:
                                active = False
                                break
                        if active:
                            return False
    return True


def _literal_rule_two_eligible(
    graph: TinyADMG,
    *,
    outcomes: NodeSet,
    actions: NodeSet,
    conditions: NodeSet,
    candidate: int,
) -> bool:
    backdoor = literal_rule_two_eligible_by_backdoor_paths(
        graph,
        outcomes=outcomes,
        actions=actions,
        conditions=conditions,
        candidate=candidate,
    )
    mutilated = literal_rule_two_eligible_by_mutilated_paths(
        graph,
        outcomes=outcomes,
        actions=actions,
        conditions=conditions,
        candidate=candidate,
    )
    if backdoor != mutilated:
        raise AssertionError("independent literal Rule-2 routes disagree")
    return backdoor


def _finish_literal_idc_reduction(
    graph: TinyADMG,
    *,
    outcomes: NodeSet,
    original_interventions: NodeSet,
    original_conditions: NodeSet,
    moved: tuple[int, ...],
    interventions: NodeSet,
    residual_conditions: NodeSet,
) -> LiteralIDCReduction:
    joint_outcomes = outcomes | residual_conditions
    id_failed = literal_id_fails(
        graph,
        interventions=interventions,
        outcomes=joint_outcomes,
    )
    return LiteralIDCReduction(
        outcomes=outcomes,
        original_interventions=original_interventions,
        original_conditions=original_conditions,
        rule_two_interventions=moved,
        interventions=interventions,
        residual_conditions=residual_conditions,
        joint_outcomes=joint_outcomes,
        id_failed=id_failed,
    )


def literal_idc_reduction(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
    conditions: NodeSet,
) -> LiteralIDCReduction:
    """Run UAI-06 Figure 3 to a fixed point, choosing the first eligible vertex."""

    _validate_conditional_query(
        graph,
        interventions=interventions,
        outcomes=outcomes,
        conditions=conditions,
    )
    current_interventions = interventions
    current_conditions = conditions
    moved: list[int] = []
    while True:
        candidate = next(
            (
                vertex
                for vertex in graph.nodes
                if vertex in current_conditions
                and _literal_rule_two_eligible(
                    graph,
                    outcomes=outcomes,
                    actions=current_interventions,
                    conditions=current_conditions,
                    candidate=vertex,
                )
            ),
            None,
        )
        if candidate is None:
            break
        moved.append(candidate)
        current_interventions = current_interventions | {candidate}
        current_conditions = current_conditions - {candidate}
    return _finish_literal_idc_reduction(
        graph,
        outcomes=outcomes,
        original_interventions=interventions,
        original_conditions=conditions,
        moved=tuple(moved),
        interventions=current_interventions,
        residual_conditions=current_conditions,
    )


def literal_idc_one_pass_reduction(
    graph: TinyADMG,
    *,
    interventions: NodeSet,
    outcomes: NodeSet,
    conditions: NodeSet,
) -> LiteralIDCReduction:
    """Move the singleton Rule-2 set computed from the unchanged original split."""

    _validate_conditional_query(
        graph,
        interventions=interventions,
        outcomes=outcomes,
        conditions=conditions,
    )
    moved = tuple(
        candidate
        for candidate in graph.nodes
        if candidate in conditions
        and _literal_rule_two_eligible(
            graph,
            outcomes=outcomes,
            actions=interventions,
            conditions=conditions,
            candidate=candidate,
        )
    )
    moved_set = frozenset(moved)
    return _finish_literal_idc_reduction(
        graph,
        outcomes=outcomes,
        original_interventions=interventions,
        original_conditions=conditions,
        moved=moved,
        interventions=interventions | moved_set,
        residual_conditions=conditions - moved_set,
    )


def obsolete_idc_corollary_two_flag(
    graph: TinyADMG,
    reduction: LiteralIDCReduction,
) -> bool:
    """Evaluate the obsolete subset-hedge quantifier in UAI-06 Corollary 2."""

    return obsolete_subset_literal_hedge_criterion(
        graph,
        interventions=reduction.interventions,
        outcomes=reduction.joint_outcomes,
    )
