"""Exact finite-SCM semantics and literal R-115 constructions for the calculus regressions.

Every model has one independent private root per vertex and, by default, one
independent finite latent per bidirected edge. A caller may instead list latent
groups: each group is a bidirected clique (for example a triangle) fed by one
shared latent, which is the documented clique-root model class. A private-root
value is an explicit table from (parent states, values of the vertex's latents)
to the vertex state, so every copy of a vertex in every world reads the same
table and the same latents. Weights are exact integers or Fractions and need
not be normalized: the cross-product independence test is homogeneous.

``definition_2_4`` and ``algorithm_1`` construct R-115's counterfactual
ancestors and ancestral multi-world network literally, seed by seed, with their
own path search. Only counterfactual data types are imported; ancestry,
exclusion, AMWN construction and separation are never taken from the library
under test.
"""
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
import random

from causal_graph_algorithms.counterfactual import CounterfactualTerm, ValueLiteral


def _topological_order(nodes, directed):
    waiting = {v: {p for p, c in directed if c == v} for v in nodes}
    order = []
    while waiting:
        ready = [v for v in nodes if v in waiting and not waiting[v]]
        if not ready:
            raise ValueError("directed cycle")
        for v in ready:
            order.append(v)
            del waiting[v]
        for parents in waiting.values():
            parents.difference_update(ready)
    return tuple(order)


def clamps(term):
    """The hard intervention of a term; nested assignments are outside this oracle."""
    assert isinstance(term, CounterfactualTerm)
    result = {}
    for assignment in term.interventions:
        assert isinstance(assignment.value, ValueLiteral), "the oracle evaluates hard terms only"
        result[assignment.variable] = assignment.value.state
    return result


class FiniteSCM:
    """Explicit units (latent values, private tables) with exact positive weights.

    ``bidirected`` lists the latent groups: vertex tuples that each read one
    independent latent, whose weights are ``latents[group]``.
    """

    def __init__(self, nodes, directed, bidirected, domains, latents, roots):
        self.nodes = tuple(nodes)
        arcs = {tuple(edge) for edge in directed}
        self.edges = tuple(tuple(edge) for edge in bidirected)
        self.order = _topological_order(self.nodes, arcs)
        self.parents = {v: tuple(p for p in self.nodes if (p, v) in arcs) for v in self.nodes}
        self.incident = {v: tuple(i for i, edge in enumerate(self.edges) if v in edge)
                         for v in self.nodes}
        self.domains = {v: tuple(domains[v]) for v in self.nodes}
        self.units = []
        for latent in product(*(range(len(latents[edge])) for edge in self.edges)):
            base = 1
            for edge, value in zip(self.edges, latent):
                base *= latents[edge][value]
            for choice in product(*(roots[v] for v in self.order)):
                weight = base
                for _, root_weight in choice:
                    weight *= root_weight
                assert weight > 0
                tables = {v: table for v, (table, _) in zip(self.order, choice)}
                self.units.append((weight, latent, tables))
        self.total = sum(weight for weight, _, _ in self.units)

    def world(self, latent, tables, clamp):
        values = {}
        for v in self.order:
            if v in clamp:
                values[v] = clamp[v]
            else:
                key = (tuple(values[p] for p in self.parents[v])
                       + tuple(latent[i] for i in self.incident[v]))
                values[v] = tables[v][key]
        return values

    def joint(self, terms):
        """Unnormalized exact law of a list of hard terms, keyed by state tuples."""
        settings = [clamps(term) for term in terms]
        keys = [tuple(sorted(setting.items())) for setting in settings]
        law = defaultdict(int)
        for weight, latent, tables in self.units:
            worlds = {}
            states = []
            for term, key, setting in zip(terms, keys, settings):
                if key not in worlds:
                    worlds[key] = self.world(latent, tables, setting)
                states.append(worlds[key][term.variable])
            law[tuple(states)] += weight
        return dict(law)

    def probability(self, atoms):
        """Exact probability of a conjunction of (term, state) pairs."""
        terms = [term for term, _ in atoms]
        wanted = tuple(state for _, state in atoms)
        hits = sum((w for states, w in self.joint(terms).items() if states == wanted), 0)
        return Fraction(hits) / Fraction(self.total)

    def kernel(self, clamp):
        """Unnormalized law of the unclamped vertices, in declared order, under do(clamp)."""
        free = tuple(v for v in self.nodes if v not in clamp)
        law = defaultdict(int)
        for weight, latent, tables in self.units:
            values = self.world(latent, tables, clamp)
            law[tuple(values[v] for v in free)] += weight
        return dict(law)


def positive_kernels(model, intervened=None):
    """Whether P_s(V \\ S) is strictly positive for every listed S and every value s.

    ``intervened`` lists vertex sets S; the default is every subset, so the
    observational law (S empty) and every hard-intervention kernel are checked.
    A kernel is positive when every state combination of V \\ S has mass.
    """
    sets = (intervened if intervened is not None else
            [subset for size in range(len(model.nodes) + 1) for subset in combinations(model.nodes, size)])
    for subset in sets:
        free = [v for v in model.nodes if v not in subset]
        cells = 1
        for v in free:
            cells *= len(model.domains[v])
        for values in product(*(model.domains[v] for v in subset)):
            if len(model.kernel(dict(zip(subset, values)))) != cells:
                return False
    return True


def independent(law, left, right, given):
    """Exact (left _||_ right | given) for index tuples into the keys of ``law``."""
    def marginal(positions):
        result = defaultdict(int)
        for states, weight in law.items():
            result[tuple(states[i] for i in positions)] += weight
        return result

    joint = marginal(left + right + given)
    with_left, with_right, alone = marginal(left + given), marginal(right + given), marginal(given)
    split = len(given)
    lefts, rights = defaultdict(list), defaultdict(list)
    for key, weight in with_left.items():
        lefts[key[len(key) - split:]].append((key[:len(key) - split], weight))
    for key, weight in with_right.items():
        rights[key[len(key) - split:]].append((key[:len(key) - split], weight))
    for context, total in alone.items():
        for a, weight_a in lefts[context]:
            for b, weight_b in rights[context]:
                if joint.get(a + b + context, 0) * total != weight_a * weight_b:
                    return False
    return True


def term_independent(model, left, right, given=()):
    """Exact CI between lists of hard terms in one finite model."""
    terms = [*left, *right, *given]
    law = model.joint(terms)
    a, b = len(left), len(left) + len(right)
    return independent(law, tuple(range(a)), tuple(range(a, b)), tuple(range(b, len(terms))))


def generic_scm(nodes, directed, bidirected, domains, *, seed, latent_states=2, max_tables=16,
                latent_groups=None):
    """Seeded model with positive integer weights in 1..1000.

    ``latent_groups`` lists the vertex groups that share one independent latent
    each; the default is one group per bidirected edge. Every group must be a
    bidirected clique of the graph, such as a triangle fed by a single root, so
    the model stays in the class whose roots' children form bidirected cliques.
    A vertex's private root ranges over all tables from its parent states and
    latent values when there are at most ``max_tables`` of them (an integer, or
    a per-vertex mapping); otherwise it ranges over that many distinct seeded
    tables. Either way the model belongs to the independent-root class, in
    which an allowed edge may be inactive.
    """
    rng = random.Random(seed)
    nodes = tuple(nodes)
    arcs = {tuple(edge) for edge in directed}
    pairs = {frozenset(edge) for edge in bidirected}
    edges = tuple(tuple(group) for group in (bidirected if latent_groups is None else latent_groups))
    for group in edges:
        assert len(set(group)) == len(group) >= 2, group
        assert all(frozenset(pair) in pairs for pair in combinations(group, 2)), f"{group} is not a bidirected clique"
    latents = {edge: tuple(rng.randint(1, 1000) for _ in range(latent_states)) for edge in edges}
    roots = {}
    for v in nodes:
        budget = max_tables[v] if isinstance(max_tables, dict) else max_tables
        parents = tuple(p for p in nodes if (p, v) in arcs)
        inputs = tuple(product(*(tuple(domains[p]) for p in parents),
                               *(range(latent_states) for edge in edges if v in edge)))
        outputs = tuple(domains[v])
        if len(outputs) ** len(inputs) <= budget:
            rows = list(product(outputs, repeat=len(inputs)))
        else:
            chosen = set()
            while len(chosen) < budget:
                chosen.add(tuple(rng.choice(outputs) for _ in inputs))
            rows = sorted(chosen)
        roots[v] = tuple((dict(zip(inputs, row)), rng.randint(1, 1000)) for row in rows)
    return FiniteSCM(nodes, directed, edges, domains, latents, roots)


# ------------------------------------------- literal Definition 2.4 / Algorithm 1


def _directed_path(directed, start, end, cut):
    """Whether G with the edges into ``cut`` removed has a directed path from start to end.

    The path may have length zero, so every vertex is its own ancestor.
    """
    kept = [(a, b) for a, b in directed if b not in cut]
    seen, frontier = {start}, [start]
    while frontier:
        vertex = frontier.pop()
        if vertex == end:
            return True
        for a, b in kept:
            if a == vertex and b not in seen:
                seen.add(b)
                frontier.append(b)
    return False


def definition_2_4(nodes, directed, term):
    """R-115 Definition 2.4 (TR p.5): the counterfactual ancestors An(Y_x) of a hard term.

    Every W_z with W in An(Y) minus X and z = x n An(W), both taken in G with
    the edges into X removed; W = Y is included. Terms are returned as
    (variable, frozenset of (clamped variable, state)) pairs.
    """
    y, x = term.variable, clamps(term)
    found = set()
    for w in nodes:
        if w not in x and _directed_path(directed, w, y, x):
            found.add((w, frozenset((v, s) for v, s in x.items() if _directed_path(directed, v, w, x))))
    return found


def algorithm_1(nodes, directed, bidirected, terms):
    """R-115 Algorithm 1 (TR p.6) for hard terms, built seed by seed.

    Line 1 adds An(Y*) with the arrows witnessing ancestrality: for each seed,
    every edge P -> W of G between two of its ancestors' variables joins their
    copies. Lines 2-4 add a root for each variable with two or more copies, and
    lines 5-7 a root for each bidirected edge whose endpoints both have copies;
    a root feeds every copy of the variables it serves. Returns the term nodes,
    the arrows and a map from each root's served variables to its children.
    """
    found, arrows = set(), set()
    for term in terms:
        copy_of = {w: (w, z) for w, z in definition_2_4(nodes, directed, term)}
        found.update(copy_of.values())
        arrows.update((copy_of[p], copy_of[c]) for p, c in directed if p in copy_of and c in copy_of)
    copies = defaultdict(set)
    for node in found:
        copies[node[0]].add(node)
    roots = {frozenset((v,)): frozenset(c) for v, c in copies.items() if len(c) > 1}
    for a, b in bidirected:
        if a in copies and b in copies:
            roots[frozenset((a, b))] = frozenset(copies[a] | copies[b])
    return frozenset(found), frozenset(arrows), roots
