"""Counterfactual d-separation and the side conditions of the ctf-calculus.

Hard-intervention implementation of Correa and Bareinboim's exclusion operator
(TR R-115 Lemma 2.3), counterfactual ancestors (Definition 2.4), ancestral
multi-world network (Algorithm 1) and the side conditions of the three rules of
Theorem 3.1. A d-separation claim is sound for SCMs whose exogenous roots are
mutually independent and feed bidirected cliques; ``False`` claims nothing. The
printed soundness proof of Theorem 2.5 (TR p.13) is not relied on, and
completeness is not claimed: the criterion is not complete on fixed binary
domains, and completeness when domains may be enlarged is open. See
docs/reference/COUNTERFACTUAL_CALCULUS.md for the replacement argument.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
import json
from typing import TypeAlias, final

from ._immutable import _ImmutableValue
from .counterfactual import CounterfactualDomainError, CounterfactualEvent, CounterfactualTerm, ValueLiteral
from .directed_bidirected import AcyclicDirectedBidirectedGraph, DAG, DirectedBidirectedGraph
from .separation import d_separated_sets


CalculusGraph: TypeAlias = DAG | AcyclicDirectedBidirectedGraph
_Atom: TypeAlias = tuple[str, frozenset[tuple[str, str]], str]
_TOKEN = object()
_NESTED = ("nested counterfactual terms are outside this hard-intervention calculus; "
           "unnest them first (R-115 Corollary 2.2; CLB21 Theorem 4)")


class CounterfactualCalculusDomainError(CounterfactualDomainError):
    """A calculus input is nested, self-clamped, unknown to the graph, or malformed."""


def _clamps(term: CounterfactualTerm) -> dict[str, str]:
    result: dict[str, str] = {}
    for assignment in term.interventions:
        if not isinstance(assignment.value, ValueLiteral):
            raise CounterfactualCalculusDomainError(_NESTED)
        result[assignment.variable] = assignment.value.state
    return result


def _hard_term(value: object, vertices: frozenset[str] | None) -> CounterfactualTerm:
    if not isinstance(value, CounterfactualTerm):
        raise CounterfactualCalculusDomainError("expected CounterfactualTerm values")
    clamp = _clamps(value)
    if vertices is not None:
        unknown = [v for v in (value.variable, *clamp) if v not in vertices]
        if unknown:
            raise CounterfactualCalculusDomainError(f"unknown vertex in a counterfactual term: {unknown[0]!r}")
    if value.variable in clamp:
        raise CounterfactualCalculusDomainError(
            f"a term that clamps its own variable {value.variable!r} is the constant clamp value; "
            "remove or evaluate it before applying the calculus")
    return value


def _label(term: CounterfactualTerm) -> str:
    return json.dumps([term.variable, _clamps(term)], ensure_ascii=False, separators=(",", ":"))


def _root_label(variables: tuple[str, ...]) -> str:
    return json.dumps({"U": list(variables)}, ensure_ascii=False, separators=(",", ":"))


class _Diagram:
    """Parent lists and declared order of a validated known graph."""

    def __init__(self, graph: object) -> None:
        if not isinstance(graph, (DAG, AcyclicDirectedBidirectedGraph)):
            raise CounterfactualCalculusDomainError(
                "the calculus requires a known DAG or acyclic directed-bidirected graph")
        topology = graph.topology
        self.nodes = topology.nodes
        self.vertices = frozenset(topology.nodes)
        self.position = {v: i for i, v in enumerate(topology.nodes)}
        self.parents = {v: tuple(p for p in topology.nodes if (p, v) in topology.directed)
                        for v in topology.nodes}
        self.bidirected = tuple(sorted(topology.bidirected,
                                       key=lambda edge: (self.position[edge[0]], self.position[edge[1]])))

    def term(self, value: object) -> CounterfactualTerm:
        return _hard_term(value, self.vertices)

    def terms(self, values: object, role: str, *, nonempty: bool) -> tuple[CounterfactualTerm, ...]:
        if isinstance(values, CounterfactualTerm):
            raw: tuple[object, ...] = (values,)
        elif isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise CounterfactualCalculusDomainError(f"{role} must be a term or an iterable of terms")
        else:
            raw = tuple(values)
        checked = tuple(dict.fromkeys(self.term(value) for value in raw))
        if nonempty and not checked:
            raise CounterfactualCalculusDomainError(f"{role} must contain at least one term")
        return checked

    def ancestors(self, variable: str, cut: Iterable[str]) -> frozenset[str]:
        """Reflexive ancestors of ``variable`` with the edges into ``cut`` removed."""
        blocked = frozenset(cut)
        reached = {variable}
        pending = [variable]
        while pending:
            current = pending.pop()
            if current in blocked:
                continue
            for parent in self.parents[current]:
                if parent not in reached:
                    reached.add(parent)
                    pending.append(parent)
        return frozenset(reached)

    def reduce(self, term: CounterfactualTerm) -> CounterfactualTerm:
        """Lemma 2.3: keep the clamps that remain ancestors once edges into them are cut."""
        keep = self.ancestors(term.variable, _clamps(term))
        return CounterfactualTerm(term.variable, tuple(a for a in term.interventions if a.variable in keep))

    def key(self, term: CounterfactualTerm) -> tuple[int, tuple[tuple[int, str], ...]]:
        return self.position[term.variable], tuple(sorted(
            (self.position[v], state) for v, state in _clamps(term).items()))

    def closure(self, seeds: Iterable[CounterfactualTerm]) -> tuple[
            tuple[CounterfactualTerm, ...], tuple[tuple[CounterfactualTerm, CounterfactualTerm], ...]]:
        """Definition 2.4 ancestors of the seeds and the parent arrows of Algorithm 1, line 1.

        For an unclamped parent P of W, the arrow into W_z starts at ||P_z||.
        That is the Definition 2.4 ancestor P_{x n An(P)} of every Y_x whose
        ancestors include W_z, so the closure from the reduced seeds is An(W*).
        """
        found: set[CounterfactualTerm] = set()
        arrows: set[tuple[CounterfactualTerm, CounterfactualTerm]] = set()
        pending = [self.reduce(seed) for seed in seeds]
        while pending:
            node = pending.pop()
            if node in found:
                continue
            found.add(node)
            clamp = _clamps(node)
            for parent in self.parents[node.variable]:
                if parent not in clamp:
                    source = self.reduce(CounterfactualTerm(parent, node.interventions))
                    arrows.add((source, node))
                    pending.append(source)
        return (tuple(sorted(found, key=self.key)),
                tuple(sorted(arrows, key=lambda arrow: (self.key(arrow[0]), self.key(arrow[1])))))


@final
@dataclass(frozen=True, slots=True)
class AncestralMultiWorldNetwork(_ImmutableValue):
    """R-115 Algorithm 1 for hard-intervention terms; created by the factory only.

    ``dag`` is a DAG over canonical JSON labels: ``["Y",{"X":"0"}]`` for the
    reduced term Y_{X=0}, and ``{"U":["Y"]}`` or ``{"U":["X","Y"]}`` for the
    added roots U_Y (two or more copies of Y) and U_XY (a bidirected edge X <-> Y
    with copies of both). ``terms`` maps term labels to the counterfactual
    ancestors and ``exogenous`` maps root labels to the variables they serve.
    Terms equal under the exclusion operator are one node.
    """
    graph: CalculusGraph
    dag: DAG
    terms: tuple[tuple[str, CounterfactualTerm], ...]
    exogenous: tuple[tuple[str, tuple[str, ...]], ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("ancestral multi-world networks are factory-created")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AncestralMultiWorldNetwork cannot be subclassed")

    def node(self, term: CounterfactualTerm) -> str:
        """Return the label of ``||term||``; it must be a node of this network.

        Raises ``CounterfactualCalculusDomainError`` for a term that
        ``counterfactual_exclusion`` rejects on ``graph`` or whose reduction is
        not a counterfactual ancestor here.
        """
        diagram = _Diagram(self.graph)
        label = _label(diagram.reduce(diagram.term(term)))
        if all(existing != label for existing, _ in self.terms):
            raise CounterfactualCalculusDomainError("the term is not a counterfactual ancestor in this network")
        return label


def _network(graph: CalculusGraph, diagram: _Diagram,
             seeds: Iterable[CounterfactualTerm]) -> AncestralMultiWorldNetwork:
    nodes, arrows = diagram.closure(seeds)
    copies: dict[str, list[CounterfactualTerm]] = {}
    for node in nodes:
        copies.setdefault(node.variable, []).append(node)
    roots: list[tuple[tuple[str, ...], list[CounterfactualTerm]]] = [
        ((v,), copies[v]) for v in diagram.nodes if len(copies.get(v, ())) > 1]
    roots += [((a, b), copies[a] + copies[b]) for a, b in diagram.bidirected if a in copies and b in copies]
    labels = [_label(node) for node in nodes]
    directed = [(_label(parent), _label(child)) for parent, child in arrows]
    for variables, children in roots:
        directed.extend((_root_label(variables), _label(child)) for child in children)
    dag = DAG(DirectedBidirectedGraph(nodes=labels + [_root_label(v) for v, _ in roots], directed=directed))
    return AncestralMultiWorldNetwork(graph, dag, tuple(zip(labels, nodes)),
                                      tuple((_root_label(v), v) for v, _ in roots), _token=_TOKEN)


def counterfactual_exclusion(graph: CalculusGraph, term: CounterfactualTerm) -> CounterfactualTerm:
    """Return ``||term||`` of R-115 Lemma 2.3 (TR p.4).

    The result keeps exactly the clamps on ancestors of the response variable
    in ``graph`` with the edges into the clamped set removed. Both terms take
    the same value on every unit of every SCM compatible with ``graph``,
    whatever its exogenous law.

    Raises ``CounterfactualCalculusDomainError`` unless ``graph`` is a known
    ``DAG`` or ``AcyclicDirectedBidirectedGraph`` and ``term`` is a hard
    ``CounterfactualTerm`` on its vertices that does not clamp its own
    variable; nested terms must be unnested first.
    """
    diagram = _Diagram(graph)
    return diagram.reduce(diagram.term(term))


def counterfactual_ancestors_reflexive(graph: CalculusGraph,
                                       terms: CounterfactualTerm | Iterable[CounterfactualTerm]
                                       ) -> tuple[CounterfactualTerm, ...]:
    """Return An(W*) of R-115 Definition 2.4 (TR p.5) as reduced terms.

    The relation is reflexive: for each input term t the result contains
    ``||t||``, since Definition 2.4's An(Y_x) "includes Y itself". Each
    ancestor W_z of Y_x has W an ancestor of Y outside X in the graph with
    edges into X removed, and z the clamps of x on ancestors of W there. The
    result is deduplicated and sorted by declared vertex order, then by the
    clamps as (declared position, state) pairs. Each input term contributes at
    most one ancestor per vertex, so there are at most n * s of them for n
    vertices and s distinct input terms.

    ``terms`` is a term or an iterable of terms; it may be empty. Raises
    ``CounterfactualCalculusDomainError`` unless ``graph`` is a known ``DAG``
    or ``AcyclicDirectedBidirectedGraph`` and every term is a hard
    ``CounterfactualTerm`` on its vertices that does not clamp its own variable.
    """
    diagram = _Diagram(graph)
    return diagram.closure(diagram.terms(terms, "terms", nonempty=False))[0]


def ancestral_multi_world_network(graph: CalculusGraph,
                                  terms: CounterfactualTerm | Iterable[CounterfactualTerm]
                                  ) -> AncestralMultiWorldNetwork:
    """Build G_A(G, W*) by R-115 Algorithm 1 (TR p.6) for hard-intervention terms.

    Nodes are the counterfactual ancestors of ``terms``, in the order of
    ``counterfactual_ancestors_reflexive``, with their parent arrows; a root
    U_V feeds every copy of a variable with two or more copies, and a root
    U_VW feeds every copy of V and of W for each bidirected edge V <-> W whose
    endpoints both have copies. Root nodes follow the term nodes: U_V in
    declared vertex order, then U_VW ordered by the declared positions of V
    and W.

    Resources: by Definition 2.4 each input term contributes at most one node
    per vertex, so for n vertices, b bidirected edges and s distinct input
    terms there are at most n * s term nodes and n + b roots, and each term
    node costs one graph search per parent. The work is polynomial, so no
    resource budget is needed.

    Accepts and rejects inputs exactly as ``counterfactual_ancestors_reflexive``
    does, raising ``CounterfactualCalculusDomainError``.
    """
    diagram = _Diagram(graph)
    return _network(graph, diagram, diagram.terms(terms, "terms", nonempty=False))


def counterfactual_d_separated(graph: CalculusGraph,
                               left: CounterfactualTerm | Iterable[CounterfactualTerm],
                               right: CounterfactualTerm | Iterable[CounterfactualTerm],
                               given: CounterfactualTerm | Iterable[CounterfactualTerm] = ()) -> bool:
    """Decide counterfactual d-separation, the side condition of R-115 Rule 2.

    ``True`` means ``||left||`` and ``||right||`` are d-separated by
    ``||given||`` in the network of all three term sets (Theorem 2.5, TR p.6;
    Rule 2, TR p.7), so that left is independent of right given ``given`` in
    every SCM compatible with ``graph`` whose exogenous roots are mutually
    independent and feed bidirected cliques. Rule 2 then licenses
    P(left | right, given) = P(left | given) wherever defined. Soundness rests
    on the replacement argument in docs/reference/COUNTERFACTUAL_CALCULUS.md
    (section "Soundness basis"); R-115's printed proof (TR p.13) uses the false
    Lemma B.5 and is not relied on. ``False`` claims neither dependence nor
    anything about identification. The criterion is not complete on fixed
    binary domains, and whether it is complete when domains may be enlarged is
    open.

    Roles take a term or an iterable of hard terms on graph vertices; left and
    right are nonempty and the three roles syntactically disjoint. Terms that
    coincide after exclusion are one random variable: a side reduced into
    ``given`` is trivially independent, and an unconditioned term shared by
    both sides returns ``False``. Raises ``CounterfactualCalculusDomainError``
    for a graph or term that ``counterfactual_exclusion`` rejects, a role that
    is not a term or an iterable of terms, an empty left or right role, and
    overlapping roles.
    """
    diagram = _Diagram(graph)
    a = diagram.terms(left, "left", nonempty=True)
    b = diagram.terms(right, "right", nonempty=True)
    c = diagram.terms(given, "given", nonempty=False)
    if set(a) & set(b) or set(a) & set(c) or set(b) & set(c):
        raise CounterfactualCalculusDomainError("left, right and given terms must be pairwise disjoint")
    conditioned = {diagram.reduce(t) for t in c}
    sources = {diagram.reduce(t) for t in a} - conditioned
    targets = {diagram.reduce(t) for t in b} - conditioned
    if not sources or not targets:
        return True
    if sources & targets:
        return False
    network = _network(graph, diagram, a + b + c)
    return d_separated_sets(network.dag, {_label(t) for t in sources}, {_label(t) for t in targets},
                            {_label(t) for t in conditioned})


def _event_atoms(value: object, role: str) -> frozenset[_Atom]:
    if not isinstance(value, CounterfactualEvent):
        raise CounterfactualCalculusDomainError(f"{role} must be a CounterfactualEvent")
    atoms: set[_Atom] = set()
    for atom in value.atoms:
        term = _hard_term(atom.term, None)
        atoms.add((term.variable, frozenset(_clamps(term).items()), atom.value.state))
    return frozenset(atoms)


def _one_rule_1_step(source: frozenset[_Atom], target: frozenset[_Atom]) -> bool:
    """Whether one Rule 1 application, moving atoms from world t,x to world t, turns source into target.

    The moved atoms are exactly those of ``source`` missing from ``target``;
    they must share one world t,x, and every atom X_t = x[X] must occur in
    ``source``. Moved atoms may merge with atoms already in world t.
    """
    moved = source - target
    worlds = {world for _, world, _ in moved}
    if len(worlds) != 1:
        return False
    (world,) = worlds
    kept = source - moved
    for context in {w for _, w, _ in source}:
        if (context < world and all((v, context, state) in source for v, state in world - context)
                and kept | {(v, context, state) for v, _, state in moved} == target):
            return True
    return False


def consistency_rule_applies(first: CounterfactualEvent, second: CounterfactualEvent) -> bool:
    """Whether one application of R-115 Rule 1 turns one event into the other.

    Rule 1 (TR p.7, Eq. 15; Lemma 2.1, p.3) states
    P(y_{t x}, x_t, w*) = P(y_t, x_t, w*). One application takes a hard context
    t, a nonempty hard x on variables outside t whose atoms X_t = x[X] all occur
    in the event, and a nonempty set of atoms in world t,x, and moves each of
    them to world t. This returns ``True`` iff one such application turns
    ``first`` into ``second`` or ``second`` into ``first``, comparing events as
    sets of atoms; a moved atom may merge with an atom already present. Such
    events are the same set of units in every recursive SCM, whatever its graph
    or exogenous law, so no graph is taken and variables and states are
    arbitrary labels. Identical events are not a step and return ``False``.
    ``False`` otherwise means only that no single application relates them:
    two applications in opposite directions can relate events with the same
    units.

    Raises ``CounterfactualCalculusDomainError`` when an argument is not a
    ``CounterfactualEvent`` or contains a nested or self-clamped term.
    """
    a, b = _event_atoms(first, "first"), _event_atoms(second, "second")
    return a != b and (_one_rule_1_step(a, b) or _one_rule_1_step(b, a))


def _removed(value: object, diagram: _Diagram, clamp: dict[str, str]) -> frozenset[str]:
    raw: tuple[object, ...]
    if isinstance(value, str):
        raw = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, bytes):
        raw = tuple(value)
    else:
        raise CounterfactualCalculusDomainError("removed clamps must be a label or an iterable of labels")
    labels: set[str] = set()
    for label in raw:
        if not isinstance(label, str) or label not in diagram.vertices:
            raise CounterfactualCalculusDomainError(f"unknown vertex in removed clamps: {label!r}")
        if label not in clamp:
            raise CounterfactualCalculusDomainError(f"the term does not clamp {label!r}")
        labels.add(label)
    if not labels:
        raise CounterfactualCalculusDomainError("at least one clamp must be removed")
    return frozenset(labels)


def exclusion_rule_applies(graph: CalculusGraph, term: CounterfactualTerm, removed: str | Iterable[str]) -> bool:
    """Side condition of R-115 Rule 3 (TR p.7, Eq. 17) for removing clamps.

    With ``term`` = Y_{xz}, X the ``removed`` clamped variables and Z the other
    clamps, return whether no vertex of X is an ancestor of Y in ``graph`` with
    the edges into Z removed. ``True`` means Y_{xz} and Y_z agree on every unit
    of every SCM compatible with ``graph``, so P(y_{xz}, w*) = P(y_z, w*) for
    any w*. ``False`` means only that the rule does not license the step. A set
    of clamps is removable exactly when ``counterfactual_exclusion`` drops all
    of them.

    ``removed`` is a vertex label or an iterable of labels; it must be nonempty
    and every label must be clamped by ``term``. Raises
    ``CounterfactualCalculusDomainError`` otherwise, and for a graph or term
    that ``counterfactual_exclusion`` rejects.
    """
    diagram = _Diagram(graph)
    checked = diagram.term(term)
    clamp = _clamps(checked)
    drop = _removed(removed, diagram, clamp)
    return drop.isdisjoint(diagram.ancestors(checked.variable, frozenset(clamp) - drop))
