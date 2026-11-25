# Counterfactual calculus side conditions

This module implements, for hard interventions, the graphical objects of
Correa and Bareinboim's counterfactual calculus: the exclusion operator,
counterfactual ancestors, the **ancestral multi-world network** (AMWN) and the
side conditions of the calculus's three rules. A positive independence answer is
**sound** for the model class below. The criterion is **not complete on fixed
binary domains**, and whether it is complete when domains may be enlarged (finite
domains of any size) is open. No completeness claim is made for it or for the
calculus.

## Sources

- Correa and Bareinboim, *Counterfactual Graphical Models: Constraints and Inference*. The theorem, figures, and rules appear in the [ICML 2025 proceedings](https://proceedings.mlr.press/v267/correa25a.html). Section and page numbers below follow the [July 2025 technical report R-115](https://causalai.net/r115.pdf), SHA-256 `64dbee95584b97e4bd55f36bf528777d829ed5f710216bac3ac5adbf1699e7ba`.
- The [paper fixture](../../tests/fixtures/papers/correa_bareinboim_2025_calculus.json) records the figures, rule applications, and the checked interpretation for each case.

## Functions and outcomes

```python
counterfactual_exclusion(graph, term)                  # Lemma 2.3, p.4
counterfactual_ancestors_reflexive(graph, terms)       # Definition 2.4, p.5
ancestral_multi_world_network(graph, terms)            # Algorithm 1, p.6
counterfactual_d_separated(graph, left, right, given=())  # Theorem 2.5 / Rule 2
consistency_rule_applies(first, second)                # Rule 1, p.7
exclusion_rule_applies(graph, term, removed)           # Rule 3, p.7
```

| Result | Meaning |
| --- | --- |
| `counterfactual_exclusion` | `‖Y_x‖` keeps only the clamps on ancestors of `Y` once the edges into the clamped set are cut. It equals `Y_x` on every unit. |
| `counterfactual_ancestors_reflexive` | Definition 2.4's `An(W*)` as reduced terms, without duplicates. It is reflexive: it contains `‖t‖` for each input `t`, as `An(Y_x)` "includes Y itself". Terms are ordered by declared vertex, then by their clamps as (declared position, state) pairs. |
| `AncestralMultiWorldNetwork` | A `DAG` over canonical JSON labels: `["Y",{"X":"0"}]` for a term, and `{"U":["Y"]}` or `{"U":["X","Y"]}` for added roots. `terms`, `exogenous` and `node(term)` give the meaning of each label. Term nodes follow the order of `counterfactual_ancestors_reflexive`; then come the `U_V` roots in declared vertex order and the `U_VW` roots ordered by the declared positions of their endpoints. |
| `counterfactual_d_separated` is `True` | The left terms are independent of the right terms given `given` in every model of the class below. Rule 2 then removes or adds the right-hand evidence wherever the conditional is defined. |
| `counterfactual_d_separated` is `False` | No claim: it means neither dependence nor non-identifiability. |
| `consistency_rule_applies` is `True` | One application of Rule 1 turns one event into the other: a nonempty set of atoms in world `t,x` moves to world `t`, every `X_t = x` atom is present, and a moved atom may merge with an existing one. The events are then the same set of units in every recursive SCM. |
| `exclusion_rule_applies` is `True` | `X ∩ An(Y) = ∅` with the edges into the remaining clamps cut. Then `Y_{xz}` equals `Y_z` on every unit, and Rule 3 applies inside any event. |

A `False` from either rule check means only that the step is not licensed; for
example, two Rule 1 applications in opposite directions can relate events with
the same units, and `consistency_rule_applies` returns `False` for them. No
derivation is searched. In `counterfactual_d_separated`, terms that coincide
after exclusion are one random variable. A side that falls entirely inside
`given` is trivially independent. An unconditioned term shared by both sides
returns `False`.

## Domain and assumptions

These conditions are checked, and a violation raises
`CounterfactualCalculusDomainError` (a `CounterfactualDomainError`):

- for the functions that take a graph, it is a known `DAG` or bow-capable
  `AcyclicDirectedBidirectedGraph`, and every term is a `CounterfactualTerm`
  on its vertices with literal clamps;
- the roles of `counterfactual_d_separated` are terms or iterables of terms,
  syntactically disjoint, and the left and right roles are nonempty;
- `removed` in `exclusion_rule_applies` is a vertex label or an iterable of
  labels, nonempty, and every label is clamped by the term;
- `consistency_rule_applies` takes two `CounterfactualEvent` values and no
  graph, so it checks no vertices: variables and states are arbitrary labels.

A **nested** term is rejected; unnesting is the caller's job (R-115 Corollary
2.2). A term that clamps its own variable is the constant clamp
value and is also rejected; remove it, or replace it by that constant, first.
These two checks apply to every function, including the terms inside events.
States are opaque labels. No alphabet is required, and distinct labels denote
distinct worlds.

The caller supplies the model class. It is a recursive SCM in which each vertex
is a function of its graph parents and of exogenous roots. The roots are
**mutually independent**, and the children of each root form a bidirected clique
(private roots are allowed). Domains are finite sets of any size, and allowed
edges may be inactive. Rules 1 and 3 hold unit by unit and need only the
recursive equations.

## Soundness basis

Let `N` be the counterfactual ancestors. For `W_z` in `N` and an unclamped parent
`P`, the parent value is `P_z`, which equals `‖P_z‖` in `N` on every unit. So
each node is a function of its `N` parents and of the roots of `W`. Let `G*` be
`N` together with every root of the model, with arrows from roots to all copies
of their children. This is the ancestral part of a parallel-worlds graph
(Shpitser–Pearl 2007) with equal copies merged. Its joint law factorizes along
`G*`, with deterministic kernels, so d-separation in `G*` implies conditional
independence by the global Markov property.

Now map a path of `G*` that is d-connecting given the conditioning set. A fork
at a root becomes a fork at `U_V` when both ends are copies of `V`. When they
are copies of different variables, the root's clique gives an edge `V <-> W`,
and the fork goes through `U_VW` instead. Roots are never colliders, and term
nodes have the same term parents, hence the same term descendants, in both
graphs, so every collider keeps its relation to the conditioning set. The image
is a d-connecting walk in `G_A`: no non-collider is conditioned on, and every
collider is an ancestor of the conditioning set. Such a walk always yields a
d-connecting path between its ends (its nodes lie in the ancestral set, and
it gives a path avoiding the conditioning set in the moralized ancestral
graph), so the sets are not d-separated in `G_A`.
Hence d-separation in `G_A` implies independence.

Appendix B.4 derives soundness from Lemma B.5. For a graph
`X -> Y`, copies `Y_x0` and `Y_x1` can fall in different ancestral
components while sharing the exogenous root of `Y`. Their joint distribution
need not factorize as in Eq. (48). The argument above establishes the
criterion used by this implementation for the independent-root model class.

## Limits

- **Theorem 2.5's completeness clause fails on fixed binary domains.** In
  `X -> Y` with binary `X`, `Y_x0 ⊥ Y_x1 | X, Y` holds in every model, since
  each conditioning cell fixes one of the two terms by consistency, but it is not
  d-separated in `G_A`. With a third state of `X` it fails generically, so the
  example says nothing about larger domains. Whether completeness holds when
  domains may be enlarged is open, and no gap is exhibited here for a fixed
  domain with three or more states.
- Only hard interventions are covered: no nested, natural or soft
  interventions, and no unnesting.
- This is not an identification procedure. It does not search derivations,
  apply probability-axiom steps or decide R-115 Theorem 3.2.
- The independence of the roots is assumed, not checked. Correlated exogenous
  noise outside the clique-root class is not covered.
- Runtime is polynomial, and no budget is needed. By Definition 2.4 each input
  term contributes at most one ancestor per vertex, so `n` vertices and `s`
  distinct input terms give at most `n·s` term nodes, plus at most one root per
  vertex and per bidirected edge. Each term node costs one graph search per
  parent arrow.

## Evidence

[The regressions](../../tests/python/test_counterfactual_calculus.py) compare
these functions with [an independent oracle](../../tests/python/calculus_oracles.py)
that imports only counterfactual data types. The oracle builds explicit finite
SCMs with one private root per vertex and, by default, one latent per
bidirected edge; it can instead feed a whole bidirected clique from one root.
It evaluates each term unit by unit and checks independence by exact cross
products of integer or rational weights. It also builds Definition 2.4 and
Algorithm 1 literally, seed by seed, with its own path search.

The named cases reproduce the paper's ancestor sets, AMWN figures, and
Rule 2 applications in Eqs. (22), (27), (28), and (32). The Figure 6
dependence case also uses a finite SCM whose observational and intervention
kernels are strictly positive. The bounded and opt-in sweeps
(`CGA_CALCULUS_SWEEP=1`) test the criterion on additional independent-root
models, including one root with three observed children.
