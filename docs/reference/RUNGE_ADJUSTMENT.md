# Runge optimal adjustment in ADMGs

## Overview

The implementation provides the O-set construction in Algorithm C.1 and the
graphical optimality test in Theorem 3 of Jakob Runge, *Necessary and sufficient
graphical conditions for optimal adjustment sets in causal graphical models with
hidden variables*, NeurIPS 2021
([proceedings](https://proceedings.neurips.cc/paper/2021/file/8485ae387a981d783f8764e508151cd9-Paper.pdf)).
The detailed definitions and pseudocode were checked against the revised
[arXiv version 4](https://arxiv.org/abs/2102.10324v4), submitted June 23, 2023:
Definitions 3–4, pp. 5–6; Theorem 3, pp. 7–8; Algorithm C.1, p. 23.
The [arXiv version history](https://arxiv.org/abs/2102.10324) was rechecked on
2026-09-13 and lists v4 as latest, submitted 23 June 2023. Its revision comment
reports corrected supplementary proofs. That comment does not identify the
corrected claims; the observations below remain tied to the explicit v4 passages.
The code is an independent implementation from those mathematical definitions;
no Tigramite source code is imported.

## API and the meaning of optimality

```python
from causal_graph_algorithms import (
    AcyclicDirectedBidirectedGraph,
    runge_o_set,
    runge_optimal_adjustment,
)

graph = AcyclicDirectedBidirectedGraph.from_text("X -> Y; Z1 -> X; Z1 -> Z2; Z2 <-> Y")
construction = runge_o_set(graph, "X", "Y")
result = runge_optimal_adjustment(graph, "X", "Y")
assert result.adjustment_set == ("Z1", "Z2")
assert result.graphical_optimality is False
```

This example is Figure 2E. Its O-set is valid, but its variance ordering relative
to the empty adjustment set depends on the distribution. The name O-set by itself
does not establish optimality.

Both functions accept a validated `AcyclicDirectedBidirectedGraph` or `DAG`, a
single treatment and outcome, and `conditioned_on` for the fixed set S. Bows are
allowed. Raw topology, MAG/PAG views, multivariate treatments/outcomes, and
graphs without a directed treatment-to-outcome path are outside this API.
S must exclude the treatment and all directed descendants of the treatment.
This restriction is stronger than the paper's S-disjoint-from-forbidden
condition: the latter allows conditioning after treatment, for which the
paper's conditional causal estimand need not equal its proposed OLS coefficient.
Both public functions reject such inputs with `RungeAdjustmentDomainError`;
neither silently changes the causal question. The example and justification
are recorded below.
Ancestors and descendants in the paper are reflexive; mediators exclude X and Y.

The immutable result uses the graph's declaration order for every tuple. Results
are created only by the algorithm functions: direct construction and
`dataclasses.replace` reject attempts to manufacture or alter verified claims.

| Field | Meaning |
| --- | --- |
| `adjustment_set` / `vertices` | Additional covariates O minus S; `None` means no valid adjustment set exists, `()` means valid empty adjustment |
| `conditioned_on` | Fixed S, included in all separation checks |
| `o_set` | Definition 4 union P ∪ C ∪ P_C, which may overlap S; `None` when invalid |
| `parents`, `colliders`, `collider_parents` | P, C, P_C; diagnostics may exist even when no valid adjustment exists |
| `mediators`, `forbidden`, `valid_ancestors` | M, X ∪ des(YM), an(XYS) minus forbidden |
| `graphical_optimality` | `unique_adjustment_set or (criterion_i and criterion_ii)`; `None` if unevaluated or no adjustment exists |
| `criterion_i`, `criterion_ii` | The two conditions in Theorem 3; `None` if unevaluated or inapplicable |
| `unique_adjustment_set` | Whether only one additional adjustment set exists given S; `None` if unevaluated or inapplicable |

`runge_o_set` performs only construction and validity checks. It leaves the four
Theorem 3 fields `None`. When adjustment is possible, `runge_optimal_adjustment`
computes the graphical criterion. Runge's published Theorem 3 interprets a
`False` verdict as absence of a distribution-independent optimal adjustment set
in its domain; that interpretation relies on its necessity direction. The local
sufficiency repairs and fixture checks do not independently validate necessity.
The returned O-set remains a valid adjustment set.

The statistical result assumes a causal Markov and faithful distribution and a
nonzero causal effect. These are caller assumptions; topology cannot verify
them. The theorem optimizes the paper's adjustment information. Its translation
to minimal asymptotic variance requires the estimator relationship in
Assumptions 2, verified in the paper for the linear Gaussian/OLS setting.
This API does not promise optimal variance for arbitrary estimators, compute
sample-size behavior, or estimate variances. It does not implement Algorithm C.2
minimized O-set variants.

## Algorithm and explicit resource limits

Construction expands bidirected paths from Y and its mediators through nodes
that satisfy Definition 3. Parents of the outcome/mediators and of the valid
colliders complete the set. The two failures in Algorithm C.1 are retained.
As an additional verification, forbidden membership and m-separation in the
proper backdoor graph check O ∪ S. Only first edges of proper directed X-to-Y
paths are removed in that graph; unrelated treatment descendants remain.

Theorem 3 defines N-nodes as the bidirected neighbors of Y, its mediators M,
or C, excluding forbidden vertices, O and S:
N = sp({Y} ∪ M ∪ C) minus (forbidden ∪ O ∪ S).
Condition I searches simple bidirected paths from each N-node to
Y or a mediator, with internal vertices in C. Instead of reconstructing an
auxiliary O-set for every completed path, it tests whether the required vertices
can be included in **any** valid adjustment set. For a nonterminal path prefix,
let Q contain the starting N-node and all internal vertices visited so far; Y and mediators are
never included in Q. In the proper backdoor graph H, form

```text
W = V minus forbidden
A(Q) = an_H({X, Y} union S union Q) intersect W
```

For Q ⊆ W, a valid adjustment containing S ∪ Q exists exactly when A(Q)
m-separates X and Y in H. This is the constrained separator construction
described below. Failure means that no continuation of this prefix can work,
since every continuation requires a superset of Q. The prefix is pruned.
A surviving path that reaches Y or a mediator refutes condition I.
Importantly, this checks **existence of a valid extension**, not whether S ∪ Q
itself is valid: conditioning on a collider can invalidate that set while
additional covariates restore validity.

The equivalence to the published auxiliary O-set test uses
[Runge's Theorem 1](https://arxiv.org/html/2102.10324v4#S2.SS3): the auxiliary
O-set is graphically valid exactly when a valid adjustment exists with the
augmented fixed set S′ = S ∪ Q. S′ remains disjoint from forbidden vertices,
even when Q includes other treatment descendants. This is a graphical
existence test, not an endorsement of the source's post-treatment conditional
estimand/OLS correspondence discussed below.

Fixed-S convention: candidates are additional covariates disjoint from S,
and uniqueness is tested among these candidates, not among redundant ways
of writing the same total conditioning set. The source's Theorem 3 explicitly
excludes S from N and states its uniqueness alternative under Assumptions 1,
which already allow fixed S. For II, this API checks E in O minus (P union S):
a variable already fixed in S is not an additional covariate and contributes
zero conditional information given S. This makes explicit the fixed-conditioning
reading of the printed E-index range and conditioning expression; it is not
a separate proof of Theorem 3 for fixed S. The test suite covers this convention. Agreement with a test oracle using the same convention
does not independently adjudicate the source's notation.

The separator result is cached by A(Q), within a single query. Reflexive
ancestor sets are also cached per vertex. Search uses walks, merging equal
`(endpoint, A(Q))` states **at the same source**. Equal closures give identical
continuations when no visited-set restriction is imposed. A feasible completed
walk yields a simple indexed path by cycle erasure: removing requirements
preserves existence of a valid extension. At one endpoint, a reached feasible
closure contained in another dominates the larger closure. The implementation
keeps these incomparable minimal closures; it never substitutes a union or
intersection for two different reached states. No Tigramite code is copied.

Before searching each N-node n, polynomial preprocessing fixes B=A({n}) and
uses incremental labels L_v=(an_H(v) minus forbidden) minus B. It excludes
infeasible sources, vertices and adjacent pairs by the same extension check,
then keeps the source component only if a terminal is reachable. A synthetic
terminal joins the original bidirected adjacencies to Y or a mediator and has
empty label; it is never conditioned on. These are the original C/O/N/F,
not components recomputed on a reduced causal graph.

In this separate traversal graph, an edge a-b is deleted only when a detour
exists whose internal labels lie in L_a union L_b. Deletions are sequential.
Replacing the edge adds no ancestors; truncating at the terminal and erasing
cycles only removes requirements. Thus some feasible index exists before
deletion exactly when one exists afterward. This preserves existence, not
every original witness, and does not permit an arbitrary district spanning tree.
Source and terminal can occur internally on a replacement walk, but not on its
final simple path. A single fixed-label edge pass is sufficient, with no
minimum-edge guarantee. Mandatory-articulation iteration is not implemented.

Separation may use H[U], where U=an_H({X,Y} union S union R) and R is the retained
set of possible required vertices. All induced directed/bidirected edges and
forbidden graph vertices remain. This is safe for the ensuing canonical
conditioning sets, all contained in U, not arbitrary outside-U queries.
Directed transitive reduction is **not** used for separation. For example,
`h -> X -> Y; X -> b -> n; X -> n; n <-> Y` has Condition I true, but deleting
X -> n preserves every directed ancestor relation and makes I false: {b,n}
then becomes a valid extension of the index n <-> Y. The tests independently
check this distinction and a six-node same-district counterexample.

Condition II uses reachability through C ∩ valid ancestors, allowing
an arrowhead or a tail only at the starting E vertex. A fixed E in S is omitted
from the condition-II test because it is already conditioned on. The paper's
conditional-information comparison has no additional contribution from E.
The augmented sets in condition I are internal combinatorial tests of the
published criterion. They can include treatment descendants even though the
caller-supplied S cannot. They are not returned as separately validated causal
queries; this distinction preserves the criterion, including Figure 2F.

Uniqueness also does not enumerate the adjustment powerset. For each eligible node,
it asks whether a valid set can add it or omit it from O. Each constrained
separator query I ⊆ Z ⊆ R uses the ancestral candidate an(X ∪ Y ∪ I) ∩ R in
the proper backdoor graph, following the separator construction in van der
Zander, Liśkiewicz and Textor (2019), *Separators and adjustment sets in causal
graphs: Complete criteria and an algorithmic framework*
([Lemma 1, FindSep and Proposition 2](https://arxiv.org/html/1803.00116v2#S3.SS1)).
The same construction supplies A(Q) above, with I = S ∪ Q and R = W.
For a bow-capable ADMG, this construction can be justified in the latent DAG
that replaces each bidirected edge with a distinct unobserved common parent:
latent vertices are excluded from R, and the observed ancestral candidate and
separation statements are unchanged. No MAG-validity claim is made about the
input ADMG.

Construction, uniqueness and preprocessing use polynomially many
reachability/separation operations. Condition I can still visit exponentially
many distinct closure states; preprocessing and caching are not a general
polynomial-time guarantee. No empirical large-graph speedup is asserted.

| Keyword | Counting rule, across all N-nodes |
| --- | --- |
| `max_collider_states=None` | Optional nonnegative integer bound on dequeued distinct `(endpoint, closure)` states, summed across sources. Cache hits, infeasible states and states dominated after enqueueing count; duplicates and states dominated before enqueueing do not. Per-source preprocessing is **not counted**. `None` leaves state search unbounded. |
| `max_collider_paths=100_000` | Completed feasible-witness bound, applied **after preprocessing/pruning**. Search stops at the first witness, so only zero versus positive matters. Zero raises if a witness is found, but may return true when preprocessing or search rules out every candidate. It does not limit search work. |

For example, `runge_optimal_adjustment(graph, "X", "Y",
max_collider_states=100_000)` sets an explicit partial-search limit. Exhaustion
raises `RungeAdjustmentResourceLimitError`; it never becomes `False` or “no
adjustment.” Preprocessing/merging may avoid earlier search-limit exceptions;
even both limits zero can return true when preprocessing excludes all sources.
Neither limit bounds preprocessing, memory, wall time, O-set construction,
Condition II or uniqueness. Use `runge_o_set` if only the valid candidate is needed.

State identity is local to one source: the same `(endpoint, closure)` reached
under another source counts again. A cached feasibility answer still counts when
its state is dequeued. A feasible terminal witness establishes **Condition I
false**; the overall result still evaluates Condition II and uniqueness. The
fixed-S nondescendant restriction and public result fields are unchanged.

The [required-set tests](../../tests/python/test_runge_required_sets.py) enumerate
all eligible supersets with an independent latent-DAG oracle on exhaustive
small bow-capable graphs and fixed-seed samples. The
[search regressions](../../tests/python/test_runge_condition_i_search.py) include
repairable colliders, fixed S, mediators, a bow that reverses the answer, and
bidirected cycles with repeated vertex sets. The
[preprocessing regressions](../../tests/python/test_runge_condition_i_sparsify.py)
compare each source separately, so another source cannot hide a lost witness.
They include fixed S on a retained path, terminal-name collisions, forbidden
vertices in the induced oracle, unsafe spanning-tree/transitive reductions,
unsafe state merging, 800 adjustable ordered four-node ADMGs, and 100 fixed-S
seven-node draws. A targeted nine-vertex graph has 1,956 literal completed paths
but preprocessing rejects all six internal vertices, requiring zero walk
states. This is an operation-count example, not a general speedup claim.

## Source interpretation and reproducibility

The public conditioning domain is narrower than Assumptions 1 because the
paper's statistical interpretation needs an additional premise. Equation (3),
p. 2, defines a contrast of E[Y | do(X=x), S=s]. The text immediately after
Figure 1, p. 3, equates the unit contrast with the X coefficient in the
regression of Y on X, Z and S. Assumptions 1 on that page require only
S ∩ forbidden = empty. That condition does not establish the equivalence.

Consider the DAG X→Y, X→S, U→S, U→Y with independent unit Gaussian
X, U, ε_S and ε_Y, and equations S=X+U+ε_S and Y=X+U+ε_Y.
S is outside the forbidden set {X,Y}, and the literal construction gives O={U}.
However,

```text
E[Y | do(X=x), S=s] = x + (s-x)/2 = (x+s)/2
unit conditional causal contrast = 1/2
X coefficient in the regression Y ~ X + U + S = 1
```

Thus the source's OLS coefficient targets a different quantity in this example.
The exact-rational regression
`test_posttreatment_conditioning_estimand_mismatch_exactly` derives the
covariances from exogenous loadings, solves the population normal equations,
and verifies faithfulness by checking every pair and every conditioning subset.
This is a counterexample to the stated conditional-estimand/OLS correspondence
under the written assumptions, not to the combinatorial O-set construction or
the information-ordering criterion of Theorem 3.

The chosen repair is to require S outside all directed descendants of X in both
public APIs. They return the same causal-adjustment result type; exposing the
broader domain through construction alone would give that type two statistical
meanings. The internal Algorithm C.1 implementation and augmented-set tests in
Theorem 3 retain their literal graphical meaning. This restriction does not
exclude any of the paper's nine illustrated graph queries.

A separate minor error in the discussion of Figure S1 was independently
identified: supplement B.2, p. 13 first paragraph, asserts that terms (i) and
(ii) of Eq. (7) are both strictly positive. Here O={Z2,Z3} and
vancs={Z1,Z2,Z3,S}, so term (i) has empty first argument O minus vancs and is
exactly zero. Only term (ii), I(X;Z1 | O,S), is strictly positive under
faithfulness. The stated strict ordering J_O > J_vancs and the optimality
conclusion remain correct. The regression
`test_s1_comparison_term_i_is_zero_and_term_ii_is_positive` records the graph,
sets and independent path checks; no algorithm change is implied.

Fixed conditioning is made explicit in the API because O from Definition 4
and O ∪ S are different objects: O need not contain all of S. The generalized
adjustment check always uses their union.

All nine graph panels in the paper and theoretical supplement are executable
fixtures in [`runge_2023_figures.json`](../../tests/fixtures/papers/runge_2023_figures.json),
consumed by `tests/python/test_runge_adjustment.py`. The edges were transcribed
from rendered images of the source PDF, not inferred from its reported O-sets.
Provenance: arXiv:2102.10324v4, Figure 1A p. 3, Figure 2A–G p. 6 (discussion
pp. 4, 7–8), Figure S1 p. 12. The inspected 41-page PDF has SHA-256
`4302151468edfbed2863b30246212e0bd13c2ff1b1571458afbedf1ecba521ac`.

| Figure | Reported O | Reported graphical optimality | Independent adjudication |
| --- | --- | --- | --- |
| 1A | {Z2, Z3} | true | Agrees; all six reported valid sets reproduced exactly |
| 2A, S fixed | {Z1} | true | Agrees, including the S→X / S↔X bow |
| 2B, S fixed | {Z1, …, Z8} | true | Agrees; P, C, P_C, forbidden nodes, valid ancestors and isolated Z12 separately checked |
| 2C | {Z1, Z5} | true | Agrees; N-node Z2 cannot be included in a valid set |
| 2D | {Z2} | true | Agrees with original-path validity and Theorem 3 |
| 2E | {Z1, Z2} | false | Agrees; condition II fails |
| 2F | empty | false | Agrees; condition I fails |
| 2G, S fixed | {Z1, Z2} | true | Agrees with original-path validity and Theorem 3 |
| S1 | {Z2, Z3} | true | Agrees for the fully observed DAG version of Figure 1A |

Figure 2B's 16 vertices include the Z8→Z5 / Z8↔Z5 bow, forbidden descendants
Z9 and Z10, and isolated Z12. Every subset of eligible adjustment vertices is
checked by a separate latent-DAG/moralization oracle. Its valid sets are exactly
the 384 sets that contain Z1 and at least one of Z2 or Z11, with the other seven
eligible vertices optional. This count follows independently as
3 × 2⁷; it is a fixture assertion rather than a published result attributed to
the paper.

The remaining panels are accounted for separately:

| Panel or figure | Coverage and scope |
| --- | --- |
| 1B | The variance ordering for its Figure 1A graph is checked over all valid sets and five explicit parameter-scale combinations. The original finite-sample scatterplot is not reproduced; no estimator or sample-generation API is implemented here. |
| 1C | A Venn diagram explaining adjustment information, not a graph-algorithm input/output fixture. The analytic covariance reference and OLS comparisons test the implemented statistical implication. |
| 3 | Aggregate finite-sample simulation results, not a specific graph fixture; reproducing these estimator experiments is outside this graph-algorithm implementation. |
| S2–S16 | Linear/nonlinear simulation plots for OLS, kNN, MLP, RF and DML; not rerun. These require estimators and experimental datasets beyond the graph APIs. |

The independent oracle in `tests/python/runge_oracles.py` enumerates paths
with separate edge marks, so directed and bidirected edges on the same pair
remain distinct. It checks noncausal paths in the original graph, enumerates
all small adjustment sets, and constructs Definition 3 collider paths with
permutations; it does not import production traversal, separation, or adjustment
helpers. A second oracle expands bidirected edges into distinct latent common
parents and uses ancestral moralization with integer-indexed reachability
matrices. It covers Figure 2B and is cross-checked against original-path
enumeration on 65 random small ADMGs, including every eligible conditioning
subset. No production traversal is used in either oracle.

The existing randomized structural checks cover 110 five-vertex and 80
six-vertex ADMGs, including mediators, bows and fixed conditions. These tests
now explicitly reject caller-supplied S that contains treatment
descendants. Every original random draw is retained: these cases assert the
domain error in both public functions, including cases where unconditional
adjustment of Z ∪ S would have been possible. Supported cases continue to be
compared with the independent oracles; no failing graph is silently discarded.

Independent linear SEM covariance calculations check all valid sets for each of the nine
paper graphs. The seven graphically optimal paper graphs each use five
predeclared directed/latent coefficient-scale combinations, including mixed
signs, varied noise, and fixed S where shown. The two nonoptimal graphs use
the full 3 × 3 grid of directed and latent scales {0.1, 1, 10}; each exhibits
both signs of the variance difference against a specified alternative set.
Thus the paper-graph comparison covers 53 SEM parameterizations, including
all 384 adjustment alternatives for Figure 2B. An additional 65 generated
graphs exercise the predicted-optimality implication under six parameter draws
for each graphically optimal result. A three-node analytic covariance example
independently anchors the numerical calculation. No cases are discarded based
on their variance rankings. These tests verify implementation and sampled
implications; they do not establish the theorem for all distributions.

Run focused verification with:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/python/test_runge_adjustment.py tests/python/test_runge_sufficiency_defects.py \
  tests/python/test_runge_required_sets.py tests/python/test_runge_condition_i_search.py \
  tests/python/test_runge_condition_i_sparsify.py
```

## A defective sufficiency step and two qualified observations

The first witness contradicts a step of the **sufficiency** ("if") half of the
Theorem 3 proof with its hypotheses in force. A second witness refutes only a
hypothesis-free reading; a third refutes an attempted stronger repair, not a claim
of the paper. The proof is supplement Appendix B.10, pp. 18–21, of arXiv v4; both passages
discussed below sit on p. 19. The pagination is that of the 41-page v4 PDF whose SHA-256
this file records above. Line numbers refer to the v4 LaTeX source
`Runge_Optimal_adjustment_sets_v5_neurips.tex`, whose lines 1156–1267 are that subsection
(line 1268 begins B.11).

**The two are not of equal force and must not be cited as though they were.** The first
is a counterexample to a step with that step's own hypotheses in force. The second
refutes only a hypothesis-free reading of a step, because Condition (II) is false on its
witness graph; whether that step is complete on its own terms was not determined here.

Neither witness establishes a counterexample to the **statement** of Theorem 3, and
**no implementation change follows**: on all three witness graphs below the
implementation agrees with a separate evaluation of the graphical criterion. The observations concern how the theorem is proved and
not about what the criterion computes. Each case was adjudicated against two independent
implementations — this library for the O-set, Conditions (I) and (II) and the verdict,
and a separate implementation written from the published definitions and kept in the
external consumer repository where this work was performed, which computes the Def. 3 and
Def. 4 objects (valid ancestors, the parents/colliders/collider-parents decomposition and
the O-set), the E-set, Conditions (I) and (II), m-separation and adjustment validity. The
two agree on every recorded field.

Notation for this section: for a valid adjustment set Z, write A = O \ Z, R = Z \ O and
B = O ∩ Z, so that the chain rule splits J_O − J_Z into an outcome bracket
I(A;Y | BX) − I(R;Y | BX), which Condition (I) is meant to make non-negative, and a
treatment bracket I(X;R | B) − I(X;A | B), which Condition (II) is meant to make
non-negative. The selection set is empty throughout.

### Condition (I) step, lines 1187–1190: a counterexample with the hypotheses in force

*What the published step claims.* In the branch where N-nodes exist, term (iii') =
I(R;Y | BX) is argued to be non-zero only if the compared set contains a descendant of an
N-node together with all nodes of one of that node's collider paths; line 1190 concludes
that term (iii') is zero.

*Witness.* `X -> Y; X -> U; V -> U2; V -> Y; U <-> W; W <-> Y`. Here the mediators are
empty, forbidden is {X, Y}, `parents` is {V}, `colliders` is {W} and `collider_parents` is
empty, so O = {V, W}, and the valid ancestors are {V}. Conditions (I) and (II) both hold,
so the criterion returns graphical optimality true. Take the valid adjustment set
Z = {U, U2}: then B = empty, R = {U, U2} and A = {V, W}.

*What an independent adjudication found.* False as written. The step's conclusion is that
Y is m-separated from {U, U2} given {X}. It is not: the path U2 ← V → Y is open given
{X}. For a faithful distribution, term (iii') is positive; m-connection alone does
not imply positive mutual information for every Markov distribution. The blocking vertex V is a non-collider that does belong
to O, but it lies in A rather than in B, so the conditioning set of term (iii') — B ∪ {X}
— does not contain it, and the supporting lemmas the step leans on are stated for
conditioning sets that contain all of O. No N-node, no collider path and no descendant of
an N-node plays any role in the open path.

*What survives.* The theorem's conclusion is not contradicted on this graph. The outcome
bracket is still non-negative, by a step the paper does not take: with the subset witness
A' = {V} ⊆ A one has Y m-separated from {U, U2} given {X, V}, whence
I(R;Y | BX) ≤ I(A';Y | BX) ≤ I(A;Y | BX) by the chain rule and non-negativity of
conditional mutual information.

### Condition (II) step, lines 1199–1210: only a hypothesis-free reading is refuted

Let the E-set denote the nodes E of O outside its `parents` part for which X is not
m-separated from E given O \ {E}.

*What the published step claims.* Case (2) of the Condition (II) argument considers a path
ending with an arrowhead at E where E is not a valid ancestor, states that this implies E
m-separated from X given the valid ancestors by Definition 3, and concludes that then E is
m-separated from X given S ∪ O \ {E}, hence that E is not in the E-set.

*Witness.* `P -> C; Q -> P; Q -> X; X -> Y; C <-> Y`, where P and C are vertex names.
Here forbidden is {X, Y}, `parents` is empty, `colliders` is {C} and `collider_parents` is
{P}, so O = {C, P}, and the valid ancestors are {Q}. Thus P is not a valid ancestor, yet
X is not m-separated from P given O \ {P} = {C}, so P is in the E-set.

*What an independent adjudication found, and its limit.* The displayed implication read
**without** its Condition (II) hypothesis is false, as the witness shows. It is **not** a
direct counterexample to the published Case (2). Case (2) treats nodes whose Condition (II)
witnesses are all of type 2, that is, paths leaving E by an arrowhead; P has no bidirected
edge at all, so it has no type-2 witness, and its only witness path P → C ↔ Y leaves P by a
tail while its collider C is not a valid ancestor, so it is not an admissible Condition (II)
witness either. Condition (II) is therefore **false** on this graph and this library
correctly returns graphical optimality false. The step's own hypotheses are consequently
not met. Line 1195 invokes Condition (II) to establish the witness typology that
Case (2), lines 1199–1210, inherits. A lack of another explicit invocation within
the case would not show that the hypothesis is unused. The witness shows why the
hypothesis cannot simply be dropped; it does not determine whether the published
Case (2) is complete.

*What survives.* The statement that actually spends the hypothesis is weaker than
membership of E in Z: Condition (II), Z valid, E in the E-set and E not in Z imply X
m-separated from E given Z. That weaker statement is the repair target, and it holds on
the third witness below.

### Third witness: Condition (II) does not force the E-set into every valid set

This one refutes a repair one might attempt, not a claim of the paper. `A -> X; X -> B;
B -> Y; C -> D; D -> Y; A <-> C; B <-> D` has O = {C, D}, E-set {C}, Conditions (I) and
(II) both true, and the valid adjustment set Z = {A} — the vertex named A, not the bracket
set of the notation above — which omits C. So Condition (II) does not force every E-set
node into every valid adjustment set. The weaker m-separation statement does hold here: X
is m-separated from C given {A}.

### What each claim rests on

| Claim | Level of evidence |
| --- | --- |
| The O-set, its three parts, the valid ancestors, both conditions and the verdict on all three witness graphs; the separations the two published steps assert | Machine-checked, and agreed by two independent implementations on every recorded field |
| The lines 1187–1190 step fails with its own hypotheses in force | Deductive and hand-checkable from the definitions, given the machine-checked separation |
| The lines 1199–1210 witness refutes the hypothesis-free reading only | Deductive, but its *limit* rests on an interpretation of the source: that Case (2) treats nodes whose Condition (II) witnesses are all of type 2. Under that reading the witness does not meet the step's hypotheses |
| Validity of the two witness adjustment sets {U, U2} and {A}; exact third E-set {C} | Recomputed by repository regressions. Both adjustment sets pass the independent literal noncausal-path and latent-DAG moralization oracles in `tests/python/runge_oracles.py`. The third E-set is recomputed by literal m-connecting paths; {A} omits C and still separates X from C. These helpers use raw edge data, not production separation or adjustment traversal |

### Which half of the theorem each recorded problem concerns

- **Sufficiency half:** one in-hypothesis proof-step counterexample, one
  hypothesis-free observation, and the third witness to an attempted repair.
- **Necessity half:** two items **reported by an earlier external review of this paper and
  not adjudicated in this repository** — a logical flaw in the NeurIPS 2021 proof, said to
  have been corrected in v4, and the rigour of the v4 attenuation argument. The
  arXiv comment establishes only that the Supplementary Material was
  updated with corrected proofs, not what was wrong. Out of scope here.
- **Neither half:** the post-treatment fixed-S estimand/OLS mismatch recorded above, which
  concerns the paper's statistical interpretation of Eq. (3) under its Assumptions 1, not
  the graphical proof. It is the reason for the narrower public conditioning domain; these
  sufficiency observations imply no domain change. Also the Figure S1 term-(i)/term-(ii) error
  recorded above (supplement B.2, p. 13; regression
  `test_s1_comparison_term_i_is_zero_and_term_ii_is_positive`), which is an error in a
  worked example, on neither half of the proof.

### Fixture and regressions

The three witness graphs, the reported claim and the adjudicated finding for each, and the
flag `hypotheses_of_the_step_in_force` that keeps the cases at their unequal levels are
stored in
[`runge_2023_sufficiency_defects.json`](../../tests/fixtures/papers/runge_2023_sufficiency_defects.json)
and consumed by
[`tests/python/test_runge_sufficiency_defects.py`](../../tests/python/test_runge_sufficiency_defects.py).
The regressions assert the O-set components and criterion verdicts, both witness-set
validities, the exact third E-set, and each recorded separation. The regressions also distinguish the three claim statuses. Passing these finite checks is not a proof of the criterion for all graphs.
No production change is justified by these observations, and none was made.
