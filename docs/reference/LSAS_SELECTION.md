# LSAS selection with an exact independence oracle

`find_lsas_adjustment(observed, *, treatment, outcome, independent)` evaluates
the local sufficient rules from Li et al., *Local Learning for Covariate
Selection in Nonparametric Causal Effect Estimation with Latent Variables*
(NeurIPS 2025), with two explicit corrections. R2(i) does not require a witness
variable, and R1 searches the empty adjustment set. The [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/b43a0e8a35b1c044b18cd843b9771915-Paper-Conference.pdf) and the [reproduction fixture](../../tests/fixtures/papers/li_2025_lsas.json) give the source and the corrected cases.

This operation uses exact conditional independence, without statistical testing,
sample data, effect estimation, or fitting a graph. Deriving blankets by full
conditioning is an oracle reference construction; it does not reproduce the
paper's finite-sample Markov-blanket discovery or its CI-query efficiency.

## Assumptions and inputs

`observed` is an ordered sequence of unique nonempty strings including the
distinct `treatment` X and `outcome` Y. Its order determines deterministic tie
breaking and is copied before oracle calls. All other observed variables are
available pretreatment covariates.

The callable `independent(left, right, given)` accepts two distinct labels and a
`frozenset[str]` disjoint from those labels, and returns the built-in `bool`:
`True` means conditional independence. The implementation memoizes symmetric
queries. Exceptions raised by the callable propagate. A non-boolean answer
raises `LSASOracleError`; malformed inputs raise `LSASDomainError` before any
oracle calls. An arbitrary callable's global consistency is not tested.

The causal interpretation requires an exact CI oracle for an observational
distribution Markov and faithful to the paper's acyclic latent-variable model,
without selection bias. Y must not be an ancestor of X; every remaining observed
variable must be a nondescendant of both X and Y. These causal assumptions cannot
be verified from the input callable. The function does not extend them to cyclic
SCMs, statistical test errors, or arbitrary dependency tables.

For v in {X,Y}, the blanket is derived as

```text
MB(v) = {u in observed \ {v}: not independent(v, u, observed \ {v,u})}.
```

This is the total-conditioning characterization under the stated assumptions,
not a request for an estimated or user-certified blanket.

## Rules and result meanings

The positive-result guarantee follows from the adjustment-validity part of R1
and the zero-effect part of R2 in
[Entner, Hoyer and Spirtes (2013), Theorem 2](https://proceedings.mlr.press/v31/entner13a.pdf),
under the assumptions above. Every local witness/set pair is also an eligible
pair for those global rules, with identical CI premises. Restricting the search
can omit a valid conclusion but cannot invalidate a conclusion that satisfies
the rule. This argument does not require LSAS's locality or completeness proof.
Separating R2(i)'s quantifier restores the original EHS rule. This guarantee is
about graphical adjustment validity or absence of causal influence, not a
nonzero marginal ATE or the reliability of finite-sample CI tests.

First enumerate Z subsets of MB(Y) minus X by increasing size, including zero.
If X and Y are independent given Z, return `LSASZeroEffect` with rule `R2(i)`
and no witness. This search runs independently of whether MB(X) has any witness.

Then visit S in MB(X) minus Y in observed order. For each S, enumerate Z subsets
of MB(Y) minus {X,S}, again by increasing size including zero:

- R1: S and Y are dependent given Z and independent given Z union {X}.
  Return `LSASAdjustmentSet` with this adjustment and witness S.
- R2(ii): S and X are dependent given Z and S and Y are independent given Z.
  Return `LSASZeroEffect` with this conditioning set and witness S.

The first matching rule returns. No match after full search returns
`LSASUnresolved(reason="no_rule_applies")`. This is **not** a conclusion that
the effect is unidentified or that no valid adjustment set exists. The
implementation makes no general completeness claim for the corrected rules.

All results are immutable and factory-created. They retain `observed`,
`treatment`, and `outcome`.

| Result | Additional fields | Meaning under the assumptions |
| --- | --- | --- |
| `LSASAdjustmentSet` | `adjustment`, `witness`, `rule="R1"`, `status="adjustment"` | A valid adjustment set; the tuple `()` is a successful empty set |
| `LSASZeroEffect` | `conditioning`, `witness`, `rule`, `status="zero"` | X has no causal influence on Y; `conditioning` records a rule premise, not a claim about all adjustment sets |
| `LSASUnresolved` | `reason="no_rule_applies"`, `status="unknown"` | None of these local sufficient rules applies |

An R1 result does not assert a nonzero numerical average effect: valid adjustment
can yield zero through cancellation. The function returns one rule witness, not
all adjustment sets, and makes no minimum-cardinality or variance-optimality
promise. There is no search cap; subset search is exponential in blanket size.

The guarantee is graphical adjustment validity. Evaluating or estimating the
adjustment functional at particular treatment values additionally requires
suitable support/positivity; this selector receives no distribution or sample
with which to check it. This is a downstream requirement, not an extra condition
for the graph-level selection result.

## Small example

```python
from causal_graph_algorithms import DAG, find_lsas_adjustment, m_separated

graph = DAG.from_text("S -> X -> Y")
result = find_lsas_adjustment(
    graph.topology.nodes,
    treatment="X",
    outcome="Y",
    independent=lambda left, right, given: m_separated(graph, left, right, given),
)
assert result.status == "adjustment"
assert result.adjustment == ()
```

The graph supplies an exact oracle in this example. The operation itself is
given no edges or graph proof view and cannot verify the causal premises.

## Evidence

The [source fixtures](../../tests/fixtures/papers/li_2025_lsas.json) preserve paper,
upstream-code, and corrected outcomes separately. The
[contract tests](../../tests/python/test_lsas_selection.py) cover isolated X,
empty and nonempty adjustment, the D.4 proof example, unresolved outcomes,
strict oracle outputs, malformed inputs, immutability and query disjointness.

An [independent oracle](../../tests/python/lsas_oracles.py) expands bidirected edges
into latent common parents and tests separation by ancestral moralization. In
fixed-order three- and four-observed-node pretreatment ADMG universes, tests
check each returned adjustment by separation after deleting outgoing X edges,
and each zero result by absence of a directed X-to-Y path. The universe has
64 and 4,096 graphs respectively, including bows. It is not an exhaustive
test over all labelled four-node topological orders and does not prove general
completeness. These tests import no production separation or traversal helpers.

A separate four-observed-variable Gaussian example supplies its CI answers by
exact rational Schur complements of an analytically derived covariance matrix.
The selected adjustment {Z} gives coefficient 1, equal to the direct intervention
coefficient, while the unadjusted coefficient is 5/4. All 24 pairwise CI queries
agree with the independently constructed latent DAG. This separates one numeric
CI/causal-effect check from the traversal used in the graph-universe tests.

The earlier paper review's much larger census concerns the separately implemented
rules; it is not counted as execution evidence for this library function. The
pinned upstream implementation's failure is retained external reproduction
evidence, not recomputed by the package test suite.
