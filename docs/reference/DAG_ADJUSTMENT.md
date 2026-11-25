# Optimal covariate adjustment in a known DAG

## Overview

This module implements the DAG specialization of Henckel–Perković–Maathuis's
O-set and pruning procedure, together with Witte et al.'s forbidden projection.
It is a graph algorithm, not an estimator. CPDAG/maxPDAG construction, optimal IDA,
and PAG adjustment are not implemented by these functions.

## Sources and guarantees

- Henckel, Perković and Maathuis, *Graphical criteria for efficient total effect
  estimation via adjustment in causal linear models*. The inspected source is
  [arXiv:1907.02435v2](https://arxiv.org/abs/1907.02435v2), December 2020,
  which supplies the numbering below for
  Definition 3.12, Theorem 3.13, Algorithm 1 and Theorem 3.9.
- Witte, Henckel, Maathuis and Didelez, *On Efficient Adjustment in Causal Graphs*,
  [JMLR 21(246), 2020](https://jmlr.org/papers/v21/20-175.html), Definitions 4–5
  and Propositions 6–10. Two illustration/text discrepancies are documented in
  [the paper review](https://github.com/snu-causality-lab/open-causal-science/blob/main/docs/papers/witte-et-al-2020-efficient-adjustment.md); they are not counterexamples
  to the stated propositions.

The O-set minimizes each component's asymptotic OLS variance among valid
adjustment sets in the cited causal linear model. This is not a finite-sample MSE
guarantee, a claim about every estimator, or a matrix Loewner-order result.
The construction itself only reads the graph and requires no coefficients.
Nonparametric efficiency requires its own assumptions; see
[Smucler adjustment](SMUCLER_ADJUSTMENT.md).

## Public API

```python
from causal_graph_algorithms import (
    DAG, DAGOptimalAdjustmentSet,
    dag_adjustment_set_is_valid, dag_optimal_adjustment_set,
    dag_forbidden_projection, prune_dag_adjustment_set,
)

graph = DAG.from_text("I --> X --> M --> Y\nC --> X\nC --> M\nP --> Y")
result = dag_optimal_adjustment_set(graph, interventions="X", outcomes="Y")
assert isinstance(result, DAGOptimalAdjustmentSet)
assert result.vertices == ("C", "P")
assert prune_dag_adjustment_set(
    graph, interventions="X", outcomes="Y", adjustment=("I", "C", "P")
) == ("C", "P")
assert dag_adjustment_set_is_valid(
    graph, interventions="X", outcomes="Y", adjustment=result.vertices
)
projected = dag_forbidden_projection(graph, interventions="X", outcomes="Y")
```

All four functions require a validated `DAG`. Treatment and outcome selectors are
nonempty, disjoint strings or iterables of labels; a string is one complete label.
Results use graph declaration order, independently of set-input order.
Result records are sealed and factory-created; copies return the same immutable
value and pickle is unsupported.

| Operation | Result and interpretation |
| --- | --- |
| `dag_optimal_adjustment_set(graph, *, interventions, outcomes)` | `DAGOptimalAdjustmentSet` with `vertices`, `causal_nodes`, `forbidden_vertices`, and the graph/query; otherwise `DAGAdjustmentNotApplicable` |
| `dag_adjustment_set_is_valid(graph, *, interventions, outcomes, adjustment)` | Boolean complete adjustment criterion; supports multiple treatments/outcomes and non-descendant outcomes |
| `prune_dag_adjustment_set(graph, *, interventions, outcomes, adjustment)` | Tuple after Algorithm 1; input must already be valid. Does not necessarily return an inclusion-minimal set or the global O-set |
| `dag_forbidden_projection(graph, *, interventions, outcomes)` | `AcyclicDirectedBidirectedGraph`, possibly with bidirected edges or bows; not necessarily a DAG |

The Python O-set API assumes **every vertex is available** for measurement. A full DAG
with hidden vertices is accepted by the validity checker when the caller supplies
an observed candidate, but use `smucler_adjustment_set(..., observed=...)` when
optimizing under measurement restrictions.

`DAGAdjustmentNotApplicable.reason` separates:

- `outcomes_not_descendants`: some outcome is outside the theorem's descendant
  premise. `candidate_vertices` is only the algebraically constructed set, not a
  validity claim. A zero effect does not make unadjusted observational regression
  valid. No outcome is silently dropped.
- `no_valid_adjustment_set`: within the descendant premise, no valid adjustment
  set exists. This can occur with multiple interventions in a DAG and is not a
  claim of nonidentification by other methods.

Malformed domains and invalid pruning inputs raise `DAGAdjustmentDomainError`;
invalid/unknown vertex selectors follow graph validation errors.

## Construction and complexity

Let `cn` be the vertices after the first vertex on **proper** directed paths from
X to Y, and let `forb = de(cn) ∪ X`, with reflexive descendants. The O-set is
`pa(cn) \ forb`. Proper paths cannot pass through another intervention. The
implementation computes them by reachability in the induced graph excluding X.
Using `de(X) ∩ an(Y)` without this restriction is incorrect for joint interventions.

Validity checks the forbidden restriction and separation in the proper back-door
graph, removing just the first edge of each proper causal path. Pruning makes one
declaration-order pass through the candidate using the paper's conditional
independence test; Theorem 3.9 gives order invariance. Projection marginalizes
`forb \ (X ∪ Y)` using the existing latent-projection implementation.

Construction and validity use polynomial reachability/separation; pruning adds at
most one separation query per candidate vertex. Projection uses the existing
polynomial graph traversal implementation. No production function enumerates
adjustment subsets or simple paths. Exhaustive enumerations are test-only.

## Reproduced examples and independent checks

Exact source hashes, graph edges, printed claims and corrected expectations are in
[`henckel_witte_adjustment.json`](../../tests/fixtures/papers/henckel_witte_adjustment.json).
The figures were read visually before transcription. Executable checks are in
[`test_adjustment_paper_figures.py`](../../tests/python/test_adjustment_paper_figures.py).

| Source material | Coverage |
| --- | --- |
| Henckel Figures 1, 2a–b, 5, 7 | O-sets; all valid-set pruning cases for Figure 2a; Figure 5's exact unfaithful variance equality |
| Henckel Table 2 / Example 3.11 | Both specified linear models, all six printed population variances to two decimals; parameter-dependent limits of pruning |
| Witte Figures 1 and 3 | Full 12-node graph and exact forbidden projection, boxed O-set and three named valid alternatives |
| Witte Figure 2A–F | Every original/projection pair, retained vertices, directed/bidirected edges; B/F printed discrepancies retained, not silently adopted |
| Other figures | CPDAG/maxPDAG orientation/IDA and proof diagrams are outside this DAG operation; simulation and real-data plots are not reproduced as algorithm fixtures |

[`test_dag_adjustment.py`](../../tests/python/test_dag_adjustment.py) additionally
compares all 543 four-vertex DAGs against a literal proper/noncausal-path oracle:
26,064 singleton candidate validity checks, plus joint-query checks, exact rational
linear-SEM variance comparisons and pruning-order checks. The oracle imports no
production graph or separation helpers. Finite numerical checks do not prove the
theorem for all distributions.
