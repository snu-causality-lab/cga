# Conditional adjustment in a graph equivalence class

`conditional_adjustment_set_is_valid` and `find_conditional_adjustment_set`
implement LaPlante–Perković, [AISTATS 2024](https://proceedings.mlr.press/v238/laplante24a.html),
Definitions 1, 2 and 7, Theorems 3/9 and the constructions in Theorems 5(a)/10.

```python
from causal_graph_algorithms import DAG
from causal_graph_algorithms.conditional_adjustment import (
    ConditionalAdjustmentSet, find_conditional_adjustment_set,
)

graph = DAG.from_text("C -> X; C -> Y; X -> Y")
result = find_conditional_adjustment_set(
    graph, interventions="X", outcomes="Y", conditioned_on="C",
)
assert isinstance(result, ConditionalAdjustmentSet)
assert result.vertices == ()
```

The accepted graph is a factory-validated `DAG`, `CPDAG`, `MPDAG`, or no-selection
`CausalPAG`. Raw endpoint topology, raw directed-bidirected topology, MAGs and
selection PAGs are rejected. The labels X, Y and Z are pairwise disjoint, with
nonempty treatment/outcome sets; Z may be empty. Every conditioning vertex must
lie outside the graph's **possible descendants of X**. The condition is checked
using the source's nonlocal definition: a possible directed path has no arrowhead
at an earlier vertex from any later vertex, including a chord.

```text
conditional_adjustment_set_is_valid(
    graph, *, interventions, outcomes, conditioned_on, adjustment,
    max_path_states=100_000,
)
find_conditional_adjustment_set(
    graph, *, interventions, outcomes, conditioned_on,
    max_path_states=100_000,
)
```

These functions, `ConditionalAdjustmentSet`, `ConditionalAdjustmentNotFound`,
`ConditionalAdjustmentResult`, `ConditionalAdjustmentDomainError` and
`ConditionalAdjustmentResourceLimitError` are exported from
`causal_graph_algorithms`. A string means one vertex label; other iterables
supply label sets. DAG/CPDAG inputs use the MPDAG case of the criterion and
construction below. The candidate S is supplied as keyword `adjustment`.

The checker tests whether a supplied S identifies

```text
P(Y | do(X), Z) = sum_S P(Y | X, Z, S) P(S | Z).
```

An empty S gives `P(Y | X,Z)`. The result is complete only for this adjustment
functional under the nondescendant-conditioning premise. It is not a general
conditional-ID or CIDP algorithm. `False` rejects the supplied S and does not
reject every other adjustment set or every identifying formula. Query labels and
the graph interpretation are fully validated before mathematical testing; labels
outside the graph and malformed role overlaps raise `ConditionalAdjustmentDomainError`.
An S overlapping another query role is well formed but invalid and returns False.

The constructor returns the explicit source set. For an MPDAG it is
`[PossAn(X∪Y) ∪ An(Z)] \ [Forb(X,Y)∪X∪Y∪Z]`; for a PAG it replaces `An(Z)`
with `PossAn(Z)`. `ConditionalAdjustmentSet` contains the original graph/query,
the selected `vertices` and the immutable observational `formula` in the existing
probability IR. `ConditionalAdjustmentNotFound.reason` is `not_amenable` or
`no_adjustment_set`; both concern this adjustment class. No minimality, measurement
cost, efficiency or finite-sample claim is made. All represented labels are
eligible for construction; there is no implicit unmeasurable-variable mask.

The causal interpretation, model compatibility and positive-density/query-support
premises belong to the caller. The symbolic API has no distribution and cannot
check positivity or define off-support conditional kernels. No numerical estimator,
formula evaluator, C kernel, background-knowledge inference or learned graph is added.

Literal path traversal preserves the source definitions and can take exponential
time/space. `max_path_states=100_000` bounds cumulative enqueued path prefixes in
one call. The positive integer limit is explicit; exhaustion raises
`ConditionalAdjustmentResourceLimitError`, never a mathematical negative. This is
not a wall-time or whole-process memory bound. Returned values are frozen,
factory-created, copy by identity and reject pickle.

Verification is split between pinned Examples 1–4, source Figure 3/Example 6's
exact endpoint-roundtrip and `{V2,V4}` construction, synthetic PAG visibility
controls, independent three-node DAG-class adjustment tests and exact finite-law
controls.
