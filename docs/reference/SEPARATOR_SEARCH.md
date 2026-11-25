# Minimum-cost separators and adjustment-set enumeration

```python
from fractions import Fraction
from causal_graph_algorithms import (
    DAG, MinimumCostSeparator, find_minimum_cost_m_separator,
    find_minimum_cost_adjustment_set, iter_adjustment_sets,
)

graph = DAG.from_text("X --> A --> B --> Y")
result = find_minimum_cost_m_separator(
    graph, "X", "Y", costs={"A": 7, "B": Fraction(1, 3)},
)
assert isinstance(result, MinimumCostSeparator)
assert result.vertices == ("B",) and result.total_cost == Fraction(1, 3)

causal = DAG.from_text("Z --> X --> Y; Z --> Y; W --> Z")
best = find_minimum_cost_adjustment_set(
    causal, interventions="X", outcomes="Y", costs={"Z": 2, "W": 1},
)
assert best.vertices == ("Z",)
sets = tuple(iter_adjustment_sets(causal, interventions="X", outcomes="Y"))
assert {frozenset(s.vertices) for s in sets} == {frozenset({"Z"}), frozenset({"Z", "W"})}
```

## Operations and guarantees

All four operations accept `required=()` and `allowed=None`, interpreted as
`I ⊆ Z ⊆ R`. Query endpoints cannot be covariates. `None` permits every other
original vertex. Inputs are normalized eagerly; mutable inputs are copied.
Both endpoint sets must be nonempty and disjoint. Explicit endpoint covariates,
unknown labels and `required` outside `allowed` are malformed and raise
`SeparatorDomainError` (or `AdjustmentDomainError` in adjustment operations).
They are not scientific no-set results. A string denotes one vertex label;
other iterables supply label sets. All result/error/operation names in this
guide are exported from `causal_graph_algorithms`.

| Operation | Domain | Result |
| --- | --- | --- |
| `find_minimum_cost_m_separator(graph, left, right, *, costs=None, required=(), allowed=None, max_states=100_000)` | Validated DAG, acyclic bow-capable ADMG, ancestral graph or MAG | `MinimumCostSeparator` or `SeparatorNotFound` |
| `iter_m_separators(graph, left, right, *, required=(), allowed=None, minimal=False, max_states=100_000)` | Same separation domain | Lazy stream of existing `SeparatorSet` values |
| `find_minimum_cost_adjustment_set(graph, *, interventions, outcomes, costs=None, required=(), allowed=None, max_states=100_000)` | Existing ordinary-adjustment domain: DAG, known ADMG, no-selection MAG, no-selection `CausalPAG`, CPDAG | `MinimumCostAdjustment` or existing `AdjustmentNotFound` |
| `iter_adjustment_sets(graph, *, interventions, outcomes, required=(), allowed=None, minimal=False, max_states=100_000)` | Same ordinary-adjustment domain | Lazy stream of existing `AdjustmentSet` values |

Explicit costs must give exactly one strictly positive `int` or `Fraction` for
each allowed vertex, including required vertices and any forbidden covariates
that will subsequently be excluded. Floats, booleans, zero, negative weights,
missing and unknown keys are rejected. Omitted costs are all one. Returned
`total_cost` includes the required vertices and is an exact `Fraction`.
Results retain the graph, query, original constraints and copied cost table.
Weighted results are immutable, factory-created and not picklable, consistent
with existing semantic values. Ties are resolved deterministically by declared
node order; no particular lexicographically smallest optimum is promised.

`SeparatorNotFound` establishes no separator in the declared interval. An
`AdjustmentNotFound` establishes failure of ordinary adjustment in that interval,
not general causal nonidentification. Statistical positivity, graph-causal
interpretation and support conventions of ordinary adjustment remain caller
premises. No model fitting or effect evaluation takes place.

## Source and implementation

The source is van der Zander, Liśkiewicz and Textor,
[Sections 3.2 and 3.5 (2019)](https://arxiv.org/abs/1803.00116v3).
Minimum-cost search removes mandatory vertices from the augmented anterior
graph, splits the remaining vertices, and uses exact rational Edmonds–Karp flow.
Uncuttable vertices have a finite capacity strictly larger than the sum of all
admissible costs. Independent latent roots introduced for an ADMG remain
uncuttable. The result is globally minimum in the constrained separator domain.
Ordinary adjustment reuses the existing forbidden-set and proper-backdoor
reductions; those source-specific premises remain unchanged.

Enumeration implements the source's `ListSep` include/exclude recursion with an
exact constrained existence check at every state, using an explicit stack.
Include branches precede exclude branches in original node order. Both branches
recompute the appropriate ancestral reduction, so conditioning that activates a
collider is handled correctly. Every admissible set is yielded exactly once.
`minimal=True` retains only sets that are I-minimal relative to the original
required set. This is filtering of `ListSep`, not the source's Takata-based
`ListMinSep`, and has no polynomial-delay guarantee for minimal outputs.
I-minimal means that no proper subset which still contains every required
vertex is a separator; required vertices themselves need not be indispensable.

The [2023 corrigendum](https://doi.org/10.1016/j.artint.2023.103938) corrects
augmented-graph construction and dependent runtime claims. This implementation
does not claim the paper's cubic min-cut bound, its linear existence-check bound,
or a benchmark-derived speed guarantee. The graph transformations and exact
arithmetic have their own overhead.

`max_states` is a positive integer. In minimum-cost search it counts residual
BFS vertex expansions; in enumeration it counts popped include/exclude states.
It does not bound graph preprocessing, memory, rational bit lengths or time.
Exhaustion raises `SearchBudgetExceeded` and never yields a scientific negative.
An iterator can raise after earlier outputs: completeness requires normal
exhaustion. Copy input streams if they must be replayed.

## Examples and verification

[Figure 5 fixture](../../tests/fixtures/papers/zander_2019_separator_figure5.json)
pins the source PDF, page, rendered-figure edge transcription, printed result,
independently adjudicated result and correction status. The two minimal sets are
`{Z1}` and `{Z2}`; requiring `V1` makes `{V1,Z1}` and `{V1,Z2}` I-minimal.

[Tests](../../tests/python/test_separator_search.py) compare listing and exact
costs against original-graph simple paths and independent subset enumeration.
They include collider activation, mandatory costs, latent roots, rational
weights, eager validation, empty/impossible outcomes and interrupted streams.
