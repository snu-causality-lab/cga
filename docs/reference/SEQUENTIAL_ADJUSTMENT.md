# Sequential adjustment in a known causal graph

These Python operations implement the corrected Jung–Park–Lee sequential criterion
and a forward constructor. The [paper record](https://github.com/snu-causality-lab/open-causal-science/blob/main/docs/papers/jung-park-lee-2024-sequential-adjustment.md)
distinguishes the published statement, the authors’ correction and the formalized scope.

## Public operations

```python
sequential_adjustment_is_valid(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    *, treatment_order: Sequence[str], outcomes: str | Iterable[str],
    covariate_blocks: Sequence[str | Iterable[str]],
) -> bool

find_minimal_sequential_adjustment(
    graph: AcyclicDirectedBidirectedGraph | DAG,
    *, treatment_order: Sequence[str], outcomes: str | Iterable[str],
) -> SequentialAdjustmentResult
```

The graph is a causal DAG or causal latent-projection ADMG, including bows. Every
graph vertex is observed; hidden common causes are represented by bidirected edges.
An arbitrary observational independence graph or MAG/PAG interpretation does not
satisfy this causal contract. Callers are responsible for that interpretation;
topology validation checks acyclicity, not whether a graph is causally correct.

`treatment_order` is an explicit nonempty sequence of distinct scalar treatment
vertices in topological order. It need not include every graph vertex. A bare
string or unordered set is rejected. `outcomes` is a nonempty disjoint vertex set.
Each outcome goes in block `Y_j`, where `j` is the largest 1-based index of a
treatment ancestor, or in `Y_0` if it has none. This schedule is computed, not
supplied by the caller.

The checker requires one covariate block per treatment. Blocks must be pairwise
disjoint, outside treatments/outcomes, and contain no descendant of any later
treatment in the original graph. Current-treatment descendants are permitted.
Repeated labels within a block are normalized as a set. Empty blocks are allowed.

The supplied expression is

```text
sum_z p(y_0) product_i p(z_i | h_{i-1}) p(y_i | h_{i-1}, x_i, z_i),
h_{i-1} = (x_{<i}, y_{<i}, z_{<i}).
```

`True` means universal validity of this expression over compatible positive causal
models. The model class permits real-valued Gaussian variables; necessity for a
separately fixed finite-alphabet model class is not asserted. Positivity/overlap for
the observational kernels is an assumption, not something these graph calls check.
`False` rejects the supplied allocation and says nothing about another formula.

## Construction and results

The constructor searches all eligible observed vertices for the declared treatment
order. It finalizes each block before building the next history, starts with
`H = Y_0`, and uses ancestors of the endpoints **and mandatory history** for its
constrained separator. It prunes candidate vertices in topology declaration order.
This order breaks ties; it does not replace the explicit treatment order.

`SequentialAdjustmentResult` is the union of two immutable, factory-created values:

| Result | Fields and meaning |
| --- | --- |
| `SequentialAdjustmentSequence` | `graph`, `treatment_order`, `outcome_blocks` (including `Y_0`), `covariate_blocks`; a valid sequence minimal under componentwise deletion |
| `SequentialAdjustmentNotFound` | `graph`, `treatment_order`, `outcome_blocks`, zero-based `failed_stage`, `reason="no_valid_sequence"`; no admissible sequence for this order |

Every block is a tuple in topology declaration order. Input graphs and collections
remain unchanged. Results support copying by identity and reject pickle and direct
construction. Minimality is stagewise inclusion minimality and hence componentwise
inclusion minimality. It is neither minimum total cardinality nor minimum variance.
Failure is confined to this formula class and order; a front-door formula, for
example, can identify an effect when no adjustment sequence exists.

The implementation uses repeated m-separation tests. It is a deterministic reference
constructor, **not** an implementation or benchmark of the cited linear-time
constrained-separator algorithm. There is no C backend or estimator for these calls.

## Errors and validation order

Both calls check the validated graph view, ordered treatment container, known
treatment/outcome labels, nonempty sets, treatment uniqueness, role disjointness,
and treatment topological order, in that order. The checker then checks the block
container/count and each block's labels, disjointness and future-treatment ancestry.
Unknown labels raise `GraphValidationError`; unsupported domains and inadmissible
allocations raise `SequentialAdjustmentDomainError`. A domain exception is not a
negative theorem result. The checker validates the whole allocation before testing
stage conditions, so an early graphical failure cannot hide malformed later blocks.

## Example

```python
from causal_graph_algorithms import (
    DAG, SequentialAdjustmentSequence, find_minimal_sequential_adjustment,
    sequential_adjustment_is_valid,
)

chain = DAG.from_text("X1 -> Z -> X2 -> Y")
assert sequential_adjustment_is_valid(
    chain, treatment_order=("X1", "X2"), outcomes="Y",
    covariate_blocks=({"Z"}, ()),
)

graph = DAG.from_text("X1 -> X2 -> Y; C -> X2; C -> Y")
result = find_minimal_sequential_adjustment(
    graph, treatment_order=("X1", "X2"), outcomes="Y",
)
assert isinstance(result, SequentialAdjustmentSequence)
assert result.covariate_blocks == ((), ("C",))
```

The executable [tests](../../tests/python/test_sequential_adjustment.py) cover these
examples, early outcomes, activated history colliders, bows, malformed inputs,
exhaustive small graphs and exact finite-SCM replay.
