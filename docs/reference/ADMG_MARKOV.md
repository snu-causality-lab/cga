# ADMG Markov properties

Both operations take a `DAG` or `AcyclicDirectedBidirectedGraph`.
`ancestral_vertices` accepts a label or iterable of labels that includes
every directed parent of each selected vertex. Invalid inputs raise
`GraphValidationError`.

## Head–tail partition

`admg_head_tail_partition` returns a tuple of `ADMGHeadTail` terms for
one ancestral margin. The default margin contains every vertex. For a
distribution satisfying the graph's global Markov property, each pair
specifies the variables of one observational factor `p(X_H | X_T)`.
[Richardson (2009), Theorem 4](https://arxiv.org/pdf/1406.6764#page=4) ·
[Partition in §3.2](https://arxiv.org/pdf/1406.6764#page=5)

Each round removes one head per remaining bidirected district. The head
contains the district's barren vertices, which have no strict directed
descendant in that district. Directed paths can pass through vertices
removed earlier. The tail contains the rest of the district and its
external directed parents. Lemma 15 and Corollary 17 equate the current
district with the head's district in its ancestral closure. Within a round,
districts follow their first vertex in input graph order. The same order
applies to labels within each head and tail.
[Richardson (2009), head and tail in §3.1](https://arxiv.org/pdf/1406.6764#page=4) ·
[Lemma 15 and Corollary 17](https://arxiv.org/pdf/1406.6764#page=7)

Open Causal Science proves that the heads partition the selected
vertices and computes the terms in Figures 1 and 2. In Figure 2,
each factor conditions on a vertex in the other head.
[OCS proof record and figures](https://github.com/snu-causality-lab/open-causal-science/blob/7da03c981cf6f9e062d40e0685446235e9f11e49/docs/papers/richardson-2009-admg-factorization.md#lean-results)

## Ordered-local statements

`admg_ordered_local_ci` takes a member of the supplied ancestral set
with no directed child there. Its Markov blanket contains the other
members of its district in the induced graph and the district's external
directed parents. The returned `CIStatement` describes independence of
the vertex and the remaining vertices given that blanket. An empty
remainder gives a trivial statement. Open Causal Science proves the
blanket separation and the implication from global Markov to ordered-local
Markov.
[Richardson (2003), §3](https://doi.org/10.1111/1467-9469.00323) ·
[OCS proof record](https://github.com/snu-causality-lab/open-causal-science/blob/7da03c981cf6f9e062d40e0685446235e9f11e49/docs/papers/richardson-2003-ordered-local-markov.md#lean-results)

In Figure 1, the full margin has a joint head `("X3", "X4")` with tail
`("X1", "X2")`. In the margin with just `X1`, `X2`, and `X3`, the
ordered-local statement for `X3` is independence from `X2` given `X1`.
[Richardson (2009), Figure 1](https://arxiv.org/pdf/1406.6764#page=1)

```python
from causal_graph_algorithms import (
    ADMGHeadTail,
    AcyclicDirectedBidirectedGraph,
    CIStatement,
    admg_head_tail_partition,
    admg_ordered_local_ci,
)

graph = AcyclicDirectedBidirectedGraph.from_text(
    "X1; X2; X3; X4; X1 --> X3; X2 --> X4; X3 <-> X4"
)
assert admg_head_tail_partition(graph) == (
    ADMGHeadTail(("X1",), ()),
    ADMGHeadTail(("X2",), ()),
    ADMGHeadTail(("X3", "X4"), ("X1", "X2")),
)
assert admg_ordered_local_ci(
    graph, "X3", ancestral_vertices=("X1", "X2", "X3")
) == CIStatement("X3", "X2", "X1")
```
