# ADMG Markov properties

## Head–tail partition

`admg_head_tail_partition` returns the head and tail of every factor in
Richardson's observational factorization for one ancestral margin. The
default margin is all graph vertices. Supply `ancestral_vertices` to
select a margin closed under directed parents. Both `DAG` and
`AcyclicDirectedBidirectedGraph` are supported.
[Richardson (2009), Theorem 4 and §3.2](https://arxiv.org/pdf/1406.6764#page=4)

Each round takes the barren vertices of each remaining bidirected district
and removes those heads. A vertex is barren in its district when none of
its strict directed descendants is in that district. Descendants are
computed in the original graph, including paths through vertices already
removed. The tail is the remainder of the district together with its
external directed parents. For an ancestral margin, each round's
district agrees with the head's district in its ancestral closure.
Head–tail pairs follow partition rounds and then the first vertex of
each district in graph order. Labels within each head and tail also
follow graph order.
[Richardson (2009), §2.2, Definition 5, Lemma 15 and Corollary 17](https://arxiv.org/pdf/1406.6764)

Each pair describes the scope of a factor `p(X_H | X_T)` for a
distribution satisfying the graph's global Markov property. For
Figure 1, the joint head `("X3", "X4")` has tail `("X1", "X2")`.
In Figure 2, each factor conditions on a vertex in the other head.
[Figures 1 and 2](https://proceedings.mlr.press/r7/richardson09a.html)

## Ordered-local statements

`admg_ordered_local_ci` returns a `CIStatement` for a vertex in a supplied
ancestral set. The vertex must have no directed child in the set. Its
blanket consists of its district in the induced graph, with the vertex
removed, and the district's external directed parents. The statement
asserts independence of the vertex from the rest of the ancestral set
given that blanket. When the remainder is empty, the statement is trivial.
[Richardson (2003), §3](https://doi.org/10.1111/1467-9469.00323)

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
