# Chordal graph elimination

`chordal_elimination_forest` takes an undirected `EndpointMarkedGraph`.
With `order=`, it checks that the order lists every vertex once and that
each vertex's later neighbors form a clique. By default, it reverses
maximum cardinality search, breaking ties in input vertex order, and
checks the resulting order. A nonchordal graph or invalid order raises
`GraphValidationError`.
[Tarjan and Yannakakis (1984)](https://doi.org/10.1137/0213035)

`EliminationForest` contains the verified `order` and one entry per
vertex in `bags`. The bag's `members` are its vertex followed by its later
neighbors in that order. The earliest later neighbor is its `parent`, and the
`separator` is the intersection with the parent bag. Each graph
component has one root with `parent=None` and an empty separator.
Elimination bags can be proper subsets of maximal cliques.
[Blair and Peyton, §5.2](https://digital.library.unt.edu/ark:/67531/metadc1319152/m2/1/high_res_d/10145949.pdf#page=30)

Given a perfect elimination order, Open Causal Science proves that
the bags containing any one vertex form a connected subtree.
[OCS proof record](https://github.com/snu-causality-lab/open-causal-science/blob/7da03c981cf6f9e062d40e0685446235e9f11e49/docs/papers/lauritzen-speed-vijayan-1984-decomposable-graphs.md)

```python
from causal_graph_algorithms import EndpointMarkedGraph, chordal_elimination_forest

graph = EndpointMarkedGraph.from_text("a --- b --- c --- d --- a; a --- c")
forest = chordal_elimination_forest(graph, order=("b", "d", "a", "c"))

assert forest.order == ("b", "d", "a", "c")
assert [bag.parent for bag in forest.bags] == ["a", "a", "c", None]
assert forest.bags[0].members == ("b", "a", "c")
assert forest.bags[0].separator == ("a", "c")
```
