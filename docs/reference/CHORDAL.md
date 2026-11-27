# Chordal graph elimination

`chordal_elimination_forest` takes an `EndpointMarkedGraph` whose
edges are all undirected. Supply a perfect elimination order with
`order=`, or let the function compute one by reversing maximum
cardinality search. The search breaks ties in the graph's vertex order.
Every returned order has the property that each vertex's later neighbors
form a clique. Nonchordal input and invalid supplied orders raise
`GraphValidationError`. [Tarjan and Yannakakis (1984)](https://doi.org/10.1137/0213035)

The result has one bag per vertex, in elimination order. An
`EliminationBag` contains its `vertex` followed by its later neighbors
in `members`. Its `parent` is the earliest later neighbor, and the
`separator` is the intersection with the parent bag. A root has
`parent=None` and an empty separator.
Each graph component gives one tree and each isolated vertex gives one
singleton bag. The bags containing any one vertex form a connected
subtree. The forest can contain bags that are proper subsets of a
maximal clique. [Lauritzen, Speed and Vijayan (1984), Theorem 2](https://doi.org/10.1017/S1446788700027300)

```python
from causal_graph_algorithms import EndpointMarkedGraph, chordal_elimination_forest

graph = EndpointMarkedGraph.from_text("a --- b --- c --- d --- a; a --- c")
forest = chordal_elimination_forest(graph, order=("b", "d", "a", "c"))

assert forest.order == ("b", "d", "a", "c")
assert [bag.parent for bag in forest.bags] == ["a", "a", "c", None]
assert forest.bags[0].members == ("b", "a", "c")
assert forest.bags[0].separator == ("a", "c")
```
